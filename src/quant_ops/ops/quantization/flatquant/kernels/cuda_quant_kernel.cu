#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

// 每个 block 处理的最大 token 数
#define MAX_TOKENS_PER_BLOCK 256
#define MAX_FEATURES_PER_BLOCK 1024
#define WARP_SIZE 32

// 辅助函数：将不同精度转换为 float
__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(half x) { return __half2float(x); }
__device__ inline float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

// 原子操作：安全的 float atomicMax
__device__ inline float atomicMax_float(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// 原子操作：安全的 float atomicMin
__device__ inline float atomicMin_float(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Kronecker 矩阵乘法 CUDA 内核
template<typename T>
__global__ void kronecker_matmul_kernel(
    const T* __restrict__ input,        // [batch_tokens, M, N]
    const T* __restrict__ left_trans,   // [M, M]
    const T* __restrict__ right_trans,  // [N, N]
    T* __restrict__ output,             // [batch_tokens, M, N]
    int batch_tokens,
    int M,
    int N
) {
    int token_idx = blockIdx.x;
    if (token_idx >= batch_tokens) return;
    
    int thread_idx = threadIdx.x;
    int threads_per_block = blockDim.x;
    
    // 每个 block 处理一个 token
    const T* input_token = input + token_idx * M * N;
    T* output_token = output + token_idx * M * N;
    
    // 第一步：x @ right_trans^T (即 input @ right_trans，因为我们要 right_trans^T)
    // 计算 [M, N] @ [N, N] = [M, N]
    for (int m = 0; m < M; m++) {
        for (int n = thread_idx; n < N; n += threads_per_block) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += to_float(input_token[m * N + k]) * to_float(right_trans[k * N + n]);
            }
            output_token[m * N + n] = (T)sum;
        }
    }
    
    __syncthreads();
    
    // 第二步：left_trans^T @ temp
    // 计算 [M, M] @ [M, N] = [M, N]
    for (int m = thread_idx; m < M; m += threads_per_block) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < M; k++) {
                sum += to_float(left_trans[k * M + m]) * to_float(output_token[k * N + n]);
            }
            output_token[m * N + n] = (T)sum;
        }
    }
}

// 计算动态量化参数 (per-token scale)
template<typename T>
__global__ void compute_dynamic_scale_kernel(
    const T* __restrict__ input,        // [batch_tokens, features]
    float* __restrict__ scales,         // [batch_tokens]
    float clip_ratio,
    int batch_tokens,
    int features
) {
    int token_idx = blockIdx.x;
    if (token_idx >= batch_tokens) return;
    
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int threads_per_block = blockDim.x;
    
    const T* input_token = input + token_idx * features;
    
    // 第一步：计算每个 token 的最大最小值
    float local_max = -FLT_MAX;
    float local_min = FLT_MAX;
    
    for (int i = tid; i < features; i += threads_per_block) {
        float val = to_float(input_token[i]);
        local_max = fmaxf(local_max, val);
        local_min = fminf(local_min, val);
    }
    
    // 使用 shared memory 进行 reduction
    sdata[tid] = local_max;
    sdata[tid + threads_per_block] = local_min;
    __syncthreads();
    
    // Reduction for max
    for (int s = threads_per_block / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            sdata[tid + threads_per_block] = fminf(sdata[tid + threads_per_block], 
                                                   sdata[tid + threads_per_block + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float xmax = fmaxf(sdata[0], 0.0f);
        float xmin = fminf(sdata[threads_per_block], 0.0f);
        
        // 应用 clip_ratio
        xmax = xmax * clip_ratio;
        xmin = xmin * clip_ratio;
        
        // 对称量化：使用绝对值最大值
        float abs_max = fmaxf(fabsf(xmin), xmax);
        float scale = abs_max / 7.0f;  // int4 对称量化范围 [-8, 7]
        scale = fmaxf(scale, 1e-8f);   // 避免除零
        
        scales[token_idx] = scale;
    }
}

// 执行量化到 int4 并存储为 int8
template<typename T>
__global__ void quantize_to_int4_kernel(
    const T* __restrict__ input,        // [batch_tokens, features]
    const float* __restrict__ scales,   // [batch_tokens]
    int8_t* __restrict__ output,        // [batch_tokens, features] 存储为 int8
    int batch_tokens,
    int features
) {
    int token_idx = blockIdx.x;
    int feature_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (token_idx >= batch_tokens || feature_idx >= features) return;
    
    float scale = scales[token_idx];
    float val = to_float(input[token_idx * features + feature_idx]);
    
    // 量化：q = round(x / scale)，范围 [-8, 7]
    int quantized = __float2int_rn(val / scale);
    quantized = (quantized < -8) ? -8 : ((quantized > 7) ? 7 : quantized);
    
    output[token_idx * features + feature_idx] = (int8_t)quantized;
}

// 执行量化到 int4 并打包为 int32
template<typename T>
__global__ void quantize_and_pack_to_int32_kernel(
    const T* __restrict__ input,        // [batch_tokens, features]
    const float* __restrict__ scales,   // [batch_tokens]
    int32_t* __restrict__ output,       // [batch_tokens, features//8] 打包存储
    int batch_tokens,
    int features
) {
    int token_idx = blockIdx.x;
    int pack_idx = threadIdx.x + blockIdx.y * blockDim.x;
    int features_packed = features / 8;
    
    if (token_idx >= batch_tokens || pack_idx >= features_packed) return;
    
    float scale = scales[token_idx];
    int32_t packed_val = 0;
    
    // 处理8个连续的特征，打包到一个 int32 中
    for (int i = 0; i < 8; i++) {
        int feature_idx = pack_idx * 8 + i;
        if (feature_idx < features) {
            float val = to_float(input[token_idx * features + feature_idx]);
            
            // 量化：q = round(x / scale)，范围 [-8, 7]
            int quantized = __float2int_rn(val / scale);
            quantized = (quantized < -8) ? -8 : ((quantized > 7) ? 7 : quantized);
            
            // 转换为 uint4 [0, 15] 用于打包
            uint32_t uint4_val = (uint32_t)(quantized + 8);
            packed_val |= (uint4_val << (i * 4));
        }
    }
    
    output[token_idx * features_packed + pack_idx] = packed_val;
}

// C++ 包装函数
std::tuple<torch::Tensor, torch::Tensor> cuda_kronecker_quant_int8(
    torch::Tensor input,           // [batch_tokens, M*N] or [batch_tokens, M, N]
    torch::Tensor left_trans,      // [M, M]
    torch::Tensor right_trans,     // [N, N]
    float clip_ratio,
    bool pack_int32 = false
) {
    auto batch_tokens = input.size(0);
    auto M = left_trans.size(0);
    auto N = right_trans.size(0);
    auto features = M * N;
    
    TORCH_CHECK(input.size(-1) == features, "Input last dimension must equal M*N");
    TORCH_CHECK(features % 8 == 0 || !pack_int32, "Features must be divisible by 8 for int32 packing");
    
    // 确保输入是连续的并在 GPU 上
    input = input.contiguous().cuda();
    left_trans = left_trans.contiguous().cuda();
    right_trans = right_trans.contiguous().cuda();
    
    // 重塑输入为 [batch_tokens, M, N]
    auto input_reshaped = input.view({batch_tokens, M, N});
    
    // 创建输出张量
    auto transformed = torch::zeros_like(input_reshaped);
    auto scales = torch::zeros({batch_tokens}, torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    
    // 设置 CUDA grid 和 block 维度
    dim3 grid_kron(batch_tokens);
    dim3 block_kron((features < 256) ? features : 256);
    
    dim3 grid_scale(batch_tokens);
    dim3 block_scale((features < 512) ? features : 512);
    int shared_mem_size = 2 * block_scale.x * sizeof(float);
    
    // 执行 Kronecker 矩阵乘法
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "kronecker_matmul", ([&] {
        kronecker_matmul_kernel<scalar_t><<<grid_kron, block_kron>>>(
            input_reshaped.data_ptr<scalar_t>(),
            left_trans.data_ptr<scalar_t>(),
            right_trans.data_ptr<scalar_t>(),
            transformed.data_ptr<scalar_t>(),
            batch_tokens, M, N
        );
    }));
    
    // 计算动态量化参数
    auto transformed_flat = transformed.view({batch_tokens, features});
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "compute_dynamic_scale", ([&] {
        compute_dynamic_scale_kernel<scalar_t><<<grid_scale, block_scale, shared_mem_size>>>(
            transformed_flat.data_ptr<scalar_t>(),
            scales.data_ptr<float>(),
            clip_ratio,
            batch_tokens, features
        );
    }));
    
    torch::Tensor quantized_output;
    
    if (pack_int32) {
        // 打包为 int32
        auto features_packed = features / 8;
        quantized_output = torch::zeros({batch_tokens, features_packed}, 
                                      torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
        
        dim3 grid_quant(batch_tokens, (features_packed + 255) / 256);
        dim3 block_quant((features_packed < 256) ? features_packed : 256);
        
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "quantize_and_pack_to_int32", ([&] {
            quantize_and_pack_to_int32_kernel<scalar_t><<<grid_quant, block_quant>>>(
                transformed_flat.data_ptr<scalar_t>(),
                scales.data_ptr<float>(),
                quantized_output.data_ptr<int32_t>(),
                batch_tokens, features
            );
        }));
    } else {
        // 存储为 int8
        quantized_output = torch::zeros({batch_tokens, features}, 
                                      torch::TensorOptions().dtype(torch::kInt8).device(input.device()));
        
        dim3 grid_quant(batch_tokens, (features + 255) / 256);
        dim3 block_quant((features < 256) ? features : 256);
        
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "quantize_to_int4", ([&] {
            quantize_to_int4_kernel<scalar_t><<<grid_quant, block_quant>>>(
                transformed_flat.data_ptr<scalar_t>(),
                scales.data_ptr<float>(),
                quantized_output.data_ptr<int8_t>(),
                batch_tokens, features
            );
        }));
    }
    
    cudaDeviceSynchronize();
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel execution failed");
    
    return std::make_tuple(quantized_output, scales);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_kronecker_quant_int8", &cuda_kronecker_quant_int8, "CUDA Kronecker Quantization to Int4");
} 