#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

// 每个 block 处理的最大 token 数
#define MAX_TOKENS_PER_BLOCK 256
#define MAX_FEATURES_PER_BLOCK 4096
#define WARP_SIZE 32

// 移除类型转换函数，直接使用原始数据类型进行计算

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

// 分步骤的 Kronecker 矩阵乘法 CUDA 内核 (先右乘再左乘)
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
    
    int tid = threadIdx.x;
    int threads_per_block = blockDim.x;
    
    // 使用共享内存存储中间结果
    extern __shared__ char smem[];
    T* temp_matrix = (T*)smem;
    
    // 每个 block 处理一个 token
    const T* input_token = input + token_idx * M * N;
    T* output_token = output + token_idx * M * N;
    
    // 第一步：计算 input @ right_trans，结果存储在共享内存中
    // input[M, N] @ right_trans[N, N] = temp[M, N]
    for (int m = 0; m < M; m++) {
        for (int n = tid; n < N; n += threads_per_block) {
            T sum = T(0);
            for (int k = 0; k < N; k++) {
                sum += input_token[m * N + k] * right_trans[k * N + n];
            }
            temp_matrix[m * N + n] = sum;
        }
    }
    
    __syncthreads();
    
    // 第二步：计算 left_trans^T @ temp，写入输出
    // left_trans^T[M, M] @ temp[M, N] = output[M, N]
    for (int m = tid; m < M; m += threads_per_block) {
        for (int n = 0; n < N; n++) {
            T sum = T(0);
            for (int k = 0; k < M; k++) {
                sum += left_trans[k * M + m] * temp_matrix[k * N + n];
            }
            output_token[m * N + n] = sum;
        }
    }
}

// 优化的动态量化参数计算 (使用warp-level primitives)
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
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int threads_per_block = blockDim.x;
    int warps_per_block = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;
    
    extern __shared__ float sdata[];
    float* s_max = sdata;
    float* s_min = sdata + warps_per_block;
    
    const T* input_token = input + token_idx * features;
    
    // 第一步：每个线程计算局部最大最小值
    float local_max = -FLT_MAX;
    float local_min = FLT_MAX;
    
    for (int i = tid; i < features; i += threads_per_block) {
        float val = static_cast<float>(input_token[i]);
        local_max = fmaxf(local_max, val);
        local_min = fminf(local_min, val);
    }
    
    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        local_min = fminf(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
    }
    
    // 每个warp的第一个线程写入shared memory
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
        s_min[warp_id] = local_min;
    }
    
    __syncthreads();
    
    // 第一个warp对shared memory进行最终reduction
    if (warp_id == 0) {
        local_max = (lane_id < warps_per_block) ? s_max[lane_id] : -FLT_MAX;
        local_min = (lane_id < warps_per_block) ? s_min[lane_id] : FLT_MAX;
        
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
            local_min = fminf(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
        }
        
        if (lane_id == 0) {
            // 应用 clip_ratio, 逻辑与 PyTorch 保持一致
            float xmax_val = local_max * clip_ratio;
            float xmin_val = local_min * clip_ratio;
            
            // 对称量化：使用绝对值最大值
            float abs_max = fmaxf(fabsf(xmin_val), xmax_val);
            float scale = abs_max / 7.0f;  // int4 对称量化范围 [-8, 7]
            scale = fmaxf(scale, 1e-8f);   // 避免除零
            
            scales[token_idx] = scale;
        }
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
    float val = static_cast<float>(input[token_idx * features + feature_idx]);
    
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
            float val = static_cast<float>(input[token_idx * features + feature_idx]);
            
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
    
    // 优化的 CUDA grid 和 block 维度配置
    // Kronecker 矩阵乘法：使用适中的线程数和共享内存
    dim3 grid_kron(batch_tokens);
    dim3 block_kron(std::min(256, std::max(32, (int)N)));  // 1D block配置，至少32个线程
    size_t kron_shared_mem = M * N * sizeof(float);  // 为中间结果分配共享内存
    
    // 动态量化参数计算：优化线程数量，更好利用warp
    dim3 grid_scale(batch_tokens);
    int scale_threads = std::min(512, (int)(((features + 31) / 32) * 32));  // 对齐到warp边界
    dim3 block_scale(scale_threads);
    int warps_per_block = (scale_threads + 31) / 32;
    size_t scale_shared_mem = 2 * warps_per_block * sizeof(float);
    
    // 执行 Kronecker 矩阵乘法
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "kronecker_matmul", ([&] {
        kronecker_matmul_kernel<scalar_t><<<grid_kron, block_kron, kron_shared_mem>>>(
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
        compute_dynamic_scale_kernel<scalar_t><<<grid_scale, block_scale, scale_shared_mem>>>(
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