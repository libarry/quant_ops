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

// ================== 融合动态量化 + 量化/打包 Kernel ==================
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

template <typename T, bool PACK_INT32>
__global__ void scale_quant_kernel(
    const T* __restrict__ input,      // [batch_tokens, features]
    float* __restrict__ scales,       // [batch_tokens]
    int8_t* __restrict__ out_int8,    // 可为 nullptr
    int32_t* __restrict__ out_int32,  // 可为 nullptr
    float clip_ratio,
    int features)
{
    int token_idx = blockIdx.x;
    int tid       = threadIdx.x;

    const int warps_per_block = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    extern __shared__ float sdata[];
    float* s_max = sdata;
    float* s_min = sdata + warps_per_block;

    const T* input_token = input + token_idx * features;

    // 1. 局部 min/max
    float local_max = -FLT_MAX;
    float local_min =  FLT_MAX;
    for (int idx = tid; idx < features; idx += blockDim.x) {
        float v = static_cast<float>(input_token[idx]);
        local_max = fmaxf(local_max, v);
        local_min = fminf(local_min, v);
    }

    // 2. warp 规约
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        local_min = fminf(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
    }

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
        s_min[warp_id] = local_min;
    }
    __syncthreads();

    // 3. block 规约并计算 scale
    if (warp_id == 0) {
        local_max = (lane_id < warps_per_block) ? s_max[lane_id] : -FLT_MAX;
        local_min = (lane_id < warps_per_block) ? s_min[lane_id] :  FLT_MAX;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
            local_min = fminf(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
        }
        if (lane_id == 0) {
            float abs_max = fmaxf(fabsf(local_min * clip_ratio), local_max * clip_ratio);
            float scale   = fmaxf(abs_max / 7.f, 1e-8f);
            scales[token_idx] = scale;
            s_max[0] = scale; // reuse shared mem
        }
    }
    __syncthreads();
    float scale = s_max[0];

    // 4. 量化 & 可选打包
    if (!PACK_INT32) {
        for (int idx = tid; idx < features; idx += blockDim.x) {
            int q = __float2int_rn(static_cast<float>(input_token[idx]) / scale);
            q = (q < -8) ? -8 : ((q > 7) ? 7 : q);
            out_int8[token_idx * features + idx] = static_cast<int8_t>(q);
        }
    } else {
        int packs = features / 8;
        for (int pidx = tid; pidx < packs; pidx += blockDim.x) {
            int32_t pack_val = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float v = static_cast<float>(input_token[pidx * 8 + i]);
                int q = __float2int_rn(v / scale);
                q = (q < -8) ? -8 : ((q > 7) ? 7 : q);
                uint32_t u4 = static_cast<uint32_t>(q + 8);
                pack_val |= (u4 << (i * 4));
            }
            out_int32[token_idx * packs + pidx] = pack_val;
        }
    }
}

// C++ 包装函数 (重写，使用 cuBLAS + 融合量化)
std::tuple<torch::Tensor, torch::Tensor> cuda_kronecker_quant_int8(
    torch::Tensor input,           // [batch_tokens, M*N] 或 [B, M, N]
    torch::Tensor left_trans,      // [M, M]
    torch::Tensor right_trans,     // [N, N]
    float clip_ratio,
    bool pack_int32 = false
) {
    // 维度检查
    const int64_t batch_tokens = input.size(0);
    const int64_t M = left_trans.size(0);
    const int64_t N = right_trans.size(0);
    const int64_t features = M * N;
    TORCH_CHECK(input.size(-1) == features, "Input last dimension must equal M*N");
    TORCH_CHECK(features % 8 == 0 || !pack_int32, "Features must be divisible by 8 for int32 packing");

    // 保证数据在 GPU 且连续
    input       = input.contiguous().cuda();
    left_trans  = left_trans.contiguous().cuda();
    right_trans = right_trans.contiguous().cuda();

    // 1. Kronecker 变换 —— 使用 cuBLAS (at::matmul)
    auto input_3d = input.view({batch_tokens, M, N});
    auto transformed = torch::matmul(input_3d, right_trans);              // [B, M, N]
    transformed      = torch::matmul(left_trans.transpose(0, 1), transformed); // [B, M, N]

    auto transformed_flat = transformed.view({batch_tokens, features});

    // 2. 分配输出
    auto scales = torch::zeros({batch_tokens}, torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    torch::Tensor quantized_output;
    if (pack_int32) {
        quantized_output = torch::empty({batch_tokens, features / 8}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
    } else {
        quantized_output = torch::empty({batch_tokens, features}, torch::TensorOptions().dtype(torch::kInt8).device(input.device()));
    }

    // 3. 启动融合 kernel
    dim3 grid(batch_tokens);
    int threads = std::min(512, static_cast<int>(((features + 31) / 32) * 32)); // 对齐到 warp
    dim3 block(threads);
    int warps_per_block = (threads + 31) / 32;
    size_t shared_mem = 2 * warps_per_block * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "scale_quant_fused", ([&] {
        if (pack_int32) {
            scale_quant_kernel<scalar_t, true><<<grid, block, shared_mem>>>(
                transformed_flat.data_ptr<scalar_t>(),
                scales.data_ptr<float>(),
                nullptr,
                quantized_output.data_ptr<int32_t>(),
                clip_ratio,
                features);
        } else {
            scale_quant_kernel<scalar_t, false><<<grid, block, shared_mem>>>(
                transformed_flat.data_ptr<scalar_t>(),
                scales.data_ptr<float>(),
                quantized_output.data_ptr<int8_t>(),
                nullptr,
                clip_ratio,
                features);
        }
    }));

    cudaDeviceSynchronize();
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel execution failed");

    return std::make_tuple(quantized_output, scales);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_kronecker_quant_int8", &cuda_kronecker_quant_int8, "CUDA Kronecker Quantization to Int4");
} 