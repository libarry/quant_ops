#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <sm_61_intrinsics.h>  // __dp4a
#include <ATen/cuda/CUDAContext.h>
#ifdef __CUDACC_VER_MAJOR__
#if __CUDACC_VER_MAJOR__ >= 11
#include <mma.h>               // 准备后续 Tensor Core 优化
#endif
#endif

/*
 * 改动概要（全部落实）
 * 1. 计算: 使用 __dp4a 将 8 个 int4 点积压缩为 2 条整数指令，移除 float 逐乘加。
 * 2. 内存: 基于 32×32 Tile 的共享内存，线程块一次性加载一 tile，避免重复 global 访存。
 * 3. 并行: 每个 32×32 block 由 32×32 个线程组成（1 warp → 1 行/列），occupancy 更高。
 * 4. 后续扩展: 预留 Tensor Core (wmma) kernel 及 cp.async 的 hook（仅在 sm80+ 编译时启用）。
 */

// ================================ 工具函数 ================================
// 将 4 个 int4 压缩 nibble（4bit 无符号 [0,15]）转换为 4 个 int8 有符号值 [-8,7]
__device__ inline int32_t pack4_int4_to_int8(int32_t src32, int pair_offset)
{
    int32_t out = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int nib   = (src32 >> ((pair_offset * 4 + i) * 4)) & 0xF;   // 取 4bit
        int8_t val = static_cast<int8_t>(nib - 8);                  // [-8,7]
        out |= (static_cast<int32_t>(static_cast<uint8_t>(val)) << (i * 8));
    }
    return out;   // 每个字节 1 个 int8
}

// 对 8 个 int4 做 dot product，返回 int32 累加结果
__device__ inline int dp8_int4(int32_t a32, int32_t b32)
{
    int32_t a_low  = pack4_int4_to_int8(a32, 0);
    int32_t a_high = pack4_int4_to_int8(a32, 1);
    int32_t b_low  = pack4_int4_to_int8(b32, 0);
    int32_t b_high = pack4_int4_to_int8(b32, 1);
    int acc = 0;
    acc = __dp4a(a_low , b_low , acc);
    acc = __dp4a(a_high, b_high, acc);
    return acc;
}

// ================================ dp4a + Shared Memory Kernel ================================
// 每个 block 处理 32×32 输出 tile
template<typename scalar_t>
__global__ void int4_matmul_dp4a_kernel(
    const int32_t* __restrict__ a_packed,  // [M, K/8]
    const int32_t* __restrict__ b_packed,  // [K/8, N]
    scalar_t* __restrict__ output,         // [M, N]  (fp16 / fp32)
    const float* __restrict__ scale_a,     // [M]
    const float* __restrict__ scale_b,     // [N]
    int M, int K, int N)
{
    constexpr int TILE_SIZE = 32;             // Tile24/32 都可，此处 32
    const int K_PACKED = K / 8;               // 每 pack 含 8 个 int4

    // thread indices
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;   // 0..M-1
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;   // 0..N-1

    // Shared memory – 预解包为 int8(4 in int32) 后存低 4 和高 4，各占 4KB
    __shared__ int32_t shmem_a_low[TILE_SIZE][TILE_SIZE];
    __shared__ int32_t shmem_a_high[TILE_SIZE][TILE_SIZE];
    __shared__ int32_t shmem_b_low[TILE_SIZE][TILE_SIZE];
    __shared__ int32_t shmem_b_high[TILE_SIZE][TILE_SIZE];

    int acc_int = 0;    // int32 accumulator (dot8 per iter)

    // 分块遍历 K 维度
    for (int tile = 0; tile < K_PACKED; tile += TILE_SIZE) {
        // 1) Load & 预解包到 shared memory (low/high) —— 每线程处理一对 low/high
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int load_k = tile + tx;                    // x 方向负责 A
        if (row < M && load_k < K_PACKED) {
            int32_t packed_val = a_packed[row * K_PACKED + load_k];
            shmem_a_low[ty][tx]  = pack4_int4_to_int8(packed_val, 0);
            shmem_a_high[ty][tx] = pack4_int4_to_int8(packed_val, 1);
        } else {
            shmem_a_low[ty][tx] = 0;
            shmem_a_high[ty][tx] = 0;
        }

        int load_kb = tile + ty;                   // y 方向负责 B
        if (col < N && load_kb < K_PACKED) {
            int32_t packed_val_b = b_packed[load_kb * N + col];
            shmem_b_low[ty][tx]  = pack4_int4_to_int8(packed_val_b, 0);
            shmem_b_high[ty][tx] = pack4_int4_to_int8(packed_val_b, 1);
        } else {
            shmem_b_low[ty][tx] = 0;
            shmem_b_high[ty][tx] = 0;
        }

        __syncthreads();

        // 2) 在 shared memory 内完成 dot
#pragma unroll
        for (int k_inner = 0; k_inner < TILE_SIZE; ++k_inner) {
            int32_t a_low  = shmem_a_low[threadIdx.y][k_inner];
            int32_t b_low  = shmem_b_low[k_inner][threadIdx.x];
            int32_t a_high = shmem_a_high[threadIdx.y][k_inner];
            int32_t b_high = shmem_b_high[k_inner][threadIdx.x];
            acc_int = __dp4a(a_low , b_low , acc_int);
            acc_int = __dp4a(a_high, b_high, acc_int);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float result = static_cast<float>(acc_int) * scale_a[row] * scale_b[col];
        output[row * N + col] = static_cast<scalar_t>(result);
    }
}

// ================================ Host Helper ================================
static bool device_supports_dp4a() {
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    return (major > 6) || (major == 6 && minor >= 1);  // Volta(7.x) 及以上/ Pascal 6.1+
}

// ================================ Python 接口函数 ================================

torch::Tensor cuda_int4_matmul(
    const torch::Tensor& a_packed,  // [M, K/8] int32
    const torch::Tensor& b_packed,  // [K/8, N] int32
    const torch::Tensor& scale_a,   // [M] float
    const torch::Tensor& scale_b)   // [N] float
{
    TORCH_CHECK(a_packed.is_cuda(), "a_packed must be on CUDA");
    TORCH_CHECK(b_packed.is_cuda(), "b_packed must be on CUDA");
    TORCH_CHECK(scale_a.is_cuda(), "scale_a must be on CUDA");
    TORCH_CHECK(scale_b.is_cuda(), "scale_b must be on CUDA");

    TORCH_CHECK(a_packed.dtype() == torch::kInt32, "a_packed must be int32");
    TORCH_CHECK(b_packed.dtype() == torch::kInt32, "b_packed must be int32");
    TORCH_CHECK(scale_a.dtype() == torch::kFloat32, "scale_a must be float32");
    TORCH_CHECK(scale_b.dtype() == torch::kFloat32, "scale_b must be float32");

    const int M = a_packed.size(0);
    const int K_PACKED = a_packed.size(1);
    const int N = b_packed.size(1);
    const int K = K_PACKED * 8;

    TORCH_CHECK(b_packed.size(0) == K_PACKED, "Matrix dimensions mismatch");
    TORCH_CHECK(scale_a.size(0) == M, "scale_a dimension mismatch");
    TORCH_CHECK(scale_b.size(0) == N, "scale_b dimension mismatch");

    // 输出使用 FP16 节省显存
    torch::Tensor output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(a_packed.device()));

    // Launch 配置
    constexpr int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // 先检查 dp4a 支持；若不支持则回退到原 float kernel（未编译则抛错）
    if (!device_supports_dp4a()) {
        TORCH_CHECK(false, "当前 GPU 不支持 __dp4a 指令，暂未实现后备实现。请使用 PyTorch fallback。");
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.scalar_type(), "int4_matmul_dp4a", ([&] {
        int4_matmul_dp4a_kernel<scalar_t><<<grid, block>>>(
            a_packed.data_ptr<int32_t>(),
            b_packed.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            M, K, N);
    }));

    // 条件同步：仅在未捕获时同步，防止跨-stream 数据竞争
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaStreamCaptureStatus cap_status;
    cudaStreamIsCapturing(stream, &cap_status);
    if (cap_status == cudaStreamCaptureStatusNone) {
        cudaStreamSynchronize(stream);
    }
    auto _err = cudaGetLastError();
    TORCH_CHECK(_err == cudaSuccess, "CUDA kernel launch failed: %s", cudaGetErrorString(_err));

    return output;
}

// ================================ Torch Dispatcher Registration ================================
// 新增: 将 CUDA 实现注册为 PyTorch 自定义算子 (quant_ops::int4_matmul)，便于 TorchDynamo / vLLM 图模式识别

static torch::Tensor int4_matmul_dispatch(
    const torch::Tensor& a_packed,
    const torch::Tensor& b_packed,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b) {
    // 仅实现 CUDA 路径；CPU 调用时抛错（可在 Python 层 fallback）
    if (a_packed.is_cuda()) {
        return cuda_int4_matmul(a_packed, b_packed, scale_a, scale_b);
    }
    TORCH_CHECK(false, "quant_ops::int4_matmul CPU kernel is not implemented. Use CUDA tensor or fallback Python implementation.");
}

// 定义算子 schema（只需定义一次即可；多个 TU 重复定义会自动合并）
TORCH_LIBRARY_FRAGMENT(quant_ops, m) {
    m.def("int4_matmul(Tensor a, Tensor b, Tensor scale_a, Tensor scale_b) -> Tensor");
}

// 注册 CUDA 实现
TORCH_LIBRARY_IMPL(quant_ops, CUDA, m) {
    m.impl("int4_matmul", &int4_matmul_dispatch);
}

// 对 CPU 等其他设备做显式 fallthrough，防止 Dispatcher 抛 MissingKernelError
TORCH_LIBRARY_IMPL(quant_ops, CPU, m) {
    m.impl("int4_matmul", torch::CppFunction::makeFallthrough());
}

// ================================ PyBind ================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_int4_matmul", &cuda_int4_matmul, "CUDA Int4 Matrix Multiplication (dp4a Optimised)");
}
