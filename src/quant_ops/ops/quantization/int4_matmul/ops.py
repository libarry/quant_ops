"""
Int4 Matrix Multiplication 算子实现
"""

import torch
import warnings
from typing import Tuple

from .kernels import cuda_int4_matmul_ops, CUDA_KERNELS_AVAILABLE

# 允许 torch.compile 捕获 C++ int4 matmul PyCapsule
try:
    import torch
    if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'allow_in_graph'):
        torch.compiler.allow_in_graph(cuda_int4_matmul_ops.cuda_int4_matmul)
    else:
        from torch._dynamo import allow_in_graph  # type: ignore
        allow_in_graph(cuda_int4_matmul_ops.cuda_int4_matmul)
except Exception:
    pass


def pack_int4_to_int32(tensor: torch.Tensor) -> torch.Tensor:
    """
    将int4 tensor (存储为int8, 值范围[-8,7]) 打包为int32
    
    Args:
        tensor: [batch, features] int8 tensor，值范围 [-8, 7]
        
    Returns:
        packed tensor: [batch, features//8] int32 tensor
    """
    batch, features = tensor.shape
    assert features % 8 == 0, f"features ({features}) must be divisible by 8"
    
    # 将 [-8, 7] 转换为 [0, 15]
    tensor_uint4 = (tensor + 8).to(torch.uint8)
    tensor_reshaped = tensor_uint4.view(batch, features // 8, 8)
    
    # 打包为int32
    packed = torch.zeros(batch, features // 8, dtype=torch.int32, device=tensor.device)
    for i in range(8):
        packed += (tensor_reshaped[:, :, i].to(torch.int32) << (i * 4))
    
    return packed


def unpack_int32_to_int4(packed: torch.Tensor) -> torch.Tensor:
    """
    将int32 tensor解包为int4 tensor
    
    Args:
        packed: [batch, features//8] int32 tensor
        
    Returns:
        unpacked tensor: [batch, features] int8 tensor，值范围 [-8, 7]
    """
    batch, features_packed = packed.shape
    features = features_packed * 8
    
    unpacked = torch.zeros(batch, features, dtype=torch.int8, device=packed.device)
    for i in range(8):
        mask = (packed >> (i * 4)) & 0xF
        unpacked[:, i::8] = (mask - 8).to(torch.int8)
    
    return unpacked


def int4_matmul_pytorch(
    a_packed: torch.Tensor,  # [M, K//8] int32
    b_packed: torch.Tensor,  # [K//8, N] int32  
    scale_a: torch.Tensor,   # [M] float32
    scale_b: torch.Tensor    # [N] float32
) -> torch.Tensor:
    """
    Int4 矩阵乘法的PyTorch实现
    
    Args:
        a_packed: 矩阵A，打包的int4数据 [M, K//8]
        b_packed: 矩阵B，打包的int4数据 [K//8, N]
        scale_a: A的per-row scale [M]
        scale_b: B的per-column scale [N]
        
    Returns:
        output: 矩阵乘法结果 [M, N]
    """
    M, K_packed = a_packed.shape
    _, N = b_packed.shape
    K = K_packed * 8
    
    # 1. 解包int32到int4 (存储为int8)
    a_int4 = unpack_int32_to_int4(a_packed.view(-1, K_packed)).view(M, K)
    b_int4 = unpack_int32_to_int4(b_packed.T).T.view(K, N)
    
    # 2. 执行矩阵乘法 (转换为float以避免溢出)
    result = torch.matmul(a_int4.float(), b_int4.float())
    
    # 3. 应用per-channel scale进行反量化
    scale_a_expanded = scale_a.view(M, 1)  # [M, 1]
    scale_b_expanded = scale_b.view(1, N)  # [1, N]
    result = result * scale_a_expanded * scale_b_expanded
    
    return result.half()  # 返回fp16以节省内存


def int4_matmul_cuda(
    a_packed: torch.Tensor,  # [M, K//8] int32
    b_packed: torch.Tensor,  # [K//8, N] int32
    scale_a: torch.Tensor,   # [M] float32
    scale_b: torch.Tensor    # [N] float32
) -> torch.Tensor:
    """
    Int4 矩阵乘法的CUDA实现
    """
    if not CUDA_KERNELS_AVAILABLE or cuda_int4_matmul_ops is None:
        raise RuntimeError("CUDA 算子不可用，请编译 CUDA 扩展或使用 int4_matmul_pytorch")
    
    # 确保在 GPU 上
    a_packed = a_packed.cuda()
    b_packed = b_packed.cuda()
    scale_a = scale_a.cuda()
    scale_b = scale_b.cuda()

    # --- 连续性预检 ---
    if not a_packed.is_contiguous():
        warnings.warn("a_packed 张量非连续，已自动调用 contiguous()，可能产生额外的拷贝开销。", RuntimeWarning)
        a_packed = a_packed.contiguous()
    if not b_packed.is_contiguous():
        warnings.warn("b_packed 张量非连续，已自动调用 contiguous()，可能产生额外的拷贝开销。", RuntimeWarning)
        b_packed = b_packed.contiguous()
    if not scale_a.is_contiguous():
        scale_a = scale_a.contiguous()
    if not scale_b.is_contiguous():
        scale_b = scale_b.contiguous()
    
    # 统一走 Dispatcher，FakeTensor 会命中 meta kernel；实张量自动落到 CUDA 实现
    return torch.ops.quant_ops.int4_matmul(a_packed, b_packed, scale_a, scale_b)


def int4_matmul(
    a_packed: torch.Tensor,  # [M, K//8] int32
    b_packed: torch.Tensor,  # [K//8, N] int32
    scale_a: torch.Tensor,   # [M] float32
    scale_b: torch.Tensor    # [N] float32
) -> torch.Tensor:
    """
    Int4 矩阵乘法主接口 - 自动选择最佳实现
    """
    # 优先使用通过 C++ Dispatcher 注册的自定义算子 —— 这可以被 TorchDynamo / vLLM 捕获为原子op
    try:
        return torch.ops.quant_ops.int4_matmul(a_packed, b_packed, scale_a, scale_b)
    except (RuntimeError, AttributeError):
        # 若算子尚未成功加载，则回退到原有实现
        if CUDA_KERNELS_AVAILABLE and cuda_int4_matmul_ops is not None:
            return int4_matmul_cuda(a_packed, b_packed, scale_a, scale_b)
        else:
            return int4_matmul_pytorch(a_packed, b_packed, scale_a, scale_b)

