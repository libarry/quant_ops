"""
FlatQuant 动态量化算子实现 - 精简版
"""

import torch
from typing import Tuple
from torch.autograd import Function

from .kernels import cuda_quant_ops, CUDA_KERNELS_AVAILABLE


class _FlatQuantCUDA(Function):
    """
    FlatQuant CUDA aotugrad function
    """
    @staticmethod
    def forward(ctx, input_tensor, left_trans, right_trans, clip_ratio, pack_int32):
        # 无需保存 ctx，因为 backward 不会被实现
        return cuda_quant_ops.cuda_kronecker_quant_int8(
            input_tensor, left_trans, right_trans, clip_ratio, pack_int32
        )

    @staticmethod
    def backward(ctx, grad_output, grad_scales):
        # 推理场景下，量化操作无需梯度
        return None, None, None, None, None


def flatquant_cuda(
    input_tensor: torch.Tensor,
    left_trans: torch.Tensor,
    right_trans: torch.Tensor,
    clip_ratio: float = 1.0,
    pack_int32: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FlatQuant CUDA 算子实现
    """
    if not CUDA_KERNELS_AVAILABLE or cuda_quant_ops is None:
        raise RuntimeError("CUDA 算子不可用，请编译 CUDA 扩展或使用 flatquant_pytorch")
    
    # 确保在 GPU 上
    input_tensor = input_tensor.cuda()
    left_trans = left_trans.cuda()
    right_trans = right_trans.cuda()
    
    # 调用 autograd function
    return _FlatQuantCUDA.apply(
        input_tensor, left_trans, right_trans, clip_ratio, pack_int32
    )


def flatquant_pytorch(
    input_tensor: torch.Tensor,
    left_trans: torch.Tensor,
    right_trans: torch.Tensor,
    clip_ratio: float = 1.0,
    pack_int32: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FlatQuant PyTorch 实现
    
    Args:
        input_tensor: 输入张量 [batch_tokens, features]
        left_trans: 左变换矩阵 [M, M]
        right_trans: 右变换矩阵 [N, N]
        clip_ratio: 量化截断比例
        pack_int32: 是否打包为 int32
        
    Returns:
        Tuple[量化结果, scale值]
    """
    # 1. Kronecker 矩阵乘法
    init_shape = input_tensor.shape
    x = input_tensor.reshape(-1, left_trans.shape[0], right_trans.shape[0])
    x = torch.matmul(x, right_trans)
    x = torch.matmul(left_trans.T, x)
    x_transformed = x.reshape(init_shape)
    
    # 2. 计算动态量化参数 (per-token)
    batch_tokens, features = x_transformed.shape
    reshaped_x = x_transformed.view(batch_tokens, -1)
    xmax = reshaped_x.amax(1, keepdim=True)
    xmin = reshaped_x.amin(1, keepdim=True)
    
    # 应用 clip_ratio 和对称量化
    xmax = torch.maximum(torch.abs(xmin * clip_ratio), xmax * clip_ratio)
    scale_per_token = (xmax / 7.0).clamp(min=1e-8).squeeze(1)  # int4 对称量化范围 [-8, 7]
    
    # 3. 量化
    scale_expanded = scale_per_token.view(-1, 1).expand_as(x_transformed)
    x_quantized = torch.round(x_transformed / scale_expanded).clamp(-8, 7)
    
    if pack_int32:
        # 打包到 int32
        x_quantized_reshaped = x_quantized.view(batch_tokens, features // 8, 8)
        x_uint4 = (x_quantized + 8).to(torch.uint8).view(batch_tokens, features // 8, 8)
        
        packed = torch.zeros(batch_tokens, features // 8, dtype=torch.int32, device=input_tensor.device)
        for i in range(8):
            packed += (x_uint4[:, :, i].to(torch.int32) << (i * 4))
        
        return packed, scale_per_token
    else:
        return x_quantized.to(torch.int8), scale_per_token


def flatquant_dynamic_quantize(
    input_tensor: torch.Tensor,
    left_trans: torch.Tensor,
    right_trans: torch.Tensor,
    clip_ratio: float = 1.0,
    pack_int32: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FlatQuant 动态量化主接口 - 默认使用 CUDA 算子
    """
    # 优先调用使用 C++ Dispatcher 注册的算子，便于 graph 捕获
    try:
        return torch.ops.quant_ops.flatquant_dynamic_quantize(
            input_tensor, left_trans, right_trans, float(clip_ratio), pack_int32
        )
    except (RuntimeError, AttributeError):
        # Dispatcher 未加载时回退
        if CUDA_KERNELS_AVAILABLE and cuda_quant_ops is not None:
            return flatquant_cuda(input_tensor, left_trans, right_trans, clip_ratio, pack_int32)
        else:
            return flatquant_pytorch(input_tensor, left_trans, right_trans, clip_ratio, pack_int32)


def dequantize_int4(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    packed_int32: bool = False
) -> torch.Tensor:
    """
    反量化 int4 数据回浮点数
    """
    if packed_int32:
        # 解包 int32 到 int4
        batch_tokens, features_packed = quantized.shape
        features = features_packed * 8
        
        unpacked = torch.zeros(batch_tokens, features, dtype=torch.int8, device=quantized.device)
        for i in range(8):
            mask = (quantized >> (i * 4)) & 0xF
            unpacked[:, i::8] = (mask - 8).to(torch.int8)
        
        quantized = unpacked
    
    # 反量化
    scales_expanded = scales.view(-1, 1).expand_as(quantized)
    return quantized.float() * scales_expanded 