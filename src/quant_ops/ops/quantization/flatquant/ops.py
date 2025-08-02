"""
FlatQuant 动态量化算子实现

提供FlatQuant算法的完整实现，包括CUDA加速版本和PyTorch后备实现。
"""

import torch
from typing import Tuple, Optional

from .kernels import cuda_quant_ops, CUDA_KERNELS_AVAILABLE
from ....utils.decompose import get_decompose_dim


def kronecker_matmul_torch(x: torch.Tensor, hadL: torch.Tensor, hadR: torch.Tensor) -> torch.Tensor:
    """
    Kronecker 乘积矩阵乘法的 PyTorch 实现 (作为后备方案)
    
    Args:
        x: 输入张量
        hadL: 左变换矩阵 
        hadR: 右变换矩阵
        
    Returns:
        变换后的张量
    """
    init_shape = x.shape
    x = x.reshape(-1, hadL.shape[0], hadR.shape[0])
    x = torch.matmul(x, hadR)
    x = torch.matmul(hadL.T, x)
    return x.reshape(init_shape)


def get_scale_zero_torch(x: torch.Tensor, clip_ratio: float, sym: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    获取动态量化的 scale 和 zero_point 参数的 PyTorch 实现 (作为后备方案)
    
    Args:
        x: 输入张量
        clip_ratio: 裁剪比例
        sym: 是否使用对称量化
        
    Returns:
        Tuple[scale, zero_point]
    """
    q_max, q_min = (7, -8) if sym else (15, 0)  # int4 范围
    
    init_shape = x.shape
    reshaped_x = x.reshape((-1, x.shape[-1]))
    xmax, xmin = reshaped_x.amax(1, keepdim=True), reshaped_x.amin(1, keepdim=True)
    tmp = torch.zeros_like(xmax)
    xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)

    # 应用 clip_ratio
    xmax = xmax * clip_ratio
    xmin = xmin * clip_ratio
    
    if sym:
        # 对称量化：使用绝对值最大值
        xmax = torch.maximum(torch.abs(xmin), xmax)
        scale = (xmax / q_max).clamp(min=1e-8)  # 避免除零
        scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
        zero = torch.zeros_like(scale)
    else:
        # 非对称量化
        scale = (xmax - xmin) / (q_max - q_min)
        zero = torch.round(-xmin / scale)
        scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)   
        zero = zero.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

    return scale, zero


def flatquant_dynamic_quantize(
    input_tensor: torch.Tensor,
    left_trans: torch.Tensor,
    right_trans: torch.Tensor,
    clip_ratio: float = 1.0,
    pack_int32: bool = False,
    use_cuda: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    执行 FlatQuant 动态量化到 int4
    
    Args:
        input_tensor: 输入张量 [batch_tokens, features]，支持 fp16/bf16/fp32
        left_trans: 左变换矩阵 [M, M]
        right_trans: 右变换矩阵 [N, N]  
        clip_ratio: 量化截断比例
        pack_int32: 是否打包为 int32 (8个 int4 打包到 1个 int32)，否则使用 int8 存储
        use_cuda: 是否使用 CUDA 加速
        
    Returns:
        Tuple[量化结果, scale值]
        - 量化结果: [batch_tokens, features] (int8) 或 [batch_tokens, features//8] (int32)
        - scale值: [batch_tokens] (float32)
    """
    if not input_tensor.is_cuda:
        input_tensor = input_tensor.cuda()
    if not left_trans.is_cuda:
        left_trans = left_trans.cuda()
    if not right_trans.is_cuda:
        right_trans = right_trans.cuda()
    
    # 检查维度兼容性
    M, N = left_trans.shape[0], right_trans.shape[0]
    features = input_tensor.shape[-1]
    assert M * N == features, f"变换矩阵维度不匹配: M({M}) * N({N}) != features({features})"
    
    if pack_int32:
        assert features % 8 == 0, f"pack_int32=True 时特征维度必须是8的倍数，当前为 {features}"
    
    if CUDA_KERNELS_AVAILABLE and use_cuda and cuda_quant_ops is not None:
        # 使用 CUDA 加速版本
        try:
            quantized, scales = cuda_quant_ops.cuda_kronecker_quant_int8(
                input_tensor, left_trans, right_trans, clip_ratio, pack_int32
            )
            return quantized, scales
        except Exception as e:
            print(f"CUDA 算子执行失败，切换到 PyTorch 实现: {e}")
            use_cuda = False
    
    if not use_cuda:
        # 使用 PyTorch 后备实现
        print("使用 PyTorch 后备实现...")
        
        # 1. Kronecker 矩阵乘法变换
        x_transformed = kronecker_matmul_torch(input_tensor, left_trans, right_trans)
        
        # 2. 计算动态量化参数
        scale, _ = get_scale_zero_torch(x_transformed, clip_ratio, sym=True)
        
        # 3. 量化到 int4 范围 [-8, 7]
        x_quantized = torch.round(x_transformed / scale).clamp(-8, 7)
        
        # 计算 per-token 的 scale
        batch_tokens = x_transformed.shape[0]
        scale_per_token = scale.view(batch_tokens, -1).amax(dim=1)  # 每个 token 的最大 scale
        
        if pack_int32:
            # 手动打包到 int32
            x_quantized_reshaped = x_quantized.view(batch_tokens, features // 8, 8)
            # 转换为 uint4 [0, 15] 用于打包
            x_uint4 = (x_quantized + 8).to(torch.uint8)
            x_uint4_reshaped = x_uint4.view(batch_tokens, features // 8, 8)
            
            # 打包到 int32
            packed = torch.zeros(batch_tokens, features // 8, dtype=torch.int32, device=input_tensor.device)
            for i in range(8):
                packed += (x_uint4_reshaped[:, :, i].to(torch.int32) << (i * 4))
            
            return packed, scale_per_token
        else:
            # 直接返回 int8 存储的结果
            return x_quantized.to(torch.int8), scale_per_token


def dequantize_int4(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    packed_int32: bool = False
) -> torch.Tensor:
    """
    反量化 int4 数据回浮点数
    
    Args:
        quantized: 量化后的数据
        scales: 量化 scale 值
        packed_int32: 输入是否为 int32 打包格式
        
    Returns:
        反量化后的浮点张量
    """
    if packed_int32:
        # 解包 int32 到 int4
        batch_tokens, features_packed = quantized.shape
        features = features_packed * 8
        
        # 解包
        unpacked = torch.zeros(batch_tokens, features, dtype=torch.int8, device=quantized.device)
        for i in range(8):
            mask = (quantized >> (i * 4)) & 0xF  # 提取第 i 个 4-bit
            unpacked[:, i::8] = (mask - 8).to(torch.int8)  # 转换回 [-8, 7]
        
        quantized = unpacked
    
    # 反量化
    scales_expanded = scales.view(-1, 1).expand_as(quantized)
    return quantized.float() * scales_expanded 