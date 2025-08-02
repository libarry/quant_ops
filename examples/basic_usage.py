#!/usr/bin/env python3
"""
FlatQuant 量化算子基础使用示例

这个脚本展示了如何使用FlatQuant动态量化算子。
"""

import torch
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from quant_ops import flatquant_dynamic_quantize, dequantize_int4, get_decompose_dim, CUDA_AVAILABLE


def main():
    print("FlatQuant 动态量化示例")
    print("=" * 50)
    
    # 检查环境
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"CUDA 算子可用: {CUDA_AVAILABLE}")
    
    if not torch.cuda.is_available():
        print("错误: 需要CUDA支持")
        return
    
    # 设置参数
    batch_tokens = 128
    features = 4096
    M, N = get_decompose_dim(features)
    
    print(f"\n配置参数:")
    print(f"  batch_tokens: {batch_tokens}")
    print(f"  features: {features}")
    print(f"  分解维度: M={M}, N={N}")
    
    # 创建测试数据
    print(f"\n创建测试数据...")
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_tokens, features, dtype=torch.float16, device='cuda')
    left_trans = torch.randn(M, M, dtype=torch.float16, device='cuda')
    right_trans = torch.randn(N, N, dtype=torch.float16, device='cuda')
    
    print(f"  输入张量形状: {input_tensor.shape}")
    print(f"  左变换矩阵形状: {left_trans.shape}")
    print(f"  右变换矩阵形状: {right_trans.shape}")
    
    # 执行量化 (int8存储)
    print(f"\n执行量化 (int8存储)...")
    quantized_int8, scales = flatquant_dynamic_quantize(
        input_tensor,
        left_trans,
        right_trans,
        clip_ratio=1.0,
        pack_int32=False,
        use_cuda=CUDA_AVAILABLE
    )
    
    print(f"  量化结果形状: {quantized_int8.shape}")
    print(f"  量化数据类型: {quantized_int8.dtype}")
    print(f"  Scale形状: {scales.shape}")
    print(f"  量化值范围: [{quantized_int8.min().item()}, {quantized_int8.max().item()}]")
    
    # 反量化
    print(f"\n执行反量化...")
    dequantized_int8 = dequantize_int4(quantized_int8, scales, packed_int32=False)
    print(f"  反量化结果形状: {dequantized_int8.shape}")
    print(f"  反量化数据类型: {dequantized_int8.dtype}")
    
    # 计算量化误差
    error = (input_tensor.float() - dequantized_int8).abs()
    relative_error = (error / input_tensor.float().abs().clamp(min=1e-8)).mean()
    print(f"  平均绝对误差: {error.mean().item():.6f}")
    print(f"  最大绝对误差: {error.max().item():.6f}")
    print(f"  平均相对误差: {relative_error.item():.6f}")
    
    # 执行量化 (int32打包存储)
    if features % 8 == 0:
        print(f"\n执行量化 (int32打包存储)...")
        quantized_int32, scales_packed = flatquant_dynamic_quantize(
            input_tensor,
            left_trans,
            right_trans,
            clip_ratio=1.0,
            pack_int32=True,
            use_cuda=CUDA_AVAILABLE
        )
        
        print(f"  打包量化结果形状: {quantized_int32.shape}")
        print(f"  打包数据类型: {quantized_int32.dtype}")
        
        # 计算存储效率
        memory_int8 = quantized_int8.numel() * quantized_int8.element_size()
        memory_int32 = quantized_int32.numel() * quantized_int32.element_size()
        compression_ratio = memory_int8 / memory_int32
        
        print(f"  存储效率:")
        print(f"    int8存储: {memory_int8} bytes")
        print(f"    int32打包: {memory_int32} bytes")
        print(f"    压缩比: {compression_ratio:.2f}x")
        
        # 反量化打包数据
        dequantized_int32 = dequantize_int4(quantized_int32, scales_packed, packed_int32=True)
        
        # 验证打包前后一致性
        consistency_error = (dequantized_int8 - dequantized_int32).abs()
        print(f"  打包一致性:")
        print(f"    最大差异: {consistency_error.max().item():.6f}")
        print(f"    平均差异: {consistency_error.mean().item():.6f}")
    
    print(f"\n✅ 示例运行完成！")


if __name__ == "__main__":
    main() 