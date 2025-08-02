#!/usr/bin/env python3
"""
FlatQuant 精度测试模块

专门用于测试CUDA实现与PyTorch参考实现的精度一致性。
"""

import torch
import sys
import os

# 添加src目录到Python路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from quant_ops import (
    flatquant_dynamic_quantize, 
    dequantize_int4, 
    get_decompose_dim,
    CUDA_AVAILABLE
)


def test_basic_accuracy():
    """基础精度测试"""
    print("=" * 60)
    print("基础精度测试")
    print("=" * 60)
    
    # 测试参数
    batch_tokens = 128
    features = 4096
    M, N = get_decompose_dim(features)
    
    print(f"测试配置: batch_tokens={batch_tokens}, features={features}, M={M}, N={N}")
    
    # 创建测试数据
    torch.manual_seed(42)
    input_fp16 = torch.randn(batch_tokens, features, dtype=torch.float16, device='cuda')
    left_trans = torch.randn(M, M, dtype=torch.float16, device='cuda')
    right_trans = torch.randn(N, N, dtype=torch.float16, device='cuda')
    clip_ratio = 1.0
    
    # PyTorch 实现
    print("\n测试 PyTorch 参考实现...")
    torch_quant, torch_scales = flatquant_dynamic_quantize(
        input_fp16, left_trans, right_trans, clip_ratio, pack_int32=False, use_cuda=False
    )
    torch_dequant = dequantize_int4(torch_quant, torch_scales, packed_int32=False)
    
    if not CUDA_AVAILABLE:
        print("CUDA 实现不可用，跳过精度对比")
        return True
    
    # CUDA 实现
    print("测试 CUDA 加速实现...")
    try:
        cuda_quant, cuda_scales = flatquant_dynamic_quantize(
            input_fp16, left_trans, right_trans, clip_ratio, pack_int32=False, use_cuda=True
        )
        cuda_dequant = dequantize_int4(cuda_quant, cuda_scales, packed_int32=False)
        
        # 精度对比
        print("\n精度对比结果:")
        quant_diff = (cuda_quant.float() - torch_quant.float()).abs()
        scale_diff = (cuda_scales - torch_scales).abs()
        dequant_diff = (cuda_dequant - torch_dequant).abs()
        
        print(f"量化结果最大差异: {quant_diff.max().item():.6f}")
        print(f"量化结果平均差异: {quant_diff.mean().item():.6f}")
        print(f"Scale 最大差异: {scale_diff.max().item():.6f}")
        print(f"Scale 平均差异: {scale_diff.mean().item():.6f}")
        print(f"反量化结果最大差异: {dequant_diff.max().item():.6f}")
        print(f"反量化结果平均差异: {dequant_diff.mean().item():.6f}")
        print(f"反量化结果相对误差: {(dequant_diff / torch_dequant.abs().clamp(min=1e-8)).mean().item():.6f}")
        
        # 判断精度是否可接受
        if quant_diff.max() <= 2 and scale_diff.max() <= 1e-4:
            print("✅ 精度测试通过")
            return True
        else:
            print("❌ 精度测试失败")
            return False
            
    except Exception as e:
        print(f"CUDA 实现执行失败: {e}")
        return False


def test_packing_accuracy():
    """测试打包功能精度"""
    print("\n" + "=" * 60)
    print("Int32 打包功能精度测试")
    print("=" * 60)
    
    if not CUDA_AVAILABLE:
        print("CUDA 实现不可用，跳过打包测试")
        return True
    
    batch_tokens, features = 128, 4096
    torch.manual_seed(42)
    M, N = get_decompose_dim(features)
    input_tensor = torch.randn(batch_tokens, features, dtype=torch.float16, device='cuda')
    left_trans = torch.randn(M, M, dtype=torch.float16, device='cuda') 
    right_trans = torch.randn(N, N, dtype=torch.float16, device='cuda')
    clip_ratio = 1.0
    
    try:
        # 测试不打包
        quant_int8, scales_int8 = flatquant_dynamic_quantize(
            input_tensor, left_trans, right_trans, clip_ratio, pack_int32=False, use_cuda=True
        )
        dequant_int8 = dequantize_int4(quant_int8, scales_int8, packed_int32=False)
        
        # 测试打包
        quant_int32, scales_int32 = flatquant_dynamic_quantize(
            input_tensor, left_trans, right_trans, clip_ratio, pack_int32=True, use_cuda=True
        )
        dequant_int32 = dequantize_int4(quant_int32, scales_int32, packed_int32=True)
        
        # 对比两种方式的结果
        dequant_diff = (dequant_int8 - dequant_int32).abs()
        scale_diff = (scales_int8 - scales_int32).abs()
        
        print(f"打包前后反量化结果差异 - 最大: {dequant_diff.max().item():.6f}, 平均: {dequant_diff.mean().item():.6f}")
        print(f"打包前后Scale值差异 - 最大: {scale_diff.max().item():.6f}, 平均: {scale_diff.mean().item():.6f}")
        
        # 检查存储效率
        memory_int8 = quant_int8.numel() * quant_int8.element_size()
        memory_int32 = quant_int32.numel() * quant_int32.element_size()
        compression_ratio = memory_int8 / memory_int32
        
        print(f"存储效率: int8={memory_int8}bytes, int32={memory_int32}bytes, 压缩比={compression_ratio:.2f}x")
        
        if dequant_diff.max() <= 1e-6 and compression_ratio >= 1.9:  # 理论上应该是2x压缩
            print("✅ 打包功能测试通过")
            return True
        else:
            print("❌ 打包功能测试失败")
            return False
            
    except Exception as e:
        print(f"打包功能测试失败: {e}")
        return False


def main():
    """运行所有精度测试"""
    print("FlatQuant 精度测试")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"CUDA 算子可用: {CUDA_AVAILABLE}")
    
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用，无法运行测试")
        return False
    
    # 运行测试
    all_passed = True
    all_passed &= test_basic_accuracy()
    all_passed &= test_packing_accuracy()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有精度测试通过！")
    else:
        print("❌ 部分测试失败")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 