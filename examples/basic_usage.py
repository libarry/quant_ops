#!/usr/bin/env python3
"""
FlatQuant CUDA vs PyTorch 性能和精度对比
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from quant_ops.ops.quantization.flatquant.ops import flatquant_cuda, flatquant_pytorch
from quant_ops.ops.quantization.flatquant.kernels import CUDA_KERNELS_AVAILABLE


def main():
    print("FlatQuant CUDA vs PyTorch 对比测试")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("错误: 需要CUDA支持")
        return
    
    print(f"CUDA 算子可用: {CUDA_KERNELS_AVAILABLE}")
    
    # 测试参数
    batch_tokens = 512
    features = 4096
    M, N = 64, 64  # M * N = features
    
    print(f"测试配置: batch_tokens={batch_tokens}, features={features}")
    
    # 创建测试数据
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_tokens, features, dtype=torch.float16, device='cuda') 
    left_trans = torch.randn(M, M, dtype=torch.float16, device='cuda') / M
    right_trans = torch.randn(N, N, dtype=torch.float16, device='cuda') / N
    
    # 预热 PyTorch 实现
    print("预热 PyTorch 实现...")
    for _ in range(5):
        _ = flatquant_pytorch(input_tensor, left_trans, right_trans)
    torch.cuda.synchronize()
    
    if CUDA_KERNELS_AVAILABLE:
        print("预热 CUDA 实现...")
        for _ in range(5):
            _ = flatquant_cuda(input_tensor, left_trans, right_trans)
        torch.cuda.synchronize()
    
    # 性能测试
    n_runs = 20
    
    # PyTorch 性能测试
    print("\n测试 PyTorch 实现...")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        pytorch_quantized, pytorch_scales = flatquant_pytorch(input_tensor, left_trans, right_trans)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / n_runs
    
    print(f"PyTorch 实现:")
    print(f"  时间: {pytorch_time*1000:.2f}ms")
    print(f"  量化结果形状: {pytorch_quantized.shape}")
    print(f"  量化值范围: [{pytorch_quantized.min().item()}, {pytorch_quantized.max().item()}]")
    print(f"  Scale 形状: {pytorch_scales.shape}")
    
    if CUDA_KERNELS_AVAILABLE:
        # CUDA 性能测试
        print("\n测试 CUDA 实现...")
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_runs):
            cuda_quantized, cuda_scales = flatquant_cuda(input_tensor, left_trans, right_trans)
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) / n_runs

        print(f"CUDA 实现:")
        print(f"  时间: {cuda_time*1000:.2f}ms")
        print(f"  加速比: {pytorch_time/cuda_time:.1f}x")

        # 精度对比
        max_diff_quantized = (cuda_quantized.float() - pytorch_quantized.float()).abs().max().item()
        max_diff_scales = (cuda_scales - pytorch_scales).abs().max().item()
        mean_diff_quantized = (cuda_quantized.float() - pytorch_quantized.float()).abs().mean().item()
        mean_diff_scales = (cuda_scales - pytorch_scales).abs().mean().item()
        
        print(f"\n精度对比:")
        print(f"  量化结果最大差异: {max_diff_quantized:.6f}")
        print(f"  量化结果平均差异: {mean_diff_quantized:.6f}")
        print(f"  Scale最大差异:    {max_diff_scales:.6f}")
        print(f"  Scale平均差异:    {mean_diff_scales:.6f}")
    else:
        print("\n⚠️ CUDA 算子不可用，仅测试 PyTorch 实现")
    
    print(f"\n✅ 测试完成")


if __name__ == "__main__":
    main() 