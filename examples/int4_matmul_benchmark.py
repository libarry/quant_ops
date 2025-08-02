#!/usr/bin/env python3
"""
Int4 矩阵乘法 CUDA vs PyTorch 性能和精度对比 (简化版)
"""
import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from quant_ops.ops.quantization.int4_matmul.ops import int4_matmul_cuda, int4_matmul_pytorch, pack_int4_to_int32
from quant_ops.ops.quantization.int4_matmul.kernels import CUDA_KERNELS_AVAILABLE

def generate_test_data(M: int, K: int, N: int, device: str = 'cuda'):
    """生成测试数据"""
    torch.manual_seed(42)
    a_original = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=device)
    b_original = torch.randint(-8, 8, (K, N), dtype=torch.int8, device=device)
    
    a_packed = pack_int4_to_int32(a_original).contiguous()
    b_packed = pack_int4_to_int32(b_original.T).T.contiguous()
    
    scale_a = torch.rand(M, device=device) * 0.1 + 0.01
    scale_b = torch.rand(N, device=device) * 0.1 + 0.01
    
    return a_packed, b_packed, scale_a, scale_b, a_original, b_original

def benchmark_pytorch_baseline(a_original, b_original, scale_a, scale_b, n_runs=10):
    """PyTorch标准fp16矩阵乘法基准"""
    a_fp16 = a_original.half() 
    b_fp16 = b_original.half()
    
    # 预热
    for _ in range(5):
        _ = torch.matmul(a_fp16, b_fp16)
    torch.cuda.synchronize()
    scale_a_expanded = scale_a.view(-1, 1)
    scale_b_expanded = scale_b.view(1, -1)
    # 测试
    start_time = time.time()
    for _ in range(n_runs):
        result = torch.matmul(a_fp16, b_fp16)
        result = result * scale_a_expanded * scale_b_expanded
    torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    return elapsed_time / n_runs, result

def main():
    """主函数，运行基准测试"""
    if not torch.cuda.is_available():
        print("错误: 需要CUDA支持")
        return

    print("Int4 矩阵乘法 CUDA vs PyTorch 对比测试 (简化版)")
    print("=" * 90)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA 算子可用: {CUDA_KERNELS_AVAILABLE}")
    print("=" * 90)

    test_configs = [
        (1024, 256, 256),
        (1024, 512, 512),
        (1024, 1024, 1024),
    ]
    
    header = f"{'Config (M,K,N)':<20} | {'Implementation':<15} | {'Time (ms)':<15} | {'Speedup vs FP16':<20} | {'Max Abs Diff':<15}"
    print(header)
    print("-" * 90)

    for M, K, N in test_configs:

        # Generate data
        a_packed, b_packed, scale_a, scale_b, a_original, b_original = generate_test_data(M, K, N)
        n_runs = 10 if M <= 2048 else 5

        # --- FP16 baseline ---
        baseline_time, baseline_result = benchmark_pytorch_baseline(a_original, b_original, scale_a, scale_b, n_runs)
        print(f"{(M, K, N)!s:<20} | {'PyTorch FP16':<15} | {baseline_time * 1000:<15.2f} | {'1.00x':<20} | {'-':<15}")

        # --- PyTorch int4 ---
        # Warmup
        for _ in range(5):
            _ = int4_matmul_pytorch(a_packed, b_packed, scale_a, scale_b)
        torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(n_runs):
            pytorch_result = int4_matmul_pytorch(a_packed, b_packed, scale_a, scale_b)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / n_runs
        
        accuracy_diff = (pytorch_result - baseline_result).abs().max().item()
        speedup = f"{baseline_time/pytorch_time:.2f}x"
        print(f"{'':<20} | {'PyTorch Int4':<15} | {pytorch_time * 1000:<15.2f} | {speedup:<20} | {accuracy_diff:<15.6f}")

        # --- CUDA int4 ---
        if CUDA_KERNELS_AVAILABLE:
            # Warmup
            # for _ in range(5):
            #     _ = int4_matmul_cuda(a_packed, b_packed, scale_a, scale_b)
            # torch.cuda.synchronize()

            start_time = time.time()
            for _ in range(n_runs):
                cuda_result = int4_matmul_cuda(a_packed, b_packed, scale_a, scale_b)
            torch.cuda.synchronize()
            cuda_time = (time.time() - start_time) / n_runs

            cuda_accuracy_diff = (cuda_result - baseline_result).abs().max().item()
            cuda_speedup = f"{baseline_time/cuda_time:.2f}x"
            print(f"{'':<20} | {'CUDA Int4':<15} | {cuda_time * 1000:<15.2f} | {cuda_speedup:<20} | {cuda_accuracy_diff:<15.6f}")
        else:
                print(f"{'':<20} | {'CUDA Int4':<15} | {'N/A':<15} | {'N/A':<20} | {'N/A':<15}")
        print("-" * 90)

            
    print("\n✅ 所有测试完成")


if __name__ == "__main__":
    main()
