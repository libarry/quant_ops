#!/usr/bin/env python3
"""
FlatQuant 动态量化 CUDA 算子性能测试和精度对比脚本

用法:
    python test_performance.py --test accuracy       # 只运行精度测试
    python test_performance.py --test performance    # 只运行性能测试
    python test_performance.py --test all            # 运行所有测试 (默认)
    python test_performance.py --detailed            # 详细测试不同数据类型
"""

import argparse
import torch
import time
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

def create_test_data(batch_tokens, features, dtype=torch.float16, device='cuda'):
    """创建测试数据"""
    M, N = get_decompose_dim(features)
    
    torch.manual_seed(42)  # 固定随机种子确保可重复性
    input_tensor = torch.randn(batch_tokens, features, dtype=dtype, device=device)
    left_trans = torch.randn(M, M, dtype=dtype, device=device)
    right_trans = torch.randn(N, N, dtype=dtype, device=device)
    
    return input_tensor, left_trans, right_trans

def test_accuracy_single_config(batch_tokens, features, dtype, clip_ratio=1.0):
    """测试单个配置的精度"""
    print(f"\n--- 精度测试: batch_tokens={batch_tokens}, features={features}, dtype={dtype} ---")
    
    # 创建测试数据
    input_tensor, left_trans, right_trans = create_test_data(batch_tokens, features, dtype)
    
    # PyTorch 实现 (参考实现)
    print("运行 PyTorch 参考实现...")
    torch_quant, torch_scales = flatquant_dynamic_quantize(
        input_tensor, left_trans, right_trans, clip_ratio, pack_int32=False, use_cuda=False
    )
    torch_dequant = dequantize_int4(torch_quant, torch_scales, packed_int32=False)
    
    if not CUDA_AVAILABLE:
        print("CUDA 实现不可用，跳过 CUDA 精度测试")
        return
    
    # CUDA 实现
    print("运行 CUDA 加速实现...")
    try:
        cuda_quant, cuda_scales = flatquant_dynamic_quantize(
            input_tensor, left_trans, right_trans, clip_ratio, pack_int32=False, use_cuda=True
        )
        cuda_dequant = dequantize_int4(cuda_quant, cuda_scales, packed_int32=False)
        
        # 精度对比
        print("\n精度对比结果:")
        quant_diff = (cuda_quant.float() - torch_quant.float()).abs()
        scale_diff = (cuda_scales - torch_scales).abs()
        dequant_diff = (cuda_dequant - torch_dequant).abs()
        
        print(f"  量化结果 - 最大差异: {quant_diff.max().item():.6f}, 平均差异: {quant_diff.mean().item():.6f}")
        print(f"  Scale值  - 最大差异: {scale_diff.max().item():.6f}, 平均差异: {scale_diff.mean().item():.6f}")
        print(f"  反量化   - 最大差异: {dequant_diff.max().item():.6f}, 平均差异: {dequant_diff.mean().item():.6f}")
        
        # 相对误差
        rel_error = (dequant_diff / torch_dequant.abs().clamp(min=1e-8)).mean().item()
        print(f"  反量化相对误差: {rel_error:.6f}")
        
        # 判断精度是否可接受
        if quant_diff.max() <= 2 and scale_diff.max() <= 1e-4 and rel_error <= 1e-3:
            print("  ✅ 精度测试通过")
        else:
            print("  ❌ 精度测试失败")
            
    except Exception as e:
        print(f"CUDA 实现执行失败: {e}")

def test_packing_accuracy():
    """测试 int32 打包功能的精度"""
    print("\n--- Int32 打包功能精度测试 ---")
    
    batch_tokens, features = 128, 4096
    input_tensor, left_trans, right_trans = create_test_data(batch_tokens, features)
    clip_ratio = 1.0
    
    if not CUDA_AVAILABLE:
        print("CUDA 实现不可用，跳过打包测试")
        return
    
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
        quant_diff = (quant_int8.float() - dequant_int32).abs()
        scale_diff = (scales_int8 - scales_int32).abs()
        
        print(f"打包前后量化结果差异 - 最大: {quant_diff.max().item():.6f}, 平均: {quant_diff.mean().item():.6f}")
        print(f"打包前后Scale值差异 - 最大: {scale_diff.max().item():.6f}, 平均: {scale_diff.mean().item():.6f}")
        
        # 检查打包存储效率
        memory_int8 = quant_int8.numel() * quant_int8.element_size()
        memory_int32 = quant_int32.numel() * quant_int32.element_size()
        compression_ratio = memory_int8 / memory_int32
        
        print(f"存储效率: int8={memory_int8}bytes, int32={memory_int32}bytes, 压缩比={compression_ratio:.2f}x")
        
        if quant_diff.max() <= 1e-6 and compression_ratio >= 1.9:  # 理论上应该是2x压缩
            print("  ✅ 打包功能测试通过")
        else:
            print("  ❌ 打包功能测试失败")
            
    except Exception as e:
        print(f"打包功能测试失败: {e}")

def benchmark_single_config(batch_tokens, features, dtype, warmup_runs=5, benchmark_runs=20):
    """基准测试单个配置"""
    print(f"\n--- 性能测试: batch_tokens={batch_tokens}, features={features}, dtype={dtype} ---")
    
    # 创建测试数据
    input_tensor, left_trans, right_trans = create_test_data(batch_tokens, features, dtype)
    clip_ratio = 1.0
    
    # 预热
    print("预热中...")
    for _ in range(warmup_runs):
        _ = flatquant_dynamic_quantize(input_tensor, left_trans, right_trans, clip_ratio, use_cuda=False)
        if CUDA_AVAILABLE:
            _ = flatquant_dynamic_quantize(input_tensor, left_trans, right_trans, clip_ratio, use_cuda=True)
    
    torch.cuda.synchronize()
    
    # PyTorch 实现基准测试
    print("测试 PyTorch 实现性能...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    for _ in range(benchmark_runs):
        _ = flatquant_dynamic_quantize(input_tensor, left_trans, right_trans, clip_ratio, use_cuda=False)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) / benchmark_runs
    torch_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    
    print(f"  PyTorch - 平均时间: {torch_time*1000:.3f} ms, 峰值内存: {torch_memory:.1f} MB")
    
    # CUDA 实现基准测试
    if CUDA_AVAILABLE:
        print("测试 CUDA 实现性能...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        for _ in range(benchmark_runs):
            _ = flatquant_dynamic_quantize(input_tensor, left_trans, right_trans, clip_ratio, use_cuda=True)
        torch.cuda.synchronize()
        cuda_time = (time.time() - start_time) / benchmark_runs
        cuda_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        
        speedup = torch_time / cuda_time
        memory_ratio = torch_memory / cuda_memory
        
        print(f"  CUDA    - 平均时间: {cuda_time*1000:.3f} ms, 峰值内存: {cuda_memory:.1f} MB")
        print(f"  加速比: {speedup:.2f}x, 内存效率: {memory_ratio:.2f}x")
    else:
        print("  CUDA 实现不可用")

def main():
    parser = argparse.ArgumentParser(description="FlatQuant CUDA 算子测试")
    parser.add_argument('--test', choices=['accuracy', 'performance', 'all'], default='all',
                        help='选择测试类型')
    parser.add_argument('--detailed', action='store_true',
                        help='运行详细测试（不同数据类型）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FlatQuant 动态量化 CUDA 算子测试")
    print("=" * 80)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"CUDA 算子可用: {CUDA_AVAILABLE}")
    
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用，无法运行测试")
        sys.exit(1)
    
    # 精度测试
    if args.test in ['accuracy', 'all']:
        print("\n" + "=" * 60)
        print("精度测试")
        print("=" * 60)
        
        # 基础精度测试
        test_accuracy_single_config(128, 4096, torch.float16)
        
        if args.detailed:
            # 详细精度测试
            test_configs = [
                (64, 2048, torch.float16),
                (256, 4096, torch.float32),
                (512, 4096, torch.bfloat16),
            ]
            
            for batch_tokens, features, dtype in test_configs:
                test_accuracy_single_config(batch_tokens, features, dtype)
        
        # 打包功能测试
        test_packing_accuracy()
    
    # 性能测试
    if args.test in ['performance', 'all']:
        print("\n" + "=" * 60)
        print("性能基准测试")
        print("=" * 60)
        
        # 标准性能测试配置
        test_configs = [
            (64, 2048, torch.float16),
            (128, 4096, torch.float16),
            (256, 4096, torch.float16),
            (512, 4096, torch.float16),
            (1024, 4096, torch.float16),
        ]
        
        if args.detailed:
            # 详细性能测试：不同数据类型
            test_configs.extend([
                (256, 4096, torch.float32),
                (256, 4096, torch.bfloat16),
            ])
        
        for batch_tokens, features, dtype in test_configs:
            benchmark_single_config(batch_tokens, features, dtype)
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main() 