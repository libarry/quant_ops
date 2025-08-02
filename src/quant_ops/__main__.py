#!/usr/bin/env python3
"""
Quant Ops 模块主入口

用于验证安装和运行基本测试。

用法:
    python -m quant_ops               # 验证安装
    python -m quant_ops --test        # 运行基本测试
    python -m quant_ops --version     # 显示版本信息
"""

import argparse
import sys
import torch

def check_installation():
    """检查安装状态"""
    print("=" * 50)
    print("Quant Ops 安装验证")
    print("=" * 50)
    
    # 检查基本环境
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"当前GPU: {torch.cuda.get_device_name()}")
    
    print()
    
    # 检查包导入
    try:
        from . import __version__
        print(f"✅ Quant Ops 版本: {__version__}")
    except ImportError as e:
        print(f"❌ 无法导入版本信息: {e}")
        return False
    
    # 检查核心功能
    try:
        from . import flatquant_dynamic_quantize, dequantize_int4, get_decompose_dim
        print("✅ 核心算子导入成功")
    except ImportError as e:
        print(f"❌ 核心算子导入失败: {e}")
        return False
    
    # 检查CUDA支持
    try:
        from . import CUDA_AVAILABLE
        if CUDA_AVAILABLE:
            print("✅ CUDA 算子可用")
        else:
            print("⚠️ CUDA 算子不可用，将使用PyTorch后备实现")
    except ImportError as e:
        print(f"⚠️ CUDA 状态检查失败: {e}")
    
    print()
    print("🎉 安装验证完成！")
    return True


def run_basic_test():
    """运行基本功能测试"""
    print("=" * 50)
    print("Quant Ops 基本功能测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ 测试需要CUDA支持")
        return False
    
    try:
        from . import flatquant_dynamic_quantize, dequantize_int4, get_decompose_dim, CUDA_AVAILABLE
        
        # 创建小规模测试数据
        batch_tokens = 32
        features = 256
        M, N = get_decompose_dim(features)
        
        print(f"测试配置: batch_tokens={batch_tokens}, features={features}, M={M}, N={N}")
        
        # 创建测试数据
        torch.manual_seed(42)
        input_tensor = torch.randn(batch_tokens, features, dtype=torch.float16, device='cuda')
        left_trans = torch.randn(M, M, dtype=torch.float16, device='cuda')
        right_trans = torch.randn(N, N, dtype=torch.float16, device='cuda')
        
        # 测试量化
        print("执行量化测试...")
        quantized, scales = flatquant_dynamic_quantize(
            input_tensor, left_trans, right_trans, 
            clip_ratio=1.0, pack_int32=False, use_cuda=CUDA_AVAILABLE
        )
        
        # 测试反量化
        print("执行反量化测试...")
        dequantized = dequantize_int4(quantized, scales, packed_int32=False)
        
        # 计算误差
        error = (input_tensor.float() - dequantized).abs().mean()
        print(f"平均量化误差: {error.item():.6f}")
        
        if error.item() < 0.1:  # 合理的误差范围
            print("✅ 基本功能测试通过！")
            return True
        else:
            print("❌ 量化误差过大")
            return False
            
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_version():
    """显示版本信息"""
    try:
        from . import __version__, __author__
        print(f"Quant Ops {__version__}")
        print(f"作者: {__author__}")
    except ImportError:
        print("无法获取版本信息")


def main():
    parser = argparse.ArgumentParser(description="Quant Ops 验证和测试工具")
    parser.add_argument('--test', action='store_true', help='运行基本功能测试')
    parser.add_argument('--version', action='store_true', help='显示版本信息')
    
    args = parser.parse_args()
    
    if args.version:
        show_version()
    elif args.test:
        success = check_installation() and run_basic_test()
        sys.exit(0 if success else 1)
    else:
        success = check_installation()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 