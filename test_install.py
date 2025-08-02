#!/usr/bin/env python3
"""
安装后的快速验证脚本

用于测试 pip install 后的基本功能。
"""

def main():
    print("测试 Quant Ops 安装...")
    
    try:
        # 测试基本导入
        import quant_ops
        print("✅ 基本导入成功")
        
        # 测试版本信息
        print(f"📦 版本: {quant_ops.__version__}")
        print(f"👥 作者: {quant_ops.__author__}")
        
        # 测试CUDA状态
        if quant_ops.CUDA_AVAILABLE:
            print("🚀 CUDA 算子可用")
        else:
            print("⚠️ CUDA 算子不可用，使用PyTorch后备实现")
        
        # 测试基本功能
        import torch
        if torch.cuda.is_available():
            print("🧪 开始功能测试...")
            
            # 创建小规模测试数据
            batch_tokens, features = 32, 256
            M, N = quant_ops.get_decompose_dim(features)
            
            torch.manual_seed(42)
            input_tensor = torch.randn(batch_tokens, features, dtype=torch.float16, device='cuda')
            left_trans = torch.randn(M, M, dtype=torch.float16, device='cuda')
            right_trans = torch.randn(N, N, dtype=torch.float16, device='cuda')
            
            # 执行量化
            quantized, scales = quant_ops.flatquant_dynamic_quantize(
                input_tensor, left_trans, right_trans, clip_ratio=1.0
            )
            
            # 反量化
            dequantized = quant_ops.dequantize_int4(quantized, scales)
            
            # 计算误差
            error = (input_tensor.float() - dequantized).abs().mean()
            print(f"📊 量化误差: {error.item():.6f}")
            
            if error.item() < 0.1:
                print("✅ 功能测试通过")
            else:
                print("❌ 功能测试失败：误差过大")
                return False
        else:
            print("⚠️ 无CUDA环境，跳过功能测试")
        
        print("🎉 安装验证完成！")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确认已正确安装: pip install .")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1) 