"""
Int4 Matrix Multiplication CUDA内核模块
"""

import importlib.util
import os

# 获取当前目录和CUDA扩展文件路径
current_dir = os.path.dirname(__file__)
cuda_lib_path = os.path.join(current_dir, "cuda_int4_matmul_ops.cpython-310-x86_64-linux-gnu.so")

# 尝试加载CUDA扩展
cuda_int4_matmul_ops = None
CUDA_KERNELS_AVAILABLE = False

try:
    if os.path.exists(cuda_lib_path):
        spec = importlib.util.spec_from_file_location("cuda_int4_matmul_ops", cuda_lib_path)
        cuda_int4_matmul_ops = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cuda_int4_matmul_ops)
        
        # 验证必要的函数存在
        if hasattr(cuda_int4_matmul_ops, 'cuda_int4_matmul'):
            CUDA_KERNELS_AVAILABLE = True
            if os.environ.get('QUANT_OPS_DEBUG', '0') == '1':
                print("✅ Int4 MatMul CUDA扩展加载成功")
        else:
            print("⚠️ CUDA扩展模块缺少必要的函数: cuda_int4_matmul")
    else:
        print(f"⚠️ CUDA扩展文件不存在: {cuda_lib_path}")
        print("   请先运行构建脚本编译CUDA扩展")
        
except Exception as e:
    print(f"⚠️ 加载 Int4 MatMul CUDA扩展失败: {e}")
    print("   将回退到PyTorch实现")

# 调试信息
if os.environ.get('QUANT_OPS_DEBUG', '0') == '1':
    if CUDA_KERNELS_AVAILABLE:
        print(f"✅ 可用函数: {dir(cuda_int4_matmul_ops)}")
    else:
        print("⚠️ CUDA内核不可用，将使用PyTorch实现")

__all__ = ["cuda_int4_matmul_ops", "CUDA_KERNELS_AVAILABLE"]