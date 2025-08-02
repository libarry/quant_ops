"""
FlatQuant CUDA内核模块

包含FlatQuant算法的CUDA加速实现。
"""

import importlib.util
import os

# 获取当前目录和CUDA扩展文件路径
current_dir = os.path.dirname(__file__)
cuda_lib_path = os.path.join(current_dir, "cuda_quant_ops.cpython-310-x86_64-linux-gnu.so")

# 直接加载CUDA扩展，如果失败就报错
if not os.path.exists(cuda_lib_path):
    raise ImportError(f"CUDA扩展文件不存在: {cuda_lib_path}")

spec = importlib.util.spec_from_file_location("cuda_quant_ops", cuda_lib_path)
cuda_quant_ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cuda_quant_ops)

# 验证必要的函数存在
if not hasattr(cuda_quant_ops, 'cuda_kronecker_quant_int8'):
    raise ImportError("CUDA扩展模块缺少必要的函数: cuda_kronecker_quant_int8")

# 设置可用性标志
CUDA_KERNELS_AVAILABLE = True

# 调试信息
if os.environ.get('QUANT_OPS_DEBUG', '0') == '1':
    print("✅ CUDA扩展加载成功")
    print(f"✅ 可用函数: {dir(cuda_quant_ops)}")

__all__ = ["cuda_quant_ops", "CUDA_KERNELS_AVAILABLE"] 