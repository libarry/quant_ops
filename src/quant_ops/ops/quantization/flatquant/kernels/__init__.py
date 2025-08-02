"""
FlatQuant CUDA内核模块

包含FlatQuant算法的CUDA加速实现。
"""

# 尝试导入编译的CUDA扩展
cuda_quant_ops = None
CUDA_KERNELS_AVAILABLE = False
_import_error_msg = None

try:
    # 首先尝试导入安装后的扩展
    from . import cuda_quant_ops as _cuda_ops
    # 验证模块确实存在且有需要的函数
    if hasattr(_cuda_ops, 'cuda_kronecker_quant_int8'):
        cuda_quant_ops = _cuda_ops
        CUDA_KERNELS_AVAILABLE = True
    else:
        cuda_quant_ops = None
        CUDA_KERNELS_AVAILABLE = False
        _import_error_msg = "CUDA扩展模块缺少必要的函数"
except ImportError as e1:
    try:
        # 回退到直接导入（开发时）
        import cuda_quant_ops as _cuda_ops
        # 验证模块确实存在且有需要的函数
        if hasattr(_cuda_ops, 'cuda_kronecker_quant_int8'):
            cuda_quant_ops = _cuda_ops
            CUDA_KERNELS_AVAILABLE = True
        else:
            cuda_quant_ops = None
            CUDA_KERNELS_AVAILABLE = False
            _import_error_msg = "CUDA扩展模块缺少必要的函数"
    except ImportError as e2:
        # 都失败了，设置为None
        cuda_quant_ops = None
        CUDA_KERNELS_AVAILABLE = False
        _import_error_msg = f"CUDA扩展导入失败: {e2}"

# 可选：输出调试信息（仅在开发时启用）
import os
if os.environ.get('QUANT_OPS_DEBUG', '0') == '1':
    if CUDA_KERNELS_AVAILABLE:
        print("✅ CUDA扩展加载成功")
    else:
        print(f"⚠️ CUDA扩展不可用: {_import_error_msg}")

__all__ = ["cuda_quant_ops", "CUDA_KERNELS_AVAILABLE"] 