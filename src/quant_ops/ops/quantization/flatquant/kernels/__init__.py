"""
FlatQuant CUDA内核模块

包含FlatQuant算法的CUDA加速实现。
"""

# 尝试导入编译的CUDA扩展
cuda_quant_ops = None
CUDA_KERNELS_AVAILABLE = False

try:
    # 首先尝试导入安装后的扩展
    from . import cuda_quant_ops
    CUDA_KERNELS_AVAILABLE = True
except ImportError:
    try:
        # 回退到直接导入（开发时）
        import cuda_quant_ops
        CUDA_KERNELS_AVAILABLE = True
    except ImportError as e:
        # 都失败了，设置为None
        cuda_quant_ops = None
        CUDA_KERNELS_AVAILABLE = False

__all__ = ["cuda_quant_ops", "CUDA_KERNELS_AVAILABLE"] 