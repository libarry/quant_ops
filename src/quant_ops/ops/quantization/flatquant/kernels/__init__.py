"""
FlatQuant CUDA内核模块

包含FlatQuant算法的CUDA加速实现。
"""

# 尝试导入编译的CUDA扩展
try:
    import cuda_quant_ops
    CUDA_KERNELS_AVAILABLE = True
except ImportError as e:
    cuda_quant_ops = None
    CUDA_KERNELS_AVAILABLE = False

__all__ = ["cuda_quant_ops", "CUDA_KERNELS_AVAILABLE"] 