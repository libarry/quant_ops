"""
Quant Ops - 精简版量化算子库
"""

__version__ = "0.1.0"

# 导入核心功能
from .ops.quantization.flatquant import (
    flatquant_cuda,
    flatquant_pytorch,
    flatquant_dynamic_quantize,
    dequantize_int4,
)
from .utils.decompose import get_decompose_dim

# 检查CUDA算子可用性
try:
    from .ops.quantization.flatquant.kernels import CUDA_KERNELS_AVAILABLE
    CUDA_AVAILABLE = CUDA_KERNELS_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

__all__ = [
    "flatquant_cuda",
    "flatquant_pytorch",
    "flatquant_dynamic_quantize",
    "dequantize_int4",
    "CUDA_AVAILABLE",
    "get_decompose_dim",
] 