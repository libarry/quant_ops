"""
Quant Ops - 高性能量化算子库

这个库提供了各种量化算法的高性能CUDA实现，包括：
- FlatQuant动态量化
- W4A4量化方案
- 更多量化算子（待扩展）

主要功能：
- 多精度支持 (fp16/bf16/fp32)
- CUDA加速
- 高精度实现
- 易于扩展的模块化设计
"""

__version__ = "0.1.0"
__author__ = "Quant Ops Team"

# 导入核心功能
from .ops.quantization.flatquant import (
    flatquant_dynamic_quantize,
    dequantize_int4,
)
from .utils.decompose import get_decompose_dim

# 检查CUDA算子可用性
try:
    from .ops.quantization.flatquant.kernels import cuda_quant_ops
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

__all__ = [
    "flatquant_dynamic_quantize",
    "dequantize_int4", 
    "get_decompose_dim",
    "CUDA_AVAILABLE",
] 