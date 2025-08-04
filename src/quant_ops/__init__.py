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

# Int4 MatMul 便捷导出
from .ops.quantization.int4_matmul.ops import int4_matmul
from .utils.decompose import get_decompose_dim

# 注册 meta kernels（确保在 import 时执行装饰器）
from . import meta_ops  # noqa: F401

# 检查CUDA算子可用性
try:
    from .ops.quantization.flatquant.kernels import CUDA_KERNELS_AVAILABLE, cuda_quant_ops
    CUDA_AVAILABLE = CUDA_KERNELS_AVAILABLE and cuda_quant_ops is not None
except ImportError:
    CUDA_AVAILABLE = False

__all__ = [
    "flatquant_dynamic_quantize",
    "dequantize_int4", 
    "int4_matmul", 
    "get_decompose_dim",
    "CUDA_AVAILABLE",
] 