"""
FlatQuant 动态量化算子 - 精简版
"""

from .ops import (
    flatquant_cuda,
    flatquant_pytorch,
    flatquant_dynamic_quantize,
    dequantize_int4,
)

__all__ = [
    "flatquant_cuda",
    "flatquant_pytorch", 
    "flatquant_dynamic_quantize",
    "dequantize_int4",
] 