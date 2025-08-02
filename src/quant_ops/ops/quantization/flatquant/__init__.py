"""
FlatQuant 动态量化算子

FlatQuant是一种基于Kronecker矩阵分解的动态量化方法，能够有效平滑激活分布，
提高量化精度。支持将浮点输入量化为int4格式。

主要功能：
- 动态per-token量化
- Kronecker矩阵变换预处理  
- int4量化与可选int32打包
- CUDA加速实现
"""

from .ops import (
    flatquant_dynamic_quantize,
    dequantize_int4,
)

__all__ = [
    "flatquant_dynamic_quantize", 
    "dequantize_int4",
] 