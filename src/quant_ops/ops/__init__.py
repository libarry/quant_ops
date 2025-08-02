"""
量化算子模块

包含各种量化算法的实现，按照算法类型组织：
- quantization: 各种量化算法
  - flatquant: FlatQuant相关算子
  - 其他量化方法（待扩展）
"""

from .quantization import flatquant

__all__ = ["flatquant"] 