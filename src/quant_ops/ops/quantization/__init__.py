"""
量化算法模块

包含各种量化算法的实现：
- flatquant: FlatQuant动态量化算法
- 其他量化方法（待扩展，如GPTQ、AWQ等）
"""

from . import flatquant

__all__ = ["flatquant"] 