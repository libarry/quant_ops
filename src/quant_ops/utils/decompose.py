"""
矩阵分解工具模块

提供FlatQuant等算法需要的矩阵分解功能。
"""

import math
from typing import Tuple


def get_decompose_dim(n: int) -> Tuple[int, int]:
    """
    计算 FlatQuant 变换矩阵的分解维度
    
    找到满足 M*N = n 的 M, N，使得存在整数 a, b 满足 a^2 - b^2 = M*N - n = 0
    这种分解方式有助于优化 Kronecker 矩阵乘法的性能。
    
    Args:
        n (int): 需要分解的维度大小
        
    Returns:
        Tuple[int, int]: (M, N) 使得 M*N = n
        
    Example:
        >>> get_decompose_dim(4096)
        (64, 64)
        >>> get_decompose_dim(2048)
        (32, 64)
    """
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            break
        a += 1
    return a - b, a + b 