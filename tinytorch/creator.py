""" 快速创建器

用于快速创建零矩阵、一矩阵等
"""
from .tensor import Tensor
import numpy as np


def ones(shape, dtype=None):
    return Tensor(np.ones(shape, dtype))
 