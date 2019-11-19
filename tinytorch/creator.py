""" 快速创建器

用于快速创建零矩阵、全一矩阵等
"""
from .tensor import Tensor
import numpy as np


def ones(shape, dtype=None):
    return Tensor(np.ones(shape, dtype))


def zeros(shape, dtype=None):
    return Tensor(np.zeros(shape, dtype))
