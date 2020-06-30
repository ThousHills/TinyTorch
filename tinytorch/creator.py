""" 快速创建器

用于快速创建零矩阵、全一矩阵等
"""
import numpy as np

from .tensor import Tensor


def ones(shape, dtype=None):
    return Tensor(np.ones(shape, dtype))


def zeros(shape, dtype=None):
    return Tensor(np.zeros(shape, dtype))


def rand(*dims):
    return Tensor(np.random.rand(*dims))


def randn(*dims):
    return Tensor(np.random.randn(*dims))
