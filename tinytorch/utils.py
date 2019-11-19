""" 工具
"""
import numpy as np
from typing import List


def convert_to_np(data):
    """ 将数据转为 np 数据
    """
    if not isinstance(data, np.ndarray):
        return np.array(data)
    else:
        return data
