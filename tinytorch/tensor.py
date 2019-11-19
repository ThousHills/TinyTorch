""" 
数据对象
"""
import numpy as np

from . import utils
from . import op


class Tensor(object):
    """ 基本数据类型
    """

    def __init__(self, data, from_tensors=None, op=None, grad=None):
        self.data = utils.convert_to_np(data)
        self.from_tensors = from_tensors
        self.op = op

        self.grad_fn = op.backward if op else None

        if grad:
            self.grad = grad
        else:
            if isinstance(self.data, np.ndarray):
                self.grad = np.zeros(self.data.shape)
            else:
                self.grad = 0

    def backward(self, grad=None):
        # 判断y的梯度是否存在，如果不存在初始化和y.data一样类型的1的数据
        if grad is None:
            if isinstance(self.data, np.ndarray):
                # 顺便给grad赋值，下面还要用
                self.grad = grad = np.ones(self.data.shape)
            else:
                self.grad = grad = 1

        # 如果op不存在，则说明该Tensor为根节点，其from_tensors也必然不存在，否则计算梯度
        if self.op:
            grad = self.op.backward(self.from_tensors, grad)

        if self.from_tensors:
            for i in range(len(grad)):
                tensor = self.from_tensors[i]
                # 把梯度加给对应的子Tensor，因为该Tensor可能参与多个运算
                tensor.grad += grad[i]
                # 子Tensor进行后向过程
                tensor.backward(grad[i])

    def __add__(self, other):
        if isinstance(other, Tensor):
            return op.add([self, other])
        else:
            return op.add_by_const([self, other])
    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return op.sub([self, other])
        else:
            return op.add_by_const([self, -other])

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return op.sub([self, other])
        else:
            return op.rsub_by_const([self, other])

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return op.mul([self, other])
        else:
            return op.mul_by_const([self, other])

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return op.div([self, other])
        else:
            return op.mul_by_const([self, 1./other])

    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            return op.div([self, other])
        else:
            return op.rdiv_by_const([self, other])

    def mean(self):
        return op.mean([self])

    def log(self):
        return op.log([self])

    def exp(self):
        return op.exp([self])

    def sum(self):
        return op.sum([self])

    def __str__(self):
        # Backward function
        if self.op != None:
            return f"Tensor({self.data.__str__()})" + \
                f", grad_fn=<{self.op.__class__.__name__}Backward>"

        return f"Tensor({self.data.__str__()})"
