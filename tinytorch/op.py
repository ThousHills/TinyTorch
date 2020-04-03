from . import tensor
import numpy as np
from typing import List, Any


class OP:
    """ 运算符基类
    """

    def __call__(self, from_tensors: List[Any]):
        """ 使用forward方法 """
        return self.forward(from_tensors)

    def forward(self, from_tensors: List[Any]):
        """ 前向计算

        :param from_tensors: 参与计算的数据
        """
        raise NotImplementedError

    def backward(self, from_tensors: List[Any], grad: np.float):
        """ 求梯度

        :param from_tensors: 参与运算的张量，需要将梯度传递给对应的张量
        :param grad: 上一个运算传过来的梯度数据
        """
        raise NotImplementedError


class Add(OP):
    def forward(self, from_tensors):
        return tensor.Tensor(
            from_tensors[0].data + from_tensors[1].data,
            from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad, grad]


class AddByConst(OP):
    def forward(self, from_tensors):
        return tensor.Tensor(
            from_tensors[0].data + from_tensors[1],
            from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad]


class Sub(OP):
    def forward(self, from_tensors):
        return tensor.Tensor(
            from_tensors[0].data - from_tensors[1].data,
            from_tensors, self
        )

    def backward(self, from_tensors, grad):
        return [grad[0], -grad[1]]


class RSubByConst(OP):
    """ 右减法

    例如：2 - x，减法只用处理右减法，因为左减法x - 2 等于 x + (-2) 因此不用单独处理
    """

    def forward(self, from_tensors):
        return tensor.Tensor(
            -(from_tensors[0].data - from_tensors[1]),
            from_tensors, self
        )

    def backward(self, from_tensors, grad):
        return [-grad]


class Mul(OP):
    def forward(self, from_tensors):
        return tensor.Tensor(
            from_tensors[0].data * from_tensors[1].data,
            from_tensors, self)

    def backward(self, from_tensors, grad):
        return [from_tensors[1].data * grad,
                from_tensors[0].data * grad]


class MulByConst(OP):
    def forward(self, from_tensors):
        return tensor.Tensor(
            from_tensors[0].data * from_tensors[1],
            from_tensors, self)

    def backward(self, from_tensors, grad):
        return [from_tensors[1] * grad]


class Div(OP):
    def forward(self, from_tensors):
        return tensor.Tensor(
            from_tensors[0].data / from_tensors[1].data,
            from_tensors, self
        )

    def backward(self, from_tensors, grad):
        r""" 除法的偏导公式
        """
        return [1/from_tensors[1].data*grad, -from_tensors[0].data*grad]


class RDivByConst(OP):
    def forward(self, from_tensors):
        return tensor.Tensor(
            from_tensors[1] / from_tensors[0].data,
            from_tensors, self
        )

    def backward(self, from_tensors, grad):
        return [grad * (-from_tensors[1]) / (from_tensors[0].data**2)]


class Sum(OP):
    def forward(self, from_tensors):
        _sum = 0.0
        for _tensor in from_tensors:
            _sum += np.sum(_tensor.data)
        return tensor.Tensor(
            _sum,
            from_tensors, self
        )

    def backward(self, from_tensors, grad):
        return [grad * np.ones(from_tensors[0].data.shape)]


class Mean(OP):
    def forward(self, from_tensors):
        if not isinstance(from_tensors, List):
            from_tensors = [from_tensors]
        return tensor.Tensor(
            sum(from_tensors) / from_tensors[0].data.size,
            from_tensors, self
        )

    def backward(self, from_tensors, grad):
        return [grad * np.ones(from_tensors[0].data.shape) /
                from_tensors[0].data.size]


class Exp(OP):
    def forward(self, from_tensors):
        if not isinstance(from_tensors, List):
            from_tensors = [from_tensors]
        return tensor.Tensor(np.exp(from_tensors[0].data), from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad * np.exp(from_tensors[0].data)]


class Log(OP):
    def forward(self, from_tensors):
        if not isinstance(from_tensors, List):
            from_tensors = [from_tensors]
        return tensor.Tensor(np.log(from_tensors[0].data), from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad / from_tensors[0].data]


class MatMul(OP):
    """ 矩阵乘法
    """
    def forward(self, from_tensors):

        return tensor.Tensor(np.matmul(from_tensors[0].data, from_tensors[1].data), self)

    def backward(self, from_tensors, grad):
        return [grad*]

# 基本运算符
add = Add()
add_by_const = AddByConst()
sub = Sub()
rsub_by_const = RSubByConst()
mul = Mul()
mul_by_const = MulByConst()
div = Div()
rdiv_by_const = RDivByConst()
matmul = MatMul()

# 基本运算方法
sum = Sum()
mean = Mean()
exp = Exp()
log = Log()
