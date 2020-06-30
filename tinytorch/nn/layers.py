""" 一些基础的层
"""
import tinytorch

from .module import Module


class Linear(Module):
    """ 线性层
    """

    def __init__(self, input_dim, output_dim):
        self.W = tinytorch.rand(input_dim, output_dim)
        self.b = tinytorch.rand(1)

    def forward(self, x):
        return x.matmul(self.W) + self.b
