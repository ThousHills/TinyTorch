from .op import exp
from .creator import zeros


def sigmoid(x):
    return 1/(1 + exp(-x))


def tanh(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
