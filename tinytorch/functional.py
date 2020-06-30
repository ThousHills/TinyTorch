from .op import exp


def sigmoid(x):
    return 1/(1 + exp(-x))
