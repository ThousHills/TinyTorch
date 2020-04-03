import tinytorch
import tinytorch.functional as F


if __name__ == "__main__":
    x = tinytorch.Tensor([[1, 2, 3]])
    print(x.shape)
    W = tinytorch.randn(3, 2)
    b = tinytorch.rand(1)

    o = x.matmul(W)
    print(o)
