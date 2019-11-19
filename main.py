import tinytorch
from tinytorch import functional as F


if __name__ == "__main__":
    x = F.sigmoid(tinytorch.Tensor(3))
    print(x)
