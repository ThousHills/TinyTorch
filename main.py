import tinytorch
import tinytorch.functional as F
from tinytorch import nn
import numpy as np

if __name__ == "__main__":
    x = tinytorch.Tensor([[1, 2, 3], [2, 2, 2]])
    
    linear = nn.layers.Linear(3, 1)

    y = linear(x)
    pred = F.sigmoid(y)
    target = np.array([[0], [1]])
    loss_func = nn.BCELoss()
    loss = loss_func(pred, target)
    print(loss)
