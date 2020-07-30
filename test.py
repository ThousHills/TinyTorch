
import numpy as np

import tinytorch
import tinytorch.creator as creator
import tinytorch.functional as F
import tinytorch.nn as nn


class MLP(nn.Module):

    def __init__(self, h1, h2, h3):
        self.linear1 = nn.Linear(h1, h2)
        self.linear2 = nn.Linear(h2, h3)

    def forward(self, x):
        out = self.linear1(x)
        out = F.tanh(out)
        out = self.linear2(out)
        out = F.sigmoid(out)
        return out


def main():
    x = creator.randn(2, 2)
    model = MLP(2, 3, 2)
    out = model(x)
    # print(out.shape)
    out.backward()
    print(out)


if __name__ == '__main__':
    main()