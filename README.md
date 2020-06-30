# TinyTorch

一个自己编写的自微分框架，基于 Numpy 实现，整体 API 类似于 PyTorch。本项目用于学习自动微分框架实现，并不适用于一些生产环境。

Python 需要 3.6 版本以上

## 安装

```bash
git clone https://github.com/Codle/TinyTorch
cd TinyTorch
```

## 用法

```python
import tinytorch as ttorch
import tinytorch.nn as nn
import tinytorch.functional as F


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        out = self.output(out)
        out = F.sigmoid(out)
        return out

if __name__ == '__main__':
    model = MLP(10, 5, 2)
```
