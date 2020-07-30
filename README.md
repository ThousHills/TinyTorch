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

if __name__ == '__main__':
    model = MLP(10, 5, 2)
```
