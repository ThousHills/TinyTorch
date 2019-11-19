# TinyTorch

一个自己编写的自微分框架，基于 Numpy 实现，整体 API 类似于 PyTorch。本项目用于学习自动微分框架实现，并不适用于一些生产环境。

本代码在以下环境中实现，其余版本未经过太多测试：

+ Python 3.80
+ Numpy

## 安装

```bash
git clone https://github.com/Codle/TinyTorch
cd TinyTorch
pip install -e .
```

## 用法

```python
import tinytorch as ttorch

if __name__ == '__main__':
    a = ttorch.ones((1, 1))
```
