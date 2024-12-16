import torch
from torch import nn
from d2l import torch as d2l
from loguru import logger

# 用随机数初始化 X 和 Y
X = torch.randn(1, 1, 6, 8)
Y = torch.randn(1, 1, 6, 7)

conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

lr = 3e-2

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')