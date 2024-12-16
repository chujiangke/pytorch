import torch
from torch import nn
from loguru import logger


fc = nn.Linear(3, 4)
logger.info('fc.bias:{}', fc.bias) 

x = torch.randn(2,4) # 2 个样本，特征长度为 4 的张量
w = torch.ones(4, 3) # 定义 W 张量
b = torch.zeros(3) # 定义 b 张量

o = x@w+b # X@W+b 运算