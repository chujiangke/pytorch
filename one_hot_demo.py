import torch
from torch import nn
from torch.nn import  functional as F
from torch import optim

import torch.utils
import torch.utils.data
import torchvision
from matplotlib import pyplot as plt
import torch
from matplotlib import pyplot as plt


def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out

y = torch.tensor([0, 1, 2, 3])# 数字编码的4个样本
y = one_hot(y, depth=10) # one-hot 编码,指定类别总数为10
print(y)

