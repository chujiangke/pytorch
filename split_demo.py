import torch
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchvision
from torch import nn
from matplotlib import pyplot as plt
from loguru import logger
from torch.nn import  functional as F
from torch import optim
from loguru import logger

x = torch.randn([10, 35, 8])
result = torch.split(x, split_size_or_sections=1, dim=0)
logger.info(len(result))

x = torch.randn([4,3,32,32])
x2 = x.repeat([2,1,3,3]) # 数据复制
logger.info(x2.shape)

x = torch.arange(9)
x = torch.max(x, torch.tensor(2))
logger.info(x)

