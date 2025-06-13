import torch
import torch.utils
import torch.utils.data
import torchvision

from torch import nn
from torch.nn import  functional as F
from torch import optim
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from torch.nn import functional as F # 导入函数库
from loguru import logger


out = torch.randn(4,10) # 随机模拟网络输出
y = torch.tensor([2,3,2,0]) # 随机构造样本真实标签
y = F.one_hot(y, num_classes=10) # one-hot 编码
logger.info('y:{}', y) 

loss = F.mse_loss(y, out) # 计算每个样本的 MSE
logger.info('loss:{}', loss) 
