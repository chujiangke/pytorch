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


def relu(x):
    return torch.max(x, torch.tensor(0))
x = torch.arange(-4, 4)
logger.info(x)
logger.info("relu:{}".format(relu(x)))


x = torch.arange(9)
x = torch.min(torch.max(x, torch.tensor(2)), torch.tensor(7)) # 限幅为 2~7
logger.info("x:{}".format(x))
