import torch
from torch import nn
from torch.nn import  functional as F
from torch import optim
from loguru import logger

x = torch.tensor([1.,2.,3.])
logger.info('2**x :{}', 2**x) # 打印转换后的精度2**x