import torch
from torch import nn
from torch.nn import  functional as F
from torch import optim
import numpy as np

import torch.utils
import torch.utils.data
import torchvision
from matplotlib import pyplot as plt
import torch
from matplotlib import pyplot as plt
from loguru import logger


a = torch.tensor(np.pi, dtype=torch.float32)
logger.info("%.20f"%a)

if a.dtype != torch.float32: # 如果精度不符合要求，则进行转换
    a = a.type(torch.float32) # tensor.type 函数可以完成精度转换
logger.info('after :{}',a.dtype) # 打印转换后的精度
