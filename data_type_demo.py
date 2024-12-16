import torch
from torch import nn
from torch.nn import  functional as F
from torch import optim
from loguru import logger
 
import torch.utils
import torch.utils.data
import torchvision
from matplotlib import pyplot as plt
import  torch
from matplotlib import pyplot as plt

a = 1.2
aa = torch.tensor(1.2)
logger.info("{} {} {}".format(type(a), type(aa), torch.is_tensor(aa)))

x = torch.tensor([1, 2, 3])
logger.info("x.shape:{} x.device:{} x.dtype:{}".format(x.shape, x.device, x.dtype))

a = torch.tensor([[1, 2], [3, 4]])
logger.info("x.shape:{} x.device:{} x.dtype:{}".format(x.shape, x.device, x.dtype))


a = torch.tensor([True, False])
