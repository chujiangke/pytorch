import torch
from torch import nn
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchvision
from matplotlib import pyplot as plt
from loguru import logger
from torch.nn import  functional as F
from torch import optim

logger.info('{}', torch.full([], -1)) 
logger.info('{}', torch.full([1], 9)) 
logger.info('{}', torch.full([2,2], 99)) 
logger.info('{}', torch.randn(2,2)) 
