import torch
from torch import nn
from torch.nn import  functional as F
from torch import optim
from loguru import logger
 
import torch.utils
import torch.utils.data
import torchvision
from matplotlib import pyplot as plt
import torch
from  matplotlib import pyplot as plt
from torch import autograd
import numpy as np


x = torch.tensor(1.0, requires_grad=False)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
y = x*w + b

dy_dw, dy_db = autograd.grad(y, [w, b])
print(dy_dw, dy_db)
logger.info("x {}".format(torch.from_numpy(np.array([1,2,3.]))))

