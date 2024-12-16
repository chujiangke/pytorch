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
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
    

x = torch.linspace(-8.0, 8, 100)
y = torch.linspace(-8.0, 8, 100)

x, y = torch.meshgrid(x, y)
x.shape, y.shape

z = torch.sqrt(x**2+y**2)
z= torch.sin(z)/z

fig = plt.figure()
ax = Axes3D(fig)
ax.contour3D(x.numpy(), y.numpy(), z.numpy(), 50)
plt.show()
