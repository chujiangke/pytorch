import torch
from torch import nn
from loguru import logger


x = torch.randn(4, 3, 32, 32)
layer = nn.Conv2d(3, 16, kernel_size=3)
out = layer(x)

logger.info("out shape:{}".format(out.shape))
