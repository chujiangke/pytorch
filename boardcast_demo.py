
import torch
from loguru import logger


a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))

before = id(a)
a = a + b
after = id(a)

logger.info(before)
logger.info(after)