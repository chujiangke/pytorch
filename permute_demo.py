import torch
import torch.utils
import torch.utils.data
from loguru import logger


x = torch.randn([2, 3, 32, 32])
x = x.permute(0,2,3,1) # 交换维度
logger.info('x.shape:{}', x.shape) 

x = x.transpose(2, 2)
logger.info('x.shape:{}', x.shape) 
