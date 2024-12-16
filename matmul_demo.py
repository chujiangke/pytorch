import torch 
from torch.nn import functional as F

x = torch.randn([2, 784])
w1 = torch.randn([784, 256])
b1 = torch.zeros([256])
ol = torch.matmul(x, w1) + b1 # 线性变换
ol = F.relu(ol)

print("output:", ol.shape)
