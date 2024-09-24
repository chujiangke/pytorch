import torch
from torch import autograd

a = torch.tensor(1.)
b = torch.tensor(2.)
c = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)

y = a * w **2 + b * w + c
w_grad = autograd.grad(y, [w])
print('w_grad :', w_grad)