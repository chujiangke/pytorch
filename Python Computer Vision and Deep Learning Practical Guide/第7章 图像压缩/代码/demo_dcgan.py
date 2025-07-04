import torch
from torchvision import transforms
from model import generator
import matplotlib.pyplot as plt
from config import checkpoint_G
import os.path as osp

topil = transforms.ToPILImage()
net = generator()
if osp.exists(checkpoint_G):
    net.load_state_dict(torch.load(checkpoint_G))
    print("model loaded")
for i in range(9):
    input_array = torch.randn(1, 100, 1, 1)
    out_tensor = net(input_array).squeeze(0)
    out_img = topil(out_tensor)
    plt.subplot(330 + i + 1)
    plt.imshow(out_img,cmap = "gray")
plt.show()
