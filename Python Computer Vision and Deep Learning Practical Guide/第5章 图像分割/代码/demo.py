import torch
from torchvision.transforms import ToPILImage

import matplotlib.pyplot as plt

from model import ResNet18Unet
from data import SegmentationData
from config import checkpoint, device

net = ResNet18Unet().to(device)
net.load_state_dict(torch.load(checkpoint)["params"])

test_data = SegmentationData(subset="test")
img, _ = test_data[10]
mask = net(img.unsqueeze(0).to(device))
topil = ToPILImage()
mask_img = torch.argmax(mask, dim=1).squeeze(0).squeeze(0)
plt.subplot(121)
plt.imshow(topil(img))
plt.subplot(122)
plt.imshow(mask_img.data.cpu().numpy())
plt.savefig("img/result.jpg")
plt.show()
