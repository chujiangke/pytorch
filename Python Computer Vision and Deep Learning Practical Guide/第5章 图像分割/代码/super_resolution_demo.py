import torch
from torch import nn
from torchvision.transforms import ToPILImage

import matplotlib.pyplot as plt
from PIL import Image

from model import ResNet18Unet
from super_resolution_data import SuperResolutionData
from config import sr_checkpoint, device

# net = ResNet18Unet(num_classes=3).to(device)
net = ResNet18Unet(num_classes=1)
net.firstconv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
net = net.to(device)
net.load_state_dict(torch.load(sr_checkpoint)["params"])

test_data = SuperResolutionData(subset="test", demo=True)
low, high, (high_cb, high_cr, low_cb, low_cr) = test_data[0]
mask = net(low.unsqueeze(0).to(device)).squeeze(0).data.cpu()
topil = ToPILImage()
plt.subplot(131)
plt.title("low")
low_rgb = Image.merge("YCbCr", [topil(low), low_cb, low_cr]).convert("RGB")
plt.imshow(low_rgb)
plt.subplot(132)
plt.title("rebuilt")
rebuilt = mask + low
rebuilt_rgb = Image.merge("YCbCr", [topil(rebuilt), low_cb, low_cr]).convert(
    "RGB"
)
plt.imshow(rebuilt_rgb)
plt.subplot(133)
plt.title("high")
high_rgb = Image.merge("YCbCr", [topil(high), high_cb, high_cr]).convert("RGB")
plt.imshow(high_rgb)
plt.savefig("img/sr_result.jpg")
plt.show()
