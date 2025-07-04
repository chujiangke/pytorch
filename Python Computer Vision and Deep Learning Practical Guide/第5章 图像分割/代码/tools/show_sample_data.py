import torch
from torch import nn
from torchvision.transforms import ToPILImage

import matplotlib.pyplot as plt
from PIL import Image
import sys

sys.path.append("..")
from super_resolution_data import SuperResolutionData


test_data = SuperResolutionData(subset="test", demo=True)
low, high, (high_cb, high_cr, low_cb, low_cr) = test_data[0]
topil = ToPILImage()
plt.subplot(121)
plt.title("low")
low_rgb = Image.merge("YCbCr", [topil(low), low_cb, low_cr]).convert("RGB")
plt.imshow(low_rgb)
plt.subplot(122)
plt.title("high")
high_rgb = Image.merge("YCbCr", [topil(high), high_cb, high_cr]).convert("RGB")
plt.imshow(high_rgb)
plt.savefig("../img/sr_sample.jpg")
plt.show()
