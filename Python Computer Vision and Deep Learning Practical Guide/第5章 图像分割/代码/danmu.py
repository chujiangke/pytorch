# 演示如何制作能够过滤前景的弹幕

import torch
from PIL import ImageDraw, ImageFont, Image
from data import SegmentationData
from model import ResNet18Unet
from torchvision import transforms
from config import device, checkpoint
import numpy as np

# 加载模型
net = ResNet18Unet().to(device)
net.load_state_dict(torch.load(checkpoint)["params"])

# 从验证集中取图片
test_data = SegmentationData(subset="test")
img, _ = test_data[10]
mask = net(img.unsqueeze(0).to(device))
topil = transforms.ToPILImage()
mask_img = torch.argmax(mask, dim=1).squeeze(0).squeeze(0)
im = topil(img)
imcopy = im.copy()
font = ImageFont.truetype("simsun.ttf", size=15)
draw = ImageDraw.Draw(im)
mask = mask_img.cpu().data.numpy()
for j in range(10):
    draw.text((20, 20 * j), u"好可爱的小狗子", font=font, fill=(0, 0, 0))
im.save("img/danmu1.jpg")
im_array = np.array(im)
im_copy_array = np.array(imcopy)
im_array[mask == 1] = im_copy_array[mask == 1]
im = Image.fromarray(im_array)
im.show()
im.save("img/danmu2.jpg")
