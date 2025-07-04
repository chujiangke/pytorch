import torch
from torch import nn
from torchvision.transforms import ToTensor

from PIL import Image, ImageDraw
from numpy import random
import numpy as np
import os.path as osp

from model import resnet18
from generate_data import expand
from config import checkpoint_folder


def expand(img, background=(128, 128, 128), show=False):
    height, width, depth = img.shape
    ratio = random.uniform(1, 2)
    left = random.uniform(0.3 * width, width * ratio - width)
    top = random.uniform(0.3 * width, width * ratio - width)

    while int(left + width) > int(width * ratio) or int(top + height) > int(
        height * ratio
    ):
        ratio = random.uniform(1, 2)
        left = random.uniform(0.3 * width, width * ratio - width)
        top = random.uniform(0.3 * width, width * ratio - width)

    expand_img = np.zeros(
        (int(height * ratio), int(width * ratio), depth), dtype=img.dtype
    )
    expand_img[:, :, :] = background
    expand_img[
        int(top) : int(top + height), int(left) : int(left + width)
    ] = img
    return expand_img


if __name__ == "__main__":
    # 加载模型
    net = resnet18()
    net.linear = nn.Linear(in_features=512, out_features=4, bias=True)
    net.eval()
    totensor = ToTensor()
    net.load_state_dict(torch.load(osp.join(checkpoint_folder, "reg")))

    # 读取图片
    img_path = "img/plane.jpeg"
    img = Image.open(img_path)
    img = np.array(img)
    expand_img = expand(img)
    height, width = expand_img.shape[:2]

    # 对坐标进行解码
    inp = totensor(Image.fromarray((expand_img)).resize((32, 32))).unsqueeze(0)
    out = net(inp)
    xmin, ymin, xmax, ymax = out.view(-1)
    xmin, ymin, xmax, ymax = (
        xmin * width,
        ymin * height,
        xmax * width,
        ymax * height,
    )

    # 绘制预测图像
    expand_img = Image.fromarray(expand_img)
    draw = ImageDraw.ImageDraw(expand_img)
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0), width=10)
    expand_img.save("img/plane_bound_pred.jpg")
    expand_img.show()
