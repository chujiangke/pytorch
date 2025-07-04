import torch
from torch import nn
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.models import resnet18

import matplotlib.pyplot as plt
from PIL import ImageDraw

from data import DetectionData, TestTransform
from config import checkpoint


import numpy as np


def py_cpu_nms(boxes, scores, thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按分数排序
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        # 保留分数最高的box
        i = order[0]
        keep.append(i)
        # 计算box之间的交并比
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算交并比
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留与高分box交并比低于阈值的box
        inds = np.where(ovr <= thresh)[0]
        # ovr中的index比order中小1
        order = order[inds + 1]

    return keep


# torchvision要求是0.2.2版本
# 因为0.2.2中的resnet18最后的池化层是AdaptiveAvgPool，可以适应尺寸变化，免去尺寸不匹配的烦恼

dataset = DetectionData(subset="test", transform=TestTransform())
img, boxes = dataset[6]  # 此图片出现重叠窗口

net = resnet18(pretrained=False)
net.fc = nn.Linear(512, 54)
net.load_state_dict(torch.load(checkpoint))
net.eval()

out = net(img.unsqueeze(0))
out_label = out[:, :18].view(-1, 9, 2)
out_offset = out[:, 18:45]
out_score = out[:, 45:]

predict_label = torch.argmax(out_label, dim=2)

predict_offset = out_offset.view(-1, 9, 3)

anchors = torch.Tensor(
    [
        [100, 100, 300],
        [200, 100, 300],
        [300, 100, 300],
        [100, 200, 300],
        [200, 200, 300],
        [300, 200, 300],
        [100, 300, 300],
        [200, 300, 300],
        [300, 300, 300],
    ]
)

predict_box = predict_offset * anchors


topil = ToPILImage()
img_pil = topil(img)
img_pil_nms = img_pil.copy()
draw = ImageDraw.Draw(img_pil)
positive_boxes = []
positive_scores = []
for i, b in enumerate(predict_box[0]):
    if predict_label[0][i] == 1:
        xmin = b[0] - b[2] / 2
        ymin = b[1] - b[2] / 2
        xmax = b[0] + b[2] / 2
        ymax = b[1] + b[2] / 2
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0))
        draw.text((xmin, ymin), "{}".format(out_score[:, i].item()))
        positive_boxes.append(
            [xmin.item(), ymin.item(), xmax.item(), ymax.item()]
        )
        positive_scores.append(out_score[:, i].item())
plt.figure(figsize=(5, 5))
plt.imshow(img_pil)
plt.savefig("img/dog_detect.jpg")


draw_nms = ImageDraw.Draw(img_pil_nms)
boxes = np.array(positive_boxes)
scores = np.array(positive_scores)
keep_idx = py_cpu_nms(boxes, scores, 0.4)
keep_box = boxes[keep_idx]
for i, b in enumerate(keep_box):
    xmin, ymin, xmax, ymax = b
    draw_nms.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0))
    draw_nms.text((xmin, ymin), "{}".format(scores[i].item()))
plt.figure(figsize=(5, 5))
plt.imshow(img_pil_nms)
plt.savefig("img/dog_detect_nms.jpg")

plt.show()

