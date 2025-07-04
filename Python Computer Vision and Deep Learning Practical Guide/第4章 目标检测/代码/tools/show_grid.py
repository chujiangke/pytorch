# 展示九宫格的标记情况
from torchvision import transforms
from PIL import Image, ImageDraw
import sys

sys.path.append("..")
from data import DetectionData, TrainTransform
from mark_data import mark_data


data = DetectionData(subset="train", transform=TrainTransform())
# 选择一张展示效果较好的图片
img, boxes = data[11]
topil = transforms.ToPILImage()
labels, _, _ = mark_data(boxes)
img = topil(img)
# width和height相等
width, height = img.size
for i in range(9):
    xmin = (i % 3) * (width // 3)
    ymin = (i // 3) * (height // 3)
    xmax = xmin + (width // 3)
    ymax = ymin + (height // 3)
    draw = ImageDraw.Draw(img)
    if labels[0, i].item() == 1:
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 0, 255), width=6)
    else:
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(255, 0, 0), width=2)
for box in boxes:
    cx, cy, w = box
    xmin = cx - w / 2
    ymin = cy - w / 2
    xmax = cx + w / 2
    ymax = cy + w / 2
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0), width=3)
img.save("../img/grids.jpg")
img.show()
