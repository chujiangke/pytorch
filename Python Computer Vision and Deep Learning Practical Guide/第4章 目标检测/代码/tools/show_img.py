# 展示合并之后的图片
# 在tools目录下运行
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import sys

sys.path.append("..")

from generate_data import get_background, combine_img

background_paths = get_background()
dog = Image.open("../img/dog.png").convert("RGB")
combined_img, box, _ = combine_img(background_paths[0], dog)
img = Image.fromarray(combined_img)
draw = ImageDraw.Draw(img)
for b in box:
    cx, cy, w = b
    xmin = cx - w / 2
    ymin = cy - w / 2
    xmax = cx + w / 2
    ymax = cy + w / 2
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 0, 255), width=5)
plt.imshow(img)
plt.savefig("../img/object.jpg")
plt.show()
