# 使用dlib库验证图片是否正常,不正常的图片清理掉
# 如果dlib下载失败，可以尝试直接下载dlib的whl文件进行安装
# dlib需要cmake

from glob import glob
from PIL import Image
import numpy as np
import dlib
import os
from tqdm import tqdm


def remove(path):
    try:
        os.remove(path)
    except:
        pass


folder = "D:\\datasets\\pubfig"

image_paths = glob(os.path.join(folder, "*\\*.jpg"))

face_detector = dlib.get_frontal_face_detector()
# broken_image_paths = []
f = open("D:\\datasets\\pubfig\\broken_list.txt", "w")
for path in tqdm(image_paths):
    try:
        img = np.array(Image.open(path))
    except:
        remove(path)
        continue
    try:
        face_rects = face_detector(img)
    except:
        f.write(path + "\n")
        remove(path)
        continue
    if len(face_rects) == 0:
        remove(path)
        continue
    # for rect in face_rects:
    # (xmin,ymin),(xmax,ymax)
    # face = Image.fromarray(img[ymin:ymax, xmin:xmax]).show()
    # print("rect")
f.close()

