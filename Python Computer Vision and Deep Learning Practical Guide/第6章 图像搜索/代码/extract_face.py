# 因为很多图片中不止一个人脸，所以提取人脸这一步不能直接使用dlib检测
# 只能通过原始的标注文件坐标来提取
import pandas as pd
import os

# from PIL import Image
import numpy as np
import re
import time
import cv2
from tqdm import tqdm

path = "/data/pubfig/dev_urls.txt"
folder = "/data/pubfig"
# 第一行是注释，第二行标题前面有个#号
df = pd.read_table(path, header=1)
cols = df.columns[1:]
df = df.drop(["md5sum"], axis=1)
df.columns = cols
print(df.head())

# cnt = 0
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    src_folder = os.path.join(folder, row["person"])
    src_path = os.path.join(
        src_folder, row["person"] + str(row["imagenum"]) + ".jpg"
    )
    target_folder = re.sub("pubfig", "pubfig_faces", src_folder)
    target_path = re.sub("pubfig", "pubfig_faces", src_path)
    if os.path.exists(src_path):
        # img = np.array(Image.open(src_path).convert("BGR"))
        img = cv2.imread(src_path)
        rect = row["rect"]
        # xmin,ymin,xmax,ymax
        rect = [int(r) for r in rect.split(",")]
        face = img[rect[1] : rect[3], rect[0] : rect[2], :]
        # face = Image.fromarray(face)
        # print(src_path)
        # print(rect)
        # face.show()
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        # face.save(target_path)
        # cv2.imshow("face", face)
        cv2.imwrite(target_path, face)
        # if cnt > 10:
        # break
        # cnt += 1

