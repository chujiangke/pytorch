import numpy as np 
from glob import glob
import os
from tqdm import tqdm
import re
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import resize

img_paths = sorted(glob(r"D:\datasets\EnglishHnd\English\Hnd\Img\*\*.png"))


def binary(img):
    # 二值化
    rows,cols = img.shape
    for i in range(rows):
        for j in range(cols):
            if img[i,j] < 0.5:
                img[i,j] = 0
            else:
                img[i,j] = 1
    return img

def preprocess(img):
    width ,height = img.shape
    rows,cols = np.where(img < 1.)
    x_min,x_max = min(rows),max(rows)
    y_min,y_max = min(cols),max(cols)
    # 边长
    size = max(y_max - y_min,x_max - x_min)
    # 字符旁边留一定范围的空白
    x_empty = (size-(x_max - x_min)) // 2
    y_empty = (size-(y_max - y_min)) // 2  
    # 裁剪图片
    img = img[max(x_min-x_empty,0):min(x_max+x_empty,width),max(y_min-y_empty,0):min(y_max+y_empty,height)]
    img = resize(img,(64,64))
    return img

def hog_features(img_path):
    img = imread(img_path,as_grey=True) # 读取灰度图片
    img = binary(img)
    img = preprocess(img)
    hog_feat = hog(img,orientations=9,pixels_per_cell=[5,5],cells_per_block=[3,3])
    return hog_feat

for img_path in tqdm(img_paths):
    if os.path.exists(re.sub(r".png",".npy",img_path)):
        continue
    np.save(re.sub(r".png",".npy",img_path),hog_features(img_path)) 
