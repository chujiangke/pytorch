# 感知哈希算法

from load_cifar import Cifar
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np


def hash(img):
    # 灰度化
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 缩小尺寸
    img = cv2.resize(img, (8, 8))
    # 简化色彩
    # 255 / 64 = 4
    img = (img / 4).astype(np.uint8) * 4
    # 计算均值
    m = np.mean(img)
    img[img <= m] = 0
    img[img > m] = 1
    print(img.shape)
    plt.imshow(img * 255, cmap="gray")
    return img.reshape(-1)


img1 = cv2.imread("img/panda1.jpg", 0)
img2 = cv2.imread("img/panda2.jpg", 0)
img3 = cv2.imread("img/husky1.jpg", 0)

hash_img1 = hash(img1)
hash_img2 = hash(img2)
hash_img3 = hash(img3)

distance1 = np.sum(hash_img1 == hash_img2) / hash_img1.shape[0]
distance2 = np.sum(hash_img1 == hash_img3) / hash_img1.shape[0]

plt.subplot(131)
plt.imshow(img1)
plt.title("source ")
plt.subplot(132)
plt.imshow(img2)
plt.title("distance: {}".format(distance1))
plt.subplot(133)
plt.imshow(img3)
plt.title("distance: {}".format(distance2))
plt.savefig("img/compare.png")
plt.show()

# 接下来就可以按照图像搜索的方法去搜索最相思图片了。
