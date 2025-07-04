from torchvision.models import resnet18
from torch import nn
import torch
import os
from config import CHECKPOINT, DATA_FOLDER, SIZE, device
from glob import glob
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

cos = True
ckpt = os.path.join(CHECKPOINT, "face_cos_{}.pth".format(cos))
# 搭建模型
net = resnet18()
net.fc = nn.Linear(512, 512)
net.to(device)
net.load_state_dict(torch.load(ckpt))

transform = transforms.Compose(
    [transforms.Resize((SIZE, SIZE)), transforms.ToTensor()]
)

# clf = KMeans(10)
clf = DBSCAN(eps=2)


def load_img(root=DATA_FOLDER):
    img_paths = glob(os.path.join(root, "*/*.jpg"))
    return img_paths


def extract_feature(img_path, transform=transform):
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0).to(device)
    return net(img)


def cluster(clf):
    img_paths = load_img()  # [:1000]
    features = []
    for path in tqdm(img_paths):
        feature = extract_feature(path)
        features.append(feature.cpu().data.numpy().reshape(-1))
    print("fitting")
    clf.fit(features)
    # 只挑3个聚类的结果进行展示
    for i in range(3):
        indices, = np.where(clf.labels_ == i)
        print(indices[:9])
        plt.figure()
        for j in range(9):
            plt.subplot(3, 3, j + 1)
            # 消除刻度
            plt.xticks([])
            plt.yticks([])
            # 抽取前18张图片画个图
            try:
                plt.imshow(Image.open(img_paths[indices[j]]))
            except:
                continue
        plt.show()


if __name__ == "__main__":
    cluster(clf)
