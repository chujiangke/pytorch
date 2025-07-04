# 使用scikit-learn中的kd-tree进行搜索

from PIL import Image
from sklearn.neighbors import KDTree
from torchvision import transforms
from glob import glob
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt

# from classification import resnet18
from cluster import net

# from compare import cls_net, enc_net, extract_feature_cls, extract_feature_enc
from config import CHECKPOINT, DATA_FOLDER, device

method = "classification"
# method = "autoencoder"
img_path = "/data/pubfig_faces/Ali Landry/Ali Landry67.jpg"

transform = transforms.Compose(
    [transforms.Resize((128, 128)), transforms.ToTensor()]
)


def extract_feature(img_path):
    img = Image.open(img_path)
    # if method == "autoencoder":
    #     img = img.convert("L")
    img_tensor = transform(img).unsqueeze(0).to(device)
    # if method == "classification":
    #     feature = extract_feature_cls(cls_net, img_tensor)
    # elif method == "autoencoder":
    #     feature = extract_feature_enc(enc_net, img_tensor)
    feature = net(img_tensor)
    return feature.view(-1)


if __name__ == "__main__":
    img_paths = glob(os.path.join(DATA_FOLDER, "*/*.jpg"))
    x = []
    for path in tqdm(img_paths):
        x.append(extract_feature(path).cpu().data.numpy())
    tree = KDTree(np.array(x), leaf_size=2)
    v = extract_feature(img_path).unsqueeze(0).cpu().data.numpy()
    dist, ind = tree.query(v, k=9)
    cnt = 1
    for i in ind.reshape(-1):
        plt.subplot(3, 3, cnt)
        plt.title(os.path.basename(img_paths[i]).split(".")[0])
        # 消除刻度
        plt.xticks([])
        plt.yticks([])
        plt.imshow(Image.open(img_paths[i]))
        cnt += 1
    plt.show()
