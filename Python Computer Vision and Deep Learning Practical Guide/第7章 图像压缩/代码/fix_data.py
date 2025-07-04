from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from config import DATA_FOLDER, BATCH_SIZE, SIZE
from glob import glob
import os.path as osp
from PIL import Image
from torchvision import transforms
import random
import torch


class FixData(Dataset):
    def __init__(self, folder=DATA_FOLDER, subset="train", transform=None):
        img_paths = glob(osp.join(DATA_FOLDER, "*/*.jpg"))
        train_paths, test_paths = train_test_split(
            img_paths, test_size=0.2, random_state=10
        )
        if subset == "train":
            self.img_paths = train_paths
        else:
            self.img_paths = test_paths

        if transform is None:
            self.transform = transforms.Compose(
                [transforms.Resize((SIZE, SIZE)), transforms.ToTensor()]
            )
        else:
            self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert("L")
        img = self.transform(img)
        # 随机选择顶点
        w = int(SIZE / 3)
        xmin, ymin = (
            int(random.random() * (SIZE - w)),
            int(random.random() * (SIZE - w)),
        )
        img_src = img.clone()
        img_src[:, ymin : ymin + w, xmin : xmin + w] = torch.rand((1, w, w))
        return img_src, img

    def __len__(self):
        return len(self.img_paths)


transform = transforms.Compose(
    [
        transforms.RandomRotation(15),
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
    ]
)

train_data = FixData(subset="train", transform=transform)
val_data = FixData(subset="test")
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE * 2, shuffle=True)


if __name__ == "__main__":
    img_src, img_tgt = train_data[0]
    topil = transforms.ToPILImage()
    img_src = topil(img_src)
    img_tgt = topil(img_tgt)
    import matplotlib.pyplot as plt

    plt.subplot(121)
    plt.imshow(img_src, cmap="gray")
    plt.subplot(122)
    plt.imshow(img_tgt, cmap="gray")
    plt.show()
