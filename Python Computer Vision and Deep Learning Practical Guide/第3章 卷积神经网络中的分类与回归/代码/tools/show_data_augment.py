# tools/show_data_augment.py
# 展示增强操作前后的图像变化
# 请在当前目录下运行此文件
import torchvision
from torchvision import transforms
import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import sys

# 将上级目录加入系统目录
sys.path.append("..")
from config import data_folder


# 批量显示图片
def show_batch(display_transform=None):
    # 重新定义一个不带Normalize的dataloader，因为Normalize处理后的图片很难辨认
    if display_transform is None:
        display_transform = transforms.ToTensor()
    display_set = torchvision.datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=display_transform
    )
    display_loader = torch.utils.data.DataLoader(display_set, batch_size=32)
    topil = transforms.ToPILImage()
    # DataLoader对象无法直接取index，可以通过这种方式取其中元素
    for batch_img, batch_label in display_loader:
        # 建立张量网格
        grid = make_grid(batch_img, nrow=8)
        # 将张量转成图片
        grid_img = topil(grid)
        plt.figure(figsize=(15, 15))
        plt.imshow(grid_img)
        grid_img.save("../img/trans_cifar10.png")
        plt.show()
        break


if __name__ == "__main__":
    # 训练过程中的图像增强与数据转换
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    show_batch(transform_train)
