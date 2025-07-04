import torchvision
from torchvision import transforms
import torch

from config import data_folder, batch_size

# 创建数据集
def create_datasets(data_folder, transform_train=None, transform_test=None):
    # 训练过程中的图像增强与数据转换
    if transform_train is None:
        transform_train = transforms.Compose(
            [
                # 扩张之后再随机剪裁
                transforms.RandomCrop(32, padding=4),
                # 随即翻转
                transforms.RandomHorizontalFlip(),
                # 将图片转换成张量
                transforms.ToTensor(),
                # 根据Cifar10数据集的各个通道上的像素均值和方差进行归一化处理，使模型更易拟合
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    # 测试过程中的数据转换
    if transform_test is None:
        transform_test = transforms.Compose(
            [
                # 测试过程中无需做图形变换
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    # 训练集
    trainset = torchvision.datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform_train
    )
    # 训练集loader
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    # 测试集
    testset = torchvision.datasets.CIFAR10(
        root=data_folder, train=False, download=True, transform=transform_test
    )
    # 测试集loader
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return trainloader, testloader

