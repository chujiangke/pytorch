import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import device, data_folder
from model import vgg11
from data import create_datasets


def lr_find(
    net,
    optimizer_class,
    dataloader,
    criteron,
    lr_list=[1 * 10 ** (i / 2) for i in range(-20, 0)],
    show=False,
    test_times=10,
):
    """
    net: 模型
    optimizer_class: 优化器类
    dataloader: 数据
    criteron: 损失函数
    lr_list: 学习率列表
    show: 是否显示结果
    test_time: 实验次数
    """
    # 复制模型参数
    params = net.state_dict().copy()
    # 损失值矩阵
    loss_matrix = []
    for i, (img, label) in enumerate(dataloader):
        img, label = img.to(device), label.to(device)
        loss_list = []
        for lr in tqdm(lr_list):
            # 重新加载原始参数
            net.load_state_dict(params)
            # 训练模型
            out = net(img)
            optimizer = optimizer_class(
                net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
            )
            loss = criteron(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 计算更新模型之后的损失
            new_out = net(img)
            new_loss = criteron(new_out, label)
            loss_list.append(new_loss.item())

        loss_matrix.append(loss_list)
        # plt.plot([np.log(lr) for lr in lr_list],loss_list)
        if i + 1 == test_times:
            break
    loss_matrix = np.array(loss_matrix)
    loss_matrix = np.mean(loss_matrix, axis=0)
    if show:
        plt.plot([np.log10(lr) for lr in lr_list], loss_matrix)
        plt.savefig("img/lr_find.jpg")
        plt.show()

    # 计算loss下降幅度，寻找最佳学习率
    decrease = [
        loss_matrix[i + 1] - loss_matrix[i] for i in range(len(lr_list) - 1)
    ]
    max_decrease = np.argmin(decrease)
    best_lr = lr_list[max_decrease]
    return best_lr


if __name__ == "__main__":
    net = vgg11().to(device)
    trainloader, _ = create_datasets(data_folder)
    criteron = CrossEntropyLoss()
    lr_list = [1 * 10 ** (i / 3) for i in range(-30, 0)]
    lr_find(net, SGD, trainloader, criteron, show=True)

