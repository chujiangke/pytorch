import sys
import os


import torch
from torch import nn, optim
from captcha_model import net
from captcha_data import dataloader, val_loader
from tqdm import tqdm

epoch_lr = [
    (1000, 0.1),
    (100, 0.01),
    (100, 0.001),
    (100, 0.0001),
]  # [(300,0.05),(100,0.001),(100,0.0001)]
# 自动检测GPU可用性并选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criteron = nn.MultiLabelSoftMarginLoss()


def train():
    net.to(device)
    accuracies = []
    losses = []
    val_accuracies = []
    val_losses = []
    for n, (num_epoch, lr) in enumerate(epoch_lr):
        # 优化器也可以多尝试一下，一般来说使用SGD的对应的学习率会比Adam大一个数量级
        optimizer = optim.SGD(
            net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )
        for epoch in range(num_epoch):
            epoch_loss = 0.0
            epoch_acc = 0.0
            for i, (img, label) in tqdm(enumerate(dataloader)):
                out = net(img.to(device))
                label = label.to(device)
                # 清空net里面所有参数的梯度
                optimizer.zero_grad()
                # 计算预测值与目标值之间的损失
                loss = criteron(out, label.to(device))
                # 计算梯度
                loss.backward()
                # 根据梯度调整net中的参数
                optimizer.step()
                # 整理输出，方便与标签进行对比
                predict = torch.argmax(out.view(-1, 36), dim=1)
                true_label = torch.argmax(label.view(-1, 36), dim=1)
                epoch_acc += torch.sum(predict == true_label).item()
                epoch_loss += loss.item()
            # 每训练三次验证一次
            if epoch % 3 == 0:
                # no_grad()模式不计算梯度，可以跑得快一点
                with torch.no_grad():
                    val_loss = 0.0
                    val_acc = 0.0
                    for i, (img, label) in tqdm_notebook(enumerate(val_loader)):
                        out = net(img.to(device))
                        label = label.to(device)
                        loss = criteron(out, label.to(device))
                        predict = torch.argmax(out.view(-1, 36), dim=1)
                        true_label = torch.argmax(label.view(-1, 36), dim=1)
                        val_acc += torch.sum(predict == true_label).item()
                        val_loss += loss.item()
                val_acc /= len(val_loader.dataset) * 4
                val_loss /= len(val_loader)
            epoch_acc /= len(dataloader.dataset) * 4
            epoch_loss /= len(dataloader)
            print(
                "epoch : {} , epoch loss : {} , epoch accuracy : {}".format(
                    epoch + sum([e[0] for e in epoch_lr[:n]]),
                    epoch_loss,
                    epoch_acc,
                )
            )
            if epoch % 3 == 0:
                print(
                    "epoch : {} , val loss : {} , val accuracy : {}".format(
                        epoch + sum([e[0] for e in epoch_lr[:n]]),
                        val_loss,
                        val_acc,
                    )
                )
                for i in range(3):
                    val_accuracies.append(val_acc)
                    val_losses.append(val_loss)
            accuracies.append(epoch_acc)
            losses.append(epoch_loss)


if __name__ == "__main__":
    train()
