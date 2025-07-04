import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from lr_find import lr_find
from model import resnet18
from data import create_datasets
from config import data_folder, batch_size, device, epochs
from generate_data import BoxData
from train_val import train_val

net = resnet18()
net.linear = nn.Linear(in_features=512, out_features=4, bias=True)

train_loader, val_loader = create_datasets(data_folder)

net.to(device)

traindata = BoxData(train_loader.dataset)
trainloader = DataLoader(
    traindata, batch_size=batch_size, shuffle=True, num_workers=4
)
criteron = nn.L1Loss()

valdata = BoxData(val_loader.dataset)
valloader = DataLoader(
    valdata, batch_size=batch_size, shuffle=True, num_workers=4
)

# 可以预先进行学习率搜索，根据曲线确定初始学习率
# best_lr = lr_find(net, optim.SGD, train_loader, criteron)
# print("best_lr", best_lr)

train_val(
    net, trainloader, valloader, criteron, epochs, device, model_name="reg"
)
