from torch import nn, optim
import torch
import os.path as osp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data import train_loader, val_loader
from auto_encoder import AutoEncoder
from config import BATCH_SIZE, EPOCH_LR, device, CHECKPOINT


def train():
    net = AutoEncoder(pretrained=True).to(device)
    criteron = nn.L1Loss()
    ckpt = osp.join(CHECKPOINT, "net.pth")
    writer = SummaryWriter("log")
    if osp.exists(ckpt):
        net.load_state_dict(torch.load(ckpt))
    for n, (num_epoch, lr) in enumerate(EPOCH_LR):
        optimizer = optim.Adam(net.parameters(), lr=lr)
        for epoch in range(num_epoch):
            epoch_loss = 0.0
            for i, (src, target) in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):
                optimizer.zero_grad()
                src, target = src.to(device), target.to(device)
                out = net(src)
                loss = criteron(out, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(
                "epoch: {} epoch_loss {}".format(
                    sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
                    epoch_loss / len(train_loader),
                )
            )
            writer.add_scalar(
                "epoch_loss",
                epoch_loss / len(train_loader),
                sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
            )
            with torch.no_grad():
                val_loss = 0.0
                for i, (src, target) in tqdm(
                    enumerate(val_loader), total=len(val_loader)
                ):
                    src, target = src.to(device), target.to(device)
                    out = net(src)
                    loss = criteron(out, target)
                    val_loss += loss.item()
            print(
                "val: {} val_loss {}".format(
                    sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
                    val_loss / len(val_loader),
                )
            )
            writer.add_scalar(
                "val_loss",
                val_loss / len(val_loader),
                sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
            )
            torch.save(net.state_dict(), ckpt)
    writer.close()


if __name__ == "__main__":
    train()
