from torch import nn, optim
from auto_encoder import AutoEncoder
from config import device, EPOCH_LR, CHECKPOINT

# from data import train_loader, val_loader
from fix_data import train_loader, val_loader

from torchvision.models import resnet18
from tqdm import tqdm
import os
import torch
from torch.utils.tensorboard import SummaryWriter


def feature_map_loss(D, fake_img, img):
    fm_criteron = nn.MSELoss()
    fm_loss = 0.0
    f1 = D.maxpool(D.relu(D.bn1(D.conv1(img))))
    f1_fake = D.maxpool(D.relu(D.bn1(D.conv1(fake_img))))
    fm_loss += fm_criteron(f1_fake, f1)
    f2 = D.layer1(f1)
    f2_fake = D.layer1(f1_fake)
    fm_loss += fm_criteron(f2_fake, f2)
    f3 = D.layer2(f2)
    f3_fake = D.layer2(f2_fake)
    fm_loss += fm_criteron(f3_fake, f3)
    f4 = D.layer3(f3)
    f4_fake = D.layer3(f3_fake)
    fm_loss += fm_criteron(f4_fake, f4)
    f5 = D.layer4(f4)
    f5_fake = D.layer4(f4_fake)
    fm_loss += fm_criteron(f5_fake, f5)
    return fm_loss


G = AutoEncoder().to(device)
D = resnet18(num_classes=1)
D.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
D = D.to(device)

# 图像压缩
# checkpoint_G = os.path.join(CHECKPOINT, "G.pth")
# checkpoint_D = os.path.join(CHECKPOINT, "D.pth")
# 图像修复
checkpoint_G = os.path.join(CHECKPOINT, "G_fix.pth")
checkpoint_D = os.path.join(CHECKPOINT, "D_fix.pth")

if os.path.exists(checkpoint_G):
    G.load_state_dict(torch.load(checkpoint_G))
if os.path.exists(checkpoint_G):
    D.load_state_dict(torch.load(checkpoint_D))

BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss()
writer = SummaryWriter("log")
for n, (num_epoch, lr) in enumerate(EPOCH_LR):
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(num_epoch):
        D.train()
        G.train()
        for i, (img_src, img_tgt) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            # 训练判别器
            D_optimizer.zero_grad()
            mini_batch = img_src.size()[0]
            # 建立标签
            y_real = torch.ones(mini_batch)
            y_fake = torch.zeros(mini_batch)
            # 计算真实图片误差
            img_src, img_tgt, y_real, y_fake = (
                img_src.to(device),
                img_tgt.to(device),
                y_real.to(device),
                y_fake.to(device),
            )
            D_result = torch.sigmoid(D(img_tgt)).squeeze()
            D_real_loss = BCE_loss(D_result, y_real)
            # 计算伪图片误差
            img_fake = G(img_src)
            D_result = torch.sigmoid(D(img_fake)).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake)
            # 反向传播
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()

            # 训练AutoEncoder
            G_optimizer.zero_grad()
            img_fake = G(img_src)
            AE_train_loss = MSE_loss(img_fake, img_tgt)
            # AE_train_loss.backward()
            # G_optimizer.step()

            # 训练生成器
            # G_optimizer.zero_grad()
            img_fake = G(img_src)
            D_result = torch.sigmoid(D(img_fake)).squeeze()
            # G_train_loss = BCE_loss(D_result, y_real) + feature_map_loss(
            #     D, img_fake, img
            # )
            G_train_loss = AE_train_loss + feature_map_loss(
                D, img_fake, img_tgt
            )
            G_train_loss.backward()
            G_optimizer.step()

        print(
            "D train loss : {} , G train loss : {}, AE train Loss : {}".format(
                D_train_loss, G_train_loss, AE_train_loss
            )
        )
        writer.add_scalar(
            "D_train_loss",
            D_train_loss / len(train_loader),
            sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
        )
        writer.add_scalar(
            "G_train_loss",
            G_train_loss / len(train_loader),
            sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
        )
        writer.add_scalar(
            "AE_train_loss",
            AE_train_loss / len(train_loader),
            sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
        )
        with torch.no_grad():
            D.eval()
            G.eval()
            for i, (img_src, img_tgt) in tqdm(
                enumerate(val_loader), total=len(val_loader)
            ):
                mini_batch = img_src.size()[0]

                y_real = torch.ones(mini_batch)
                y_fake = torch.zeros(mini_batch)

                img_src, img_tgt, y_real, y_fake = (
                    img_src.to(device),
                    img_tgt.to(device),
                    y_real.to(device),
                    y_fake.to(device),
                )
                D_result = torch.sigmoid(D(img_tgt)).squeeze()
                D_real_loss = BCE_loss(D_result, y_real)

                # noise = torch.randn((mini_batch, noise_length)).view((-1, noise_length, 1, 1)).to(device)
                img_fake = G(img_src)
                D_result = torch.sigmoid(D(img_fake)).squeeze()
                D_fake_loss = BCE_loss(D_result, y_fake)

                D_val_loss = D_real_loss + D_fake_loss

                AE_val_loss = MSE_loss(img_fake, img_tgt)

                # noise = torch.randn((mini_batch, noise_length)).view((-1, 100, 1, 1)).to(device)
                img_fake = G(img_src)
                D_result = torch.sigmoid(D(img_fake)).squeeze()
                G_val_loss = BCE_loss(D_result, y_real)

        print(
            "D val loss : {} , G val loss : {} , AE val loss : {} ".format(
                D_val_loss, G_val_loss, AE_val_loss
            )
        )
        writer.add_scalar(
            "D_val_loss",
            D_val_loss / len(val_loader),
            sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
        )
        writer.add_scalar(
            "G_val_loss",
            G_val_loss / len(val_loader),
            sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
        )
        writer.add_scalar(
            "AE_val_loss",
            AE_val_loss / len(val_loader),
            sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
        )
        torch.save(G.state_dict(), checkpoint_G)
        torch.save(D.state_dict(), checkpoint_D)
writer.close()
