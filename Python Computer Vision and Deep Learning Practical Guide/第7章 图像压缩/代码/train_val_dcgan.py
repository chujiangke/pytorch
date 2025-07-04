from model import generator, discriminator
from config import (
    lr,
    num_epoch,
    batch_size,
    noise_length,
    device,
    checkpoint_D,
    checkpoint_G,
)

# from data import GanData
from data import train_loader, val_loader

from torch import optim, nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


resume_path_G = checkpoint_G
resume_path_D = checkpoint_D

G = generator(128).to(device)
D = discriminator(128).to(device)
if resume_path_D:
    D.load_state_dict(torch.load(resume_path_D))
    print("loaded model D")
if resume_path_G:
    G.load_state_dict(torch.load(resume_path_G))
    print("loaded model G")
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)


# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


# # dataloader
# train_set = GanData(subset="train")
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_set = GanData(subset="test")
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


def train():
    for epoch in range(num_epoch):
        D.train()
        G.train()
        for i, (img, _) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            # 训练判别器
            D_optimizer.zero_grad()
            mini_batch = img.size()[0]

            y_real = torch.ones(mini_batch)
            y_fake = torch.zeros(mini_batch)

            img, y_real, y_fake = (
                img.to(device),
                y_real.to(device),
                y_fake.to(device),
            )
            D_result = D(img).squeeze()
            D_real_loss = BCE_loss(D_result, y_real)

            noise = (
                torch.randn((mini_batch, noise_length))
                .view((-1, noise_length, 1, 1))
                .to(device)
            )
            img_fake = G(noise)
            D_result = D(img_fake).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake)

            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()

            # 训练生成器
            G_optimizer.zero_grad()
            noise = (
                torch.randn((mini_batch, noise_length))
                .view((-1, 100, 1, 1))
                .to(device)
            )
            img_fake = G(noise)
            D_result = D(img_fake).squeeze()
            G_train_loss = BCE_loss(D_result, y_real)
            G_train_loss.backward()
            G_optimizer.step()

        print(
            "D train loss : {} , G train loss : {}".format(
                D_train_loss, G_train_loss
            )
        )

        with torch.no_grad():
            D.eval()
            G.eval()
            for i, (img, _) in tqdm(
                enumerate(val_loader), total=len(val_loader)
            ):
                mini_batch = img.size()[0]

                y_real = torch.ones(mini_batch)
                y_fake = torch.zeros(mini_batch)

                img, y_real, y_fake = (
                    img.to(device),
                    y_real.to(device),
                    y_fake.to(device),
                )
                D_result = D(img).squeeze()
                D_real_loss = BCE_loss(D_result, y_real)

                noise = (
                    torch.randn((mini_batch, noise_length))
                    .view((-1, noise_length, 1, 1))
                    .to(device)
                )
                img_fake = G(noise)
                D_result = D(img_fake).squeeze()
                D_fake_loss = BCE_loss(D_result, y_fake)

                D_test_loss = D_real_loss + D_fake_loss

                # 训练生成器
                G_optimizer.zero_grad()
                noise = (
                    torch.randn((mini_batch, noise_length))
                    .view((-1, 100, 1, 1))
                    .to(device)
                )
                img_fake = G(noise)
                D_result = D(img_fake).squeeze()
                G_test_loss = BCE_loss(D_result, y_real)
        print(
            "D test loss : {} , G test loss : {}".format(
                D_test_loss, G_test_loss
            )
        )

        torch.save(G.state_dict(), checkpoint_G)
        torch.save(D.state_dict(), checkpoint_D)


if __name__ == "__main__":
    train()
