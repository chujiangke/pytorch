import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import os.path as osp

from super_resolution_data import SuperResolutionData, transform
from model import ResNet18Unet
from config import device, sr_checkpoint, batch_size, epoch_lr
from transform import TrainTransform, TestTransform


def train():

    # 只训练Y通道
    net = ResNet18Unet(num_classes=1)
    net.firstconv = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    net = net.to(device)

    trainset = SuperResolutionData(subset="train", transform=TrainTransform)
    testset = SuperResolutionData(subset="test", transform=TestTransform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    criteron = nn.MSELoss()

    best_loss = 1e9

    if osp.exists(sr_checkpoint):
        ckpt = torch.load(sr_checkpoint)
        best_loss = ckpt["loss"]
        net.load_state_dict(ckpt["params"])
        print("checkpoint loaded ...")

    writer = SummaryWriter("super_log")
    for n, (num_epochs, lr) in enumerate(epoch_lr):
        optimizer = optim.SGD(
            net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3
        )
        for epoch in range(num_epochs):
            net.train()
            pbar = tqdm(enumerate(trainloader), total=len(trainloader))
            epoch_loss = 0.0
            for i, (img, mask) in pbar:
                img = img.to(device)
                mask = mask.to(device)
                out = net(img)
                # 训练残差
                loss = criteron(out + img, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    pbar.set_description("loss: {}".format(loss))
                epoch_loss += loss.item()
            print("Epoch_loss:{}".format(epoch_loss / len(trainloader.dataset)))
            writer.add_scalar(
                "super_epoch_loss",
                epoch_loss / len(trainloader.dataset),
                sum([e[0] for e in epoch_lr[:n]]) + epoch,
            )
            with torch.no_grad():
                net.eval()
                test_loss = 0.0
                for i, (img, mask) in tqdm(
                    enumerate(testloader), total=len(testloader)
                ):
                    img = img.to(device)
                    mask = mask.to(device)
                    out = net(img)
                    loss = criteron(out + img, mask)
                    test_loss += loss.item()
                print(
                    "Test_loss:{}".format(test_loss / len(testloader.dataset))
                )
                writer.add_scalar(
                    "super_test_loss",
                    test_loss / len(testloader.dataset),
                    sum([e[0] for e in epoch_lr[:n]]) + epoch,
                )
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(
                    {"params": net.state_dict(), "loss": test_loss},
                    sr_checkpoint,
                )
    writer.close()


if __name__ == "__main__":
    train()
