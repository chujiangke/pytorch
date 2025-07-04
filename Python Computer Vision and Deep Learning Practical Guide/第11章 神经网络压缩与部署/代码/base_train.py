from config import device, CHECKPOINT, init_epoch_lr
from data import trainloader, trainset, testloader, testset
from model import VGG_11_prune

import torch
from torch import optim

# Pytorch 1.1.0以上版本才有自带的tensorboard，较低版本可以使用单独安装的tensorboard
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm


def train_epoch(net, optimizer, criteron):
    epoch_loss = 0.0
    epoch_acc = 0.0
    for j, (img, label) in enumerate(trainloader):
        img, label = img.to(device), label.to(device)
        out = net(img)
        optimizer.zero_grad()
        loss = criteron(out, label)
        loss.backward()
        optimizer.step()
        pred = torch.argmax(out, dim=1)
        acc = torch.sum(pred == label)
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    epoch_acc /= len(trainset)
    epoch_loss /= len(trainloader)
    print("Epoch loss : {:8f}  Epoch accuracy : {:8f}".format(epoch_loss, epoch_acc))
    return epoch_acc, epoch_loss, net


def validation(net, criteron, device):
    with torch.no_grad():
        test_loss = 0.0
        test_acc = 0.0
        for k, (img, label) in tqdm(enumerate(testloader), total=len(testloader)):
            img, label = img.to(device), label.to(device)
            out = net(img)
            loss = criteron(out, label)
            pred = torch.argmax(out, dim=1)
            acc = torch.sum(pred == label)
            test_loss += loss.item()
            test_acc += acc.item()
        test_acc /= len(testset)
        test_loss /= len(testloader)
        print("Test loss : {:8f}  Test accuracy : {:8f}".format(test_loss, test_acc))
    return test_acc, test_loss


def init_train(net):
    if os.path.exists(os.path.join(CHECKPOINT, "best_model.pth")):
        saved_model = torch.load(os.path.join(CHECKPOINT, "best_model.pth"))
        net.load_state_dict(torch.load(os.path.join(CHECKPOINT, "best_model.pth"))["net"])
        if saved_model["best_accuracy"] > 0.9:
            print(" break init train ... ")
            return
        best_accuracy = saved_model["best_accuracy"]
        best_loss = saved_model["best_loss"]
    else:
        best_accuracy = 0.0
        best_loss = 10.0
    writer = SummaryWriter("logs/")
    criteron = torch.nn.CrossEntropyLoss()

    for i, (num_epoch, lr) in enumerate(init_epoch_lr):
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)
        for epoch in range(num_epoch):
            epoch_acc, epoch_loss, net = train_epoch(net, optimizer, criteron)

            writer.add_scalar("epoch_acc", epoch_acc, sum([e[0] for e in init_epoch_lr[:i]]) + epoch)
            writer.add_scalar("epoch_loss", epoch_loss, sum([e[0] for e in init_epoch_lr[:i]]) + epoch)

            test_acc, test_loss = validation(net, criteron)
            if test_loss <= best_loss:
                if test_acc >= best_accuracy:
                    best_accuracy = test_acc
                best_loss = test_loss
                best_model_weights = net.state_dict().copy()
                best_optimizer_params = optimizer.state_dict().copy()
                torch.save(
                    {
                        "net": best_model_weights,
                        "optimizer": best_optimizer_params,
                        "best_accuracy": best_accuracy,
                        "best_loss": best_loss,
                    },
                    os.path.join(CHECKPOINT, "best_model.pth"),
                )

            writer.add_scalar("test_acc", test_acc, sum([e[0] for e in init_epoch_lr[:i]]) + epoch)
            writer.add_scalar("test_loss", test_loss, sum([e[0] for e in init_epoch_lr[:i]]) + epoch)

    writer.close()
    return net


if __name__ == "__main__":
    net = VGG_11_prune().to(device)
    init_train(net)

