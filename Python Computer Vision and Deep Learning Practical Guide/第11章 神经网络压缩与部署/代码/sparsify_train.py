import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

from base_train import validation, train_epoch
from config import finetune_epoch_lr, CHECKPOINT, device
from model import VGG_11_prune


def fine_tune(net, sparisity, epoch_lr):
    writer = SummaryWriter("logs/")
    criteron = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    best_loss = 10.0

    for i, (num_epoch, lr) in enumerate(epoch_lr):
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)
        for epoch in range(num_epoch):
            epoch_acc, epoch_loss, net = train_epoch(net, optimizer, criteron)

            writer.add_scalar("fine_acc", epoch_acc, sum([e[0] for e in epoch_lr[:i]]) + epoch)
            writer.add_scalar("fine_loss", epoch_loss, sum([e[0] for e in epoch_lr[:i]]) + epoch)

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
                    os.path.join(CHECKPOINT, "fine_tune_sparse_{}.pth".format(sparisity)),
                )

            writer.add_scalar("fine_test_acc", test_acc, sum([e[0] for e in epoch_lr[:i]]) + epoch)
            writer.add_scalar("fine_test_loss", test_loss, sum([e[0] for e in epoch_lr[:i]]) + epoch)

    writer.close()
    return net


def sparsify(net, sparsity_level=50.0):
    # 将一部分较小的weights值修改为0
    for name, param in net.named_parameters():
        # weight和bias都要修剪
        # 因为在通道修剪时要同时参考两者才能保证键之后的精度
        if "weight" in name:
            threshold = np.percentile(torch.abs(param.data).cpu().numpy(), sparsity_level)
            mask = torch.gt(torch.abs(param.data), threshold).float()
            param.data *= mask
        if "bias" in name:
            threshold = np.percentile(torch.abs(param.data).cpu().numpy(), sparsity_level)
            mask = torch.gt(torch.abs(param.data), threshold).float()
            param.data *= mask
    return net


def sparsify_train(net):
    sparse_model = VGG_11_prune().to(device)
    sparse_model.load_state_dict(net.state_dict())
    for i, sparsity_level in enumerate([50.0, 60.0, 70.0, 80.0, 90.0]):
        print("pruning ...")
        epoch_lr = finetune_epoch_lr[i]
        sparse_model = sparsify(sparse_model, sparsity_level)
        net = fine_tune(sparse_model, sparsity_level, epoch_lr)
    return net


if __name__ == "__main__":

    net = VGG_11_prune()
    net.load_state_dict(torch.load(os.path.join(CHECKPOINT, "best_model.pth"))["net"])
    sparsify_train(net)
