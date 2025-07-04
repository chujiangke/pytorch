import torch
from torch import optim
import os
from torch.utils.tensorboard import SummaryWriter

# from prune import compress_model
from model import VGG_11_prune
from config import CHECKPOINT, device, init_epoch_lr
from base_train import train_epoch, validation


def retrain():
    channels = [14, "M", 26, "M", 46, 45, "M", 101, 98, "M", 99, 99, "M"]
    compressed_net = VGG_11_prune(channels)
    # net.load_state_dict(torch.load(os.path.join(CHECKPOINT, "fine_tune_sparse_90.0.pth"))["net"])

    # compressed_net = compress_model(net)
    compressed_net.load_state_dict(torch.load(os.path.join(CHECKPOINT, "compressed_net.pth")))
    compressed_net.to(device)

    if os.path.exists(os.path.join(CHECKPOINT, "best_retrain_model.pth")):
        saved_model = torch.load(os.path.join(CHECKPOINT, "best_retrain_model.pth"))
        compressed_net.load_state_dict(torch.load(os.path.join(CHECKPOINT, "best_retrain_model.pth"))["compressed_net"])
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
        optimizer = optim.SGD(compressed_net.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)
        for epoch in range(num_epoch):
            epoch_acc, epoch_loss, compressed_net = train_epoch(compressed_net, optimizer, criteron)

            writer.add_scalar("epoch_acc", epoch_acc, sum([e[0] for e in init_epoch_lr[:i]]) + epoch)
            writer.add_scalar("epoch_loss", epoch_loss, sum([e[0] for e in init_epoch_lr[:i]]) + epoch)

            test_acc, test_loss = validation(compressed_net, criteron)
            if test_loss <= best_loss:
                if test_acc >= best_accuracy:
                    best_accuracy = test_acc
                best_loss = test_loss
                best_model_weights = compressed_net.state_dict().copy()
                best_optimizer_params = optimizer.state_dict().copy()
                torch.save(
                    {
                        "compressed_net": best_model_weights,
                        "optimizer": best_optimizer_params,
                        "best_accuracy": best_accuracy,
                        "best_loss": best_loss,
                    },
                    os.path.join(CHECKPOINT, "best_retrain_model.pth"),
                )

            writer.add_scalar("test_acc", test_acc, sum([e[0] for e in init_epoch_lr[:i]]) + epoch)
            writer.add_scalar("test_loss", test_loss, sum([e[0] for e in init_epoch_lr[:i]]) + epoch)

    writer.close()
    return compressed_net


if __name__ == "__main__":
    retrain()
