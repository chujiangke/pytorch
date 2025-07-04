from torch.utils.data import DataLoader
from torch import nn, optim
import torch
from tqdm import tqdm
import os.path as osp
import pickle
from torch.utils.tensorboard import SummaryWriter

from vocab import Vocab
from data import imdb, collate_cnn, collate_rnn
from cnn_model import TextCNN
from rnn_model import TextRNN
from config import (
    FOLDER,
    device,
    CHECKPOINT,
    EPOCH_LR_CNN,
    EPOCH_LR_RNN,
    BATCH_SIZE_RNN,
    BATCH_SIZE_CNN,
)
from pretrained_vector import get_pretrained_weight

# 因为vocab加载时间较长，可以预先将vocab保存下来，方便代码调试
vocab_path = osp.join(CHECKPOINT, "vocab.pkl")
weight_path = osp.join(CHECKPOINT, "weight.pth")
if osp.exists(vocab_path):
    f = open(vocab_path, "rb")
    vocab = pickle.load(f)
    f.close()
    print("Vocab loaded")
else:
    vocab = Vocab(min_freq=0)
    f = open(vocab_path, "wb")
    pickle.dump(vocab, f, 0)
    f.close()
    print("Vocab created and saved ")

if osp.exists(weight_path):
    pretrained_weight = torch.load(weight_path)
else:
    pretrained_weight = get_pretrained_weight(vocab)
    torch.save(pretrained_weight, weight_path)


def train(model="RNN"):
    text_train = imdb(FOLDER, vocab=vocab, subset="train")
    text_test = imdb(FOLDER, vocab=vocab, subset="test")
    if model == "CNN":
        # net = TextCNN(
        #     vocab_size=text_train.vocab.num_words,
        #     pretrained_weight=pretrained_weight,
        # ).to(device)
        net = TextCNN(vocab_size=text_train.vocab.num_words, embed_size=300).to(
            device
        )
        batch_size = BATCH_SIZE_CNN
        train_loader = DataLoader(
            text_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_cnn,
        )
        test_loader = DataLoader(
            text_test,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_cnn,
        )
    else:
        # net = TextRNN(
        #     text_train.vocab.num_words,300 pretrained_weight=pretrained_weight
        # ).to(device)
        net = TextRNN(text_train.vocab.num_words, 300).to(device)
        batch_size = BATCH_SIZE_RNN
        train_loader = DataLoader(
            text_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_rnn,
        )
        test_loader = DataLoader(
            text_test,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_rnn,
        )
    criteron = nn.CrossEntropyLoss()
    writer = SummaryWriter("log")
    best_loss = 1e9

    ckpt = osp.join(CHECKPOINT, "{}.pth".format(model))
    if osp.exists(ckpt):
        ckpt = torch.load(ckpt)
        best_loss = ckpt["loss"]
        net.load_state_dict(ckpt["params"])
        print("checkpoint loaded ...")

    if model == "CNN":
        epoch_lr = EPOCH_LR_CNN
    else:
        epoch_lr = EPOCH_LR_RNN

    for n, (num_epoch, lr) in enumerate(epoch_lr):
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        for epoch in range(num_epoch):
            net.train()
            epoch_loss = 0
            epoch_acc = 0
            for i, (text, label) in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):
                text = text.to(device)
                label = label.to(device)
                output = net(text)
                pred = torch.argmax(output, dim=1)
                acc = torch.sum(pred == label)
                epoch_acc += acc.item()
                loss = criteron(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if i % 10 == 0:
                # print(loss.item())
                epoch_loss += loss.item()

            print(
                "train    loss",
                epoch_loss / len(train_loader),
                "accuracy",
                epoch_acc / (len(train_loader.dataset)),
            )
            writer.add_scalar(
                "{}_train_loss".format(model),
                epoch_loss / len(train_loader),
                sum([e[0] for e in epoch_lr[:n]]) + epoch,
            )
            writer.add_scalar(
                "{}_train_acc".format(model),
                epoch_acc / len(train_loader.dataset),
                sum([e[0] for e in epoch_lr[:n]]) + epoch,
            )

            with torch.no_grad():
                net.eval()
                test_acc = 0
                test_loss = 0
                for j, (text, label) in tqdm(
                    enumerate(test_loader), total=len(test_loader)
                ):
                    text = text.to(device)
                    label = label.to(device)
                    output = net(text)
                    pred = torch.argmax(output, dim=1)
                    acc = torch.sum(pred == label)
                    loss = criteron(output, label)
                    test_acc += acc.item()
                    test_loss += loss.item()
                print(
                    "test    loss",
                    test_loss / (len(test_loader)),
                    "accuracy",
                    test_acc / (len(test_loader) * batch_size),
                )
                writer.add_scalar(
                    "{}_test_loss".format(model),
                    test_loss / len(test_loader),
                    sum([e[0] for e in epoch_lr[:n]]) + epoch,
                )
                writer.add_scalar(
                    "{}_test_acc".format(model),
                    test_acc / len(test_loader.dataset),
                    sum([e[0] for e in epoch_lr[:n]]) + epoch,
                )
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(
                        {"params": net.state_dict(), "loss": test_loss}, ckpt
                    )
    writer.close()


if __name__ == "__main__":
    train("CNN")
    
