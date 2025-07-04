from torch import nn, optim
import torch
from torch.utils.tensorboard import SummaryWriter
import random
from tqdm import tqdm
import os.path as osp

from data import train_data, train_loader
from config import (
    ATTENTION,
    HIDDEN_SIZE,
    device,
    EPOCH_LR,
    DECODER_LR_RATIO,
    SOS_token,
    BATCH_SIZE,
    CLIP,
    CHECKPOINT,
    TEACHER_FORCING_RATIO,
)
from model import Encoder, Decoder, AttenDecoder


def MaskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(
        torch.gather(inp, 1, target.view(-1, 1)).squeeze(1)
    )
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train():
    encoder = Encoder(train_data.en_data.num_words, HIDDEN_SIZE).to(device)
    if ATTENTION:
        decoder = AttenDecoder(train_data.zh_data.num_words, HIDDEN_SIZE).to(
            device
        )
    else:
        decoder = Decoder(train_data.zh_data.num_words, HIDDEN_SIZE).to(device)
    en_optim = optim.Adam(encoder.parameters(), lr=0.001)
    de_optim = optim.Adam(decoder.parameters(), lr=0.001 * DECODER_LR_RATIO)
    writer = SummaryWriter("log")
    best_loss = 1e9
    ckpt = osp.join(CHECKPOINT, "attention:{}".format(ATTENTION))
    if osp.exists(ckpt):
        ckpt_model = torch.load(ckpt)
        encoder.load_state_dict(ckpt_model["encoder"])
        decoder.load_state_dict(ckpt_model["decoder"])
        print("Model loaded ...")
    for n, (num_epoch, lr) in enumerate(EPOCH_LR):
        for p in en_optim.param_groups:
            p["lr"] = lr
        for p in de_optim.param_groups:
            p["lr"] = lr * DECODER_LR_RATIO
        # en_optim.lr = lr
        # de_optim.lr = lr * DECODER_LR_RATIO
        for epoch in range(num_epoch):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            n_samples = 0
            n_equals = 0
            for i, (x, y, mask) in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):
                # print(x[:, 0], y[:, 0])
                batch_loss = []
                loss = 0
                n_totals = 0
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                en_optim.zero_grad()
                de_optim.zero_grad()
                en_out, en_hid = encoder(x)
                de_inp = (
                    torch.Tensor([[SOS_token for _ in range(BATCH_SIZE)]])
                    .to(device)
                    .long()
                )
                de_hid = en_hid
                use_teacher_forcing = (
                    True if random.random() < TEACHER_FORCING_RATIO else False
                )
                if use_teacher_forcing:
                    for t in range(y.shape[0]):
                        if ATTENTION:
                            de_out, de_hid, _ = decoder(de_inp, de_hid, en_out)
                        else:
                            de_out, de_hid = decoder(de_inp, de_hid)
                        # 计算准确率
                        label = torch.argmax(de_out, dim=1)
                        # 因为pad不加入loss计算，所以预测结果中几乎不会出现pad，所以计算accuracy时不考虑mask的话，结果差别也不大。
                        # equals = torch.sum(label == y[t]).item()
                        equals = torch.sum(
                            (label == y[t]) * (mask[t] == 1)
                        ).item()

                        de_inp = y[t].view(1, -1)
                        if sum(mask[t]) != 0:
                            mask_loss, nTotal = MaskNLLLoss(
                                de_out.squeeze(0), y[t], mask[t]
                            )
                            # label = torch.argmax(de_out, dim=1)
                            # equals = torch.sum(label == de_inp)
                        else:
                            mask_loss, nTotal = (
                                torch.zeros(1).mean().to(device),
                                0,
                            )
                            # equals = 0
                        loss += mask_loss
                        n_equals += equals
                        n_samples += torch.sum(mask[t]).item()
                        n_totals += nTotal
                        batch_loss.append(mask_loss.item() * nTotal)
                else:
                    for t in range(y.shape[0]):
                        if ATTENTION:
                            de_out, de_hid, _ = decoder(de_inp, de_hid, en_out)
                        else:
                            de_out, de_hid = decoder(de_inp, de_hid)
                        _, topi = de_out.topk(1)
                        # 计算准确率
                        label = torch.argmax(de_out, dim=1)
                        # equals = torch.sum(label == y[t]).item()
                        equals = torch.sum(
                            (label == y[t]) * (mask[t] == 1)
                        ).item()

                        de_inp = torch.LongTensor(
                            [[topi[i][0] for i in range(BATCH_SIZE)]]
                        )
                        de_inp = de_inp.to(device)
                        if torch.sum(mask[t]) != 0:
                            mask_loss, nTotal = MaskNLLLoss(
                                de_out, y[t], mask[t]
                            )
                        else:
                            mask_loss, nTotal = (
                                torch.zeros(1).mean().to(device),
                                0,
                            )
                        loss += mask_loss
                        n_equals += equals
                        n_samples += torch.sum(mask[t]).item()
                        n_totals += nTotal
                        batch_loss.append(mask_loss.item() * nTotal)
                loss.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(), CLIP)
                nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)
                en_optim.step()
                de_optim.step()
                epoch_loss += sum(batch_loss) / n_totals

            print(
                "epoch {} loss {} accuracy {}".format(
                    sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
                    epoch_loss / len(train_loader),
                    n_equals / n_samples,
                )
            )
            writer.add_scalar(
                "train_loss_atten_{}".format(ATTENTION),
                epoch_loss / len(train_loader),
                sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
            )
            writer.add_scalar(
                "train_accuracy_atten_{}".format(ATTENTION),
                n_equals / n_samples,
                sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
            )
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                    },
                    ckpt,
                )
    writer.close()


if __name__ == "__main__":
    train()
