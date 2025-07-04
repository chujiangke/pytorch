import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from time_rnn import TimeRNN

window_size = 100


class StackData(Dataset):
    def __init__(self, data, window_size=window_size):
        self.data = data
        self.window_size = window_size

    def __getitem__(self, index):
        x = (
            self.data[index : index + self.window_size].reshape(-1, 1)
            / self.data[0]
            - 1
        )
        y = (
            self.data[index + self.window_size].reshape(-1, 1) / self.data[0]
            - 1
        )
        return x, y

    def __len__(self):
        return len(self.data) - self.window_size


def train(d):
    net = TimeRNN(1).cuda()
    criteron = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    dl = DataLoader(d, batch_size=100, shuffle=False)
    loss_curve = []
    for i in range(100):
        epoch_loss = 0.0
        for x, y in tqdm(dl):
            x = x.cuda()
            y = y.cuda()
            x = x.permute(1, 0, 2).float()
            y = y.float()
            optimizer.zero_grad()
            out = net(x)
            loss = criteron(out, y.squeeze(2))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("epoch_loss", epoch_loss / len(dl))
        loss_curve.append(epoch_loss / len(dl))
    plt.figure()
    plt.title("Loss curve")
    plt.plot(loss_curve)
    plt.savefig("img/Loss curve.jpg")
    return net


def predict_by_point(net, data):
    init_input = (
        torch.from_numpy(data[:window_size]).view(-1, 1, 1).float().cuda()
    )
    outputs = []
    for i in range(len(data) - window_size):
        output = net(init_input / init_input[0] - 1)
        true_output = (output + 1) * init_input[0]
        outputs.append(true_output.item())
        init_input[:window_size, :, :] = (
            torch.from_numpy(data[i : i + window_size])
            .view(-1, 1, 1)
            .float()
            .cuda()
        )
    plt.plot(outputs, color="g", linestyle="--")
    # 因为真实值与预测值比较接近，为了画图更清晰，把真实值右移了10个单位
    plt.plot(data[window_size - 10 :], color="r")


def predict_by_step(net, data):
    window_size = 50
    outputs_list = []
    indices_list = []
    i = 0
    while i + window_size * 2 < len(data):
        data_i = data[i : i + window_size * 2]
        init_input = (
            torch.from_numpy(data_i[:window_size]).view(-1, 1, 1).float().cuda()
        )
        indices = []
        outputs = []
        for j in range(window_size):
            output = net(init_input / init_input[0] - 1)
            true_output = (output + 1) * init_input[0]
            outputs.append(true_output.item())
            indices.append(i + j)
            init_input[0 : window_size - 1, :, :] = init_input[
                1:window_size, :, :
            ]
            init_input[window_size - 1, :, :] = true_output
        i += 2 * window_size
        indices_list.append(indices)
        outputs_list.append(outputs)
    for indices, outputs in zip(indices_list, outputs_list):
        plt.plot(indices, outputs, color="g", linestyle="--")
    plt.plot(data[2 * window_size :], color="r")


def predict_from_start(net, data):
    init_input = (
        torch.from_numpy(data[:window_size]).view(-1, 1, 1).float().cuda()
    )
    outputs = []
    for i in range(len(data) - window_size):
        output = net(init_input / init_input[0] - 1)
        true_output = (output + 1) * init_input[0]
        outputs.append(true_output.item())
        init_input[0 : window_size - 1, :, :] = init_input[1:window_size, :, :]
        init_input[window_size - 1, :, :] = true_output
    plt.plot(outputs, color="g", linestyle="--")
    plt.plot(data[window_size:], color="r")


if __name__ == "__main__":
    import tushare as ts

    df = ts.get_hist_data("600848")
    op = df["open"]
    plt.figure()
    plt.title("Stack data")
    plt.plot(op)
    plt.savefig("img/Stack_data.jpg")
    d = StackData(op.values)
    net = train(d)
    # 另起一个图幅
    plt.figure()
    plt.subplot(131)
    plt.title("by point")
    predict_by_point(net, op.values)
    plt.subplot(132)
    plt.title("by step")
    predict_by_step(net, op.values)
    plt.subplot(133)
    plt.title("from start")
    predict_from_start(net, op.values)
    plt.savefig("img/result.jpg")
    plt.show()
