import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from time_rnn import TimeRNN
from torch import nn
import torch

sample_num = 1000
window_size = 50
data_x = np.linspace(0, sample_num, 1000) / 2
data_y = np.sin(data_x)


class SinData(Dataset):
    def __init__(self, data, window_size=window_size):
        self.data = data
        self.window_size = window_size

    def __getitem__(self, index):
        x = self.data[index : index + self.window_size].reshape(-1, 1)
        y = self.data[index + self.window_size].reshape(-1, 1)
        return x, y

    def __len__(self):
        return len(self.data) - self.window_size


d = SinData(data_y)
dl = DataLoader(d, batch_size=4, shuffle=True)


def show_data():
    plt.figure()
    for x, y in dl:
        for arr in x.cpu().data.numpy():
            plt.plot(arr)
        plt.scatter(
            [x.shape[1] for i in range(x.shape[0])],
            y.data.numpy(),
            color="black",
        )
        break
    plt.show()


def train():
    net = TimeRNN(1).cuda()
    criteron = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    dl = DataLoader(d, batch_size=100, shuffle=False)
    loss_curve = []
    # 这个模型拟合非常容易，所以这里刻意少训练几个epoch，方便查看预测效果
    for i in range(5):
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
    plt.plot(loss_curve)
    plt.show()
    return net


def show_result(net, data):
    window_size = 50
    init_input = (
        torch.from_numpy(data_y[:window_size]).view(-1, 1, 1).float().cuda()
    )
    outputs = []
    for i in range(len(data) - window_size - 800):
        output = net(init_input)
        outputs.append(output)
        init_input[0 : window_size - 1, :, :] = init_input[1:window_size, :, :]
        init_input[window_size - 1, :, :] = output
    plt.figure(figsize=(24, 8))
    plt.plot(outputs, color="g")
    plt.plot(data[: len(data) - window_size - 800], color="r")

    plt.show()


if __name__ == "__main__":
    # show_data()
    net = train()
    show_result(net, data_y)
