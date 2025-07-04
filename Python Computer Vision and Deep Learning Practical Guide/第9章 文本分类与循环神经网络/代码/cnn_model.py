from torch import nn
import torch
import torch.nn.functional as F

from torchsummary import summary


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size=100, m=3, n_cls=2, pretrained_weight=None):
        super(TextCNN, self).__init__()

        if pretrained_weight is not None:
            self.embedding = nn.Embedding(vocab_size, embed_size).from_pretrained(pretrained_weight)
            print("pretrained weight loaded ... ")
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)
        self.cnn = nn.ModuleList()
        self.pool = []
        self.m = m
        self.n_cls = n_cls
        for i in range(m):
            self.cnn.append(nn.Conv2d(1, 1, (i + 2, embed_size)))
            # self.pool.append(nn.MaxPool2d((max_sentence_length-i-1,1)))
        self.fc1 = nn.Linear(m, 10)
        self.fc2 = nn.Linear(10, n_cls)

    def forward(self, x):
        x = self.embedding(x)
        batch = x.shape[0]
        xs = []
        for i in range(self.m):
            x_ = self.cnn[i](x)
            x_ = nn.MaxPool2d((x.shape[2] - i - 1, 1))(x_)
            xs.append(x_)
        x = torch.cat(xs, -1)
        x = x.view(batch, 1, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, self.n_cls)
        return F.sigmoid(x)


if __name__ == "__main__":
    from vocab import Vocab
    from config import device

    voc = Vocab()
    net = TextCNN(voc.num_words).to(device)

    print(net)
