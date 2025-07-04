from torch import nn


class TextRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=100,
        hidden_size=512,
        num_classes=2,
        num_layers=2,
        pad_idx=0,
        pretrained_weight=None,
    ):
        super(TextRNN, self).__init__()
        if pretrained_weight is not None:
            self.embedding = nn.Embedding(vocab_size, embed_size).from_pretrained(pretrained_weight)
            print("pretrained weight loaded ... ")
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        # self.rnn = nn.RNN(embed_size, hidden_size, num_layers)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.rnn(out)
        out = self.fc(out[-1])
        return out
