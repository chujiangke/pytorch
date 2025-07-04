from torch import nn
import torch
import torch.nn.functional as F

from config import MAX_LENGTH


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)  # num_layers默认为1

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        return output, hidden


# 训练和推理过程中，decoder是一个一个单词输入的
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output[0])
        return self.softmax(output), hidden


class AttenDecoder(nn.Module):
    def __init__(
        self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH
    ):
        super(AttenDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), 2)), dim=2
        )
        attn_applied = torch.bmm(
            attn_weights.permute(1, 0, 2), encoder_outputs.permute(1, 0, 2)
        )
        output = torch.cat((embedded, attn_applied.permute(1, 0, 2)), 2)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.softmax(self.out(output).squeeze(0), dim=1)
        return output, hidden, attn_weights
