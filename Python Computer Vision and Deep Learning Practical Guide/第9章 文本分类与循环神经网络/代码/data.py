from torch.utils.data import Dataset
import os.path as osp
from glob import glob
import torch
from torch.nn.utils.rnn import pad_sequence

from config import FOLDER, device
from vocab import Vocab


class imdb(Dataset):
    def __init__(self, folder=FOLDER, vocab=None, subset="train"):
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = Vocab()
        self.data_folder = osp.join(folder, subset)
        self.pos_folder = osp.join(self.data_folder, "pos")
        self.neg_folder = osp.join(self.data_folder, "neg")
        self.pos_files = sorted(glob(osp.join(self.pos_folder, "*.txt")))
        self.neg_files = sorted(glob(osp.join(self.neg_folder, "*.txt")))
        self.labels = torch.Tensor(
            [1 for i in range(len(self.pos_files))] + [0 for i in range(len(self.neg_files))]
        ).long()
        self.files = self.pos_files + self.neg_files

    def __getitem__(self, index):
        label = self.labels[index]
        sentence = self.vocab.read_file(self.files[index])
        text = self.vocab.sentence2tensor(sentence)
        return text, label

    def __len__(self):
        return len(self.files)


def collate_cnn(batch):
    text = [d[0] for d in batch]
    label = [d[1] for d in batch]
    text = pad_sequence(text, batch_first=True, padding_value=0).to(device).long()
    # cnn需要输入二维数据，所以在这里添加一个维度
    text = text.unsqueeze(1)
    label = torch.Tensor(label).to(device).long()
    return text, label


def collate_rnn(batch):
    text = [d[0] for d in batch]
    label = [d[1] for d in batch]
    text = pad_sequence(text, batch_first=False, padding_value=0).to(device).long()
    label = torch.Tensor(label).to(device).long()
    return text, label
