from torch.utils.data import Dataset, DataLoader
import jieba
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from config import MAX_LENGTH, regrex, EOS_token, DATA_PATH, BATCH_SIZE
from vocab import En, Zh


class NMTData(Dataset):
    def __init__(self, file):
        self.en_data, self.zh_data, self.pairs = self.load_data(file)

    def filter_pair(self, pair):
        # 用小于而不是小于等于，是因为后面还需要添加EOS_token
        return (
            len(pair[0].split(" ")) < MAX_LENGTH
            and len(list(jieba.cut(pair[1]))) < MAX_LENGTH
        )

    def load_data(self, file):
        # 行
        lines = open(file, "r", encoding="utf-8").read().strip().split("\n")
        # 句子对
        # pairs = [[regrex.sub(" ", s.lower()) for s in l.split("\t")] for l in lines]
        # pairs = [[s.lower() for s in l.split("\t")] for l in lines]
        # pairs = []
        pairs = [[s.lower() for s in l.split("\t")] for l in lines]
        # pairs = [[regrex.sub(" ", l.split("\t")[0]), regrex.sub("", l.split("\t")[1])]]
        pairs = [[regrex.sub(" ", p[0]), regrex.sub("", p[1])] for p in pairs]
        pairs = [pair for pair in pairs if self.filter_pair(pair)]
        en_data = En()
        zh_data = Zh()
        print("Loading data ...")
        for p in tqdm(pairs):
            en_data.add_sentence(p[0])
            zh_data.add_sentence(p[1])
        return en_data, zh_data, pairs

    def sentence2tensor(self, pair):
        en_index = [
            self.en_data.word2index[word] for word in pair[0].strip().split()
        ]
        en_index.append(EOS_token)
        zh_index = [
            self.zh_data.word2index[word] for word in jieba.cut(pair[1].strip())
        ]
        zh_index.append(EOS_token)
        input_tensor = torch.Tensor(en_index).view(-1, 1).long()
        target_tensor = torch.Tensor(zh_index).view(-1, 1).long()
        return input_tensor, target_tensor

    def __getitem__(self, index):
        pair = self.pairs[index]
        input_tensor, target_tensor = self.sentence2tensor(pair)
        return input_tensor, target_tensor

    def __len__(self):
        return len(self.pairs)


def collate(batch):
    # x和y都需要做padding
    x = [b[0] for b in batch]
    y = [b[1] for b in batch]
    x = pad_sequence(x, batch_first=False, padding_value=0).squeeze_(2)
    y = pad_sequence(y, batch_first=False, padding_value=0).squeeze_(2)
    xb = torch.zeros((MAX_LENGTH, len(batch))).long()
    yb = torch.zeros((MAX_LENGTH, len(batch))).long()
    xb[: x.shape[0], : x.shape[1]] = x
    yb[: y.shape[0], : y.shape[1]] = y
    mask = yb != 0
    return xb, yb, mask


train_data = NMTData(DATA_PATH)
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate,
    drop_last=True,
)

print(
    "en word num: {} \n zh word num: {}".format(
        train_data.en_data.num_words, train_data.zh_data.num_words
    )
)

