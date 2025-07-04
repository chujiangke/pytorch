from glob import glob
import os.path as osp
from collections import defaultdict
from tqdm import tqdm
import re
import torch
import pickle

from config import STOP_WORDS, FOLDER, device, CHECKPOINT,MIN_FREQ


class Vocab:
    def __init__(self, folder=FOLDER, stop_words=STOP_WORDS, min_freq=MIN_FREQ):

        # 建立vocab的时候不需要区分正面与负面
        self.files = sorted(glob(osp.join(folder, "*/*/*.txt")))

        # PAD 用于RNN批量计算时补全张量
        self.word2index = {"PAD": 0}
        self.index2word = {0: "PAD"}
        self.num_words = 1
        self.wordcount = defaultdict(int)
        self.stop_words = stop_words
        self.min_freq = min_freq
        self.max_length = 0

        print("Loading files for vocab ... ")
        for file in tqdm(self.files):
            sentence = self.read_file(file)
            length = self.add_sentence(sentence)
            if length > self.max_length:
                self.max_length = length
        print("max num words", max(self.wordcount.values()))
        print("num of words: {}".format(self.num_words))
        self.filter()
        print("num of words: {}".format(self.num_words))

    def read_file(self, file):
        f = open(file, "r")
        content = f.read()
        f.close()
        return content

    def add_sentence(self, sentence):
        # 将句子中的词汇逐个加入到vocab中
        # sentence = re.sub(r"^[\w\s]", " ", sentence)
        words = sentence.split(" ")
        length = 0
        for word in words:
            word = word.lower()
            if word not in self.word2index and word not in self.stop_words and re.search(r"[a-zA-Z]", word):
                self.word2index[word] = self.num_words
                self.index2word[self.num_words] = word
                self.num_words += 1
            if word not in self.stop_words and re.search(r"[a-zA-Z]", word):
                length += 1
            self.wordcount[word] += 1

        return length

    def filter(self):
        words = self.word2index.keys()
        # 单词编号从1开始，因为后面要用0来做padding
        # 筛选过后保留下拉地单词对应的数量并不会改变，所以self.word_count 不做修改
        new_word2index = {"PAD": 0}
        new_index2word = {0: "PAD"}
        new_num_words = 1
        for word in words:
            if self.wordcount[word] > self.min_freq:
                new_word2index[word] = new_num_words
                new_index2word[new_num_words] = word
                new_num_words += 1
        self.word2index = new_word2index
        self.index2word = new_index2word
        self.num_words = new_num_words

    def sentence2tensor(self, sentence):
        # 此方法用于dataset中取元素
        result = []
        for word in sentence.split(" "):
            # 不在字典中的词汇直接跳过
            if word in self.word2index:
                result.append(self.word2index[word])
        return torch.Tensor(result).to(device).long()

