import jieba


class Vocab:
    def __init__(self):
        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}  # start of sentence/end of sentence
        self.word_count = {}
        self.num_words = 3

    def add_sentence(self, sentence):
        raise Exception("Not implemented")

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word_count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word_count[word] += 1


class En(Vocab):
    def add_sentence(self, sentence):
        # 英文分词直接使用空格来分割
        # split() 会自动删除多余的空格，比split(" ")更有效
        for word in sentence.strip().split():
            self.addWord(word)


class Zh(Vocab):
    def add_sentence(self, sentence):
        # 中文分词
        for word in jieba.cut(sentence.strip()):
            self.addWord(word)
