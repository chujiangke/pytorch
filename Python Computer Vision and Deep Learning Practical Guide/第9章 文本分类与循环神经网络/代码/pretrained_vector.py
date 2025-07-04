from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import torch
import re
from tqdm import tqdm


def get_pretrained_weight(vocab=None, file="/data/glove.6B/glove.6B.100d.txt"):
    print("extracting pretrained weight")

    # glove源文件
    # glove_file = datapath("/data/glove.6B/glove.6B.300d.txt")

    glove_file = datapath(file)
    # 目标model文件
    # tmp_file = get_tmpfile("/data/glove.6B/glove.6B.300d.vec.txt")
    tmp_file = get_tmpfile(re.sub(r"\.txt", ".vec.txt", file))

    glove2word2vec(glove_file, tmp_file)

    model = KeyedVectors.load_word2vec_format(tmp_file)

    word_to_idx = vocab.word2index
    idx_to_word = vocab.index2word

    pretrained_weight = torch.zeros(vocab.num_words + 1, 100)

    # model.index2word 是一个list
    for i in tqdm(range(len(model.index2word))):
        try:
            # 查找model中的第i个词在vocab中的位置
            index = word_to_idx[model.index2word[i]]
        except:
            continue
        pretrained_weight[index, :] = torch.from_numpy(model.get_vector(idx_to_word[word_to_idx[model.index2word[i]]]))

    return pretrained_weight
