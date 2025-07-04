import torch
from string import punctuation
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PAD = 0
SOS_token = 1
EOS_token = 2
ATTENTION = True  # 是否使用attention模型

BATCH_SIZE = 16  # 不要为了追求训练速度一味地调高batch_size

EPOCH_LR = [(50, 0.001), (50, 0.001)]
# if not ATTENTION:
#     EPOCH_LR = [(50, 0.001), (50, 0.0001)]
# else:
#     EPOCH_LR = [(50, 0.001), (50, 0.0001)]
DECODER_LR_RATIO = 5.0
CLIP = 50.0
MAX_LENGTH = 10
HIDDEN_SIZE = 100  # 使用相同的embed_size 和hidden_size
DATA_PATH = "/data/cmn.txt"
# DATA_PATH = "D:\\datasets\\cmn-eng\\cmn.txt"
CHECKPOINT = "/data/nmt_model"
TEACHER_FORCING_RATIO = 0.5

add_punc = "，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥"  # 加了个空格
all_punc = punctuation + add_punc
regrex = re.compile("[%s]" % re.escape(all_punc))
print(all_punc)
