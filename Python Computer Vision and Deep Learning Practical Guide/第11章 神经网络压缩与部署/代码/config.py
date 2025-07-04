import torch

# 训练初始模型时的学习率
init_epoch_lr = [(10, 0.01), (20, 0.001), (20, 0.0001)]
# 每次稀疏化的参数占比
SPARISITY_LIST = [50, 60, 70, 80, 90]
# 稀疏化之后微调模型时的学习率
finetune_epoch_lr = [
    # 50
    [(3, 0.01),(3, 0.001), (3, 0.0001)],
    # 60
    [(6, 0.01),(6, 0.001), (6, 0.0001)],
    # 70
    [(9, 0.01),(9, 0.001), (9, 0.0001)],
    # 80
    [(12, 0.01),(12, 0.001), (12, 0.0001)],
    # 90
    [(20, 0.01),(20, 0.001), (20, 0.0001)],
]

CHECKPOINT = "/data/chapter_seven"
DATA_FOLDER = "/data/cifar10"
BATCH_SIZE = 128


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
