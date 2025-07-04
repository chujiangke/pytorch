import torch

# 定义数据存储的设备，在没有可用GPU的时候使用CPU
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# cifar-10-python.tar.gz文件下载完成之后直接放入data_folder文件夹中即可
data_folder = "/data/cifar10"
# 模型存储目录
checkpoint_folder = "/data/chapter_one"

# dataloader中每一个批次的图片数量
batch_size = 64
# 随着训练的次数增长逐步缩小学习率
epochs = [(30, 0.001), (20, 0.001), (10, 0.0001)]

# 标签列表
label_list = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
