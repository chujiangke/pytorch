import torch
from torch import nn
from torch.nn import  functional as F
from torch import optim

import torch.utils
import torch.utils.data
import torchvision
from matplotlib import pyplot as plt
import  torch
from    matplotlib import pyplot as plt
 

def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()
 
def plot_image(img, label, name):#把图片和对应的标签显示出来
 
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()
 
def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out

batch_size = 512
train_db = torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                    # 图片的预处理步骤
                                    transform=torchvision.transforms.Compose([
                                    # 转换为张量
                                    torchvision.transforms.ToTensor(),
                                    # 标准化
                                    torchvision.transforms.Normalize(
                                    (0.5,), (0.5,))
                                ]))
# 创建 Dataloader 对象，方便以批量形式训练，随机打乱顺序
train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)
x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'Image') # 观察图片