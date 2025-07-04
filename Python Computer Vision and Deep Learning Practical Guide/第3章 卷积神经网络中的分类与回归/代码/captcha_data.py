from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from captcha.image import ImageCaptcha
from random import randint, seed
import matplotlib.pyplot as plt
from tqdm import tqdm


char_list = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]


class CaptchaData(Dataset):
    def __init__(self, char_list, num=10000):
        self.char_list = char_list
        self.char2index = {
            self.char_list[i]: i for i in range(len(self.char_list))
        }
        self.label_list = []
        self.img_list = []
        self.num = num
        for i in tqdm(range(self.num)):
            chars = ""
            for i in range(4):
                chars += self.char_list[randint(0, 35)]
            image = ImageCaptcha().generate_image(chars)
            self.img_list.append(image)
            # 不区分大小写
            self.label_list.append(chars)  # .lower())

    def __getitem__(self, index):
        chars = self.label_list[index]
        image = self.img_list[index].convert("L")
        chars_tensor = self._numerical(chars)
        image_tensor = self._totensor(image)

        # 把标签转化为onehot编码，以适应多标签损失函数的输入
        label = chars_tensor.long().unsqueeze(1)
        label_onehot = torch.zeros(4, 36)
        label_onehot.scatter_(1, label, 1)
        label = label_onehot.view(-1)
        return image_tensor, label

    def _numerical(self, chars):
        # 标签字符转id
        chars_tensor = torch.zeros(4)
        for i in range(len(chars)):
            chars_tensor[i] = self.char2index[chars[i]]
        return chars_tensor

    def _totensor(self, image):
        # resize到224x224，可以直接使用pytorch自带的resnet模型
        # 一般卷积都是在正方形图片下进行
        image = image  # .resize((224,224))
        # 图片转tensor
        return transforms.ToTensor()(image)

    def __len__(self):
        # 必须指定dataset的长度
        return self.num


# 实例化一个dataset,大概要10000个样本才能训练得有模有样
data = CaptchaData(char_list, num=10000)
dataloader = DataLoader(
    data, batch_size=128, shuffle=True, num_workers=4
)  

val_data = CaptchaData(char_list, num=2000)
val_loader = DataLoader(
    val_data, batch_size=256, shuffle=True, num_workers=4
)  

if __name__ == "__main__":
    # 可以通过如下方式从数据集中获取图片和对应的标签：
    img, label = data[10]
    predict = torch.argmax(label.view(-1, 36), dim=1)
    plt.title("-".join([char_list[lab.int()] for lab in predict]))
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()
