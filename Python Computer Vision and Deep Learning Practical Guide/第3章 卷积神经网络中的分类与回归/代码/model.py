# 本文件中包含vgg11和resnet18两种模型结构，在学习过程中可以任选其一进行练习
import torch
from torch import nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(512, num_classes)

    # 根据cfg配置参数逐步叠加网络层
    def _make_layers(self, cfg):
        layers = []
        # 输入通道，彩色图片的通道数量是3
        in_channels = 3
        for x in cfg:
            # 如果x==M，则添加一个最大池化层
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # 如果不是M则添加一套卷积（卷积+batchnorm+relu）
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        # 加入平均池化
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        # 计算特征网络
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # 计算分类网络
        out = self.classifier(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, stride=1):
        """
        in_channels: 输入通道数
        mid_channels: 中间及输出通道数
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # 定义短接网络，如果不需要调整维度的话，shortcut就是一个空的nn.Sequential
        self.shortcut = nn.Sequential()
        # 因为shortcut后需要做加法，要求维度匹配
        # 所以input_channels与最终的channels不匹配时，则需要通过1×1卷积进行升维
        if stride != 1 or in_channels != mid_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    mid_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(mid_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        # 搭建basicblock
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # 最后的线性层
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, mid_channels, num_blocks, stride):
        strides = [stride] + [1] * (
            num_blocks - 1
        )  # stride 只指定第一个block的stride，后面的stride都是1
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, mid_channels, stride))
            self.in_channels = mid_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# 构建resnet18模型
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# 构建vgg11模型
def vgg11():
    cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
    return VGG(cfg)


if __name__ == "__main__":
    from torchsummary import summary

    vggnet = vgg11().cuda()
    resnet = resnet18().cuda()

    summary(vggnet, (3, 32, 32))
    summary(resnet, (3, 32, 32))

