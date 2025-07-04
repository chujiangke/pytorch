import torch
from torch import nn
from torchvision.models import resnet18


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, in_channels // 4, kernel_size, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            in_channels // 4,
            out_channels,
            kernel_size=kernel_size,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.deconv(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x


class ResNet18Unet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet18Unet, self).__init__()
        base = resnet18(pretrained=pretrained)

        self.firstconv = base.conv1
        self.firstbn = base.bn1
        self.firstrelu = base.relu
        self.firstmaxpool = base.maxpool
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        out_channels = [64, 128, 256, 512]

        self.center = DecoderBlock(
            in_channels=out_channels[3],
            out_channels=out_channels[3],
            kernel_size=3,
        )
        self.decoder4 = DecoderBlock(
            in_channels=out_channels[3] + out_channels[2],
            out_channels=out_channels[2],
            kernel_size=3,
        )
        self.decoder3 = DecoderBlock(
            in_channels=out_channels[2] + out_channels[1],
            out_channels=out_channels[1],
            kernel_size=3,
        )
        self.decoder2 = DecoderBlock(
            in_channels=out_channels[1] + out_channels[0],
            out_channels=out_channels[0],
            kernel_size=3,
        )
        self.decoder1 = DecoderBlock(
            in_channels=out_channels[0] + out_channels[0],
            out_channels=out_channels[0],
            kernel_size=3,
        )

        self.finalconv = nn.Sequential(
            nn.Conv2d(out_channels[0], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        # stem
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)

        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))

        f = self.finalconv(d1)

        return f


if __name__ == "__main__":
    net = ResNet18Unet(pretrained=False)
    img = torch.rand(1, 3, 320, 320)
    out = net(img)
    print(out.shape)
