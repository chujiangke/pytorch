import torch
from torch.nn import Conv2d, BatchNorm2d, Linear, Sequential, ReLU, MaxPool2d, AvgPool2d
import numpy as np
import os
from torchsummary import summary
import time

from model import VGG_11_prune
from config import device, CHECKPOINT
from base_train import validation


def expand(model, layers=[]):
    for layer in model.children():
        if len(list(layer.children())) > 0:
            expand(layer, layers)
        else:
            if not isinstance(layer, ReLU) and not isinstance(layer, MaxPool2d) and not isinstance(layer, AvgPool2d):
                layers.append(layer)
    return layers


def zero_indices(layer):
    weight = layer.weight.data
    bias = layer.bias.data
    indices = []
    for idx, w in enumerate(weight.data):
        # 可以剪去全为0的通道,这样几乎不会有精度损失
        if torch.sum(w) != 0 and torch.sum(bias[idx]) != 0:
            # 为了追求更大的压缩比，可以考虑剪去数值较小的层
            # if torch.sum(torch.abs(w)) > 3:
            indices.append(idx)
    return indices


def compress_conv(model):
    layers = expand(model, [])
    channels = []
    for l1, l2 in zip(layers, layers[1:]):
        if isinstance(l1, torch.nn.Conv2d):
            indices = zero_indices(l1)
            channels.append(len(indices))
            channel_size = l1.kernel_size[0] * l1.kernel_size[1]
            prune_conv(indices, l1, conv_input=False)
            if isinstance(l2, torch.nn.Conv2d):
                prune_conv(indices, l2, conv_input=True)
            elif isinstance(l2, torch.nn.Linear):
                prune_fc(indices, channel_size, l2)
        elif isinstance(l1, torch.nn.BatchNorm2d):
            prune_bn(indices, l1)
            if isinstance(l2, torch.nn.Conv2d):
                prune_conv(indices, l2, conv_input=True)
            elif isinstance(l2, torch.nn.Linear):
                prune_fc(indices, channel_size, l2)
        else:
            pass
    return layers, channels


def prune_conv(indices, layer, conv_input=False):
    # 剪切输入
    if conv_input:
        layer._parameters["weight"].data = layer._parameters["weight"].data[:, indices]
    else:
        layer._parameters["weight"].data = layer._parameters["weight"].data[indices]
        if layer._parameters["bias"] is not None:
            layer._parameters["bias"].data = layer._parameters["bias"].data[indices]


def prune_fc(indices, channel_size, layer):
    layer.weight.data = torch.from_numpy(layer.weight.data.cpu().numpy()[:, indices])


def prune_bn(indices, layer):
    layer.weight.data = torch.from_numpy(layer.weight.data.cpu().numpy()[indices])
    layer.bias.data = torch.from_numpy(layer.bias.data.cpu().numpy()[indices])

    layer.running_mean = torch.from_numpy(layer.running_mean.cpu().numpy()[indices])
    layer.running_var = torch.from_numpy(layer.running_var.cpu().numpy()[indices])


def compress_model(net):
    layers, channels = compress_conv(net)
    for i in [1, 3, 6, 9, 12]:
        channels.insert(i, "M")
    print("channels:", channels)
    compressed_net = VGG_11_prune(channels)
    compressed_layers = expand(compressed_net, [])
    for origin, compressed in zip(layers, compressed_layers):
        if hasattr(origin, "weight"):
            if origin.weight is not None:
                compressed.weight.data = origin.weight.data
            if origin.bias is not None:
                compressed.bias.data = origin.bias.data
    return compressed_net


if __name__ == "__main__":
    # 加载并验证模型
    net = VGG_11_prune()
    net.load_state_dict(torch.load(os.path.join(CHECKPOINT, "fine_tune_sparse_90.0.pth"))["net"])
    net.eval()
    net.to(device)
    s1 = time.time()
    validation(net, torch.nn.CrossEntropyLoss())
    print("压缩前计算耗时：{:.4f}".format(time.time() - s1))
    print(summary(net.to(device), (3, 32, 32)))

    compressed_net = compress_model(net)
    compressed_net.to(device)
    s2 = time.time()
    validation(compressed_net, torch.nn.CrossEntropyLoss())
    print("压缩后计算耗时：{:.4f}".format(time.time() - s2))

    print(summary(compressed_net.to(device), (3, 32, 32)))

    torch.save(compressed_net.state_dict(), os.path.join(CHECKPOINT, "compressed_net.pth"))
