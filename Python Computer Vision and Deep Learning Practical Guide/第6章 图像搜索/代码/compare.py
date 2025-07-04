# 选择三张照片
img_path1 = "/data/pubfig_faces/Anderson Cooper/Anderson Cooper77.jpg"
img_path2 = "/data/pubfig_faces/Anderson Cooper/Anderson Cooper104.jpg"
# img_path3 = "/data/pubfig_faces/Donald Faison/Donald Faison75.jpg"
img_path3 = "/data/pubfig_faces/Hugh Laurie/Hugh Laurie205.jpg"

# 分类模型
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn.functional as F

from config import CHECKPOINT, device, SIZE
from auto_encoder import AutoEncoder


cls_ckpt = os.path.join(CHECKPOINT, "face.pth")
cls_net = resnet18().to(device)
cls_net.load_state_dict(torch.load(cls_ckpt))
cls_net.eval()


def extract_feature_cls(net, img_tensor):
    feature = net.layer4(
        net.layer3(
            net.layer2(
                net.layer1(
                    net.maxpool(net.relu(net.bn1(net.conv1(img_tensor))))
                )
            )
        )
    )
    return feature


# AutoEncoder

enc_net = AutoEncoder().to(device)
enc_ckpt = os.path.join(CHECKPOINT, "net.pth")
enc_net.load_state_dict(torch.load(enc_ckpt))
enc_net.eval()


def extract_feature_enc(net, img_tensor):
    feature = net.forward(img_tensor, extract_feature=True)
    return feature


def compare(img1, img2, method="classification"):

    if method == "autoencoder":
        img1 = img1.convert("L")
        img2 = img2.convert("L")
    transform = transforms.Compose(
        [transforms.Resize((SIZE, SIZE)), transforms.ToTensor()]
    )
    img_tensor1 = transform(img1).unsqueeze(0).to(device)
    img_tensor2 = transform(img2).unsqueeze(0).to(device)
    if method == "classification":
        feature1 = F.normalize(
            extract_feature_cls(cls_net, img_tensor1).view(1, -1)
        )
        feature2 = F.normalize(
            extract_feature_cls(cls_net, img_tensor2).view(1, -1)
        )
    elif method == "autoencoder":
        feature1 = F.normalize(
            extract_feature_enc(enc_net, img_tensor1).view(1, -1)
        )
        feature2 = F.normalize(
            extract_feature_enc(enc_net, img_tensor2).view(1, -1)
        )
    else:
        raise Exception("Wrong method")
    similarity = feature1.mm(feature2.t())
    # similarity = torch.sqrt(torch.sum((feature1 - feature2) ** 2))
    return similarity


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # method = "classification"
    method = "autoencoder"
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)
    img3 = Image.open(img_path3)
    if method == "classification":
        similarity_cls_1 = compare(img1, img2)
        similarity_cls_2 = compare(img1, img3)
    elif method == "autoencoder":
        similarity_cls_1 = compare(img1, img2, method=method)
        similarity_cls_2 = compare(img1, img3, method=method)
    else:
        raise Exception("Wrong method")
    plt.figure()
    plt.suptitle(method)
    plt.subplot(131)
    plt.title("src")
    plt.imshow(img1)
    plt.subplot(132)
    plt.title("%.4f" % similarity_cls_1.item())
    plt.imshow(img2)
    plt.subplot(133)
    plt.title("%.4f" % similarity_cls_2.item())
    plt.imshow(img3)
    plt.savefig("img/{} compare.png".format(method))
    plt.show()
