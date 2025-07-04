import torch
import os.path as osp
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from model import vgg11
from config import checkpoint_folder, label_list
from data import create_datasets


def demo(img_path):
    totensor = transforms.ToTensor()
    # 输入前需要调整尺寸
    img = Image.open(img_path).resize((32, 32))
    img_tensor = totensor(img).unsqueeze(0)
    net = vgg11()
    net.load_state_dict(torch.load(osp.join(checkpoint_folder, "net.pth")))
    net.eval()
    output = net(img_tensor)
    label = torch.argmax(output, dim=1)
    plt.imshow(np.array(img))
    plt.title(str(label_list[label]))
    plt.savefig("img/plane.jpg")
    plt.show()


if __name__ == "__main__":
    demo("img/plane.jpeg")
