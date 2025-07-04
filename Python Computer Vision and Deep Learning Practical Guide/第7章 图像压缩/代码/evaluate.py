from auto_encoder import AutoEncoder
from config import CHECKPOINT
from fix_data import val_data, train_data
import torch
import os.path as osp
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


net = AutoEncoder()
# ckpt = osp.join(CHECKPOINT, "net.pth")
# ckpt = osp.join(CHECKPOINT, "G.pth")
ckpt = osp.join(CHECKPOINT, "G_fix.pth")
net.load_state_dict(torch.load(ckpt))
net.eval()
for i in range(6):
    src, _ = val_data[i]
    img = transforms.ToPILImage()(src)
    print(i)
    plt.subplot(3, 4, (i + 1) * 2 - 1)
    plt.title("src_img")
    plt.imshow(img, cmap="gray")
    out = net(src.unsqueeze(0)).squeeze(0)
    out_img = transforms.ToPILImage()(out)
    plt.subplot(3, 4, (i + 1) * 2)
    plt.title("out_img")
    plt.imshow(out_img, cmap="gray")
    plt.savefig("img/auto_encoder_face.jpg")
plt.show()
