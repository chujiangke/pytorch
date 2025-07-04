from captcha_model import net
from captcha_data import val_data, char_list
from captcha_train import device
import matplotlib.pyplot as plt
from torchvision import transforms
import torch


net.eval()
# 训练集
img, label = val_data[12]
prediction = net(img.unsqueeze(0).to(device)).view(4, 36)
predict = torch.argmax(prediction, dim=1)
print(
    "Predicte Label: {}".format(
        "-".join([char_list[lab.int()] for lab in predict])
    )
)
plt.imshow(transforms.ToPILImage()(img))
plt.show()
