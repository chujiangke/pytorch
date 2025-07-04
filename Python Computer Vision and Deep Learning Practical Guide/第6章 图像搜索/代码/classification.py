from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torch import nn, optim
from torchvision import transforms
from config import DATA_FOLDER, BATCH_SIZE, device, CHECKPOINT, EPOCH_LR
import torch
import os
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from cosface import MarginCosineProduct


class FaceData(Dataset):
    def __init__(self, root=DATA_FOLDER, transform=None, subset="train"):
        label_list = sorted(glob(os.path.join(root, "*")))
        self.label2index = {
            k.split("/")[-1]: v for v, k in enumerate(label_list)
        }
        # print(self.label2index)
        img_paths = glob(os.path.join(root, "*/*.jpg"))
        self.train_paths, self.test_paths = train_test_split(
            img_paths, test_size=0.15, random_state=10
        )
        if subset == "train":
            self.img_paths = self.train_paths
        else:
            self.img_paths = self.test_paths

        self.labels = [
            self.label2index[path.split("/")[-2]] for path in self.img_paths
        ]
        # print(self.label2index)

        if transform is None:
            self.transform = transforms.Compose(
                [transforms.Resize((128, 128)), transforms.ToTensor()]
            )

    def __getitem__(self, index):
        img = self.transform(Image.open(self.img_paths[index]))
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.img_paths)


def train(cos=True):

    # face_data = ImageFolder(root=DATA_FOLDER, transform=transform)
    # face_loader = DataLoader(face_data, batch_size=BATCH_SIZE, shuffle=True)

    train_data = FaceData(subset="train")
    val_data = FaceData(subset="val")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE * 2)

    net = resnet18(pretrained=True)
    net.fc = nn.Linear(512, 512)
    net.to(device)

    if cos:
        classifier = MarginCosineProduct(512, 200).to(device)
    else:
        classifier = nn.Linear(512, 200).to(device)
    criteron = nn.CrossEntropyLoss()
    writer = SummaryWriter("log")

    ckpt = os.path.join(CHECKPOINT, "face_cos_{}.pth".format(cos))
    if os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt))
    for n, (num_epoch, lr) in enumerate(EPOCH_LR):
        optimizer = optim.Adam(net.parameters(), lr=lr)
        for epoch in range(num_epoch):
            epoch_loss = 0.0
            epoch_acc = 0.0
            for img, label in tqdm(train_loader, total=len(train_loader)):
                optimizer.zero_grad()
                img, label = img.to(device), label.to(device)
                out = net(img)
                if cos:
                    out = classifier(out, label)
                else:
                    out = classifier(out)
                loss = criteron(out, label)
                pred = torch.argmax(out, dim=1)
                epoch_acc += torch.sum(pred == label).item()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(
                "epoch_loss : {} acc ： {}".format(
                    epoch_loss / len(train_loader), epoch_acc / len(train_data)
                )
            )
            writer.add_scalar(
                "epoch_acc : cos {}".format(cos),
                epoch_acc / len(train_data),
                sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
            )
            with torch.no_grad():
                val_loss = 0.0
                val_acc = 0.0
                for i, (img, label) in tqdm(
                    enumerate(val_loader), total=len(val_loader)
                ):
                    img, label = img.to(device), label.to(device)
                    out = net(img)
                    if cos:
                        out = classifier(out, label)
                    else:
                        out = classifier(out)
                    pred = torch.argmax(out, dim=1)
                    val_acc += torch.sum(pred == label).item()
                    loss = criteron(out, label)
                    val_loss += loss.item()
                print(
                    "val: {} val_loss {} val_acc : {}".format(
                        sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
                        val_loss / len(val_loader),
                        val_acc / len(val_data),
                    )
                )
                writer.add_scalar(
                    "val_acc : cos {}".format(cos),
                    val_acc / len(val_data),
                    sum([e[0] for e in EPOCH_LR[:n]]) + epoch,
                )
                torch.save(net.state_dict(), ckpt)


if __name__ == "__main__":
    train()
