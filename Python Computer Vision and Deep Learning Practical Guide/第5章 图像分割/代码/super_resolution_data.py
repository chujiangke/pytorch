from torch.utils.data import Dataset
from torchvision import transforms

from glob import glob
import os.path as osp
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split

from config import sr_data_folder

# transform可选
# transform = transforms.Compose(
#     [transforms.RandomRotation(45), transforms.RandomAffine(30), transforms.RandomVerticalFlip(), transforms.ToTensor()]
# )
transform = None


class SuperResolutionData(Dataset):
    def __init__(
        self,
        data_folder=sr_data_folder,
        subset="train",
        transform=None,
        demo=False,
    ):
        self.img_paths = sorted(glob(osp.join(sr_data_folder, "*.jpg")))

        train_paths, test_paths = train_test_split(
            self.img_paths, test_size=0.2, random_state=10
        )
        if subset == "train":
            self.img_paths = train_paths
        else:
            self.img_paths = test_paths
        self.subset = subset
        self.demo = demo
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        high = (
            Image.open(self.img_paths[index])
            .resize((256, 256))
            .convert("YCbCr")
        )
        high_y, high_cb, high_cr = high.split()
        low = high.filter(ImageFilter.BLUR())
        low_y, low_cb, low_cr = low.split()
        if self.subset == "train":
            if self.demo:
                return (
                    self.transform(low_y),
                    self.transform(high_y),
                    (high_cb, high_cr, low_cb, low_cr),
                )
            else:
                return self.transform(low_y), self.transform(high_y)
        else:
            totensor = transforms.ToTensor()
            if self.demo:
                return (
                    totensor(low_y),
                    totensor(high_y),
                    (high_cb, high_cr, low_cb, low_cr),
                )
            else:
                return totensor(low_y), totensor(high_y)

    def __len__(self):
        return len(self.img_paths)

