from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from PIL import Image
from glob import glob
import os.path as osp
import re
from sklearn.model_selection import train_test_split

from transform import TrainTransform, TestTransform
from config import data_folder, mask_folder


class SegmentationData(Dataset):
    def __init__(
        self,
        data_folder=data_folder,
        mask_folder=mask_folder,
        subset="train",
        transform=None,
    ):
        image_paths = sorted(glob(osp.join(data_folder, "*.jpg")))
        # mask_paths = [re.sub("object_detection", "segmentation", p) for p in image_paths]
        mask_paths = sorted(glob(osp.join(mask_folder, "*.jpg")))
        for i in range(len(image_paths)):
            assert osp.basename(image_paths[i]) == osp.basename(mask_paths[i])
        image_paths_train, image_paths_test, mask_paths_train, mask_paths_test = train_test_split(
            image_paths, mask_paths, test_size=0.2, random_state=20
        )
        if subset == "train":
            self.image_paths = image_paths_train
            self.mask_paths = mask_paths_train
        else:
            self.image_paths = image_paths_test
            self.mask_paths = mask_paths_test

        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).resize((224, 224))
        # annotation_path = self.annotation_paths[index]
        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path).resize((224, 224)).convert("L")
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image, mask = ToTensor()(image), ToTensor()(mask)
        return image, mask

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    from torchvision.transforms import ToPILImage

    topil = ToPILImage()
    data = SegmentationData(transform=TrainTransform())
    image, mask = data[11]
    image, mask = topil(image), topil(mask)
    image.save("img/sample.jpg")
    mask.save("img/sample_mask.jpg")
    image.show()
    mask.show()
