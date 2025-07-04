# 图像到图像模型的数据增强
import torch
import numpy as np
import cv2


class Compose:
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, img, mask):
        for transform in self.transform_list:
            img, mask = transform(img, mask)
        return img, mask


class ToArraySegment:
    def __call__(self, img, mask):
        img = np.array(img)
        mask = np.array(mask)
        return img, mask


class ToTensorSegment:
    def __call__(self, img, mask):
        return (
            torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
            torch.from_numpy(mask).float() / 255.0,
        )


class Resize:
    def __init__(self, size=300):
        self.size = size

    def __call__(self, img, mask):
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))
        return img, mask


class Expand:
    def __call__(self, img, mask):
        if np.random.randint(2):
            width, _, channels = img.shape
            ratio = np.random.uniform()
            expand_img = np.zeros(
                (int(width * (1 + ratio)), int(width * (1 + ratio)), channels)
            )
            expand_mask = np.zeros(
                (int(width * (1 + ratio)), int(width * (1 + ratio)))
            )
            left = np.random.uniform(0, width * ratio)
            top = np.random.uniform(0, width * ratio)
            left = int(left)
            top = int(top)
            expand_img[top : top + width, left : left + width, :] = img
            expand_mask[top : top + width, left : left + width] = mask
            return expand_img, expand_mask
        else:
            return img, mask


class Mirror:
    def __call__(self, img, mask):
        # 在绝对坐标下运行
        if np.random.randint(2):
            width = img.shape[0]
            img = img[:, ::-1]
            mask = mask[:, ::-1]
        return img, mask


class TrainTransform:
    def __init__(self, size=300):
        self.size = size
        self.augment = Compose(
            [
                ToArraySegment(),
                Mirror(),
                Expand(),
                Resize(self.size),
                ToTensorSegment(),
            ]
        )
        # self.augment = Compose([ToArraySegment(), ToTensorSegment()])

    def __call__(self, img, mask):
        img, mask = self.augment(img, mask)
        return img, mask


class TestTransform:
    def __init__(self, size=300):
        self.size = size
        self.augment = Compose(
            [ToArraySegment(), Resize(self.size), ToTensorSegment()]
        )

    def __call__(self, img, boxes):
        img, mask = self.augment(img, mask)
        return img, mask
