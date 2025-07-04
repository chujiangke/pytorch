from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os.path as osp
import os
import re
from tqdm import tqdm


from config import (
    background_folder,
    object_path,
    scale,
    num,
    img_size,
    target_folder,
)


def get_background():
    background_paths = glob(osp.join(background_folder, "*.jpg"))
    return background_paths


def extract_dog(dog):
    dog = np.array(dog)
    return np.where(np.mean(dog, axis=2) < 250)


def combine_img(background_path, dog):
    dog_num = np.random.choice(num)
    background = np.array(
        Image.open(background_path).convert("RGB").resize((300, 300))
    )
    location = []
    coordinates = []
    for n in range(dog_num):
        located = False
        while not located:
            s = np.random.random() * (scale[1] - scale[0]) + scale[0]
            dog_size = int(img_size * s)
            dog = dog.resize((dog_size, dog_size))
            single_dog = extract_dog(dog)

            cx = np.random.random() * img_size
            cy = np.random.random() * img_size
            if (
                cx + dog_size / 2 >= img_size
                or cy + dog_size / 2 >= img_size
                or cx - dog_size / 2 < 0
                or cy - dog_size / 2 < 0
            ):
                continue
            # 判断是否有重合
            overlap = False
            for loc in location:
                p_dog_size = loc[2]
                p1x = loc[0] - p_dog_size / 2
                p1y = loc[1] - p_dog_size / 2
                p2x = loc[0] + p_dog_size / 2
                p2y = loc[1] + p_dog_size / 2
                p3x = cx - dog_size / 2
                p3y = cy - dog_size / 2
                p4x = cx + dog_size / 2
                p4y = cy + dog_size / 2
                if (p1y < p4y) and (p3y < p2y) and (p1x < p4x) and (p2x > p3x):
                    overlap = True
                    break
            if overlap:
                continue
            located = True
            location.append((int(cx), int(cy), dog_size))

        # cy 对应 列
        dog_coords_x = single_dog[0] + int(cy - dog_size / 2)
        # single_dog[0] += int(cy)
        # cx 对应 行
        dog_coords_y = single_dog[1] + int(cx - dog_size / 2)
        # single_dog[1] += int(cx)
        dog_coords = tuple((dog_coords_x, dog_coords_y))
        background[dog_coords] = np.array(dog)[single_dog]
        # 用于图像分割
        coordinates.append(dog_coords)
    return background, location, coordinates


def generate_data():
    background_paths = get_background()
    dog = Image.open(object_path).convert("RGB")
    if not osp.exists(target_folder):
        os.makedirs(target_folder)
    segmentation_folder = re.sub(
        "object_detection\/$", "segmentation", target_folder
    )
    if not osp.exists(segmentation_folder):
        os.makedirs(segmentation_folder)
    for i, item in tqdm(
        enumerate(background_paths), total=len(background_paths)
    ):
        combined_img, loc, coord = combine_img(item, dog)
        target_path = osp.join(target_folder, "{:0>3d}.jpg".format(i))
        plt.imsave(target_path, combined_img)
        with open(re.sub(".jpg", ".txt", target_path), "w") as f:
            f.write(str(loc))
        mask = np.zeros((img_size, img_size, 3))
        for c in coord:
            mask[c] = 1
        segmentation_path = osp.join(
            segmentation_folder, "{:0>3d}.jpg".format(i)
        )
        plt.imsave(segmentation_path, mask)


if __name__ == "__main__":
    generate_data()
