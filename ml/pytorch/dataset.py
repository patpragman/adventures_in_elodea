from torch.utils.data import Dataset

import os
import pandas as pd
from torchvision.io import read_image

class ElodeaImages(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):

        self.image_paths = []
        self.image_labels = []
        for root, _, filenames in os.walk(img_dir, topdown=True):
            for filename in filenames:
                path = f"{root}/{filename}"

                if ".JPG" in filename:
                    self.image_paths.append(
                        path
                    )
                    self.image_labels.append(
                        root.split("/")[-1]
                    )

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(img_path)
        label = self.image_labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    file = "../data"

    elodea = ElodeaImages(file)

    for file, value in elodea:
        print(file, value)