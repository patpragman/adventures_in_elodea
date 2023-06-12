import torch
from torch.utils.data import Dataset
import numpy as np

import os
from torchvision.io import read_image


class ElodeaImages(Dataset):
    def __init__(self,
                 img_dir,
                 transforms=[],
                 target_transforms=[]):

        onehot_dictionary = {"score_0": float(0), "score_3": float(1)}
        onehot = lambda s: onehot_dictionary[s]

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
                        onehot(root.split("/")[-1])
                    )

        self.img_dir = img_dir
        self.transforms = transforms
        self.target_transform = target_transforms

        self.image_labels = torch.tensor(np.array(self.image_labels, dtype="float32"))

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(img_path)
        label = self.image_labels[idx]

        for transform in self.transforms:
            image = transform(image)

        for transform in self.target_transform:
            label = transform(label)

        return image.clone().detach(), label


if __name__ == "__main__":
    file = "../data"

    elodea = ElodeaImages(file, transforms=[])

    for file, value in elodea:
        print(file, value)