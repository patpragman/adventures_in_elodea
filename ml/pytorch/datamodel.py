import torch
from torch.utils.data import Dataset  # custom datasets inherit from this
from torch import is_tensor
import os
import pandas as pd
from torchvision.io import read_image
import copy

class FloatImageDataset(Dataset):
    """floatplane floats image dataset"""

    def __init__(self, directory_path,
                 true_folder_name: str = "score_0",
                 false_folder_name: str = "score_3",
                 augmentations=None,
                 transform=None):
        self.directory_path = directory_path
        self.transform = transform
        self.augmentations = augmentations

        # simple one-hot encoder
        def onehot(path: str) -> int:
            score_str = path.split("/")[-2]
            if score_str == true_folder_name:
                return 1
            elif score_str == false_folder_name:
                return 0
            else:
                raise Exception("score value not found, are you sure the true and false folder names are right?")


        # if there's a labels file, great, otherwise, build one
        labels_path = f"{self.directory_path}/labels.csv"
        if os.path.exists(labels_path):
            self.labels_df = pd.read_csv(labels_path)
        else:
            data = {"path": [],
                    "score": []}

            for root, _, filenames in os.walk(self.directory_path, topdown=True):
                for filename in filenames:
                    full_path = os.path.join(root, filename)
                    score = onehot(full_path)
                    data['path'].append(full_path)
                    data['score'].append(score)
                    self.labels_df = pd.DataFrame(data)
                    self.labels_df.to_csv(f"{directory_path}/labels.csv", index=False, index_label=False)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):

        if is_tensor(index):
            index = index.tolist()

        image_path = self.labels_df.iloc[index, 0]

        image = read_image(image_path).to(torch.float32)  # make sure we're working with a float32 tensor
        label = self.labels_df.iloc[index, 1]

        if self.transform:
            for transform in self.transform:
                image = transform(image)

        if self.augmentations:
            for augmentation in self.augmentations:
                imate = augmentation(image)


        return image, label

    def deepcopy(self):
        return copy.deepcopy(self)

def train_test_split(full_dataset: FloatImageDataset,
                     train_size=0.75,
                     random_state=None) -> tuple[FloatImageDataset]:

    training_dataset = full_dataset.deepcopy()
    testing_dataset = full_dataset.deepcopy()

    df = training_dataset.labels_df

    sample_df = df.sample(frac=train_size, random_state=random_state)
    non_sample_df = df[~df.index.isin(sample_df.index)]

    training_dataset.labels_df = sample_df
    testing_dataset.labels_df = non_sample_df

    return training_dataset, testing_dataset






if __name__ == "__main__":
    sizes = [1024, 512, 256, 128, 64]

    for size in sizes:
        # construct a labels.csv in each folder
        dataset = FloatImageDataset(directory_path=f"../../resized_data/data_{size}")

        # make a train-test split

        training_dataset, testing_dataset = train_test_split(dataset)

        assert set(training_dataset).intersection(testing_dataset) == set()

