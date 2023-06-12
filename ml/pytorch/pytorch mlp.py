import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ElodeaImages

from trainer import train
from tester import test

# some hyperparams
BATCH_SIZE = 32
WORKERS = 0

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, target_x, target_y, batch_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=3 * target_x * target_y,  # 3 for 3 channels
                      out_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":

    target_x, target_y = 512, 512

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(target_x, target_y)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # resize = lambda img: Resize(img, size=(512, 512))
    elodea_images = ElodeaImages(img_dir="../data",
                                 transforms=[data_transform], target_transforms=[])

    train_size = int(0.8 * len(elodea_images))
    test_size = len(elodea_images) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(elodea_images, [train_size, test_size])

    train_dataloader_custom = DataLoader(dataset=train_dataset,  # use custom created train Dataset
                                         batch_size=BATCH_SIZE,  # how many samples per batch?
                                         num_workers=WORKERS,
                                         # how many subprocesses to use for data loading? (higher = more)
                                         shuffle=True)  # shuffle the data?

    test_dataloader_custom = DataLoader(dataset=test_dataset,  # use custom created test Dataset
                                        batch_size=BATCH_SIZE,
                                        num_workers=WORKERS,
                                        shuffle=False)  # don't usually need to shuffle testing data

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    # device = 'cpu'   # manual set for debugging
    print(f"Using {device} device")

    model = NeuralNetwork(target_x,
                          target_y,
                          batch_size=BATCH_SIZE)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader_custom,
              model,
              loss_fn,
              optimizer,
              batch_size=BATCH_SIZE,
              device=device,
              progress_bar=True)

    test(test_dataloader_custom,
         model,
         loss_fn,)

    print("Done!")
