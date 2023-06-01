# import the necessary packages

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from dataset import ElodeaImages

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
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # 5568 × 4176
            nn.Linear(5568*4176, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

if __name__ == "__main__":



    #resize = lambda img: Resize(img, size=(512, 512))
    elodea_images = ElodeaImages(img_dir="../data",
                                 transforms=[], target_transforms=[])

    train_size = int(0.8 * len(elodea_images))
    test_size = len(elodea_images) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(elodea_images, [train_size, test_size])

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


    def train(dataset, model, loss_fn, optimizer):
        size = len(dataset)
        model.train()
        for batch, (X, y) in enumerate(dataset):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test(dataset, model, loss_fn):
        size = len(dataset)
        num_batches = len(dataset)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataset:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataset, model, loss_fn, optimizer)
        test(test_dataset, model, loss_fn)
    print("Done!")
