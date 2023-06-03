import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ElodeaImages
from sklearn.metrics import confusion_matrix, classification_report

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
            nn.Linear(in_features=3*target_x*target_y,  # 3 for 3 channels
                      out_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
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

    #resize = lambda img: Resize(img, size=(512, 512))
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
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork(target_x, target_y, batch_size=BATCH_SIZE).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


    def train(dataset, model, loss_fn, optimizer):
        size = len(dataset)
        model.train()
        for batch, (X, y) in enumerate(dataset):
            #X = X.type(torch.LongTensor)
            y = y.type(torch.LongTensor)
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
        num_batches = 0
        model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataset:
                y = y.type(torch.LongTensor)  # cast to GPU appropriate type
                X, y = X.to(device), y.to(device)

                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                size += 1

        y_preds = np.array([model(X) for X, _ in dataset])
        y_true = np.array([y for _, y in dataset])


        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        cm = confusion_matrix(y_true, y_preds)
        r = classification_report(y_true, y_preds)

        print(f'Confusion Matrix:  \n{cm}')
        print(r)


    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader_custom, model, loss_fn, optimizer)
        test(test_dataloader_custom, model, loss_fn)
    print("Done!")
