import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np
from datamodel import FloatImageDataset, train_test_split
from torchvision.transforms import ToTensor, Grayscale

def train(dataloader: DataLoader,
          model: nn.Module,
          loss_fn: nn.modules.loss._Loss,  # type hints are stupid and difficult for this
          optimizer: torch.optim.Optimizer,
          device: str,
          verbose: bool = False) -> float:
    size = len(dataloader.dataset)
    model.train()  # note we didn't define this, it must be in the parent class

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # send the work to the device

        prediction = model(X)  # compute the prediction
        loss = loss_fn(prediction, y)  # compute the loss value

        # now do back prop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # zero the gradients

        if batch % 100 == 0 and verbose:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current: >5d}/{size:>5d}]")

    return loss.item()  # return the final loss


def test(dataloader: DataLoader,
         model: nn.Module,
         loss_fn: nn.modules.loss._Loss,  # type hints are stupid and difficult for this
         device: str,
         verbose: bool = False) -> tuple:

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            test_loss += loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()  # wtf is this doing?!

    test_loss /= num_batches
    correct /= size

    if verbose:
        print(f"Test Error: \n Accuracy: {(100 * correct): >0.1f}%, avg loss: {test_loss: >8f} \n")

    return correct, test_loss

def train_and_test_model(
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.modules.loss._Loss,  # type hints are stupid and difficult for pytorch
        optimizer: torch.optim.Optimizer,
        device: str,
        epochs: int = 10,
        silent: bool = False,
        verbose: bool = False,) -> dict:
    training_losses = []
    testing_losses = []
    testing_accuracies = []
    epoch = []

    iterator = tqdm(range(epochs)) if not verbose or not silent else range(epochs)
    for t in iterator:
        if verbose:
            print(f"Epoch {t + 1}:\n ----------------------------------------------------------")

        training_loss = train(train_dataloader, model, loss_fn, optimizer, device, verbose=verbose)
        training_losses.append(training_loss)

        test_acc, test_loss = test(test_dataloader, model, loss_fn, device, verbose=verbose)
        testing_losses.append(test_loss)
        testing_accuracies.append(test_acc)
        epoch.append(t)

        if not silent:
            if not verbose:
                iterator.set_description(
                    f"Training Loss: {training_loss:.2f} Testing Loss: {test_loss:.2f} Accuracy {test_acc:.2f}"
                )

    return {"training_loss": training_losses,
            "testing_loss": testing_losses,
            "testing_accuracy": testing_accuracies,
            "epoch": epoch}


def saved_model_tester(model: nn.Module,
                       test_dataloader: DataLoader,
                       device: str) -> str:



    # evaluate the trained model
    y_pred = []
    y_true = []

    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)
        prediction = model(X).detach().cpu().numpy()
        prediction = np.argmax(prediction, axis=1)
        y_pred.extend(prediction)
        y_true.extend(y.detach().cpu().numpy())

    return classification_report(y_true, y_pred)