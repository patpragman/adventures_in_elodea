# doing the tutorial from the website

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Grayscale
from train_test_suite import train_and_test_model
from datamodel import FloatImageDataset, train_test_split
from torch.nn import BatchNorm2d

import plotly.express as px
import pandas as pd
"""
these are the raw datasets - these apparently cannot be directly loaded into the model
"""

sizes = [1024, 512, 256, 128, 64]
sizes.reverse()
for size in sizes:
    # construct a labels.csv in each folder
    dataset = FloatImageDataset(directory_path=f"/home/patrickpragman/resized_data/data_{size}", transform=[Grayscale()])

    # make a train-test split

    training_dataset, testing_dataset = train_test_split(dataset, train_size=0.8, random_state=42)

    batch_size = 32

    """
    create some dataloaders - in PyTorch dataloaders are the tools that help you load the data into the model - this is
    roughly analogous to the DataGenerator classes that exist in Keras apparently, regardless, you need a tool to load the
    data into the model, and separate things out etc.
    """

    # note the camel case
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)

    # let's iterate through the test_dataloader and see the shape of the objects, note that it's an iterable
    for X, y in test_dataloader:
        print(f"shape of X [N, C, H, W]: {X.shape}")
        print(f"shape of y: {y.shape}")
        print(f"datatype of X: {X.dtype}")
        print(f"datatype of y: {y.dtype}")
        break

    # ok, now we can start creating models

    # get the cpu, gpu, or mps device for the heavy lifting
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"using {device} device")




    class NeuralNetwork(nn.Module):

        def __init__(self):
            super().__init__()  # initialize the parent class

            self.flatten = nn.Flatten()

            self.linear_relu_stack = nn.Sequential(
                nn.Linear(size * size, 4096),  # you can't see it here, but in the linear layers, the bias is true
                nn.ReLU(),
                nn.Linear(4096, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2),

            )

        def forward(self, x):
            # this is required for the forward pass I am pretty sure - which makes sense
            x = self.flatten(x)  # flatten the image into a 28x28 vector
            logits = self.linear_relu_stack(x)  # calculate the logits for the linear_relu_stack for that vector
            return logits


    model = NeuralNetwork().to(device)  # we have to send this nn to whatever device we're using
    print(model)


    loss_fn = nn.CrossEntropyLoss()  # I don't know why these aren't just functions
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)  # set the optimizer to SGD

    print("type of loss_fn:", type(loss_fn))
    print("type of optimizer:", type(optimizer))
    """
    now I need some functions to train and test the data I'm working with
    
    """

    history = train_and_test_model(train_dataloader,
                                   test_dataloader,
                                   model, loss_fn, optimizer,
                                   device, epochs=150, verbose=False)


    df = pd.DataFrame(history)
    figs = [
        px.line(df, x="epoch", y=key, title=f"{key.replace('_', ' ')}") for key in history if key != "epoch"
    ]
    for fig in figs:
        fig.show()

    print('done!')
