import torch
from tqdm import tqdm


def train(dataset,
          model,
          loss_fn,
          optimizer,
          batch_size=32,
          device='cpu',
          progress_bar=False
          ) -> None:

    running_loss = 0.
    last_loss = 0.
    model.to(device)

    if progress_bar:
        wrapper = tqdm(enumerate(dataset), total=len(dataset))
    else:
        wrapper = enumerate(dataset)

    for batch, (X, y) in wrapper:

        # X = X.type(torch.LongTensor)
        X, y = X.to(device), y.to(device)

        # zero out the gradients for every batch!
        optimizer.zero_grad()

        # Compute outputs for the batch
        output = model(X)

        # get the loss
        loss = loss_fn(output, y.reshape(-1, 1))
        loss.backward()


        # Backpropagation
        optimizer.step()

        running_loss += loss.item()
