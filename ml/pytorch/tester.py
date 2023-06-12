import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def test(dataset,
         model,
         loss_fn,
         device="cpu"):
    print(f'Initiating test on {device}')

    model.to(device)

    size = len(dataset)
    num_batches = 0
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():

        xs = [X for X, _ in dataset]
        print(xs)

        y_preds = torch.tensor([model(x) for x in xs]).resize(-1, 1)
        print(y_preds)
        y_true = torch.tensor([y for _, y in dataset])
        print(y_true)


        cm = confusion_matrix(y_true, y_preds)
        r = classification_report(y_true, y_preds)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    print(f'Confusion Matrix:  \n{cm}')
    print(r)