import torch
import torch.nn as nn  # neural network class
import torch.nn.functional as F  # for activation functions

from tqdm import tqdm
from time import sleep
output = []

# playing around with tqdm
for i in tqdm(range(0, 101)):
    divisible_by_3 = i % 3 == 0
    divisible_5 = i % 5 == 0

    if divisible_5 and divisible_by_3:
        output.append("fizzbuzz")
    elif divisible_5:
        output.append("buzz")
    elif divisible_by_3:
        output.append("fizz")
    else:
        output.append(i)

    sleep(0.10)

print(output)


