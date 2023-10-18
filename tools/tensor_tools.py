import torch
import numpy as np


def logit2label(x: torch.Tensor, yuzhi, device='cpu'):
    return torch.where(torch.softmax(x.to(device), dim=-1) > yuzhi, torch.tensor(1).to(device),
                       torch.tensor(0).to(device))


def one_hot(shape, index):
    try:
        if index.shape == shape:
            return index
        a = np.zeros(shape)
        a[np.arange(shape[0]), index.astype(int)] = 1
        return a
    except IndexError:
        print(index)
        raise IndexError
