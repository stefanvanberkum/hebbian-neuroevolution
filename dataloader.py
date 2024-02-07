"""This module provides methods for loading and preproccesing data.

Functions
=========
- :func:`load`: Load and preprocess a dataset.
"""
import torch
from torch import Generator, tensor
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, MNIST


class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, device: str):
        # Put both data and targets on device in advance and switch to (N, C, H, W) shape.
        self.data = tensor(self.data, dtype=torch.float, device=device)
        self.targets = tensor(self.targets, device=device)
        self.data = torch.movedim(self.data, -1, 1)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class FastCIFAR10(CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, device: str):
        # Put both data and targets on device in advance and switch to (N, C, H, W) shape.
        self.data = tensor(self.data, dtype=torch.float, device=device)
        self.targets = tensor(self.targets, device=device)
        self.data = torch.movedim(self.data, -1, 1)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def load(dataset: str, validation=False):
    # Check validity of input argument.
    valid = {'MNIST', 'CIFAR10'}
    if dataset not in valid:
        raise ValueError(f"Dataset {dataset} not found.")

    # Create directory.
    path = f"data/{dataset}"
    # makedirs(path, exist_ok=True)

    if dataset == 'MNIST':
        train = FastMNIST(path, train=True, download=True)
        test = FastMNIST(path, train=False)
    elif dataset == 'CIFAR10':
        train = FastCIFAR10(path, train=True, download=True)
        test = FastCIFAR10(path, train=False)
    else:
        raise RuntimeError(f"Dataset {dataset} not found (internal error).")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train.preprocess(device)
    test.preprocess(device)

    if validation:
        generator = Generator()
        train, val = random_split(train, [0.9, 0.1], generator)
        return train, val, test

    return train, test
