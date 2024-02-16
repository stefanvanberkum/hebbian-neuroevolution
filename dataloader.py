"""This module provides methods for loading and preproccesing data.

Classes
=======
- :class:`FastMNIST`:
- :class:`FastCIFAR`:

Functions
=========
- :func:`load`: Load and preprocess a dataset.
"""
import torch
from torch import Generator, tensor
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage
from torchvision.transforms.v2.functional import resize


class FastMNIST(MNIST):
    """Fast version of PyTorch MNIST dataset.

    Eliminates the PIL image format following [1]_ based on
    https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist.

    References
    ==========
    .. [1] Journé, A., Garcia Rodriguez, H., Guo, Q., & Moraitis, T. (2023). Hebbian deep learning without feedback.
        *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=8gd4M-_Rj1
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, reduce: bool, device: str):
        """Put both data and targets on device in advance and switch to (N, C, H, W) shape.

        If put on GPU, the number of workers for the dataloaders should be set to zero (i.e., the main process).

        :param reduce: True if the images should be resized to 8x8 resolution.
        :param device: Device to put data on.
        """

        self.data.unsqueeze_(dim=1)
        self.data = self.data.type(dtype=torch.float).to(device)
        self.targets.to(device)
        if reduce:
            self.data = resize(self.data, size=[8, 8])

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class FastCIFAR10(CIFAR10):
    """Fast version of PyTorch MNIST dataset.

    Eliminates the PIL image format following [1]_ based on
    https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist.

    References
    ==========
    .. [1] Journé, A., Garcia Rodriguez, H., Guo, Q., & Moraitis, T. (2023). Hebbian deep learning without feedback.
        *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=8gd4M-_Rj1
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, reduce: bool, device: str):
        """Put both data and targets on device in advance and switch to (N, C, H, W) shape.

        If put on GPU, the number of workers for the dataloaders should be set to zero (i.e., the main process).

        :param reduce: True if the images should be resized to 8x8 resolution.
        :param device: Device to put data on.
        """

        self.data = tensor(self.data, dtype=torch.float, device=device)
        self.targets = tensor(self.targets, device=device)
        self.data = torch.movedim(self.data, -1, 1)
        if reduce:
            self.data = resize(self.data, size=[8, 8])

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def load(dataset: str, validation=False, fast=True, reduce=False):
    """Load a dataset.

    For MNIST and CIFAR, a fast version can be used that pre-loads the data onto GPU if available. In this case,
    the number of workers for the dataloaders should be set to zero (i.e., the main process).

    :param dataset: The dataset to load, one of: {MNIST, CIFAR10}.
    :param validation: True if 10% of the training set should be split off to from a validation set (default: False).
    :param fast: True if a fast version of the dataset should be used. Only works for MNIST and CIFAR10 (default: True).
    :param reduce: True if the images should be resized to a smaller resolution. Only works for MNIST and CIFAR10
        (default: False).
    :return: A tuple (train, test) if ``validation`` is ``False`` and a tuple (train, val, test) otherwise.
    """

    # Check validity of input argument.
    valid = {'MNIST', 'CIFAR10'}
    if dataset not in valid:
        raise ValueError(f"Dataset {dataset} not found.")

    # Create directory.
    path = f"data/{dataset}"

    # Load dataset.
    if dataset == 'MNIST' and fast:
        train = FastMNIST(path, train=True, download=True)
        test = FastMNIST(path, train=False)

        # Preprocess data.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train.preprocess(reduce, device)
        test.preprocess(reduce, device)
    elif dataset == 'MNIST':
        if reduce:
            transform = Compose([Resize([8, 8]), ToImage(), ToDtype(torch.float)])
        else:
            transform = Compose([ToImage(), ToDtype(torch.float)])
        train = MNIST(path, train=True, download=True, transform=transform)
        test = MNIST(path, train=False, transform=transform)
    elif dataset == 'CIFAR10' and fast:
        train = FastCIFAR10(path, train=True, download=True)
        test = FastCIFAR10(path, train=False)

        # Preprocess data.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train.preprocess(reduce, device)
        test.preprocess(reduce, device)
    elif dataset == 'CIFAR10':
        if reduce:
            transform = Compose([Resize([8, 8]), ToImage(), ToDtype(torch.float)])
        else:
            transform = Compose([ToImage(), ToDtype(torch.float)])
        train = CIFAR10(path, train=True, download=True, transform=transform)
        test = CIFAR10(path, train=False, transform=transform)
    else:
        raise RuntimeError(f"Dataset {dataset} not found (internal error).")

    if validation:
        # Split 10% from training set to form a validation set.
        generator = Generator()
        train, val = random_split(train, [0.9, 0.1], generator)
        return train, val, test

    return train, test
