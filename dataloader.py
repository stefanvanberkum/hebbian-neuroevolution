"""This module provides methods for loading and preprocessing data.

The fast versions of datasets eliminate the PIL image format following [1]_ based on
https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist.

Classes
=======
- :class:`FastMNIST`: A fast version of MNIST.
- :class:`FastCIFAR10`: A fast version of CIFAR-10.
- :class:`FastCIFAR100`: A fast version of CIFAR-100.
- :class:`FastSVHN`: A fast version of SVHN.

Functions
=========
- :func:`load`: Load and preprocess a dataset.

References
==========
.. [1] Journé, A., Garcia Rodriguez, H., Guo, Q., & Moraitis, T. (2023). Hebbian deep learning without feedback.
    *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=8gd4M-_Rj1
"""
import torch
from torch import Generator, tensor
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage
from torchvision.transforms.v2.functional import resize


class FastMNIST(MNIST):
    """Fast version of the PyTorch MNIST dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, reduce: bool, device: str):
        """Put both data and targets on device in advance and switch to (N, C, H, W) shape.

        If put on GPU, the number of workers for the dataloaders should be set to zero (i.e., the main process).

        :param reduce: True if the images should be resized to 16x16 resolution.
        :param device: Device to put data on.
        """

        self.data.unsqueeze_(dim=1)
        self.data = self.data.type(dtype=torch.float).to(device)
        self.targets.to(device)
        if reduce:
            self.data = resize(self.data, size=[16, 16])

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class FastCIFAR10(CIFAR10):
    """Fast version of the PyTorch CIFAR-10 dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, reduce: bool, device: str):
        """Put both data and targets on device in advance and switch to (N, C, H, W) shape.

        If put on GPU, the number of workers for the dataloaders should be set to zero (i.e., the main process).

        :param reduce: True if the images should be resized to 16x16 resolution.
        :param device: Device to put data on.
        """

        self.data = tensor(self.data, dtype=torch.float, device=device)
        self.targets = tensor(self.targets, device=device)
        self.data = torch.movedim(self.data, -1, 1)
        if reduce:
            self.data = resize(self.data, size=[16, 16])

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class FastCIFAR100(CIFAR100):
    """Fast version of the PyTorch CIFAR-100 dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, device: str):
        """Put both data and targets on device in advance and switch to (N, C, H, W) shape.

        If put on GPU, the number of workers for the dataloaders should be set to zero (i.e., the main process).

        :param device: Device to put data on.
        """

        self.data = tensor(self.data, dtype=torch.float, device=device)
        self.targets = tensor(self.targets, device=device)
        self.data = torch.movedim(self.data, -1, 1)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class FastSVHN(SVHN):
    """Fast version of the PyTorch SVHN dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.targets = []

    def preprocess(self, device: str):
        """Put both data and targets on device in advance.

        If put on GPU, the number of workers for the dataloaders should be set to zero (i.e., the main process).

        :param device: Device to put data on.
        """

        self.data = tensor(self.data, dtype=torch.float, device=device)
        self.targets = tensor(self.labels, device=device)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def load(dataset: str, validation=False, fast=True, reduce=False, seed: int | None = None):
    """Load a dataset.

    For MNIST and CIFAR, a fast version can be used that pre-loads the data onto GPU if available. In this case,
    the number of workers for the dataloaders should be set to zero (i.e., the main process).

    :param dataset: The dataset to load, one of: {MNIST, CIFAR10, CIFAR100, SVHN}.
    :param validation: True if 10% of the training set should be split off to form a validation set (default: False).
    :param fast: True if a fast version of the dataset should be used (default: True).
    :param reduce: True if the images should be resized to a smaller resolution. Only works for MNIST and CIFAR10
        (default: False).
    :param seed: Optional seed for the validation split.
    :return: A tuple (train, test) if ``validation`` is ``False`` and a tuple (train, val, test) otherwise.
    """

    # Check validity of input argument.
    valid = {'MNIST', 'CIFAR10', 'CIFAR100', 'SVHN'}
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
    elif dataset == 'CIFAR100' and fast:
        train = FastCIFAR100(path, train=True, download=True)
        test = FastCIFAR100(path, train=False)

        # Preprocess data.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train.preprocess(device)
        test.preprocess(device)
    elif dataset == 'CIFAR100':
        transform = Compose([ToImage(), ToDtype(torch.float)])
        train = CIFAR100(path, train=True, download=True, transform=transform)
        test = CIFAR100(path, train=False, transform=transform)
    elif dataset == 'SVHN' and fast:
        train = FastSVHN(path, split='train', download=True)
        test = FastSVHN(path, split='test', download=True)

        # Preprocess data.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train.preprocess(device)
        test.preprocess(device)
    elif dataset == 'SVHN':
        transform = Compose([ToImage(), ToDtype(torch.float)])
        train = SVHN(path, split='train', download=True, transform=transform)
        test = SVHN(path, split='test', transform=transform)
    else:
        raise RuntimeError(f"Dataset {dataset} not found (internal error).")

    if validation:
        # Split 10% from training set to form a validation set.
        if seed is not None:
            generator = Generator().manual_seed(seed)
        else:
            generator = Generator()
        train, val = random_split(train, [0.9, 0.1], generator)
        return train, val, test

    return train, test
