"""This module provides methods for loading and preproccesing data.

Functions
=========
- :func:`load`: Load and preprocess a dataset.
"""

from torchvision.datasets import CIFAR10, MNIST


class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Compute mean and variance.
        mu = self.data.mean()
        sigma = self.data.std()

        # Normalize.
        self.data = self.data.sub_(mu).div_(sigma)

        # Put both data and targets on device in advance.
        self.data, self.targets = self.data.cuda(), self.targets.cuda()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class FastCIFAR10(CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Compute mean and variance.
        mu = self.data.mean(dim=-1)
        sigma = self.data.std(dim=-1)

        # Normalize.
        self.data = self.data.sub_(mu).div_(sigma)

        # Put both data and targets on device in advance.
        self.data, self.targets = self.data.cuda(), self.targets.cuda()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def load(dataset: str):
    # Check validity of input argument.
    valid = {'MNIST', 'CIFAR10'}
    if dataset not in valid:
        raise ValueError(f"Dataset {dataset} not found.")

    # Create directory.
    path = f"data/{dataset}"
    # makedirs(path, exist_ok=True)

    if dataset == 'MNIST':
        train = FastMNIST(path, train=True)
        test = FastMNIST(path, train=False)
    elif dataset == 'CIFAR10':
        train = FastCIFAR10(path, train=True)
        test = FastCIFAR10(path, train=False)
    else:
        raise RuntimeError(f"Dataset {dataset} not found (internal error).")

    return train, test
