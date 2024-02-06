"""This module provides models.

Classes
=======
- :class:`SoftHebbSmall`: The small SoftHebb encoder network for CIFAR-10.
"""

import torch
from torch import Tensor
from torch.nn import AvgPool2d, Dropout, Flatten, Linear, MaxPool2d, Module

from layers import BNConvTriangle


class HebbNet(Module):
    """Hebbian encoder with final classifier.

    This network can only be used for inference.
    """

    def __init__(self, encoder: Module, classifier: Module):
        super(HebbNet, self).__init__()

        self.encoder = encoder
        self.classifier = classifier

    @torch.inference_mode()
    def forward(self, x: Tensor):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class HebbianEncoder(Module):
    """Modular Hebbian encoder network.

    This network is constructed using evolved cells.
    """


class Classifier(Module):
    """Linear classifier module for Hebbian networks.

    Returns logits.
    """

    def __init__(self, in_features: int, out_features: int):
        super(Classifier, self).__init__()

        self.flatten = Flatten(start_dim=-3, end_dim=-1)
        self.dropout = Dropout()
        self.linear = Linear(in_features, out_features)

    def forward(self, x: Tensor):
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class SoftHebbSmall(Module):
    """The small SoftHebb encoder network for CIFAR-10.

    Includes tuned hyperparameter settings.
    """

    def __init__(self):
        super(SoftHebbSmall, self).__init__()

        self.layer_1 = BNConvTriangle(in_channels=3, out_channels=96, kernel_size=5, eta=0.08, temp=1, p=0.7)
        self.pool_1 = MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.layer_2 = BNConvTriangle(in_channels=96, out_channels=384, kernel_size=3, eta=0.005, temp=1 / 0.65, p=1.4)
        self.pool_2 = MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.layer_3 = BNConvTriangle(in_channels=384, out_channels=1536, kernel_size=3, eta=0.01, temp=1 / 0.25)
        self.pool_3 = AvgPool2d(kernel_size=2, stride=2)

    @torch.no_grad()
    def forward(self, x: Tensor):
        x = self.layer_1(x)
        x = self.pool_1(x)
        x = self.layer_2(x)
        x = self.pool_2(x)
        x = self.layer_3(x)
        x = self.pool_3(x)
        return x
