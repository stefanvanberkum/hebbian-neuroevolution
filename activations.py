"""This module provides activation functions.

Classes
========
- :class:`ScaledSoftmax`: The temperature-scaled softmax.
- :class:`Triangle`: The triangle activation function.
"""

import torch
from torch import Tensor
from torch.nn import ReLU, Softmax


class ScaledSoftmax(Softmax):
    """The temperature-scaled softmax.

    This class extends PyTorch's Softmax class but scales the logits by a temperature parameter.

    :ivar temp: The temperature.
    """

    def __init__(self, dim=None, temp=1):
        super(ScaledSoftmax, self).__init__(dim)

        self.temp = temp

    def forward(self, input: Tensor):
        return super(ScaledSoftmax, self).forward(input / self.temp)


class Triangle(ReLU):
    """The triangle activation.

    This class extends PyTorch's ReLU class but subtracts the channel-wise mean from each input. It works for image
    data with shape (C_in, H, W) or a batched version (N, C_in, H, W).

    :ivar p: Power to apply before ReLU (optional).
    """

    def __init__(self, p=None):
        super(Triangle, self).__init__()

        self.p = p

    def forward(self, input: Tensor):
        if len(input.shape) == 3:
            # One sample, input shape: (C_in, H, W).
            if self.p is None or self.p == 1:
                return super(Triangle, self).forward(input - torch.mean(input, dim=0))
            else:
                return super(Triangle, self).forward(torch.pow(input - torch.mean(input, dim=0), self.p))
        else:
            # Batch, input shape: (N, C_in, H, W).
            if self.p is None or self.p == 1:
                return super(Triangle, self).forward(input - torch.mean(input, dim=1))
            else:
                return super(Triangle, self).forward(torch.pow(input - torch.mean(input, dim=1), self.p))
