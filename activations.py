"""This module provides activation functions.

Classes
=======
- :class:`ScaledSoftmax`: The temperature-scaled softmax.
- :class:`Triangle`: The triangle activation function.
"""

import torch
from torch import Tensor
from torch.nn import ReLU, Softmax2d


class ScaledSoftmax2d(Softmax2d):
    """The temperature-scaled 2D softmax.

    This class extends PyTorch's Softmax2d class but scales the logits by a temperature parameter. It applies the
    softmax operation to each spatial location.

    :ivar temp: The temperature (default: 1).
    """

    def __init__(self, temp=1.0):
        super(ScaledSoftmax2d, self).__init__()

        self.temp = temp

    def forward(self, x: Tensor):
        return super(ScaledSoftmax2d, self).forward(x / self.temp)


class Triangle(ReLU):
    """The triangle activation.

    This class extends PyTorch's ReLU class but subtracts the channel-wise mean from each input. It works for image
    data with shape (C_in, H, W) or a batched version (N, C_in, H, W).
    """

    def __init__(self):
        super(Triangle, self).__init__()

    def forward(self, x: Tensor):
        # Channel dimension is 0 if x.dim() == 3 and 1 if x.dim() == 4.
        return super(Triangle, self).forward(x - torch.mean(x, dim=x.dim() - 3, keepdim=True))


class RePUTriangle(ReLU):
    """The RePU triangle activation.

    This class extends PyTorch's ReLU class but subtracts the channel-wise mean from each input. It works for image
    data with shape (C_in, H, W) or a batched version (N, C_in, H, W).

    :ivar p: Power to apply before ReLU.
    """

    def __init__(self, p):
        super(RePUTriangle, self).__init__()

        self.p = p

    def forward(self, x: Tensor):
        # Channel dimension is 0 if x.dim() == 3 and 1 if x.dim() == 4.
        return torch.pow(super(RePUTriangle, self).forward(x - torch.mean(x, dim=x.dim() - 3, keepdim=True)), self.p)
