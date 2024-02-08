"""This module provides activation functions.

Classes
=======
- :class:`ScaledSoftmax`: The temperature-scaled softmax.
- :class:`Triangle`: The triangle activation function.
- :class:`RePUTriangle`: The RePU triangle activation.
"""

import torch
from torch import Tensor
from torch.nn import ReLU, Softmax2d


class ScaledSoftmax2d(Softmax2d):
    """The temperature-scaled 2D softmax.

    This class extends PyTorch's Softmax2d class but scales the logits by a temperature parameter:

    .. math:: \mathrm{softmax}(x; \tau) = \mathrm{softmax}(\frac{x}{\tau}).

    It applies the softmax operation to each spatial location.

    :ivar temp: The temperature (default: 1).
    """

    def __init__(self, temp=1.0):
        super(ScaledSoftmax2d, self).__init__()

        self.temp = temp

    def forward(self, x: Tensor):
        """Forward pass.

        :param x: Tensor of shape (N, C_in, H, W) or (C_in, H, W).
        :return: Output tensor of shape (N, C_out, H, W).
        """

        return super(ScaledSoftmax2d, self).forward(x / self.temp)


class Triangle(ReLU):
    """The triangle activation.

    This class extends PyTorch's ReLU class but subtracts the channel-wise mean from each input first:

    .. math:: \mathrm{triangle}(x) = \mathrm{ReLU}(0, x - \bar{x}).

    It works for image data with shape (C_in, H, W) or a batched version (N, C_in, H, W).
    """

    def __init__(self):
        super(Triangle, self).__init__()

    def forward(self, x: Tensor):
        """Forward pass.

        :param x: Tensor of shape (N, C_in, H, W) or (C_in, H, W).
        :return: Output tensor of shape (N, C_out, H, W).
        """

        # Compute channel mean. Note that the channel dimension is 0 if x.dim() == 3 and 1 if x.dim() == 4.
        channel_mean = torch.mean(x, dim=x.dim() - 3, keepdim=True)

        # Apply ReLU(x - mu) and return result.
        return super(Triangle, self).forward(x - channel_mean)


class RePUTriangle(ReLU):
    """The RePU triangle activation.

    This class extends PyTorch's ReLU class but subtracts the channel-wise mean from each input and then raises the
    result to some user-specified power:

    .. math:: \mathrm{triangle}(x) = [\mathrm{ReLU}(0, x - \bar{x})]^p.

    It works for image data with shape (C_in, H, W) or a batched version (N, C_in, H, W).

    :ivar p: Power to apply before ReLU.
    """

    def __init__(self, p):
        super(RePUTriangle, self).__init__()

        self.p = p

    def forward(self, x: Tensor):
        """Forward pass.

        :param x: Tensor of shape (N, C_in, H, W) or (C_in, H, W).
        :return: Output tensor of shape (N, C_out, H, W).
        """

        # Compute channel mean. Note that the channel dimension is 0 if x.dim() == 3 and 1 if x.dim() == 4.
        channel_mean = torch.mean(x, dim=x.dim() - 3, keepdim=True)

        # Apply [ReLU(x - mu)]^p and return result.
        return torch.pow(super(RePUTriangle, self).forward(x - channel_mean), self.p)
