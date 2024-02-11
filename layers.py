"""This module provides a PyTorch implementation of the SoftHebb convolution.

The methods in this module are based the paper and corresponding code by [1]_.

Classes
=======
- :class:`BNConvTriangle`: Combination of batch normalization, Hebbian convolution, and triangle activation.
- :class:`HebbConv2d`: Hebbian convolution.

References
==========
.. [1] Journé, A., Garcia Rodriguez, H., Guo, Q., & Moraitis, T. (2023). Hebbian deep learning without feedback.
    *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=8gd4M-_Rj1
"""

import math

import torch
from torch import Tensor
from torch.nn import BatchNorm2d, Module, ReflectionPad2d
from torch.nn.functional import conv2d

from activations import RePUTriangle, ScaledSoftmax2d, Triangle


class BNConvTriangle(Module):
    """Combination of batch normalization, Hebbian convolution, and (RePU) triangle activation.

    This combination is used in SoftHebb networks. A RePU triangle is used if ``p`` is not ``None``.

    :param in_channels: The number of input channels.
    :param out_channels: The number of output channels.
    :param kernel_size: The kernel size.
    :param eta: The base learning rate.
    :param stride: The stride for convolution (default: 1).
    :param dilation: The dilation for convolution (default: 1).
    :param temp: The temperature for the softmax operation (default: 1).
    :param p: The power for the RePU triangle (optional).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], eta: float,
                 stride: int | tuple[int, int] = 1, dilation=1, temp=1.0, p: float | None = None):
        super(BNConvTriangle, self).__init__()

        self.bn = BatchNorm2d(num_features=in_channels, affine=False)
        self.conv = HebbConv2d(in_channels, out_channels, kernel_size, eta, stride, dilation, temp)

        if p is None:
            self.activation = Triangle()
        else:
            self.activation = RePUTriangle(p)

    def forward(self, x: Tensor):
        """Forward pass.

        :param x: Tensor of shape (N, C_in, H, W) or (C_in, H, W).
        :return: Output tensor of shape (N, C_out, H, W).
        """

        x = self.bn(x)
        x = self.conv(x)
        x = self.activation(x)
        return x


class HebbConv2d(Module):
    """Hebbian convolution.

    Applies reflective padding to ensure that the input shape equals the output shape.

    :param in_channels: The number of input channels.
    :param out_channels: The number of output channels.
    :param kernel_size: The kernel size.
    :param eta: The base learning rate.
    :param stride: The stride for convolution (default: 1).
    :param dilation: The dilation for convolution (default: 1).
    :param temp: The temperature for the softmax operation (default: 1).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], eta: float,
                 stride: int | tuple[int, int] = 1, dilation=1, temp=1.0):
        super(HebbConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.eta = eta
        self.stride = stride
        self.dilation = dilation

        # Initialize softmax.
        self.softmax = ScaledSoftmax2d(temp)

        # Compute padding.
        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError(f"Kernel size must be odd, received {self.kernel_size}.")
        effective_height = self.kernel_size[0] + (self.kernel_size[0] - 1) * (self.dilation - 1)
        vertical_pad = (effective_height - 1) // 2
        effective_width = self.kernel_size[1] + (self.kernel_size[1] - 1) * (self.dilation - 1)
        horizontal_pad = (effective_width - 1) // 2
        self.pad = ReflectionPad2d((horizontal_pad, horizontal_pad, vertical_pad, vertical_pad))

        # Initialize and register weights.
        self.register_buffer('weight', self._initialize())

        # Register weight norm and retrieve it.
        self.register_buffer("weight_norm", torch.ones(self.out_channels), persistent=False)
        self._get_norm(update=True)

        # Initialize and register adaptive learning rate.
        self.register_buffer("lr", torch.ones(self.out_channels), persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor):
        """Forward pass.

        :param x: Tensor of shape (N, C_in, H, W) or (C_in, H, W).
        :return: Output tensor of shape (N, C_out, H, W).
        """

        # Pad input.
        x = self.pad(x)

        # Compute pre-activation.
        u = conv2d(x, self.weight, stride=self.stride, dilation=self.dilation)

        # Update if in training mode.
        if self.training:
            self._update(x, u)
        return u

    def _initialize(self):
        """Initialize weights.

        :return: The initialized weights.
        """

        initial_r = 25  # Initial radius.
        n_weights = self.in_channels * self.kernel_size[0] * self.kernel_size[1]

        # Note: Original code uses R * sqrt(sqrt(pi / 2) / N) for some reason. Below follows the paper.
        sigma = initial_r * math.sqrt(math.pi / (2 * n_weights))

        return sigma * torch.randn((self.out_channels, self.in_channels, *self.kernel_size))

    def _update(self, x: Tensor, u: Tensor):
        """Apply the soft Hebbian WTA update.

        :param x: Tensor of shape (N, C_in, H, W) or (C_in, H, W).
        :param u: Pre-activation.
        """

        # Compute activation.
        y = self.softmax(u)

        # Negate update scaling for all but the winner.
        neg_y = -y  # Negate for all.
        winners = torch.argmax(y, dim=-3).unsqueeze(dim=-3)  # Winner index tensor of shape (1, H, W) or (N, 1, H, W).
        y = neg_y.scatter(dim=-3, index=winners, src=y.gather(dim=-3, index=winners))  # Un-negate for winners.

        # This step from the original implementation first transforms x to (C_in, N, H + vertical_pad, W +
        # horizontal_pad) and y to (C_out, N, H, W). Then the convolution of these tensors yields a tensor of shape
        # (C_in, C_out, kernel_height, kernel_width) which corresponds to the patch-summed yx for each weight (c_in,
        # c_out, h, w). To illustrate: The first element (1, 1, 1, 1) gives the average yx for neuron 1 and input
        # channel 1 at kernel position (1, 1). That is, it gives
        #   x_1111 * y_1111 + x_1112 * y_1112 + ...,
        # which we can see as yx for neuron 1 at the top-left kernel position for the first patch, plus that of the
        # second patch (moved one to the right in x and to activation y_1112). The second element(1, 1, 1, 2) gives the
        # average yx for neuron 1 and input channel 1 at kernel position (1, 2). That is, it gives
        #   x_1112 * y_1111 + x_1113 * y_1112 + ...,
        # which we can see as yx for neuron 1 at kernel position (1, 2) for the first patch, plus that of the
        # second patch (moved one to the right in x and to activation y_1112). Transposed, this gives a patch-summed
        # tensor yx for each neuron of shape (C_in, kernel_height, kernel_width).
        yx = conv2d(x.transpose(0, 1), y.transpose(0, 1), stride=self.stride, dilation=self.dilation).transpose(0, 1)

        # The product yu is simpler as y and u both have shape (N, C_out, H, W). We take the element-wise product
        # which gives the activation times pre-activation for each neuron at each patch position. Then, the result is
        # summed over all patches like before to obtain a vector of length C_out and unsqueezed to match the shape of
        # the weights (C_out, C_in, kernel_height, kernel_width)
        yu = torch.sum(torch.mul(y, u), dim=(0, 2, 3)).view(-1, 1, 1, 1)

        # The update is now computed using the learning rule: delta_w = y * (x - uw) = yx - yu * w.
        delta_w = yx - yu * self.weight

        # Updates are normalized by dividing by the maximum absolute value.
        max_val = torch.abs(delta_w).max()
        delta_w.div_(max_val + 1e-30)

        # Apply update and update learning rate.
        self.weight.add_(self.lr.view(-1, 1, 1, 1) * delta_w)
        self._update_lr()

    def _get_norm(self, update=False):
        """Retrieve weight norm and update if necessary.

        :param update: True if the norm needs to be updated.
        :return: The weight norm.
        """

        if update:
            # Compute norm for each neuron (first dimension).
            self.weight_norm = torch.linalg.norm(self.weight.view(self.weight.shape[0], -1), dim=1)
        return self.weight_norm

    def _update_lr(self):
        """Update the neuron-specific adaptive learning rate."""

        epsilon = 1e-10  # Small number for numerical stability, as in original work.

        weight_norm = self._get_norm(update=True)
        self.lr = self.eta * torch.sqrt(torch.abs(weight_norm - 1) + epsilon)


class Zero(Module):
    """Module that zeroes out the input.

    :param stride: The stride to be used for the operation.
    """

    def __init__(self, stride=1):
        super(Zero, self).__init__()

        self.stride = stride

    def forward(self, x: Tensor):
        """Forward pass.

        :param x: Input tensor.
        :return: A zero tensor.
        """

        out_shape = x.shape

        # Divide height and width by the stride.
        out_shape[2] //= self.stride
        out_shape[3] //= self.stride

        return torch.zeros(out_shape)


class Identity(Module):
    """Module that applies an identity operation.

    :param stride: The stride to be used for the operation.
    """

    def __init__(self, stride=1):
        super(Identity, self).__init__()

        self.stride = stride

    def forward(self, x: Tensor):
        """Forward pass.

        :param x: Input tensor.
        :return: The output tensor.
        """

        return x[:, :, ::self.stride, ::self.stride]
