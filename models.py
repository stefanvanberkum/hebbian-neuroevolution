"""This module provides models.

Classes
=======
- :class:`HebbNet`: Hebbian encoder with final classifier.
- :class:`HebbianEncoder`: Modular Hebbian encoder network.
- :class:`Classifier`: Linear classifier module for Hebbian networks.
- :class:`SoftHebbSmall`: The small SoftHebb encoder network for CIFAR-10.
"""

import torch
from networkx import topological_generations, topological_sort
from torch import Tensor
from torch.nn import AvgPool2d, Dropout, Flatten, Linear, MaxPool2d, Module, ModuleList, Sequential

from evolution import Architecture, Cell
from layers import BNConvTriangle, Identity, Zero


class HebbNet(Module):
    """Hebbian encoder with final classifier.

    This network can only be used for inference. To train the individual components, use the corresponding methods in
    ``training.py``.

    :ivar encoder: The encoder network.
    :ivar classifier: The final classifier.

    :param encoder: The encoder network.
    :param classifier: The final classifier.
    """

    def __init__(self, encoder: Module, classifier: Module):
        super(HebbNet, self).__init__()

        self.encoder = encoder
        self.classifier = classifier
        self.eval()

    @torch.inference_mode()
    def forward(self, x: Tensor):
        """Forward pass.

        Runs the sample through the encoder and classifier in inference mode.

        :param x: Tensor of shape (N, C_in, H, W) or (C_in, H, W).
        :return: Output logits tensor of size N.
        """

        x = self.encoder(x)
        x = self.classifier(x)
        return x


class HebbianEncoder(Module):
    """Modular Hebbian encoder network.

    This network is constructed using evolved cells.
    """

    def __init__(self, in_features: int, architecture: Architecture, n_channels: int, stack_size: int):
        super(HebbianEncoder, self).__init__()

        # Translate cells into modules.


class HebbianCell(Module):
    """Module from a network cell."""

    def __int__(self, cell: Cell, in_channels: int, out_channels: int, stride: int):
        super(HebbianCell, self).__init__()

        self.cell = cell
        self.generations = list(topological_generations(cell))
        self.nodes = list(topological_sort(cell))  # Ensures appropriate order of operations.
        self.inputs = []  # List of (left, right) tuples recording each node's inputs.
        self.n_ops = len(self.nodes) - 2

        # Translate pairwise operations.
        self.layers = ModuleList()
        for node in self.nodes:
            if node != 0 and node != 1:
                # Get inputs and corresponding operations.
                (left, _, op_left), (right, _, op_right) = list(cell.in_edges(node, data=True))
                self.inputs += [(left, right)]

                # TODO: Does it apply stride to the first inputs or preprocess?

                # Translate and register operations.
                if left == 0 or left == 1:
                    # Apply the given stride (i.e., one for a normal cell and two for a reduction cell).
                    self.layers.append(self.translate(op_left, in_channels, out_channels, stride))
                else:
                    self.layers.append(self.translate(op_left, in_channels, out_channels, stride=1))
                if right == 0 or right == 1:
                    # Apply the given stride (i.e., one for a normal cell and two for a reduction cell).
                    self.layers.append(self.translate(op_right, in_channels, out_channels, stride))
                else:
                    self.layers.append(self.translate(op_right, in_channels, out_channels, stride=1))

    def forward(self, x_skip: Tensor, x: Tensor):

        # Record intermediate outputs.
        out = [Tensor()] * (self.n_ops + 2)
        out[0] = x_skip
        out[1] = x

        for node in range(self.n_ops):
            if node != 0 and node != 1:
                left, right = self.inputs[node]

                # Apply operation to the left input.
                x_left = self.layers[2 * node](out[left])

                # Apply operation to the right input.
                x_right = self.layers[2 * node + 1](out[right])

                # Add result.
                out[node] = torch.add(x_left, x_right)

    @staticmethod
    def translate(op: str, in_channels: int, out_channels: int, stride: int):
        """Translate an operation from string name to the corresponding PyTorch module.

        :param op: The operation name.
        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param stride: The stride to be used for the operation.
        :return: The corresponding PyTorch module.
        """

        eta = 0.01

        if op == 'zero':
            # Set everything to zero (i.e., ignore input).
            return Zero(stride=stride)
        elif op == 'identity':
            # No operation (i.e., raw input).
            return Identity(stride=stride)
        elif op == 'avg_pool_3':
            # 3x3 average pooling.
            return AvgPool2d(kernel_size=3, stride=stride, padding=1)
        elif op == 'max_pool_3':
            # 3x3 max pooling.
            return MaxPool2d(kernel_size=3, stride=stride, padding=1)
        elif op == 'conv_1':
            # 1x1 Hebbian convolution.
            return BNConvTriangle(in_channels, out_channels, kernel_size=1, eta=eta, stride=stride)
        elif op == 'conv_3':
            # 3x3 Hebbian convolution.
            return BNConvTriangle(in_channels, out_channels, kernel_size=3, eta=eta, stride=stride)
        elif op == 'conv_13_31':
            # 1x3 and then 3x1 Hebbian convolution.
            conv_13 = BNConvTriangle(in_channels, out_channels, kernel_size=(1, 3), eta=eta, stride=(1, stride))
            conv_31 = BNConvTriangle(in_channels, out_channels, kernel_size=(3, 1), eta=eta, stride=(stride, 1))
            return Sequential(conv_13, conv_31)
        elif op == 'dilated_conv_5':
            # 5x5 dilated Hebbian convolution (i.e., 3x3 with dilation=2).
            return BNConvTriangle(in_channels, out_channels, kernel_size=3, eta=eta, stride=stride, dilation=2)
        else:
            raise ValueError(f"Operation {op} not found (internal error).")


class Classifier(Module):
    """Linear classifier module for Hebbian networks.

    This module flattens the input, applies dropout, and ends with a linear layer. It returns logits.

    :param in_features: The number of input features.
    :param out_features: The number of output features.
    """

    def __init__(self, in_features: int, out_features: int):
        super(Classifier, self).__init__()

        self.flatten = Flatten(start_dim=-3, end_dim=-1)
        self.dropout = Dropout()
        self.linear = Linear(in_features, out_features)

    def forward(self, x: Tensor):
        """Forward pass.

        Runs the sample through the classifier.

        :param x: Tensor of shape (N, C_in, H, W) or (C_in, H, W).
        :return: Output logits tensor of size N.
        """

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
        """Forward pass.

        Runs the sample through the encoder network.

        :param x: Tensor of shape (N, C_in, H, W) or (C_in, H, W).
        :return: Output tensor of shape (N, C_out, H_out, W_out) or (C_out, H_out, W_out).
        """

        x = self.layer_1(x)
        x = self.pool_1(x)
        x = self.layer_2(x)
        x = self.pool_2(x)
        x = self.layer_3(x)
        x = self.pool_3(x)
        return x
