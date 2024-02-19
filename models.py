"""This module provides models.

Classes
=======
- :class:`HebbNet`: Hebbian encoder with final classifier.
- :class:`HebbianEncoder`: Modular Hebbian encoder network.
- :class:`Classifier`: Linear classifier module for Hebbian networks.
- :class:`SoftHebbSmall`: The small SoftHebb encoder network for CIFAR-10.
"""
import torch
from networkx import topological_sort
from torch import Tensor
from torch.nn import AdaptiveAvgPool2d, AvgPool2d, Dropout, Flatten, Linear, MaxPool2d, Module, ModuleList, Sequential

from architecture import Architecture, Cell
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

    This network is constructed using evolved cells. It has two reduction cells with a stack of normal cells on
    either side (i.e., N-R-N-R-N) followed up by global average pooling if the architecture contains a normal cell.
    If not, it comprises a sequence of ``n_reduction`` reduction cells followed by 2x2 average pooling.

    :param in_channels: The number of input channels.
    :param architecture: An evolved architecture.
    :param n_channels: The initial number of channels (doubled after each reduction layer).
    :param stack_size: The number of normal cells between reduction cells.
    :param eta: The base learning rate used in SoftHebb convolutions.
    :param scaling_factor: The scaling factor for the number of filters (default: 2).
    :param n_reduction: Number of reduction layers if no normal cell is included (default: 3).
    """

    def __init__(self, in_channels: int, architecture: Architecture, n_channels: int, stack_size: int, eta: float,
                 scaling_factor=2, n_reduction=3):
        super(HebbianEncoder, self).__init__()

        reduction_cell = architecture.reduction_cell

        self.cells = ModuleList()

        if architecture.normal:
            normal_cell = architecture.normal_cell

            # First stack of normal cells.
            skip_channels = in_channels
            out_channels = n_channels
            for n in range(stack_size):
                cell = HebbianCell(normal_cell, in_channels, skip_channels, out_channels, eta)
                self.cells.append(cell)
                skip_channels = in_channels  # The next skip input is the current input.
                in_channels = cell.out_channels  # The next direct input is the current output.

            # First reduction cell.
            cell = HebbianCell(reduction_cell, in_channels, skip_channels, out_channels, eta, stride=2)
            self.cells.append(cell)
            skip_channels = in_channels  # The next skip input is the current input.
            in_channels = cell.out_channels  # The next direct input is the current output.
            out_channels *= scaling_factor  # Scale the number of filters.

            # Second stack of normal cells.
            for n in range(stack_size):
                if n == 0:
                    cell = HebbianCell(normal_cell, in_channels, skip_channels, out_channels, eta,
                                       follows_reduction=True)
                else:
                    cell = HebbianCell(normal_cell, in_channels, skip_channels, out_channels, eta)
                self.cells.append(cell)
                skip_channels = in_channels  # The next skip input is the current input.
                in_channels = cell.out_channels  # The next direct input is the current output.

            # Second reduction cell.
            cell = HebbianCell(reduction_cell, in_channels, skip_channels, out_channels, eta, stride=2)
            self.cells.append(cell)
            skip_channels = in_channels  # The next skip input is the current input.
            in_channels = cell.out_channels  # The next direct input is the current output.
            out_channels *= scaling_factor  # Scale the number of filters.

            # Third stack of normal cells.
            for n in range(stack_size):
                if n == 0:
                    cell = HebbianCell(normal_cell, in_channels, skip_channels, out_channels, eta,
                                       follows_reduction=True)
                else:
                    cell = HebbianCell(normal_cell, in_channels, skip_channels, out_channels, eta)
                self.cells.append(cell)
                skip_channels = in_channels  # The next skip input is the current input.
                in_channels = cell.out_channels  # The next direct input is the current output.
            self.out_channels = in_channels  # Record the final number of output channels.

            # Global average pooling.
            self.pool = AdaptiveAvgPool2d(output_size=1)
        else:
            skip_channels = in_channels
            out_channels = n_channels
            for n in range(n_reduction):
                if n == 0:
                    # First reduction.
                    cell = HebbianCell(reduction_cell, in_channels, skip_channels, out_channels, eta, stride=2)
                else:
                    cell = HebbianCell(reduction_cell, in_channels, skip_channels, out_channels, eta, stride=2,
                                       follows_reduction=True)
                self.cells.append(cell)
                skip_channels = in_channels  # The next skip input is the current input.
                in_channels = cell.out_channels  # The next direct input is the current output.
                out_channels *= scaling_factor  # Scale the number of filters.
            self.out_channels = in_channels  # Record the final number of output channels.

            self.pool = None

    @torch.no_grad()
    def forward(self, x: Tensor):
        """Forward pass.

        :param x: The input image.
        :return: The feature encoding.
        """

        # Run input through the cells.
        x_skip = x
        for cell in self.cells:
            x_next = cell(x_skip, x)
            x_skip = x
            x = x_next

        if self.pool is not None:
            # Apply pooling.
            x = self.pool(x)

        return x


class HebbianCell(Module):
    """Module from a network cell.

    All inputs are preprocessed if necessary using either 1x1 convolutions or a factorized reduction. If this cell
    follows a reduction cell, the spatial shape of the skip input is reduced using a factorized reduction. If the
    number of input channels differs from the number of output channels, a 1x1 convolution is applied to remedy this.

    :param cell: The cell.
    :param in_channels: The number of input channels.
    :param out_channels: The number of output channels.
    :param eta: The base learning rate used in SoftHebb convolutions.
    :param stride: The stride to be used for the operation (default: 1).
    :param follows_reduction: True if this cell follows a reduction cell (default: False).
    """

    def __init__(self, cell: Cell, in_channels: int, skip_channels: int, out_channels: int, eta: float, stride=1,
                 follows_reduction=False):
        super(HebbianCell, self).__init__()

        self.nodes = list(topological_sort(cell))  # Topological sorting to ensure an appropriate order of operations.
        n_nodes = len(self.nodes)

        self.cell = cell
        self.inputs = []  # List of (left, right) tuples recording each node's inputs.
        self.used = [False] * n_nodes  # List that records whether a nodes' output is used.
        self.n_ops = n_nodes - 2

        # Preprocess inputs if necessary.
        self.preprocess_skip = None
        self.preprocess_x = None
        if follows_reduction:
            # Reduce spatial shape of the skip input using a factorized reduction.
            # self.preprocess_skip = FactorizedReduction(skip_channels, out_channels, eta)
            self.preprocess_skip = BNConvTriangle(skip_channels, out_channels, 3, eta, stride=2)
        elif skip_channels != out_channels:
            # Apply a 1x1 convolution to make the number of channels match.
            self.preprocess_skip = BNConvTriangle(skip_channels, out_channels, 3, eta)
        if in_channels != out_channels:
            # Apply a 1x1 convolution to make the number of channels match.
            self.preprocess_x = BNConvTriangle(in_channels, out_channels, 3, eta)

        # Mark input tensors as used.
        self.used[0] = True
        self.used[1] = True

        # Translate pairwise operations.
        self.layers = ModuleList()  # In order of operations.
        for node in self.nodes:
            if node != 0 and node != 1:
                # Get inputs and corresponding operations.
                (left, _, left_attr), (right, _, right_attr) = list(cell.in_edges(node, data=True))
                self.inputs += [(left, right)]
                op_left = left_attr['op']
                op_right = right_attr['op']

                # Translate and register operations. Only apply stride to original inputs.
                if left == 0 or left == 1:
                    self.layers.append(self.translate(op_left, out_channels, eta, stride))
                else:
                    self.layers.append(self.translate(op_left, out_channels, eta, stride=1))
                if right == 0 or right == 1:
                    self.layers.append(self.translate(op_right, out_channels, eta, stride))
                else:
                    self.layers.append(self.translate(op_right, out_channels, eta, stride=1))

                # Mark inputs as used.
                self.used[left] = True
                self.used[right] = True

        # Store the number of output channels after concatenation of unused intermediate outputs.
        n_unused = n_nodes - sum(self.used)
        self.out_channels = n_unused * out_channels

    @torch.no_grad()
    def forward(self, x_skip: Tensor, x: Tensor):
        """Forward pass.

        :param x_skip: Skip input.
        :param x: Direct input.
        :return: Output tensor comprising unused intermediate outputs.
        """

        # Preprocess inputs if necessary.
        if self.preprocess_skip is not None:
            x_skip = self.preprocess_skip(x_skip)
        if self.preprocess_x is not None:
            x = self.preprocess_x(x)

        # Record intermediate outputs.
        out = [Tensor()] * (self.n_ops + 2)
        out[0] = x_skip
        out[1] = x

        # Loop through, apply, and store pairwise operations.
        for op in range(self.n_ops):
            node = self.nodes[op + 2]

            left, right = self.inputs[op]

            # Apply operation to the left input.
            x_left = self.layers[2 * op](out[left])

            # Apply operation to the right input.
            x_right = self.layers[2 * op + 1](out[right])

            # Add result.
            out[node] = torch.add(x_left, x_right)

        # Concatenate unused tensors along the channel dimension and return.
        unused = [element for (element, used) in zip(out, self.used) if not used]
        return torch.cat(unused, dim=-3)

    @staticmethod
    def translate(op: str, n_channels: int, eta: float, stride: int):
        """Translate an operation from string name to the corresponding PyTorch module.

        :param op: The operation name.
        :param n_channels: The number of channels.
        :param eta: The base learning rate used in SoftHebb convolutions.
        :param stride: The stride to be used for the operation.
        :return: The corresponding PyTorch module.
        """

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
            return BNConvTriangle(n_channels, n_channels, kernel_size=1, eta=eta, stride=stride)
        elif op == 'conv_3':
            # 3x3 Hebbian convolution.
            return BNConvTriangle(n_channels, n_channels, kernel_size=3, eta=eta, stride=stride)
        elif op == 'conv_13_31':
            # 1x3 and then 3x1 Hebbian convolution.
            conv_13 = BNConvTriangle(n_channels, n_channels, kernel_size=(1, 3), eta=eta, stride=(1, stride))
            conv_31 = BNConvTriangle(n_channels, n_channels, kernel_size=(3, 1), eta=eta, stride=(stride, 1))
            return Sequential(conv_13, conv_31)
        elif op == 'dilated_conv_5':
            # 5x5 dilated Hebbian convolution (i.e., 3x3 with dilation=2).
            return BNConvTriangle(n_channels, n_channels, kernel_size=3, eta=eta, stride=stride, dilation=2)
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
