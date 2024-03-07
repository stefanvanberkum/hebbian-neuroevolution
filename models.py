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
from layers import BNConvReLU, BNConvTriangle, Identity, Padding, Zero


class HebbNetA(Module):
    """HebbNet-A."""

    def __init__(self, in_channels: int = 3, n_cells: int = 3, n_channels: int = 64, config: dict | None = None):
        super(HebbNetA, self).__init__()

        default_config = {"cell_1": {"conv_1": {"eta": 0.01, "tau": 1, "p": None}},
                          "cell_2": {"conv_1": {"eta": 0.01, "tau": 1, "p": None}},
                          "cell_3": {"conv_1": {"eta": 0.01, "tau": 1, "p": None}}}

        if config is None:
            # Set default hyperparameter settings.
            config = default_config
        else:
            # Used for tuning: Configure the last cell according to config, use default for the rest.
            cell_name = f"cell_{n_cells}"
            default_config[cell_name] = config
            config = default_config

        self.cells = ModuleList()

        if n_cells >= 1:
            cell_1 = HebbCellA(in_channels, n_channels, config["cell_1"])
            self.cells.append(cell_1)
        if n_cells >= 2:
            # TODO: What is the exact number of input channels?
            cell_2 = HebbCellA(4 * n_channels, 4 * n_channels, config["cell_2"])
            self.cells.append(cell_2)
        if n_cells >= 3:
            # TODO: What is the exact number of input channels?
            cell_3 = HebbCellA(4 ** 2 * n_channels, 4 ** 2 * n_channels, config["cell_3"])
            self.cells.append(cell_3)
        # TODO: What is the exact number of output channels?
        self.out_channels = 4 ** 3 * n_channels

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


class HebbCellA(Module):
    """The evolved reduction cell for HebbNet-A."""

    def __init__(self, in_channels: int, n_channels: int, config: dict):
        super(HebbCellA, self).__init__()

        eta, tau, p = config['conv_1']["eta"], config['conv_1']["tau"], config['conv_1']["p"]

        self.convs = ModuleList


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
    If not, it comprises a sequence of ``n_reduction`` reduction cells.

    :param in_channels: The number of input channels.
    :param architecture: An evolved architecture.
    :param n_channels: The initial number of channels (doubled after each reduction layer).
    :param stack_size: The number of normal cells between reduction cells.
    :param eta: The base learning rate used in SoftHebb convolutions.
    :param scaling_factor: The scaling factor for the number of filters (default: 2).
    :param n_reduction: Number of reduction layers if no normal cell is included (default: 3).
    """

    def __init__(self, in_channels: int, architecture: Architecture, n_channels: int, stack_size: int, eta: float,
                 scaling_factor=4, n_reduction=3):
        super(HebbianEncoder, self).__init__()

        out_channels = n_channels
        self.conv_1 = BNConvTriangle(in_channels, out_channels, kernel_size=1, eta=eta)
        self.conv_3 = BNConvTriangle(in_channels, out_channels, kernel_size=3, eta=eta)
        self.conv_5 = BNConvTriangle(in_channels, out_channels, kernel_size=5, eta=eta)
        self.max_pool = Sequential(MaxPool2d(kernel_size=3, stride=1, padding=1),
                                   BNConvTriangle(in_channels, out_channels, kernel_size=1, eta=eta))
        # self.initial_conv = BNConvTriangle(in_channels, out_channels, kernel_size=5, eta=eta)
        skip_channels = in_channels  # The next skip input is the current input.
        in_channels = 4 * out_channels  # The next direct input is the current output.
        out_channels *= scaling_factor  # Scale the number of filters.

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
            # skip_channels = in_channels
            # out_channels = n_channels
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
            self.pool = AvgPool2d(kernel_size=2, stride=2)

    @torch.no_grad()
    def forward(self, x: Tensor):
        """Forward pass.

        :param x: The input image.
        :return: The feature encoding.
        """

        x_skip = x

        # Apply initial convolution.
        # x = self.initial_conv(x)
        x_1 = self.conv_1(x)
        x_3 = self.conv_3(x)
        x_5 = self.conv_5(x)
        x_max = self.max_pool(x)
        x = torch.cat([x_1, x_3, x_5, x_max], dim=-3)

        # Run input through the cells.
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

        # TODO: Clean this up.
        # Preprocess inputs if necessary.
        self.preprocess_skip = None
        self.preprocess_x = None
        # if follows_reduction and skip_channels != out_channels:
        if follows_reduction:
            # Reduce spatial shape of the skip input using a factorized reduction.
            # self.preprocess_skip = FactorizedReduction(skip_channels, out_channels, eta)
            self.preprocess_skip = BNConvTriangle(skip_channels, out_channels, 3, eta,
                                                  stride=2)  # self.preprocess_skip = MaxPool2d(kernel_size=3,
            # stride=2, padding=1)
        # elif follows_reduction:
        #    self.preprocess_skip = MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif skip_channels != out_channels:
            # Apply a 1x1 convolution to make the number of channels match.
            self.preprocess_skip = BNConvTriangle(skip_channels, out_channels, 3,
                                                  eta)  # self.preprocess_skip = Padding(out_channels - skip_channels)
        # if in_channels != out_channels:
        # Apply a 1x1 convolution to make the number of channels match.
        #    self.preprocess_x = BNConvTriangle(in_channels, out_channels, 3,
        #                                       eta)  # self.preprocess_x = Padding(out_channels - in_channels)

        # Keep track of the output channels for each node to apply zero padding where necessary.
        node_channels = [0] * n_nodes
        # node_channels[0] = skip_channels
        node_channels[1] = in_channels
        node_channels[0] = out_channels
        # node_channels[1] = out_channels

        # Mark input tensors as used (these are never appended to the output).
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

                # Translate operations to modules. Only apply stride to original inputs.
                if left == 0 or left == 1:
                    module_left = self.translate(op_left, node_channels[left], out_channels, eta, stride)
                else:
                    module_left = self.translate(op_left, node_channels[left], out_channels, eta, stride=1)
                if right == 0 or right == 1:
                    module_right = self.translate(op_right, node_channels[right], out_channels, eta, stride)
                else:
                    module_right = self.translate(op_right, node_channels[right], out_channels, eta, stride=1)

                # Record output channels for each op (only changes for convolutions).
                if "conv" in op_left:
                    left_channels = out_channels
                else:
                    left_channels = node_channels[left]
                if "conv" in op_right:
                    right_channels = out_channels
                else:
                    right_channels = node_channels[right]

                # Apply padding if necessary.
                if left_channels < right_channels:
                    padding = Padding(right_channels - left_channels)
                    module_left = Sequential(module_left, padding)
                if right_channels < left_channels:
                    padding = Padding(left_channels - right_channels)
                    module_right = Sequential(module_right, padding)

                # Record the number of output channels for this node.
                node_channels[node] = max(left_channels, right_channels)

                # Register operations.
                self.layers.append(module_left)
                self.layers.append(module_right)

                # Mark inputs as used.
                self.used[left] = True
                self.used[right] = True

        # TODO: Clean this up.
        # Store the number of output channels after concatenation of unused intermediate outputs.
        # n_unused = n_nodes - sum(self.used)
        # self.out_channels = n_unused * out_channels
        self.out_channels = sum([node_channels[node] for node in range(n_nodes) if not self.used[node]])

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
        # if self.preprocess_x is not None:
        #    x = self.preprocess_x(x)  # TODO: Remove if unused.

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
    def translate(op: str, in_channels: int, n_channels: int, eta: float, stride: int):
        """Translate an operation from string name to the corresponding PyTorch module.

        :param op: The operation name.
        :param in_channels: The number of input channels.
        :param n_channels: The number of channels in this cell.
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
            return BNConvTriangle(in_channels, n_channels, kernel_size=1, eta=eta, stride=stride)
        elif op == 'conv_3':
            # 3x3 Hebbian convolution.
            return BNConvTriangle(in_channels, n_channels, kernel_size=3, eta=eta, stride=stride)
        elif op == 'conv_13_31':
            # 1x3 and then 3x1 Hebbian convolution.
            conv_13 = BNConvTriangle(in_channels, n_channels, kernel_size=(1, 3), eta=eta, stride=(1, stride))
            conv_31 = BNConvTriangle(n_channels, n_channels, kernel_size=(3, 1), eta=eta, stride=(stride, 1))
            return Sequential(conv_13, conv_31)
        elif op == 'dilated_conv_5':
            # 5x5 dilated Hebbian convolution (i.e., 3x3 with dilation=2).
            return BNConvTriangle(in_channels, n_channels, kernel_size=3, eta=eta, stride=stride, dilation=2)
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


class NoSkipNetA(Module):
    """HebbNet-A without skip connections."""


class SoftHebbNet(Module):
    """The small SoftHebb encoder network for CIFAR-10.

    Includes tuned hyperparameter settings.
    """

    def __init__(self):
        super(SoftHebbNet, self).__init__()

        self.layer_1 = BNConvTriangle(in_channels=3, out_channels=96, kernel_size=5, eta=0.08, temp=1, p=0.7)
        self.pool_1 = MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.layer_2 = BNConvTriangle(in_channels=96, out_channels=384, kernel_size=3, eta=0.005, temp=1 / 0.65, p=1.4)
        self.pool_2 = MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.layer_3 = BNConvTriangle(in_channels=384, out_channels=1536, kernel_size=3, eta=0.01, temp=1 / 0.25)
        self.pool_3 = AvgPool2d(kernel_size=2, stride=2)

        self.out_channels = 1536

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


class SoftHebbBPNet(Module):
    """The small backpropagation SoftHebb network for CIFAR-10."""

    def __init__(self):
        super(SoftHebbBPNet, self).__init__()

        self.layer_1 = BNConvReLU(in_channels=3, out_channels=96, kernel_size=5)
        self.pool_1 = MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.layer_2 = BNConvReLU(in_channels=96, out_channels=384, kernel_size=3)
        self.pool_2 = MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.layer_3 = BNConvReLU(in_channels=384, out_channels=1536, kernel_size=3)
        self.pool_3 = AvgPool2d(kernel_size=2, stride=2)

        self.flatten = Flatten(start_dim=-3, end_dim=-1)
        self.dropout = Dropout()
        self.linear = Linear(1536 * (32 // 2 ** 3) ** 2, 10)

    @torch.no_grad()
    def forward(self, x: Tensor):
        """Forward pass.

        Runs the sample through the network.

        :param x: Tensor of shape (N, 3, 32, 32) or (3, 32, 32).
        :return: Output logits tensor of shape (N, 10) or (10).
        """

        x = self.layer_1(x)
        x = self.pool_1(x)
        x = self.layer_2(x)
        x = self.pool_2(x)
        x = self.layer_3(x)
        x = self.pool_3(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class BPNetA(Module):
    """Backpropagation variant of HebbNet-A for CIFAR-10."""

    def __init__(self, in_channels: int = 3, n_cells: int = 3, n_channels: int = 64):
        super(BPNetA, self).__init__()

        # TODO: Adapt HebbNetA code to BP.
        self.conv = BNConvReLU(in_channels, n_channels, 3)
