"""This module provides models.

Classes
=======
- :class:`HebbNet`: Hebbian encoder with final classifier.
- :class:`HebbianEncoder`: Generic modular Hebbian encoder network used for evolution.
- :class:`HebbianCell`: Hebbian cell built from a network cell.
- :class:`Classifier`: Linear classifier module for Hebbian networks.
- :class:`HebbNetA`: The evolved HebbNet-A encoder.
- :class:`HebbCellA`: The evolved HebbNet-A reduction cell.
- :class:`BPNetA`: The backpropagation variant of the evolved HebbNet-A encoder.
- :class:`BPCellA`: The backpropagation variant of the evolved HebbNet-A reduction cell.
- :class:`SoftHebbNet`: The small SoftHebb encoder network for CIFAR-10.
- :class:`SoftHebbBPNet`: The small backpropagation variant of the SoftHebb network for CIFAR-10.
"""
import torch
from networkx import topological_sort
from torch import Tensor
from torch.nn import AdaptiveAvgPool2d, AvgPool2d, Dropout, Flatten, Linear, MaxPool2d, Module, ModuleList, Sequential

from architecture import Architecture, Cell
from layers import BNConvReLU, BNConvTriangle, Identity, Padding, Zero


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
    """Generic modular Hebbian encoder network used for evolution.

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
        self.initial_conv = BNConvTriangle(in_channels, out_channels, kernel_size=5, eta=eta)
        skip_channels = in_channels  # The next skip input is the current input.
        in_channels = out_channels
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
        x = self.initial_conv(x)

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
    """Hebbian cell built from a network cell.

    If this cell follows a reduction cell, the spatial shape of the skip input is reduced using a strided 3x3
    convolution. Otherwise, if the number of skip input channels differs from the number of output channels,
    a regular 3x3 convolution is used to preprocess the skip input. Whenever the number of channels between two
    elements of a pairwise operation is not equal, zero padding is applied to remedy this.

    :param cell: The cell.
    :param in_channels: The number of input channels.
    :param out_channels: The number of output channels.
    :param eta: The base learning rate used in SoftHebb convolutions.
    :param stride: The stride to be used for the operation (default: 1).
    :param follows_reduction: True if this cell follows a reduction cell (default: False).
    :param temp: The temperature for the softmax operation (default: 1).
    :param p: The power for the RePU triangle (optional).
    :param skip_eta: The base learning rate used in the SoftHebb convolution for preprocessing the skip input.
    :param skip_temp: The temperature for the softmax operation used in the SoftHebb convolution for preprocessing the
        skip input. (default: 1).
    :param skip_p: The power for the RePU triangle used in the SoftHebb convolution for preprocessing the skip input. (
        optional).
    """

    def __init__(self, cell: Cell, in_channels: int, skip_channels: int, out_channels: int, eta: float, stride=1,
                 follows_reduction=False, temp=1.0, p: float | None = None, skip_eta=0.01, skip_temp=1.0,
                 skip_p: float | None = None):
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
            # Reduce spatial shape of the skip input using a strided 3x3 convolution.
            self.preprocess_skip = BNConvTriangle(skip_channels, out_channels, 3, skip_eta, stride=2, temp=skip_temp,
                                                  p=skip_p)
        elif skip_channels != out_channels:
            # Apply a 3x3 convolution to make the number of channels match.
            self.preprocess_skip = BNConvTriangle(skip_channels, out_channels, 3, skip_eta, temp=skip_temp, p=skip_p)

        # Keep track of the output channels for each node to apply zero padding where necessary.
        node_channels = [0] * n_nodes
        node_channels[0] = out_channels
        node_channels[1] = in_channels

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
                    module_left = self.translate(op_left, node_channels[left], out_channels, eta, stride, temp=temp,
                                                 p=p)
                else:
                    module_left = self.translate(op_left, node_channels[left], out_channels, eta, stride=1, temp=temp,
                                                 p=p)
                if right == 0 or right == 1:
                    module_right = self.translate(op_right, node_channels[right], out_channels, eta, stride, temp=temp,
                                                  p=p)
                else:
                    module_right = self.translate(op_right, node_channels[right], out_channels, eta, stride=1,
                                                  temp=temp, p=p)

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

        # Store the number of output channels after concatenation of unused intermediate outputs.
        self.out_channels = sum([node_channels[node] for node in range(n_nodes) if not self.used[node]])

    @torch.no_grad()
    def forward(self, x_skip: Tensor, x: Tensor):
        """Forward pass.

        :param x_skip: Skip input.
        :param x: Direct input.
        :return: Output tensor comprising unused intermediate outputs.
        """

        # Preprocess skip input if necessary.
        if self.preprocess_skip is not None:
            x_skip = self.preprocess_skip(x_skip)

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
    def translate(op: str, in_channels: int, n_channels: int, eta: float, stride: int, temp=1.0,
                  p: float | None = None):
        """Translate an operation from string name to the corresponding PyTorch module.

        :param op: The operation name.
        :param in_channels: The number of input channels.
        :param n_channels: The number of channels in this cell.
        :param eta: The base learning rate used in SoftHebb convolutions.
        :param stride: The stride to be used for the operation.
        :param temp: The temperature for the softmax operation (default: 1).
        :param p: The power for the RePU triangle (optional).
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
            return BNConvTriangle(in_channels, n_channels, kernel_size=1, eta=eta, stride=stride, temp=temp, p=p)
        elif op == 'conv_3':
            # 3x3 Hebbian convolution.
            return BNConvTriangle(in_channels, n_channels, kernel_size=3, eta=eta, stride=stride, temp=temp, p=p)
        elif op == 'conv_5':
            # 5x5 Hebbian convolution.
            return BNConvTriangle(in_channels, n_channels, kernel_size=5, eta=eta, stride=stride, temp=temp, p=p)
        elif op == 'conv_13_31':
            # 1x3 and then 3x1 Hebbian convolution.
            conv_13 = BNConvTriangle(in_channels, n_channels, kernel_size=(1, 3), eta=eta, stride=(1, stride),
                                     temp=temp, p=p)
            conv_31 = BNConvTriangle(n_channels, n_channels, kernel_size=(3, 1), eta=eta, stride=(stride, 1), temp=temp,
                                     p=p)
            return Sequential(conv_13, conv_31)
        elif op == 'dilated_conv_5':
            # 5x5 dilated Hebbian convolution (i.e., 3x3 with dilation=2).
            return BNConvTriangle(in_channels, n_channels, kernel_size=3, eta=eta, stride=stride, dilation=2, temp=temp,
                                  p=p)
        else:
            raise ValueError(f"Operation {op} not found (internal error).")


class Classifier(Module):
    """Linear classifier module for Hebbian networks.

    This module flattens the input, applies dropout, and ends with a linear layer. It returns logits.

    :param in_features: The number of input features.
    :param out_features: The number of output features.
    :param dropout: The dropout rate (default: 0.5).
    """

    def __init__(self, in_features: int, out_features: int, dropout=0.5):
        super(Classifier, self).__init__()

        self.flatten = Flatten(start_dim=-3, end_dim=-1)
        self.dropout = Dropout(dropout)
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


class HebbNetA(Module):
    """HebbNet-A."""

    def __init__(self, in_channels: int = 3, config: dict | str = "tuned"):
        super(HebbNetA, self).__init__()

        if config == "default":
            config = {"n_channels": 24, "alpha": 0.001, "dropout": 0.5, "n_epochs": 50,
                      "conv_1": {"eta": 0.01, "tau_inv": 1, "p": None},
                      "conv_2": {"eta": 0.01, "tau_inv": 1, "p": None},
                      "conv_3": {"eta": 0.01, "tau_inv": 1, "p": None}}
        elif config == "tuned":
            config = {'n_channels': 40, 'alpha': 0.0017951869927118615, 'dropout': 0.2331991642665764, 'n_epochs': 33,
                      'conv_1': {'eta': 0.06203186024739623, 'p': 0.6022817150356787, 'tau_inv': 0.457869979649828},
                      'conv_2': {'eta': 0.016207688538775866, 'p': 1.350619074697909, 'tau_inv': 0.9495469692133909},
                      'conv_3': {'eta': 0.006901722666400572, 'p': 1.0948427757638093, 'tau_inv': 0.5031290494137383}}
        self.config = config

        # Initial 5x5 convolution.
        n_channels = int(config["n_channels"])
        eta, tau, p = config["conv_1"]["eta"], 1 / config["conv_1"]["tau_inv"], config["conv_1"]["p"]
        self.initial_conv = BNConvTriangle(in_channels, n_channels, kernel_size=5, eta=eta, temp=tau, p=p)

        # First reduction cell.
        skip_channels = in_channels
        in_channels = n_channels
        out_channels = 4 * n_channels
        self.cell_1 = HebbCellA(skip_channels, in_channels, out_channels, config["conv_1"], config["conv_2"])

        # Second reduction cell.
        skip_channels = n_channels
        in_channels = self.cell_1.out_channels
        out_channels = (4 ** 2) * n_channels
        self.cell_2 = HebbCellA(skip_channels, in_channels, out_channels, config["conv_2"], config["conv_3"],
                                follows_reduction=True)
        self.out_channels = self.cell_2.out_channels

        self.pool = AvgPool2d(kernel_size=2, stride=2)

    @torch.no_grad()
    def forward(self, x: Tensor):
        """Forward pass.

        :param x: The input image.
        :return: The feature encoding.
        """

        x_skip = x

        # Apply initial convolution.
        x = self.initial_conv(x)

        # Run input through the first reduction cell.
        x_next = self.cell_1(x_skip, x)
        x_skip = x
        x = x_next

        # Run input through the second reduction cell.
        x = self.cell_2(x_skip, x)

        # Apply pooling.
        x = self.pool(x)

        return x


class HebbCellA(Module):
    """The evolved reduction cell for HebbNet-A."""

    def __init__(self, skip_channels: int, in_channels: int, n_channels: int, skip_config: dict, config: dict,
                 follows_reduction=False):
        super(HebbCellA, self).__init__()

        if follows_reduction:
            eta, tau, p = skip_config["eta"], 1 / skip_config["tau_inv"], skip_config["p"]
            self.preprocess_skip = BNConvTriangle(skip_channels, n_channels, kernel_size=3, stride=2, eta=eta, temp=tau,
                                                  p=p)
        else:
            eta, tau, p = skip_config["eta"], 1 / skip_config["tau_inv"], skip_config["p"]
            self.preprocess_skip = BNConvTriangle(skip_channels, n_channels, kernel_size=3, eta=eta, temp=tau, p=p)

        self.skip_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.x_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        eta, tau, p = config["eta"], 1 / config["tau_inv"], config["p"]
        self.conv_skip = BNConvTriangle(n_channels, n_channels, kernel_size=1, stride=2, eta=eta, temp=tau, p=p)

        eta, tau, p = config["eta"], 1 / config["tau_inv"], config["p"]
        self.conv_1_add = BNConvTriangle(in_channels, n_channels, kernel_size=1, eta=eta, temp=tau, p=p)

        eta, tau, p = config["eta"], 1 / config["tau_inv"], config["p"]
        self.conv_1_cat = BNConvTriangle(in_channels, n_channels, kernel_size=1, eta=eta, temp=tau, p=p)

        eta, tau, p = config["eta"], 1 / config["tau_inv"], config["p"]
        self.dil_conv_5 = BNConvTriangle(in_channels, n_channels, kernel_size=3, dilation=2, eta=eta, temp=tau, p=p)

        self.out_channels = 4 * n_channels

    @torch.no_grad()
    def forward(self, x_skip: Tensor, x: Tensor):
        """Forward pass.

        :param x_skip: Skip input.
        :param x: Direct input.
        :return: The cell output.
        """

        # Process skip input.
        x_skip = self.preprocess_skip(x_skip)
        skip_pool = self.skip_pool(x_skip)
        x_skip = self.conv_skip(x_skip)

        # Process direct input.
        x = self.x_pool(x)
        x_1_add = self.conv_1_add(x)
        x_1_cat = self.conv_1_cat(x)
        x_dil_5 = self.dil_conv_5(x)

        # Add convolved skip input and pooled-convolved direct input.
        x_add = torch.add(x_skip, x_1_add)

        # Concatenate all unused intermediate tensors.
        return torch.cat([skip_pool, x_add, x_1_cat, x_dil_5], dim=-3)


class BPNetA(Module):
    """The backpropagation version of HebbNet-A."""

    def __init__(self, in_channels: int = 3, out_features: int = 10):
        super(BPNetA, self).__init__()

        # Initial 5x5 convolution.
        n_channels = 40
        self.initial_conv = BNConvReLU(in_channels, n_channels, kernel_size=5)

        # First reduction cell.
        skip_channels = in_channels
        in_channels = n_channels
        out_channels = 4 * n_channels
        self.cell_1 = BPCellA(skip_channels, in_channels, out_channels)

        # Second reduction cell.
        skip_channels = n_channels
        in_channels = self.cell_1.out_channels
        out_channels = (4 ** 2) * n_channels
        self.cell_2 = BPCellA(skip_channels, in_channels, out_channels, follows_reduction=True)
        self.out_channels = self.cell_2.out_channels

        self.pool = AvgPool2d(kernel_size=2, stride=2)
        self.flatten = Flatten(start_dim=-3, end_dim=-1)
        self.dropout = Dropout()
        self.linear = Linear(self.out_channels * (32 // 2 ** 3) ** 2, out_features)

    def forward(self, x: Tensor):
        """Forward pass.

        :param x: The input image.
        :return: The feature encoding.
        """

        x_skip = x

        # Apply initial convolution.
        x = self.initial_conv(x)

        # Run input through the first reduction cell.
        x_next = self.cell_1(x_skip, x)
        x_skip = x
        x = x_next

        # Run input through the second reduction cell.
        x = self.cell_2(x_skip, x)

        # Apply pooling.
        x = self.pool(x)

        # Final classifier.
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x


class BPCellA(Module):
    """The backpropagation variant of the evolved reduction cell for HebbNet-A."""

    def __init__(self, skip_channels: int, in_channels: int, n_channels: int, follows_reduction=False):
        super(BPCellA, self).__init__()

        if follows_reduction:
            self.preprocess_skip = BNConvReLU(skip_channels, n_channels, kernel_size=3, stride=2)
        else:
            self.preprocess_skip = BNConvReLU(skip_channels, n_channels, kernel_size=3)

        self.skip_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.x_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv_skip = BNConvReLU(n_channels, n_channels, kernel_size=1, stride=2)
        self.conv_1_add = BNConvReLU(in_channels, n_channels, kernel_size=1)
        self.conv_1_cat = BNConvReLU(in_channels, n_channels, kernel_size=1)
        self.dil_conv_5 = BNConvReLU(in_channels, n_channels, kernel_size=3, dilation=2)

        self.out_channels = 4 * n_channels

    def forward(self, x_skip: Tensor, x: Tensor):
        """Forward pass.

        :param x_skip: Skip input.
        :param x: Direct input.
        :return: The cell output.
        """

        # Process skip input.
        x_skip = self.preprocess_skip(x_skip)
        skip_pool = self.skip_pool(x_skip)
        x_skip = self.conv_skip(x_skip)

        # Process direct input.
        x = self.x_pool(x)
        x_1_add = self.conv_1_add(x)
        x_1_cat = self.conv_1_cat(x)
        x_dil_5 = self.dil_conv_5(x)

        # Add convolved skip input and pooled-convolved direct input.
        x_add = torch.add(x_skip, x_1_add)

        # Concatenate all unused intermediate tensors.
        return torch.cat([skip_pool, x_add, x_1_cat, x_dil_5], dim=-3)


class SoftHebbNet(Module):
    """The small SoftHebb encoder network for CIFAR-10.

    :param config: Hyperparameter configuration, either 'default' (no tuning), 'original' (original tuning),
        'tuned' (from random search), or a dictionary with custom settings.
    """

    def __init__(self, config: str | dict = "tuned"):
        super(SoftHebbNet, self).__init__()

        if config == "default":
            default_config = {"eta": 0.01, "tau_inv": 1, "p": None}
            config = {"n_channels": 96, "alpha": 0.001, "dropout": 0.5, "n_epochs": 50, "conv_1": default_config,
                      "conv_2": default_config, "conv_3": default_config}
        elif config == "original":
            config = {"n_channels": 96, "alpha": 0.001, "dropout": 0.5, "n_epochs": 50,
                      "conv_1": {"eta": 0.08, "tau_inv": 1, "p": 0.7},
                      "conv_2": {"eta": 0.005, "tau_inv": 0.65, "p": 1.4},
                      "conv_3": {"eta": 0.01, "tau_inv": 0.25, "p": None}}
        elif config == "tuned":
            config = {'n_channels': 104, 'alpha': 0.00011141993913945598, 'dropout': 0.5543794710051737, 'n_epochs': 47,
                      'conv_1': {'eta': 0.07431711472889782, 'p': 0.880224751955424, 'tau_inv': 1.0366747471998239},
                      'conv_2': {'eta': 0.00010274768914408582, 'p': 0.7551712595998065, 'tau_inv': 1.9548608559931184},
                      'conv_3': {'eta': 0.00029684728652530217, 'p': 1.5283171368979298,
                                 'tau_inv': 0.26798560666363835}}

        self.config = config

        c = int(config["n_channels"])
        params = config["conv_1"]
        eta, tau, p = params["eta"], 1 / params["tau_inv"], params["p"]
        self.layer_1 = BNConvTriangle(in_channels=3, out_channels=c, kernel_size=5, eta=eta, temp=tau, p=p)
        self.pool_1 = MaxPool2d(kernel_size=4, stride=2, padding=1)

        params = config["conv_2"]
        eta, tau, p = params["eta"], 1 / params["tau_inv"], params["p"]
        self.layer_2 = BNConvTriangle(in_channels=c, out_channels=4 * c, kernel_size=3, eta=eta, temp=tau, p=p)
        self.pool_2 = MaxPool2d(kernel_size=4, stride=2, padding=1)

        c *= 4
        params = config["conv_3"]
        eta, tau, p = params["eta"], 1 / params["tau_inv"], params["p"]
        self.layer_3 = BNConvTriangle(in_channels=c, out_channels=4 * c, kernel_size=3, eta=eta, temp=tau, p=p)
        self.pool_3 = AvgPool2d(kernel_size=2, stride=2)

        c *= 4
        self.out_channels = c

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
    """The small backpropagation variant of the SoftHebb network for CIFAR-10."""

    def __init__(self, out_features: int = 10):
        super(SoftHebbBPNet, self).__init__()

        c = 104

        self.layer_1 = BNConvReLU(in_channels=3, out_channels=c, kernel_size=5)
        self.pool_1 = MaxPool2d(kernel_size=4, stride=2, padding=1)

        self.layer_2 = BNConvReLU(in_channels=c, out_channels=4 * c, kernel_size=3)
        self.pool_2 = MaxPool2d(kernel_size=4, stride=2, padding=1)

        c *= 4
        self.layer_3 = BNConvReLU(in_channels=c, out_channels=4 * c, kernel_size=3)
        self.pool_3 = AvgPool2d(kernel_size=2, stride=2)

        c *= 4
        self.out_channels = c

        self.flatten = Flatten(start_dim=-3, end_dim=-1)
        self.dropout = Dropout()
        self.linear = Linear(self.out_channels * (32 // 2 ** 3) ** 2, out_features)

    def forward(self, x: Tensor):
        """Forward pass.

        Runs the sample through the network.

        :param x: Tensor of shape (N, 3, 32, 32) or (3, 32, 32).
        :return: Output logits tensor of shape (N, out_features) or (out_features).
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
