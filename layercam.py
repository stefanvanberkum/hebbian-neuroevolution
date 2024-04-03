"""This module provides a LayerCAM implementation based on pytorch-grad-cam:
https://github.com/jacobgil/pytorch-grad-cam.

Classes
=======
- :class:`ClassifierOutputTarget`: Class for storing the target output and getting the corresponding output.
- :class:`LayerCAM`: LayerCAM implementation.
- :class:`ActivationAndGradient`: Class for extracting activations and registering gradients.
- :class:`HebbNetGrad`: Hebbian encoder version that allows for gradient computation.
- :class:`HebbNetAGrad`: HebbNet-A version that allows for gradient computation.
- :class:`HebbCellAGrad`: HebbNet-A cell version that allows for gradient computation.
- :class:`SoftHebbNetGrad`: SoftHebb network version that allows for gradient computation.
- :class:`BNConvTriangleGrad`: BNConvTriangle version that allows for gradient computation.
- :class:`HebbConv2dGrad`: HebbConv2d version that allows for gradient computation.
"""
import math

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import AvgPool2d, BatchNorm2d, MaxPool2d, Module, ReflectionPad2d
from torch.nn.functional import conv2d

from activations import RePUTriangle, ScaledSoftmax2d, Triangle
from models import HebbNet


class ClassifierOutputTarget:
    """Class for storing the target output and getting the corresponding output."""

    def __init__(self, target):
        self.target = target

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.target]
        return model_output[:, self.target]


class LayerCAM:
    """LayerCAM implementation.

    :param model: The model.
    :param target_layer: The layer to target in the model.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.device = next(self.model.parameters()).device

        self.activation_and_grad = ActivationAndGradient(self.model, target_layer)
        self.output = None

    def forward(self, input_tensor: torch.Tensor, target: ClassifierOutputTarget):
        """Compute the LayerCAM.

        :param input_tensor: The input image as tensor.
        :param target: The target layer.
        :return: The CAM.
        """

        input_tensor = input_tensor.to(self.device)

        self.output = output = self.activation_and_grad(input_tensor)

        self.model.zero_grad()
        loss = target(output)
        loss.backward(retain_graph=True)

        activation = self.activation_and_grad.activation.cpu().data.numpy()
        grad = self.activation_and_grad.gradient.cpu().data.numpy()
        target_size = (input_tensor.size(-1), input_tensor.size(-2))

        # Compute LayerCAM.
        spatial_weighted_activations = np.maximum(grad, 0) * activation
        cam = spatial_weighted_activations.sum(axis=1)
        cam = np.maximum(cam, 0)

        # Scale CAM.
        cam = cam - np.min(cam)
        cam = cam if np.max(cam) == 0 else cam / np.max(cam)
        cam = Image.fromarray(cam.squeeze()).resize(target_size)
        cam = np.asarray(cam)
        cam = cam - np.min(cam)
        cam = cam if np.max(cam) == 0 else cam / np.max(cam)
        return cam

    def visualize(self, input_tensor: torch.Tensor, label: int, path: str):
        """Visualize the LayerCAM by overlaying it as a heatmap on top of the original image.

        :param input_tensor: The input image as tensor.
        :param label: The image label.
        :param path: The path including file name to save to.
        """

        target = ClassifierOutputTarget(label)
        grayscale_cam = self.forward(input_tensor, target)
        heatmap = np.uint8(255 * grayscale_cam)
        colormap = plt.colormaps["jet"]
        colormap = colormap(np.arange(256))[:, :3]
        heatmap = colormap[heatmap]
        img = input_tensor.squeeze().movedim(0, 2).detach().cpu().numpy() / 255
        visualization = 0.5 * img + 0.5 * heatmap
        visualization /= np.max(visualization)
        plt.imsave(path, visualization, dpi=300)

    def __del__(self):
        self.activation_and_grad.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activation_and_grad.release()
        if isinstance(exc_value, IndexError):
            print(f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


class ActivationAndGradient:
    """Class for extracting activations and registering gradients from a targeted intermediate layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradient = None
        self.activation = None
        self.handles = []
        self.handles.append(target_layer.register_forward_hook(self.save_activation))
        self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        """Activation hook."""
        self.activation = output.cpu().detach()

    def save_gradient(self, module, input, output):
        """Gradient hook."""
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # Can only register hooks on tensors that require grad.
            return

        def _store_grad(grad):
            self.gradient = grad.cpu().detach()

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradient = None
        self.activation = None
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class HebbNetGrad(HebbNet):
    """Hebbian network version that allows for gradient computation."""

    def __init__(self, encoder: Module, classifier: Module):
        super(HebbNetGrad, self).__init__(encoder, classifier)

    def forward(self, x: Tensor):
        """Forward pass.

        Runs the sample through the encoder and classifier in inference mode.

        :param x: Tensor of shape (N, C_in, H, W) or (C_in, H, W).
        :return: Output logits tensor of size N.
        """

        x = self.encoder(x)
        x = self.classifier(x)
        return x


class HebbNetAGrad(Module):
    """HebbNet-A version that allows for gradient computation."""

    def __init__(self, in_channels: int = 3, config: dict | str = "tuned"):
        super(HebbNetAGrad, self).__init__()

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
        self.initial_conv = BNConvTriangleGrad(in_channels, n_channels, kernel_size=5, eta=eta, temp=tau, p=p)

        # First reduction cell.
        skip_channels = in_channels
        in_channels = n_channels
        out_channels = 4 * n_channels
        self.cell_1 = HebbCellAGrad(skip_channels, in_channels, out_channels, config["conv_1"], config["conv_2"])

        # Second reduction cell.
        skip_channels = n_channels
        in_channels = self.cell_1.out_channels
        out_channels = (4 ** 2) * n_channels
        self.cell_2 = HebbCellAGrad(skip_channels, in_channels, out_channels, config["conv_2"], config["conv_3"],
                                    follows_reduction=True)
        self.out_channels = self.cell_2.out_channels

        self.pool = AvgPool2d(kernel_size=2, stride=2)

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


class HebbCellAGrad(Module):
    """HebbNet-A cell version that allows for gradient computation."""

    def __init__(self, skip_channels: int, in_channels: int, n_channels: int, skip_config: dict, config: dict,
                 follows_reduction=False):
        super(HebbCellAGrad, self).__init__()

        if follows_reduction:
            eta, tau, p = skip_config["eta"], 1 / skip_config["tau_inv"], skip_config["p"]
            self.preprocess_skip = BNConvTriangleGrad(skip_channels, n_channels, kernel_size=3, stride=2, eta=eta,
                                                      temp=tau, p=p)
        else:
            eta, tau, p = skip_config["eta"], 1 / skip_config["tau_inv"], skip_config["p"]
            self.preprocess_skip = BNConvTriangleGrad(skip_channels, n_channels, kernel_size=3, eta=eta, temp=tau, p=p)

        self.skip_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.x_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        eta, tau, p = config["eta"], 1 / config["tau_inv"], config["p"]
        self.conv_skip = BNConvTriangleGrad(n_channels, n_channels, kernel_size=1, stride=2, eta=eta, temp=tau, p=p)

        eta, tau, p = config["eta"], 1 / config["tau_inv"], config["p"]
        self.conv_1_add = BNConvTriangleGrad(in_channels, n_channels, kernel_size=1, eta=eta, temp=tau, p=p)

        eta, tau, p = config["eta"], 1 / config["tau_inv"], config["p"]
        self.conv_1_cat = BNConvTriangleGrad(in_channels, n_channels, kernel_size=1, eta=eta, temp=tau, p=p)

        eta, tau, p = config["eta"], 1 / config["tau_inv"], config["p"]
        self.dil_conv_5 = BNConvTriangleGrad(in_channels, n_channels, kernel_size=3, dilation=2, eta=eta, temp=tau, p=p)

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


class SoftHebbNetGrad(Module):
    """SoftHebb network version that allows for gradient computation.

    :param config: Hyperparameter configuration, either 'default' (no tuning), 'original' (original tuning),
        'tuned' (from random search), or a dictionary with custom settings.
    """

    def __init__(self, config: str | dict = "tuned"):
        super(SoftHebbNetGrad, self).__init__()

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
        self.layer_1 = BNConvTriangleGrad(in_channels=3, out_channels=c, kernel_size=5, eta=eta, temp=tau, p=p)
        self.pool_1 = MaxPool2d(kernel_size=4, stride=2, padding=1)

        params = config["conv_2"]
        eta, tau, p = params["eta"], 1 / params["tau_inv"], params["p"]
        self.layer_2 = BNConvTriangleGrad(in_channels=c, out_channels=4 * c, kernel_size=3, eta=eta, temp=tau, p=p)
        self.pool_2 = MaxPool2d(kernel_size=4, stride=2, padding=1)

        c *= 4
        params = config["conv_3"]
        eta, tau, p = params["eta"], 1 / params["tau_inv"], params["p"]
        self.layer_3 = BNConvTriangleGrad(in_channels=c, out_channels=4 * c, kernel_size=3, eta=eta, temp=tau, p=p)
        self.pool_3 = AvgPool2d(kernel_size=2, stride=2)

        c *= 4
        self.out_channels = c

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


class BNConvTriangleGrad(Module):
    """BNConvTriangle version that allows for gradient computation.

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
        super(BNConvTriangleGrad, self).__init__()

        self.bn = BatchNorm2d(num_features=in_channels, affine=False)
        self.conv = HebbConv2dGrad(in_channels, out_channels, kernel_size, eta, stride, dilation, temp)

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


class HebbConv2dGrad(Module):
    """Hebbian convolution version that allows for gradient computation.

    Applies reflective padding to ensure that the input shape equals the output shape. This method is based on the
    paper and corresponding code by [1]_. It is only tested for operations in the OpSet defined in architecture.py.

    References
    ==========
    .. [1] Journé, A., Garcia Rodriguez, H., Guo, Q., & Moraitis, T. (2023). Hebbian deep learning without feedback.
        *International Conference on Learning Representations (ICLR)*.

    :param in_channels: The number of input channels.
    :param out_channels: The number of output channels.
    :param kernel_size: The kernel size (must be odd).
    :param eta: The base learning rate.
    :param stride: The stride for convolution, can be one or two in each direction (default: 1).
    :param dilation: The dilation for convolution (default: 1).
    :param temp: The temperature for the softmax operation (default: 1).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], eta: float,
                 stride: int | tuple[int, int] = 1, dilation=1, temp=1.0):
        super(HebbConv2dGrad, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.eta = eta
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation

        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError(f"Kernel size must be odd, received {self.kernel_size}.")
        if self.stride[0] > 2 or self.stride[1] > 2:
            raise ValueError(f"Stride cannot be larger than two, received {self.stride}")

        # Initialize softmax.
        self.softmax = ScaledSoftmax2d(temp)

        # Compute padding.
        effective_height = self.kernel_size[0] + (self.kernel_size[0] - 1) * (self.dilation - 1)
        vertical_pad = (effective_height - 1) // 2
        effective_width = self.kernel_size[1] + (self.kernel_size[1] - 1) * (self.dilation - 1)
        horizontal_pad = (effective_width - 1) // 2

        # Take into account that stride of two will end up one cell before the edge of the input.
        if self.stride[0] == 2:
            bottom_pad = max(0, vertical_pad - 1)
        else:
            bottom_pad = vertical_pad
        if self.stride[1] == 2:
            right_pad = max(0, horizontal_pad - 1)
        else:
            right_pad = horizontal_pad
        self.pad = ReflectionPad2d((horizontal_pad, right_pad, vertical_pad, bottom_pad))

        # Initialize and register weights.
        self.register_buffer('weight', self._initialize())

        # Register weight norm and retrieve it.
        self.register_buffer("weight_norm", torch.ones(self.out_channels), persistent=False)
        self._get_norm(update=True)

        # Initialize and register adaptive learning rate.
        self.register_buffer("lr", torch.ones(self.out_channels), persistent=False)

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
        # second patch (moved one to the right in x and to activation y_1112). The second element (1, 1, 1, 2) gives the
        # average yx for neuron 1 and input channel 1 at kernel position (1, 2). That is, it gives
        #   x_1112 * y_1111 + x_1113 * y_1112 + ...,
        # which we can see as yx for neuron 1 at kernel position (1, 2) for the first patch, plus that of the
        # second patch (moved one to the right in x and to activation y_1112). Transposed, this gives a patch-summed
        # tensor yx for each neuron of shape (C_in, kernel_height, kernel_width).
        #   With dilation, each y-element is generated using dilated x-elements, so y should be applied to x with a
        # stride. With a stride s, each kernel position only sees every s'th x-element, so y should be applied to x
        # with a dilation.
        if self.kernel_size[0] == 1 and self.kernel_size[1] == 1 and self.stride[0] == 2 and self.stride[1] == 2:
            # 1x1 kernel with stride 2 requires a different approach: Instead of dilation we need to reduce x.
            yx = conv2d(x[:, :, ::2, ::2].transpose(0, 1), y.transpose(0, 1)).transpose(0, 1)
        else:
            yx = conv2d(x.transpose(0, 1), y.transpose(0, 1), stride=self.dilation, dilation=self.stride).transpose(0,
                                                                                                                    1)

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
