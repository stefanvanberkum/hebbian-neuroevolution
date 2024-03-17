"""This module provides a LayerCAM implementation based on pytorch-grad-cam:
https://github.com/jacobgil/pytorch-grad-cam.

Classes
=======
- :class:`LayerCAM`: LayerCAM implementation.
- :class:`ActivationAndGradient`: Class for extracting activations and registering gradients.
"""

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt


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

    def forward(self, input_tensor: torch.Tensor, target: torch.nn.Module):
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
        cam = cam / (1e-7 + np.max(cam))
        cam = Image.fromarray(cam).resize(target_size)
        cam = np.asarray(cam)
        cam = np.float32(cam)[:, None, :]

        return cam

    def visualize(self, input_tensor: torch.Tensor, target: torch.nn.Module, path: str):
        """Visualize the LayerCAM by overlaying it as a heatmap on top of the original image.

        :param input_tensor: The input image as tensor.
        :param target: The target layer.
        :param path: The path including file name to save to.
        """

        grayscale_cam = self.forward(input_tensor, target)[0, :]
        heatmap = np.uint8(255 * grayscale_cam)
        colormap = plt.colormaps["coolwarm"]
        colormap = colormap(np.arange(256))[:, :3]
        heatmap = colormap[heatmap]
        visualization = 0.5 * input_tensor + 0.5 * heatmap
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
