"""This module provides a PyTorch implementation of the SoftHebb convolution.

The methods in this module are based the code by [1]_.

Functions
=========
- :func:`train_models`: Train the models in accordance with our training scheme.
- :func:`train_model`: Train a model.
- :func:`pretrain`: Pretrain a model.
- :func:`process`: Run the input data through a model, training it if an optimizer is provided.

References
==========
.. [1] Journé, A., Garcia Rodriguez, H., Guo, Q., & Moraitis, T. (2023). Hebbian deep learning without feedback.
    *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=8gd4M-_Rj1
"""

from torch.nn import Module


class HebbConv2d(Module):
    def __init__(self):
        super(HebbConv2d, self).__init__()
