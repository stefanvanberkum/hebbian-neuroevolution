"""This module provides classes and methods for evolutionary architecture search.

Classes
=======
- :class:`OpSet`: Set of operations that can be chosen in evolution.
- :class:`Architecture`: An architecture instance.
- :class:`Cell`: A cell used in a modular architecture.
"""

from enum import StrEnum, auto, unique

from networkx import DiGraph
from numpy.random import choice


@unique
class OpSet(StrEnum):
    """Set of operations that can be chosen from during evolution."""

    ZERO = auto()  # Set everything to zero (i.e., ignore input).
    IDENTITY = auto()  # No operation (i.e., raw input).
    AVG_POOL_3 = auto()  # 3x3 average pooling.
    MAX_POOL_3 = auto()  # 3x3 max pooling.
    CONV_1 = auto()  # 1x1 Hebbian convolution.
    CONV_3 = auto()  # 3x3 Hebbian convolution.
    CONV_13_31 = auto()  # 1x3 and then 3x1 Hebbian convolution.
    DILATED_CONV_5 = auto()  # 5x5 dilated Hebbian convolution (i.e., 3x3 with dilation=2).


class Architecture:
    def __init__(self, n_ops=4):
        normal_cell = Cell(n_ops=n_ops)
        reduction_cell = Cell(n_ops=n_ops)


class Cell(DiGraph):
    def __init__(self, n_ops):
        super(Cell, self).__init__()

        # Add node for skip input (0) and direct input (1).
        self.add_nodes_from([0, 1])

        # Build random cell by sequentially sampling nodes (pairwise operations).
        for node in range(2, n_ops + 2):
            # Pick two previously added nodes at random as inputs.
            inputs = choice(node, size=2, replace=False)

            # Pick a random operation to apply to each input and add as edges.
            ops = choice(list(OpSet), size=2)
            self.add_edge(inputs[0], node, op=ops[0])
            self.add_edge(inputs[1], node, op=ops[1])
