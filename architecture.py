"""This module provides architecture and cell objects used in evolutionary architecture search.

Classes
=======
- :class:`OpSet`: Set of operations that can be chosen in evolution.
- :class:`Architecture`: An architecture defined by a normal cell and a reduction cell.
- :class:`Cell`: A network cell defined by pairwise operations.
"""

import pickle
from copy import deepcopy
from enum import StrEnum, auto, unique
from os import makedirs
from os.path import exists, join
from shutil import rmtree

from networkx import MultiDiGraph, get_edge_attributes, set_edge_attributes, topological_generations
from networkx.drawing.nx_agraph import to_agraph
from numpy.random import default_rng


@unique
class OpSet(StrEnum):
    """Set of operations that can be chosen from during evolution."""

    ZERO = auto()  # Set everything to zero (i.e., ignore input).
    IDENTITY = auto()  # No operation (i.e., raw input).
    AVG_POOL_3 = auto()  # 3x3 average pooling.
    MAX_POOL_3 = auto()  # 3x3 max pooling.
    CONV_1 = auto()  # 1x1 Hebbian convolution.
    CONV_3 = auto()  # 3x3 Hebbian convolution.
    CONV_5 = auto()  # 5x5 Hebbian convolution.
    CONV_13_31 = auto()  # 1x3 and then 3x1 Hebbian convolution.
    DILATED_CONV_5 = auto()  # 5x5 dilated Hebbian convolution (i.e., 3x3 with dilation=2).


class Architecture:
    """An architecture defined by a normal cell and a reduction cell.

    Calling the constructor generates a random architecture.

    :ivar identifier: The architecture's identifier.
    :ivar normal: True if the architecture includes a normal cell.
    :ivar normal_cell: The architecture's normal cell (if ``normal`` is True).
    :ivar reduction cell: The architecture's reduction cell.
    :ivar parent: The architecture's parent (identifier).
    """

    def __init__(self, identifier: int, n_ops=4, normal=True):
        """Generate a random architecture.

        :param identifier: The architecture's identifier.
        :param n_ops: The number of pairwise operations in each cell.
        :param normal: True if the architecture should include a normal cell.
        """

        self.identifier = identifier
        self.normal = normal

        # Randomly initialize cells.
        if self.normal:
            self.normal_cell = Cell(n_ops=n_ops)
        self.reduction_cell = Cell(n_ops=n_ops)

        self.parent = None

    def mutate(self, identifier: int):
        """Generate a randomly mutated copy of this architecture.

        With equal probability, this randomly mutates either the normal cell or reduction cell.

        :param identifier: The child's identifier.
        :return: The child architecture.
        """

        rng = default_rng()

        child = deepcopy(self)
        child.identifier = identifier
        child.parent = self.identifier

        if self.normal:
            # Randomly choose which cell to mutate.
            if rng.random() < 0.5:
                # Mutate normal cell.
                child.normal_cell.mutate()
            else:
                # Mutate reduction cell.
                child.reduction_cell.mutate()
        else:
            # Mutate reduction cell.
            child.reduction_cell.mutate()
        return child

    def save(self, path: str):
        """Save the architecture to a specified path.

        Saves the architecture to a folder named after its identifier.

        :param path: The path to save to.
        """

        stem = join(path, str(self.identifier))
        if exists(stem):
            rmtree(stem)
        makedirs(stem)
        location = join(stem, "architecture.pkl")
        pickle.dump(self, open(location, 'wb'))


class Cell(MultiDiGraph):
    """A network cell defined by pairwise operations.

    Calling the constructor generates a random cell.
    """

    def __init__(self, n_ops: int):
        """Generate a random cell.

        :param n_ops: The number of pairwise operations.
        """

        super(Cell, self).__init__()

        rng = default_rng()
        self.n_ops = n_ops

        # Add nodes. Node zero and one refer to the skip and direct input, respectively.
        self.add_nodes_from(range(self.n_ops + 2))

        # Build random cell by sequentially sampling nodes (pairwise operations).
        for node in range(2, self.n_ops + 2):
            # Pick two previously added nodes at random as inputs.
            inputs = rng.choice(node, size=2)

            # Pick a random operation to apply to each input and add as edges.
            ops = rng.choice(list(OpSet), size=2)
            self.add_edge(inputs[0], node, op=ops[0])
            self.add_edge(inputs[1], node, op=ops[1])

    def mutate(self):
        """Randomly mutate the cell.

        With equal probability, this randomly mutates either a hidden state (i.e., change input to a pairwise
        operation without forming a loop) or an operation (i.e., change operation to one in the operation set).
        """

        rng = default_rng()

        if rng.random() < 0.5:
            # Mutate hidden state: Randomly sample a node (pairwise operation).
            node = rng.integers(2, self.n_ops + 2)

            # Randomly sample an input edge (old_input, node, attr) and remove it.
            old_input, _, attr = list(self.in_edges(node, data=True))[rng.integers(2)]
            op = attr['op']
            self.remove_edge(old_input, node)

            # Get generations: A node can sample inputs from its own and previous generations without forming loops.
            candidates = []
            for _, generation in enumerate(topological_generations(self)):
                candidates += generation

                # Remove the old input from the candidates to avoid an exact copy.
                if old_input in generation:
                    candidates.remove(old_input)

                # Stop when we reach the node's generation and remove the node itself from the candidates.
                if node in generation:
                    candidates.remove(node)
                    break

            # Pick a random node from the candidates and add new edge.
            new_input = rng.choice(candidates)
            self.add_edge(new_input, node, op=op)
        else:
            # Mutate operation: Randomly sample an edge (operation).
            edge = rng.choice(list(self.edges))

            # Pick a random operation from the operation set.
            old_op = self.edges[edge]['op']
            candidates = list(OpSet)
            candidates.remove(old_op)
            self.edges[edge]['op'] = rng.choice(candidates)

    def visualize(self, path: str):
        """Visualize the cell.

        :param path: Path to save the visualization to (including filename).
        """

        # Add ops as edge label.
        ops = get_edge_attributes(self, 'op')
        set_edge_attributes(self, ops, name='label')

        graph = to_agraph(self)
        graph.layout(prog='dot')
        graph.draw(path)
