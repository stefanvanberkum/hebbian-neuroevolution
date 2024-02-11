"""This module provides classes and methods for evolutionary architecture search.

Classes
=======
- :class:`OpSet`: Set of operations that can be chosen in evolution.
- :class:`Architecture`: An architecture defined by a normal cell and a reduction cell.
- :class:`Cell`: A network cell defined by pairwise operations.
"""
import pickle
from copy import deepcopy
from datetime import datetime
from enum import StrEnum, auto, unique
from os import makedirs
from os.path import join

from networkx import DiGraph, topological_generations
from numpy.random import Generator, default_rng
from torch.utils.data import Dataset

from dataloader import load


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
    """An architecture defined by a normal cell and a reduction cell.

    Calling the constructor generates a random architecture.

    :ivar identifier: The architecture's identifier.
    :ivar rng: The random number generator.
    :ivar normal_cell: The architecture's normal cell.
    :ivar reduction cell: The architecture's reduction cell.
    :ivar parent: The architecture's parent (identifier).
    """

    def __init__(self, identifier: int, rng: Generator, n_ops=4):
        """Generate a random architecture.

        :param identifier: The architecture's identifier.
        :param rng: The random number generator.
        :param n_ops: The number of pairwise operations in each cell.
        """

        self.identifier = identifier
        self.rng = rng

        # Randomly initialize cells.
        self.normal_cell = Cell(rng, n_ops=n_ops)
        self.reduction_cell = Cell(rng, n_ops=n_ops)

        self.parent = None

    def mutate(self, identifier: int):
        """Generate a randomly mutated copy of this architecture.

        With equal probability, this randomly mutates either the normal cell or reduction cell.

        :param identifier: The child's identifier.
        :return: The child architecture.
        """

        child = deepcopy(self)
        child.identifier = identifier
        child.parent = self.identifier

        if self.rng.random() < 0.5:
            # Mutate normal cell.
            child.normal_cell.mutate()
        else:
            # Mutate reduction cell.
            child.reduction_cell.mutate()
        return child

    def save(self, path: str):
        """Save the architecture to a specified path.

        Saves the architecture to a folder named after its identifier.

        :param path: The path to save to.
        """

        location = join(path, str(self.identifier), "architecture.pkl")
        makedirs(path)
        pickle.dump(self, open(location, 'wb'))


class Cell(DiGraph):
    """A network cell defined by pairwise operations.

    Calling the constructor generates a random cell.
    """

    def __init__(self, rng: Generator, n_ops: int):
        """Generate a random cell.

        :param rng: A random generator.
        :param n_ops: The number of pairwise operations.
        """

        super(Cell, self).__init__()

        self.rng = rng
        self.n_ops = n_ops

        # Add nodes. Node zero and one refer to the skip and direct input, respectively.
        self.add_nodes_from(range(self.n_ops + 2))

        # Build random cell by sequentially sampling nodes (pairwise operations).
        for node in range(2, self.n_ops + 2):
            # Pick two previously added nodes at random as inputs.
            inputs = self.rng.choice(node, size=2, replace=False)

            # Pick a random operation to apply to each input and add as edges.
            ops = self.rng.choice(list(OpSet), size=2)
            self.add_edge(inputs[0], node, op=ops[0])
            self.add_edge(inputs[1], node, op=ops[1])

    def mutate(self):
        """Randomly mutate the cell.

        With equal probability, this randomly mutates either a hidden state (i.e., change input to a pairwise
        operation without forming a loop) or an operation (i.e., change operation to one in the operation set).
        """

        if self.rng.random() < 0.5:
            # Mutate hidden state: Randomly sample a node (pairwise operation).
            node = self.rng.integers(2, self.n_ops + 2)

            # Randomly sample an input edge (u, v, attr) and remove it.
            old_input, _, attr = list(self.in_edges(node, data=True))[self.rng.integers(2)]
            op = attr['op']
            self.remove_edge(old_input, node)

            # Get generations: A node can sample inputs from its own and previous generations without forming loops.
            candidates = []
            for _, generation in enumerate(topological_generations(self)):
                candidates += generation

                # Stop when we reach the node's generation.
                if node in generation:
                    break

            # Pick a random node from the candidates and add new edge.
            new_input = self.rng.choice(candidates)
            self.add_edge(new_input, node, op=op)
        else:
            # Mutate operation: Randomly sample an edge (operation).
            edge = self.rng.choice(list(self.edges))

            # Pick a random operation from the operation set.
            self.edges[edge]['op'] = self.rng.choice(list(OpSet))


def evolve(train: Dataset, val: Dataset, n_channels=8, stack_size=6, epochs=10, generations=100):
    # Load reduced CIFAR-10.
    train, val, _ = load('CIFAR10', validation=True, reduce=True)

    # Initialize subpopulations and accuracy list.
    P_1 = set()
    P_2 = set()
    P_3 = set()
    accuracies = []

    # Initialize random number generator.
    rng = default_rng()

    # Initialize directory.
    time = datetime.now()
    path = f"runs/{time.strftime('%Y-%m-%d-%H:%M:%S')}"
    makedirs(path)

    # Generate, train, and evaluate the initial population of random models.
    step = 0
    for i in range(50):
        # Generate architecture and add to subpopulation one.
        architecture = Architecture(step, rng)
        architecture.save(path)
        P_1.add(step)

        # Train and record accuracy.

        accuracies[step] = 0
        step += 1
