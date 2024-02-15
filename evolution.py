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

import torch
from networkx import DiGraph, topological_generations
from numpy.random import Generator, default_rng

from dataloader import load
from models import Classifier, HebbianEncoder
from training import test, train


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


def evolve(n_channels=8, stack_size=6, epochs=10, generations=100, eta=0.01, encoder_batch=16, classifier_batch=128):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load reduced CIFAR-10.
    training, validation, _ = load('CIFAR10', validation=True, reduce=True)
    in_channels = 3
    n_classes = 10

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

    def train_and_evaluate(arch: Architecture, identifier: int, scheduler_interval: tuple[int, int] = None):
        """Train, evaluate, and save a model.

        :param arch: The architecture.
        :param identifier: The architecture's identifier.
        :param scheduler_interval: Interval ``(t_min, t_max)`` for the learning rate scheduler with ``t_min >= 1`` and
        both bounds inclusive (optional).
        """

        encoder_path = join(path, str(identifier), "encoder.pt")
        classifier_path = join(path, str(identifier), "classifier.pt")
        if scheduler_interval is None:
            # Model is trained from scratch.
            load_model = False
        else:
            load_model = True

        # Train and compute validation accuracy.
        encoder = HebbianEncoder(in_channels, arch, n_channels, stack_size, eta)
        classifier = Classifier(4 * n_channels, n_classes)

        if load_model:
            # Load training checkpoint.
            encoder.load_state_dict(torch.load(encoder_path))
            classifier.load_state_dict(torch.load(classifier_path))

        model = train(encoder, classifier, training, epochs, encoder_batch, classifier_batch,
                      scheduler_interval=scheduler_interval)
        acc = test(model, validation, classifier_batch, device)

        if load_model:
            # Update accuracy.
            accuracies[identifier] = acc
        else:
            # Record new accuracy.
            accuracies.append(acc)

        # Save model.
        torch.save(model.encoder.state_dict(), encoder_path)
        torch.save(model.classifier.state_dict(), classifier_path)

    def sample_and_mutate(population: set, identifier: int):
        """Sample from a pupulation and mutate.

        :param population: A set of architecture identifiers representing the population to sample from.
        :param identifier: The child architecture's identifier.
        :return: The child architecture.
        """

        candidates = rng.choice(list(population), len(population) // 4 + 1)  # Sample ~25% of the set.
        parent_id = sorted(candidates, key=lambda x: accuracies[x])[-1]  # Pick the best out of these candidates.
        parent = pickle.load(open(join(path, str(parent_id), "architecture.pkl"), 'rb'))

        child = parent.mutate(identifier=identifier)
        child.save(path)
        return child

    # Generate, train, and evaluate the initial population of random models.
    step = 0
    for i in range(50):
        # Generate architecture and add to subpopulation one.
        architecture = Architecture(identifier=step, rng=rng)
        architecture.save(path)
        train_and_evaluate(architecture, step)
        P_1.add(step)
        step += 1

    # Run evolution for the specified number of generations.
    for generation in range(generations):
        # Sample offspring from the subpopulations if available (five from each).
        for subpopulation in {P_1, P_2, P_3}:
            for i in range(5):
                # Sample and mutate.
                architecture = sample_and_mutate(subpopulation, step)

                # Train, evaluate, and add to set.
                train_and_evaluate(architecture, step)
                P_1.add(step)
                step += 1

        # Promote the eight best architectures in P_1 to P_2 and advance their training.
        winners = sorted(P_1, key=lambda x: accuracies[x])[-8:]
        for winner in winners:
            architecture = pickle.load(open(join(path, str(winner), "architecture.pkl"), 'rb'))
            train_and_evaluate(architecture, winner, scheduler_interval=(epochs + 1, 2 * epochs))
            P_1.remove(winner)
            P_2.add(winner)

        # Promote the four best architectures in P_2 to P_3 and advance their training.
        winners = sorted(P_2, key=lambda x: accuracies[x])[-4:]
        for winner in winners:
            architecture = pickle.load(open(join(path, str(winner), "architecture.pkl"), 'rb'))
            train_and_evaluate(architecture, winner, scheduler_interval=(2 * epochs + 1, 3 * epochs))
            P_2.remove(winner)
            P_3.add(winner)

        # Bury deceased architectures.
