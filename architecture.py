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
    """An architecture defined by a reduction cell and a set of hyperparameters.

    Calling the constructor generates a random architecture.

    :ivar identifier: The architecture's identifier.
    :ivar cell_1: The architecture's first reduction cell.
    :ivar cell_2: The architecture's second reduction cell.
    :ivar params: The set of hyperparameters.
    :ivar parent: The architecture's parent (identifier).
    """

    def __init__(self, identifier: int, n_ops=5):
        """Generate a random architecture.

        :param identifier: The architecture's identifier.
        :param n_ops: The number of pairwise operations in each cell (default: 5).
        """

        self.identifier = identifier

        # TODO: Separately evolve the two cells. Run for twice as long (0.5 * 0.5 chance of mutating a cell, i.e.,
        #  half of what it was in the original). 50-50 between hyperparameters/architecture. 33-33-33 between layers if
        #  hyperparameters, 50-50 between cells if architecture. Still do 8 channels and scale up afterward. Keep
        #  learning rate and dropout to default (perhaps tune afterward).

        # Randomly initialize the cells.
        self.cell_1 = Cell(n_ops=n_ops)
        self.cell_2 = Cell(n_ops=n_ops)

        self.params = {}
        rng = default_rng()

        def random_params():
            """Generate a set of random hyperparameters for a convolution.

            :return: A random set of hyperparameters.
            """

            params = {"eta": max(0.0, rng.normal(0.05, 0.015)),  # Roughly 0-0.1.
                      "tau_inv": max(0.0, rng.normal(0.5, 0.15)),  # Roughly 0-1.
                      "p": max(0.0, rng.normal(1, 0.2))}  # Roughly 0.25-1.75.
            return params

        # Generate a random set of hyperparameters for the initial convolution.
        self.params["initial_conv"] = random_params()

        # Generate a random set of hyperparameters for the first reduction cell.
        ops = get_edge_attributes(self.cell_1, 'op')
        self.params["cell_1"] = {}
        self.params["cell_1"]["skip_conv"] = random_params()
        for op in ops:
            if "conv" in ops[op]:
                # Generate a set of random hyperparameters.
                self.params["cell_1"][op] = random_params()

        # Generate a random set of hyperparameters for the second reduction cell.
        ops = get_edge_attributes(self.cell_2, 'op')
        self.params["cell_2"] = {}
        self.params["cell_2"]["skip_conv"] = random_params()
        for op in ops:
            if "conv" in ops[op]:
                # Generate a set of random hyperparameters.
                self.params["cell_2"][op] = random_params()
        self.parent = None

    def mutate(self, identifier: int):
        """Generate a randomly mutated copy of this architecture.

        With equal probability, this randomly mutates either the architecture or its hyperparameters. If the
        architecture is mutated, one of the two reduction cells is randomly chosen for mutation. If the
        hyperparameters are mutated, the set of hyperparameters is randomly redrawn for a single convolution.

        :param identifier: The child's identifier.
        :return: The child architecture.
        """

        rng = default_rng()

        child = deepcopy(self)
        child.identifier = identifier
        child.parent = self.identifier

        def random_params():
            """Generate a set of random hyperparameters for a convolution.

            :return: A random set of hyperparameters.
            """

            params = {"eta": max(0.0, rng.normal(0.05, 0.015)),  # Roughly 0-0.1.
                      "tau_inv": max(0.0, rng.normal(0.5, 0.15)),  # Roughly 0-1.
                      "p": max(0.0, rng.normal(1, 0.2))}  # Roughly 0.25-1.75.
            return params

        # Randomly choose whether to mutate the architecture or its hyperparameters.
        if rng.random() < 0.5:
            # Mutate the architecture: Pick one of the cells at random.
            if rng.random() < 0.5:
                # Mutate the first reduction cell.
                old_edge, new_edge, old_op, new_op = child.cell_1.mutate()

                if old_op == new_op:
                    # Mutation only changed the edge so hyperparameters may need to be reassigned.
                    if "conv" in new_op:
                        child.params["cell_1"][new_edge] = child.params["cell_1"][old_edge]
                        del child.params["cell_1"][old_edge]
                else:
                    # Mutation changed the operation so new hyperparameters may need to be drawn.
                    if "conv" in old_op:
                        # Remove old hyperparameters.
                        del child.params["cell_1"][old_edge]

                    if "conv" in new_op:
                        # Draw new hyperparameters.
                        child.params["cell_1"][new_edge] = random_params()
            else:
                # Mutate the second reduction cell.
                old_edge, new_edge, old_op, new_op = child.cell_2.mutate()

                if old_op == new_op:
                    # Mutation only changed the edge so hyperparameters may need to be reassigned.
                    if "conv" in new_op:
                        child.params["cell_2"][new_edge] = child.params["cell_2"][old_edge]
                        del child.params["cell_2"][old_edge]
                else:
                    # Mutation changed the operation so new hyperparameters may need to be drawn.
                    if "conv" in old_op:
                        # Remove old hyperparameters.
                        del child.params["cell_2"][old_edge]

                    if "conv" in new_op:
                        # Draw new hyperparameters.
                        child.params["cell_2"][new_edge] = random_params()
        else:
            # Mutate the hyperparameters: Pick one of the convolutions at random.
            cell_1_convs = list(child.params["cell_1"].keys())
            cell_2_convs = list(child.params["cell_2"].keys())
            conv = rng.choice(1 + len(cell_1_convs) + len(cell_2_convs))
            if conv == 0:
                # Redraw the hyperaparameters for the initial convolution.
                child.params["initial_conv"] = random_params()
            elif conv < 1 + len(cell_1_convs):
                # Redraw the hyperaparameters for the chosen convolution from the first reduction cell.
                child.params["cell_1"][cell_1_convs[conv - 1]] = random_params()
            else:
                # Redraw the hyperaparameters for the chosen convolution from the second reduction cell.
                child.params["cell_2"][cell_2_convs[conv - 1 - len(cell_1_convs)]] = random_params()
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

        :return: A tuple comprising (old_edge, new_edge, old_op, new_op).
        """

        rng = default_rng()

        if rng.random() < 0.5:
            # Mutate hidden state: Randomly sample a node (pairwise operation).
            node = rng.integers(2, self.n_ops + 2)

            # Randomly sample an input edge (old_input, node, key, attr) and remove it.
            old_input, _, key, attr = list(self.in_edges(node, keys=True, data=True))[rng.integers(2)]
            op = attr['op']
            self.remove_edge(old_input, node, key)

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
            new_key = self.add_edge(new_input, node, op=op)

            old_edge = (old_input, node, key)
            new_edge = (new_input, node, new_key)
            old_op = op
            new_op = op
        else:
            # Mutate operation: Randomly sample an edge (operation).
            edge = rng.choice(list(self.edges(keys=True)))

            # Pick a random operation from the operation set.
            old_op = self.edges[edge]['op']
            candidates = list(OpSet)
            candidates.remove(old_op)
            new_op = rng.choice(candidates)
            self.edges[edge]['op'] = new_op

            old_edge = tuple(edge)
            new_edge = tuple(edge)
        return old_edge, new_edge, old_op, new_op

    def visualize(self, path: str):
        """Visualize the cell.

        :param path: Path to save the visualization to (including a filename).
        """

        # Add ops as edge label.
        ops = get_edge_attributes(self, 'op')
        set_edge_attributes(self, ops, name='label')

        graph = to_agraph(self)
        graph.layout(prog='dot')
        graph.draw(path + ".png")

        # Add edge index as edge label.
        indices = {key: key for key in ops}
        set_edge_attributes(self, indices, name='label')

        graph = to_agraph(self)
        graph.layout(prog='dot')
        graph.draw(path + "_map.png")

        # Save a list of edges.
        with open(path + "_edges.csv", "w") as out:
            out.write("u,v,key,op\n")
            for edge in ops:
                u, v, key = edge
                op = ops[edge]
                out.write(f"{u},{v},{key},{op}\n")
