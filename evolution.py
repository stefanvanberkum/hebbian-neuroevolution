"""This module provides the method for evolutionary architecture search.

Supports command-line use.

Functions
=========
- :func:`evolve`: Run evolution.
"""
import pickle
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
from os import makedirs, remove
from os.path import exists, join
from typing import TextIO

import numpy as np
import torch
from numpy.random import default_rng
from tqdm import tqdm

from architecture import Architecture
from dataloader import load
from models import Classifier, HebbianEncoder
from training import test, train


def evolve(dataset='CIFAR10', n_channels=8, scaling_factor=4, n_ops=5, stack_size=0, n_reduction=2, n_epochs=10,
           generations=100, eta=0.01, encoder_batch=32, classifier_batch=256, reduce=False, verbose=False,
           checkpoint=None):
    """Evolve a Hebbian encoder.

    If the stack size is set to zero, a sequence of ``n_reduction`` reduction cells is used. Otherwise, alternating
    stacks of normal cells and single reduction cells are used in the order N-R-N-R-N.

    :param dataset: The dataset to use for evolution (default: CIFAR10). One of: {MNIST, CIFAR10}.
    :param n_channels: The initial number of channels (default: 8).
    :param scaling_factor: The scaling factor for the number of filters after each reduction cell (default: 4).
    :param n_ops: The number of operations in each cell (default: 5).
    :param stack_size: The normal cell stack size (default: 0).
    :param n_reduction: The number of reduction cells, used if ``stack_size`` is set to zero (default: 2).
    :param n_epochs: The epoch increment for training the classifier (default: 10).
    :param generations: The number of generations (default 100).
    :param eta: The base learning rate used in SoftHebb convolutions (default: 0.01).
    :param encoder_batch: The batch size for training the encoder's SoftHebb convolutions (default: 32).
    :param classifier_batch: The batch size for training the classifier with SGD (default: 256).
    :param reduce: True if the spatial size of the input images should be reduced to 16x16 (default: False).
    :param verbose: True if info should be printed (default: False).
    :param checkpoint: Optional checkpoint name to continue evolution.
    """

    if dataset not in {'MNIST', 'CIFAR10'}:
        raise ValueError("Dataset not supported, should be one of {MNIST, CIFAR10}, received:", f"{dataset}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load reduced dataset.
    training, validation, _ = load(dataset, validation=True, reduce=reduce)
    in_channels = 3 if (dataset == 'CIFAR10') else 1
    n_classes = 10

    def log(message: str, console: TextIO, file: TextIO):
        """Write message to console and log file.

        :param message: The message to print.
        :param console: The console to print to.
        :param file: The log file to print to.
        """

        print(message, file=console)
        print(message, file=file)

    # Initialize evolution or load checkpoint.
    if checkpoint is None:
        # Initialize directory.
        time = datetime.now()
        path = f"runs/{time.strftime('%Y-%m-%d-%H:%M:%S')}"
        arch_path = join(path, "architectures")
        ckpt_path = join(path, "checkpoint.pkl")
        log_path = join(path, "log.txt")
        makedirs(arch_path)

        # Open log file.
        log_file = open(log_path, 'w')
        log(f"Initializing evolution run {time.strftime('%Y-%m-%d-%H:%M:%S')}", sys.stdout, log_file)

        # Initialize random number generator.
        rng = default_rng()

        # Initialize subpopulations and accuracy lists.
        P_1 = set()
        P_2 = set()
        P_3 = set()
        P_3_history = set()
        accuracies = []
        max_accuracies = []
        start = 0  # Start generation.
        step = 0  # Current iteration (architecture).
        oldest = 0  # Oldest architecture.
    else:
        # Load checkpoint.
        path = f"runs/{checkpoint}"
        arch_path = join(path, "architectures")
        ckpt_path = join(path, "checkpoint.pkl")
        log_path = join(path, "log.txt")

        # Open log file.
        log_file = open(log_path, 'a')
        log(f"Loading evolutionary run {checkpoint}", sys.stdout, log_file)

        rng, P_1, P_2, P_3, P_3_history, accuracies, max_accuracies, start, step, oldest = pickle.load(
            open(ckpt_path, 'rb'))

    def train_and_evaluate(arch: Architecture, identifier: int):
        """Train, evaluate, and save a model.

        :param arch: The architecture.
        :param identifier: The architecture's identifier.
        """

        # Train and compute validation accuracy.
        encoder = HebbianEncoder(in_channels, arch, n_channels, stack_size, eta, scaling_factor, n_reduction)
        if stack_size == 0:
            classifier = Classifier(encoder.out_channels * (32 // 2 ** 3) ** 2, n_classes)
        else:
            classifier = Classifier(encoder.out_channels, n_classes)

        # Check if a checkpoint exists.
        encoder_path = join(arch_path, str(identifier), "encoder.pt")
        classifier_path = join(arch_path, str(identifier), "classifier.pt")
        state_path = join(arch_path, str(identifier), "training_state.pt")
        if exists(state_path):
            # Load checkpoint.
            encoder.load_state_dict(torch.load(encoder_path))
            classifier.load_state_dict(torch.load(classifier_path))
            training_state = torch.load(state_path)
            update_accuracy = True

            if verbose:
                log(f"[INFO] Loaded checkpoint for architecture {identifier}: epoch {training_state['start_epoch']}/"
                    f"{training_state['max_epochs']}.", sys.stderr, log_file)
        else:
            # Initialize training state.
            training_state = {'start_epoch': 0, 'max_epochs': 3 * n_epochs, 'optimizer_state_dict': {},
                              'scheduler_state_dict': {}, 'save_path': state_path}
            update_accuracy = False

            if verbose:
                log(f"[INFO] Initialized training state for architecture {identifier}.", sys.stderr, log_file)

        model = train(encoder, classifier, training, n_epochs, encoder_batch, classifier_batch,
                      checkpoint=training_state)
        acc = test(model, validation, classifier_batch, device)

        if update_accuracy:
            # Update accuracy.
            accuracies[identifier] = acc
        else:
            # Record new accuracy.
            accuracies.append(acc)

        # Save model.
        torch.save(model.encoder.state_dict(), encoder_path)
        torch.save(model.classifier.state_dict(), classifier_path)

    def sample_and_mutate(population: set, identifier: int):
        """Sample from a population and mutate.

        :param population: A set of architecture identifiers representing the population to sample from.
        :param identifier: The child architecture's identifier.
        :return: The child architecture.
        """

        candidates = rng.choice(list(population), len(population) // 4 + 1)  # Sample ~25% of the set.
        parent_id = sorted(candidates, key=lambda x: accuracies[x])[-1]  # Pick the best out of these candidates.
        parent = pickle.load(open(join(arch_path, str(parent_id), "architecture.pkl"), 'rb'))

        if verbose:
            log(f"[INFO] Mutating architecture {parent_id} with an accuracy of {accuracies[parent_id]:.2f}%.",
                sys.stderr, log_file)

        child = parent.mutate(identifier=identifier)
        child.save(arch_path)
        return child

    if checkpoint is None:
        # Generate, train, and evaluate the initial population of random models.
        log("Generating initial population...", sys.stdout, log_file)
        progress_bar = tqdm(range(60), desc="Architecture", file=sys.stdout)
        for i in progress_bar:
            print(progress_bar, file=log_file)

            # Generate architecture and add to subpopulation one.
            if stack_size == 0:
                # No normal cell.
                architecture = Architecture(identifier=step, n_ops=n_ops, normal=False)
            else:
                architecture = Architecture(identifier=step, n_ops=n_ops)
            architecture.save(arch_path)
            train_and_evaluate(architecture, step)
            P_1.add(step)

            if verbose:
                log(f"[INFO] Architecture {step} obtained an accuracy of {accuracies[step]:.2f}%.", sys.stderr,
                    log_file)
            step += 1
        max_accuracies.append(max(accuracies))
        log("Done!\n", sys.stdout, log_file)

    # Run evolution for the specified number of generations.
    log("Running evolution...", sys.stdout, log_file)
    progress_bar = tqdm(range(start, generations), desc="Generation", file=sys.stdout)
    progress_bar.set_description(f"Best accuracy: {max(accuracies):.2f}%")
    for generation in progress_bar:
        print(progress_bar, file=log_file)

        # Sample offspring from the subpopulations if available (five from each).
        for subpopulation in [P_1, P_2, P_3]:
            if len(subpopulation) > 0:
                for i in range(5):
                    # Sample and mutate.
                    architecture = sample_and_mutate(subpopulation, step)

                    # Train, evaluate, and add to set.
                    train_and_evaluate(architecture, step)
                    P_1.add(step)

                    if verbose:
                        log(f"[INFO] Child architecture {step} obtained an accuracy of {accuracies[step]:.2f}%.",
                            sys.stderr, log_file)
                    step += 1

        # Promote the eight best architectures in P_1 to P_2 and advance their training.
        winners = sorted(P_1, key=lambda x: accuracies[x])[-8:]
        for winner in winners:
            old_acc = accuracies[winner]
            architecture = pickle.load(open(join(arch_path, str(winner), "architecture.pkl"), 'rb'))
            train_and_evaluate(architecture, winner)
            P_1.remove(winner)
            P_2.add(winner)

            if verbose:
                log(f"[INFO] Architecture {winner} was promoted to P_2, accuracy went from {old_acc:.2f}% to "
                    f"{accuracies[winner]:.2f}%.", sys.stderr, log_file)

        # Promote the four best architectures in P_2 to P_3 and advance their training.
        winners = sorted(P_2, key=lambda x: accuracies[x])[-4:]
        for winner in winners:
            old_acc = accuracies[winner]
            architecture = pickle.load(open(join(arch_path, str(winner), "architecture.pkl"), 'rb'))
            train_and_evaluate(architecture, winner)
            P_2.remove(winner)
            P_3.add(winner)
            P_3_history.add(winner)

            if verbose:
                log(f"[INFO] Architecture {winner} was promoted to P_3, accuracy went from {old_acc:.2f}% to "
                    f"{accuracies[winner]:.2f}%.", sys.stderr, log_file)

        # Remove deceased architectures.
        n_new = 5 * ((len(P_1) > 0) + (len(P_2) > 0) + (len(P_3) > 0))  # Five new architectures for each non-empty set.
        for architecture in range(oldest, oldest + n_new):
            for subpopulation in [P_1, P_2, P_3]:
                if architecture in subpopulation:
                    subpopulation.remove(architecture)
                    remove(join(arch_path, str(architecture), "encoder.pt"))
                    remove(join(arch_path, str(architecture), "classifier.pt"))
                    remove(join(arch_path, str(architecture), "training_state.pt"))

        if verbose:
            log(f"[INFO] Deceased architectures: {oldest}--{oldest + n_new}", sys.stderr, log_file)
        oldest += n_new

        # Save checkpoint.
        ckpt = (rng, P_1, P_2, P_3, P_3_history, accuracies, max_accuracies, generation, step, oldest)
        pickle.dump(ckpt, open(ckpt_path, 'wb'))

        # Update progress bar.
        progress_bar.set_description(f"Best accuracy: {max(accuracies):.2f}%")
        max_accuracies.append(max(accuracies))

        if verbose:
            log("[INFO] End of generation.", sys.stderr, log_file)
            log(f"[INFO] P_1: {P_1}.", sys.stderr, log_file)
            log(f"[INFO] P_2: {P_2}.", sys.stderr, log_file)
            log(f"[INFO] P_3: {P_3}.\n", sys.stderr, log_file)

    log("Done!\n", sys.stdout, log_file)

    # Remove all model checkpoints.
    for subpopulation in [P_1, P_2, P_3]:
        for architecture in subpopulation:
            remove(join(arch_path, str(architecture), "encoder.pt"))
            remove(join(arch_path, str(architecture), "classifier.pt"))
            remove(join(arch_path, str(architecture), "training_state.pt"))

    # Save statistics.
    np.save(join(path, "accuracies.npy"), np.array(accuracies))
    np.savetxt(join(path, "accuracies.csv"), np.array(accuracies), delimiter=',')
    np.save(join(path, "max_accuracies.npy"), np.array(max_accuracies))
    np.savetxt(join(path, "max_accuracies.csv"), np.array(max_accuracies), delimiter=',')

    # Record winners.
    winners = sorted(P_3_history, key=lambda x: accuracies[x])[-5:]
    pickle.dump(winners, open(join(path, "winners.pkl"), 'wb'))
    best_arch = None
    best_acc = 0
    with open(join(path, f"accuracies_reduced.csv"), 'w') as out:
        out.write(f"ID,Accuracy\n")
        for winner in winners:
            # Load architecture.
            architecture = pickle.load(open(join(arch_path, str(winner), "architecture.pkl"), 'rb'))

            # Save cell visualizations.
            makedirs(join(path, "winners", str(winner)), exist_ok=True)
            if architecture.normal:
                architecture.normal_cell.visualize(join(path, "winners", str(winner), "normal.png"))
            architecture.reduction_cell.visualize(join(path, "winners", str(winner), "reduction.png"))

            # Record accuracy.
            out.write(f"{winner},{accuracies[winner]}\n")

            if accuracies[winner] > best_acc:
                best_arch = winner
                best_acc = accuracies[winner]

    log("Best architecture:", sys.stdout, log_file)
    log(f"ID: {best_arch}, accuracy: {best_acc:.2f}%", sys.stdout, log_file)
    log_file.close()


if __name__ == '__main__':
    # For command-line use.
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'CIFAR10'],
                        help="The dataset to use for evolution (default: CIFAR10)")
    parser.add_argument('--n_channels', type=int, default=8, help="The initial number of channels (default: 8).")
    parser.add_argument('--scaling_factor', type=int, default=4,
                        help="The scaling factor for the number of filters after each reduction cell (default: 4).")
    parser.add_argument('--n_ops', type=int, default=5, help="The number of operations in each cell (default: 5).")
    parser.add_argument('--stack_size', type=int, default=0, help="The normal cell stack size (default: 0).")
    parser.add_argument('--n_reduction', type=int, default=2,
                        help="The number of reduction cells, used if ``stack_size`` is set to zero (default: 3).")
    parser.add_argument('--n_epochs', type=int, default=10, help="The epoch increment for training the classifier ("
                                                                 "default: 10).")
    parser.add_argument('--generations', type=int, default=100, help="The number of generations (default 100).")
    parser.add_argument('--eta', type=float, default=0.01, help="The base learning rate used in SoftHebb convolutions ("
                                                                "default: 0.01).")
    parser.add_argument('--encoder_batch', type=int, default=32,
                        help="The batch size for training the encoder's SoftHebb convolutions (default: 32).")
    parser.add_argument('--classifier_batch', type=int, default=256,
                        help="The batch size for training the classifier with SGD (default: 256).")
    parser.add_argument('--reduce', action=BooleanOptionalAction, help="Turn on this option to reduce the spatial "
                                                                       "dimension of the input images to 16x16.")
    parser.add_argument('--verbose', action=BooleanOptionalAction, help="Turn on this option for verbose info.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Optional checkpoint name to continue evolution.")
    args = parser.parse_args()

    evolve(args.dataset, args.n_channels, args.scaling_factor, args.n_ops, args.stack_size, args.n_reduction,
           args.n_epochs, args.generations, args.eta, args.encoder_batch, args.classifier_batch, args.reduce,
           args.verbose, args.checkpoint)
