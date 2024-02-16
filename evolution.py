"""This module provides the method for evolutionary architecture search.

Supports command-line use.

Functions
=========
- :func:`evolve`: Run evolution.
"""

import pickle
from argparse import ArgumentParser
from datetime import datetime
from os import makedirs, remove
from os.path import join

import numpy as np
import torch
from numpy.random import default_rng
from tqdm import tqdm

from architecture import Architecture
from dataloader import load
from models import Classifier, HebbianEncoder
from training import test, train


def evolve(dataset='CIFAR10', n_channels=8, stack_size=6, epochs=10, generations=100, eta=0.01, encoder_batch=16,
           classifier_batch=128, checkpoint=None):
    """Evolve a Hebbian encoder.

    :param dataset: The dataset to use for evolution (default: CIFAR10). One of: {MNIST, CIFAR10}.
    :param n_channels: The initial number of channels (default: 8).
    :param stack_size: The normal cell stack size (default: 6).
    :param epochs: The epoch increment for training the classifier (default: 10).
    :param generations: The number of generations (default 100).
    :param eta: The base learning rate used in SoftHebb convolutions (default: 0.01).
    :param encoder_batch: The batch size for training the encoder's SoftHebb convolutions (default: 16).
    :param classifier_batch: The batch size for training the classifier with SGD (default: 128).
    :param checkpoint: Optional checkpoint name to continue evolution.
    """

    if dataset not in {'MNIST', 'CIFAR10'}:
        raise ValueError("Dataset not supported, should be one of {MNIST, CIFAR10}, received:", f"{dataset}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load reduced dataset.
    training, validation, _ = load(dataset, validation=True)
    in_channels = 3 if (dataset == 'CIFAR10') else 1
    n_classes = 10

    # Initialize evolution or load checkpoint.
    if checkpoint is None:
        # Initialize directory.
        time = datetime.now()
        path = f"runs/{time.strftime('%Y-%m-%d-%H:%M:%S')}"
        ckpt_path = join(path, "checkpoint.pkl")
        makedirs(path)

        print(f"Initializing evolution run {time.strftime('%Y-%m-%d-%H:%M:%S')}")

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
        print(f"Loading evolutionary run {checkpoint}")

        path = f"runs/{checkpoint}"
        ckpt_path = join(path, "checkpoint.pkl")
        rng, P_1, P_2, P_3, P_3_history, accuracies, max_accuracies, start, step, oldest = pickle.load(
            open(ckpt_path, 'rb'))

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
        """Sample from a population and mutate.

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

    if checkpoint is None:
        # Generate, train, and evaluate the initial population of random models.
        print("Generating initial population...")
        for i in tqdm(range(50), desc="Architecture"):
            # Generate architecture and add to subpopulation one.
            architecture = Architecture(identifier=step, rng=rng)
            architecture.save(path)
            train_and_evaluate(architecture, step)
            P_1.add(step)
            step += 1
        max_accuracies.append(max(accuracies))
        print("Done!\n")

    # Run evolution for the specified number of generations.
    print("Running evolution...")
    progress_bar = tqdm(range(start, generations), desc="Generation")
    for generation in progress_bar:
        # Update progress bar.
        progress_bar.set_description(f"Best accuracy: {max(accuracies)}")

        # Sample offspring from the subpopulations if available (five from each).
        for subpopulation in [P_1, P_2, P_3]:
            if len(subpopulation) > 0:
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
            P_3_history.add(winner)

        # Remove deceased architectures.
        n_new = 5 * ((len(P_1) > 0) + (len(P_2) > 0) + (len(P_3) > 0))  # Five new architectures for each non-empty set.
        for architecture in range(oldest, oldest + n_new):
            for subpopulation in [P_1, P_2, P_3]:
                if architecture in subpopulation:
                    subpopulation.remove(architecture)
                    remove(join(path, str(architecture), "encoder.pt"))
                    remove(join(path, str(architecture), "classifier.pt"))
        oldest += n_new

        # Save checkpoint.
        ckpt = (rng, P_1, P_2, P_3, P_3_history, accuracies, max_accuracies, generation, step, oldest)
        pickle.dump(ckpt, open(ckpt_path, 'rb'))

    print("Done!")

    # Remove all model checkpoints.
    for subpopulation in [P_1, P_2, P_3]:
        for architecture in subpopulation:
            remove(join(path, str(architecture), "encoder.pt"))
            remove(join(path, str(architecture), "classifier.pt"))

    # Save statistics.
    np.save(join(path, "accuracies.npy"), np.array(accuracies))
    np.savetxt(join(path, "accuracies.csv"), np.array(accuracies), delimiter=',')
    np.save(join(path, "max_accuracies.npy"), np.array(max_accuracies))
    np.savetxt(join(path, "max_accuracies.csv"), np.array(max_accuracies), delimiter=',')

    # Retrain the five best models from P_3_history and return the best one.
    print("Retraining the best models...")

    # Load regular (full) CIFAR-10.
    training, validation, _ = load('CIFAR10', validation=True)

    with open(join(path, "final_accuracies.csv"), 'w') as out:
        out.write(f"ID,Accuracy")

    winners = sorted(P_3_history, key=lambda x: accuracies[x])[-5:]
    best_arch = None
    best_acc = 0
    for winner in tqdm(winners, "Architecture"):
        # Load architecture.
        architecture = pickle.load(open(join(path, str(winner), "architecture.pkl"), 'rb'))

        # Train and record validation accuracy.
        enc = HebbianEncoder(in_channels, architecture, n_channels, stack_size, eta)
        fc = Classifier(4 * n_channels, n_classes)
        m = train(enc, fc, training, 50, encoder_batch, classifier_batch)
        accuracy = test(m, validation, classifier_batch, device)

        # Save cell visualizations.
        makedirs(join(path, "winners", str(winner)), exist_ok=True)
        architecture.normal_cell.visualize(join(path, "winners", str(winner), "normal.png"))
        architecture.reduction_cell.visualize(join(path, "winners", str(winner), "reduction.png"))

        with open(join(path, "final_accuracies.csv"), 'w') as out:
            out.write(f"{winner},{accuracy}")

        if accuracy > best_acc:
            best_arch = winner
            best_acc = accuracy
    print("Done! Best architecture:")
    print(f"ID: {best_arch}, accuracy: {best_acc}")


if __name__ == '__main__':
    # For command-line use.
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'CIFAR10'],
                        help="The dataset to use for evolution (default: CIFAR10)")
    parser.add_argument('--n_channels', type=int, default=8, help="The initial number of channels (default: 8).")
    parser.add_argument('--stack_size', type=int, default=6, help="The normal cell stack size (default: 6).")
    parser.add_argument('--epochs', type=int, default=10, help="The epoch increment for training the classifier ("
                                                               "default: 10).")
    parser.add_argument('--generations', type=int, default=100, help="The number of generations (default 100).")
    parser.add_argument('--eta', type=int, default=0.01, help="The base learning rate used in SoftHebb convolutions ("
                                                              "default: 0.01).")
    parser.add_argument('--encoder_batch', type=int, default=16,
                        help="The batch size for training the encoder's SoftHebb convolutions (default: 16).")
    parser.add_argument('--classifier_batch', type=int, default=128,
                        help="The batch size for training the classifier with SGD (default: 128).")
    parser.add_argument('--checkpoint', type=str, default=None, help="Optional checkpoint name to continue evolution.")
    args = parser.parse_args()

    evolve(args.dataset, args.n_channels, args.stack_size, args.epochs, args.generations, args.eta, args.encoder_batch,
           args.classifier_batch, args.checkpoint)
