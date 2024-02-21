"""This module provides methods for hyperparameter tuning after evolution."""

import pickle
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from os.path import join

import torch
from tqdm import tqdm

from dataloader import load
from models import Classifier, HebbianEncoder
from training import test, train


def retrain_models(run: str, n_channels: int):
    """Retrains the best five models resulting from evolution.

    :param run: The evolution run.
    :param n_channels: The initial number of channels.
    """

    path = f"runs/{run}"
    winners = pickle.load(open(join(path, "winners.pkl"), 'rb'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CIFAR-10.
    training, validation, _ = load('CIFAR10', validation=True)

    with open(join(path, f"accuracies_{n_channels}.csv"), 'w') as out:
        out.write(f"ID,Accuracy\n")
        for winner in tqdm(winners, desc="Architecture", file=sys.stdout):
            # Load architecture.
            architecture = pickle.load(open(join(path, "architectures", str(winner), "architecture.pkl"), 'rb'))

            # Train and record validation accuracy.
            encoder = HebbianEncoder(3, architecture, n_channels, 0, 0.01, 4, 3)
            classifier = Classifier(encoder.out_channels * (32 // 2 ** 3) ** 2, 10)
            model = train(encoder, classifier, training, 50, 10, 64, verbose=True)
            accuracy = test(model, validation, 256, device)
            out.write(f"{winner},{accuracy}\n")

            print(f"Accuracy of model {winner}: {accuracy:.2f}%")


def tune():
    """Tune model ... from the original evolution."""


if __name__ == '__main__':
    # For command-line use.
    parser = ArgumentParser()
    parser.add_argument('--run', type=str, default=None,
                        help="The evolution run to load, used if ``tune`` is turned off.")
    parser.add_argument('--n_channels', type=int, default=16, help="The number of initial channels.")
    parser.add_argument('--tune', action=BooleanOptionalAction,
                        help="Turn on this option to tune the hyperparameters of the best model from the original "
                             "evolution.")
    args = parser.parse_args()

    if args.tune:
        tune()
    else:
        retrain_models(args.run, args.n_channels)
