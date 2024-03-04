"""This module provides methods for hyperparameter tuning after evolution."""

import pickle
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from os import makedirs
from os.path import join

import torch
from ray.train import RunConfig
from ray.tune import Tuner, grid_search
from tqdm import tqdm

from dataloader import load
from models import Classifier, HebbNetA, HebbianEncoder
from training import test, train


def retrain(run: str, n_channels: int):
    """Retrains the best five models from evolution.

    :param run: The evolution run.
    :param n_channels: The initial number of channels.
    """

    path = f"runs/{run}"
    winners = pickle.load(open(join(path, "winners.pkl"), 'rb'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CIFAR-10.
    training, validation, _ = load('CIFAR10', validation=True, seed=165326022024)

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


def tune(layer: int, n_channels: int):
    """Tunes HebbNet-A.

    :param layer: The layer to tune.
    :param n_channels: The number of initial channels.
    """

    makedirs(f"tuning/layer_{layer}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CIFAR-10.
    training, validation, _ = load('CIFAR10', validation=True)

    def objective(configuration: dict):
        """Objective used for tuning.

        :param configuration: The hyperparameter settings.
        :return: The validation accuracy.
        """

        # Initialize model.
        encoder = HebbNetA(in_channels=3, n_cells=layer, n_channels=n_channels, config=configuration)
        classifier = Classifier(encoder.out_channels * (32 // 2 ** layer) ** 2, 10)
        model = train(encoder, classifier, training, 50, 10, 64)
        accuracy = test(model, validation, 256, device)
        return accuracy

    # Stage one.
    search_space = {
        "conv_1": {"eta": grid_search([0.001, 0.005, 0.01, 0.05, 0.1]), "tau": grid_search([0.5, 1, 1.5, 2, 4]),
                   "p": None}}
    storage_path = f"tuning/layer_{layer}"
    name = "stage_one"
    run_config = RunConfig(storage_path=storage_path, name=name)
    path = join(storage_path, name)
    if Tuner.can_restore(path):
        tuner = Tuner.restore(path, objective, resume_errored=False, restart_errored=True)
    else:
        tuner = Tuner(objective, param_space=search_space, run_config=run_config)
    results = tuner.fit()

    config = results.get_best_result(metric="score", mode="min").config
    print("Best configuration found in stage one:")
    print(config)
    print("")

    # Stage two.
    search_space = config
    search_space["conv_1"]["p"] = grid_search([0.1, 0.5, 1, 1.5, 2])
    name = "stage_two"
    run_config = RunConfig(storage_path=storage_path, name=name)
    path = join(storage_path, name)
    if Tuner.can_restore(path):
        tuner = Tuner.restore(path, objective, resume_errored=False, restart_errored=True)
    else:
        tuner = Tuner(objective, param_space=search_space, run_config=run_config)
    results = tuner.fit()

    config = results.get_best_result(metric="score", mode="min").config
    print("Best configuration found in stage one:")
    print(config)
    print("")
    with open(f"tuning/layer_{layer}", 'w') as out:
        out.write(config)


if __name__ == '__main__':
    # For command-line use.
    parser = ArgumentParser()
    parser.add_argument('--run', type=str, help="The evolution run to load.")
    parser.add_argument('--n_channels', type=int, default=16, help="The number of initial channels.")
    parser.add_argument('--tune', action=BooleanOptionalAction, help="Turn on this option to tune HebbNet-A.")
    parser.add_argument('--layer', type=int, default=1,
                        help="The layer to tune for HebbNet-A if ``tune`` is switched on (default: 1).")
    args = parser.parse_args()

    if args.tune:
        tune(args.layer, args.n_channels)
    else:
        retrain(args.run, args.n_channels)
