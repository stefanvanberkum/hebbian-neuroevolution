"""This module provides methods for hyperparameter tuning after evolution."""
import csv
import pickle
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from os import makedirs
from os.path import abspath, join

import numpy as np
import torch
from ray.train import RunConfig
from ray.tune import TuneConfig, Tuner, loguniform, quniform, uniform, with_parameters, with_resources
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from dataloader import load
from models import Classifier, HebbNetA, HebbianEncoder, SoftHebbNet
from training import test, train


def retrain(run: str, n_channels: int):
    """Retrains the best five models from evolution.

    Can also evaluate SoftHebb for comparison by specifying the run as "SoftHebb".

    :param run: The evolution run or "SoftHebb".
    :param n_channels: The initial number of channels.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CIFAR-10.
    training, validation, _ = load('CIFAR10', validation=True, seed=165326022024)

    if run == 'SoftHebb':
        # Train and record validation accuracy.
        encoder = SoftHebbNet(config="default")
        classifier = Classifier(encoder.out_channels * (32 // 2 ** 3) ** 2, 10)
        model = train(encoder, classifier, training, 50, 10, 64, verbose=True)
        accuracy = test(model, validation, 256, device)
        print(f"Accuracy of the SoftHebb network: {accuracy:.2f}%")
    else:
        path = f"runs/{run}"
        winners = pickle.load(open(join(path, "winners.pkl"), 'rb'))

        with open(join(path, f"accuracies_{n_channels}.csv"), 'w') as out:
            out.write(f"ID,Accuracy\n")
            for winner in tqdm(winners, desc="Architecture", file=sys.stdout):
                # Load architecture.
                architecture = pickle.load(open(join(path, "architectures", str(winner), "architecture.pkl"), 'rb'))

                # Train and record validation accuracy.
                encoder = HebbianEncoder(3, architecture, n_channels, 0, 0.01, 4, 2)
                classifier = Classifier(encoder.out_channels * (32 // 2 ** 3) ** 2, 10)
                model = train(encoder, classifier, training, 50, 10, 64, verbose=True)
                accuracy = test(model, validation, 256, device)
                out.write(f"{winner},{accuracy}\n")

                print(f"Accuracy of model {winner}: {accuracy:.2f}%")


def cross_validate(run: str):
    """Runs 5-fold cross-validation on the two best architectures with 32 channels.

    :param run: The evolution run.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get the two best architectures.
    path = f"runs/{run}"
    n_channels = 32
    ids = []
    accuracies = []
    with open(join(path, "accuracies_32.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[0] == "ID":
                continue
            ids.append(int(row[0]))
            accuracies.append(float(row[1]))
    ids = np.array(ids, dtype=int)
    accuracies = np.array(accuracies)
    winners = np.argsort(accuracies)
    ids = ids[winners][-2:]

    # Load CIFAR-10.
    training, _ = load('CIFAR10')
    training.data = training.data.cpu()
    training.targets = training.targets.cpu()
    folds = StratifiedKFold(n_splits=5, shuffle=True)

    accuracies = np.zeros((2, 5))
    for fold, (train_ids, test_ids) in tqdm(list(enumerate(folds.split(training.data, training.targets))), desc="Fold",
                                            file=sys.stdout):
        train_data = Subset(training, train_ids)
        train_data.data = train_data.dataset.data.to(device)
        train_data.targets = train_data.dataset.targets.to(device)
        val_data = Subset(training, test_ids)
        val_data.data = val_data.dataset.data.to(device)
        val_data.targets = val_data.dataset.targets.to(device)

        # Train and evaluate the architecture.
        for i in range(2):
            architecture = pickle.load(open(join(path, "architectures", str(ids[i]), "architecture.pkl"), 'rb'))
            encoder = HebbianEncoder(3, architecture, n_channels, 0, 0.01, 4, 2)
            classifier = Classifier(encoder.out_channels * (32 // 2 ** 3) ** 2, 10)
            model = train(encoder, classifier, train_data, 50, 10, 64, verbose=True)
            accuracy = test(model, val_data, 256, device)
            print(f"Accuracy of architecture {ids[i]}: {accuracy:.2f}%\n")
            accuracies[i, fold] += accuracy
    pickle.dump(accuracies, open(join(path, f"accuracies_cv.pkl"), 'wb'))

    with open(join(path, f"accuracies_cv.csv"), 'w') as out:
        out.write(f"ID,Accuracy,SD\n")
        mus = np.mean(accuracies, axis=1)
        sigmas = np.std(accuracies, axis=1)

        for i in range(2):
            out.write(f"{ids[i]},{mus[i]},{sigmas[i]}\n")


def tune(model: str):
    """Tunes a model for CIFAR-10.

    :param model: The model to tune, one of: {HebbNet, SoftHebb}.
    """

    makedirs(f"tuning/{model}", exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CIFAR-10.
    training, validation, _ = load('CIFAR10', validation=True, seed=113813032024)

    def objective(configuration: dict, type: str, train_data: Dataset, val_data: Dataset, dev: str):
        """Objective used for tuning.

        :param configuration: The hyperparameter settings.
        :return: The validation accuracy.
        """

        # Initialize model.
        if type == "HebbNet":
            encoder = HebbNetA(in_channels=3, config=configuration)
        elif type == "SoftHebb":
            encoder = SoftHebbNet(config=configuration)
        else:
            raise ValueError(f"Model {type} not found.")
        classifier = Classifier(encoder.out_channels * (32 // 2 ** 3) ** 2, 10, dropout=configuration["dropout"])
        m = train(encoder, classifier, train_data, int(configuration["n_epochs"]), 10, 64, alpha=configuration["alpha"])
        accuracy = test(m, val_data, 256, dev)
        return {"accuracy": accuracy}

    storage_path = abspath(f"tuning/{model}")
    name = "run"
    run_config = RunConfig(storage_path=storage_path, name=name)
    path = join(storage_path, name)

    def conv_config():
        """Generate the search space for a convolution.

        :return: The search space.
        """
        return {"eta": loguniform(1e-4, 1), "tau": loguniform(0.01, 10), "p": loguniform(0.1, 10)}

    if model == "HebbNet":
        search_space = {"n_channels": quniform(8, 40, 2), "alpha": loguniform(1e-4, 1), "dropout": uniform(0, 1),
                        "n_epochs": quniform(1, 50, 1), "initial_conv": conv_config(),
                        "cell_1": {"pre_skip": conv_config(), "conv_skip": conv_config(), "conv_1_add": conv_config(),
                                   "conv_1_cat": conv_config(), "dil_conv_5": conv_config()},
                        "cell_2": {"pre_skip": conv_config(), "conv_skip": conv_config(), "conv_1_add": conv_config(),
                                   "conv_1_cat": conv_config(), "dil_conv_5": conv_config()}}
    elif model == "SoftHebb":
        search_space = {"n_channels": quniform(32, 160, 8), "alpha": loguniform(1e-4, 1), "dropout": uniform(0, 1),
                        "n_epochs": quniform(1, 50, 1), "conv_1": conv_config(), "conv_2": conv_config(),
                        "conv_3": conv_config()}
    else:
        raise ValueError(f"Model {model} not found.")
    hyperopt = HyperOptSearch(metric="accuracy", mode="max")
    tune_config = TuneConfig(search_alg=hyperopt, num_samples=400)

    obj = with_parameters(objective, type=model, train_data=training, val_data=validation, dev=device)
    trainable = with_resources(obj, {"cpu": 1, "gpu": 1})
    if Tuner.can_restore(path):
        tuner = Tuner.restore(path, trainable, resume_errored=False, restart_errored=True)
    else:
        tuner = Tuner(trainable, param_space=search_space, tune_config=tune_config, run_config=run_config)
    results = tuner.fit()

    config = results.get_best_result(metric="accuracy", mode="max").config
    print("Best configuration found:")
    print(config)
    with open(f"tuning/{model}/config.txt", 'w') as out:
        print(config, file=out)

    pickle.dump(results, open(f"tuning/{model}/results.pkl", 'wb'))
    pickle.dump(config, open(f"tuning/{model}/config.pkl", 'wb'))

    accuracies = []
    with open(f"tuning/{model}/accuracies.csv", 'w') as out:
        for i, result in enumerate(results):
            if result.error:
                print(f"Trial #{i} had an error:", result.error)
                continue
            elif "accuracy" not in result.metrics:
                print(f"Trial #{i} has not finished.")
                continue
            else:
                print(result.metrics["accuracy"], file=out)
                accuracies.append(result.metrics["accuracy"])
    pickle.dump(accuracies, open(f"tuning/{model}/accuracies.pkl", 'wb'))


if __name__ == '__main__':
    # For command-line use.
    parser = ArgumentParser()
    parser.add_argument('--run', type=str, help="The evolution run to load or model to tune (HebbNet or SoftHebb in "
                                                "that case).")
    parser.add_argument('--n_channels', type=int, default=16, help="The number of initial channels.")
    parser.add_argument('--cv', action=BooleanOptionalAction, help="Turn on this option to cross-validate the best "
                                                                   "architectures.")
    parser.add_argument('--tune', action=BooleanOptionalAction, help="Turn on this option to tune HebbNet-A or the "
                                                                     "SoftHebb network.")
    args = parser.parse_args()

    if args.tune:
        tune(args.run)
    elif args.cv:
        cross_validate(args.run)
    else:
        retrain(args.run, args.n_channels)
