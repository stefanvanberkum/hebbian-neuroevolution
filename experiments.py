"""This module provides methods for running the experiments after evolution.

Supports command-line use.

Functions
=========
- :func:`evaluate`: Perform a classical single train-test split evaluation.
- :func:`train_bp`: Train an encoder network using backpropagation.
- :func:`test`: Test a model.
- :func:`summarize`: Summarize the classical evaluation results.
- :func:`visualize`: Visualize the function of multiple neural pathways.
- :func:`bayesian_analysis`: Run a Bayesian analysis to estimate the effect size.
"""

import pickle
import statistics
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from os import makedirs
from os.path import exists, join

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import torch
from baycomp import CorrelatedTTest
from seaborn import lineplot
from sklearn.model_selection import StratifiedKFold
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

from dataloader import load
from layercam import LayerCAM
from models import BPNetA, Classifier, HebbNet, HebbNetA, NoSkipNetA, SoftHebbBPNet, SoftHebbNet
from training import train


def evaluate(model: str, mode: str, dataset: str):
    """Train and test a model on a given dataset.

    :param model: The model to test, one of: {HebbNet, SoftHebb, NoSkip}.
    :param mode: The training mode, one of: {Hebbian, BP}
    :param dataset: The dataset to test on, one of: {CIFAR10, CIFAR100, SVHN}.
    """

    makedirs("results/raw", exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data, test_data = load(dataset)
    accuracies = []
    correct = np.zeros(len(test_data))  # Keep track of how often the architecture gets each sample correct.

    print(f"Training and testing a {mode} {model} model on {dataset}...")
    with open(f"results/raw/accuracies_{dataset}_{mode}_{model}.csv", 'w') as out:
        for i in tqdm(range(10), desc="Run"):
            if model == "HebbNet" and mode == "Hebbian":
                encoder = HebbNetA()
            elif model == "HebbNet" and mode == "BP":
                encoder = BPNetA()  # Includes classifier.
            elif model == "SoftHebb" and mode == "Hebbian":
                encoder = SoftHebbNet()
            elif model == "SoftHebb" and mode == "BP":
                encoder = SoftHebbBPNet()  # Includes classifier.
            elif model == "NoSkip" and mode == "Hebbian":
                encoder = NoSkipNetA()
            else:
                raise ValueError(f"The {mode} {model} model not found.")
            if mode == "Hebbian":
                classifier = Classifier(encoder.out_channels * (32 // 2 ** 3) ** 2, 10,
                                        dropout=encoder.config["dropout"])
                model = train(encoder, classifier, train_data, int(encoder.config["n_epochs"]), 10, 64,
                              alpha=encoder.config["alpha"], verbose=True)
            else:
                model = train_bp(encoder, train_data, int(encoder.config["n_epochs"]), 64, device)
            accuracy, is_correct = test(model, test_data, 256, device)
            accuracies.append(accuracy)
            correct += is_correct

            out.write(f"{accuracy}\n")
    pickle.dump(accuracies, open(f"results/raw/accuracies_{dataset}_{mode}_{model}.pkl", 'wb'))
    np.save(f"results/raw/correct_{dataset}_{mode}_{model}.npy", correct)


def train_bp(model: Module, data: Dataset, n_epochs: int, batch_size: int, device: str, n_workers=0, verbose=True):
    """Train a model with backpropagation.

    :param model: The model to train.
    :param data: The training data.
    :param n_epochs: The number of epochs for training.
    :param batch_size: The batch size.
    :param device: The device used for PyTorch operations.
    :param n_workers: The number of workers (default: 0, i.e., run in the main process).
    :param verbose: True if progress should be printed (default: True).
    :return: The model.
    """

    # Set up dataloader.
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=True)

    # Set up loss function, optimizer, learning rate scheduler, and gradient scaler.
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    scheduler = CosineAnnealingLR(optimizer, n_epochs)
    scaler = GradScaler()
    model.train()

    # Training loop.
    cumulative_loss = 0
    n_correct = 0
    batches = 0
    samples = 0
    for epoch in range(n_epochs):
        for i, data in enumerate(loader):
            # Reset gradients.
            optimizer.zero_grad()

            with autocast(device_type=device, dtype=torch.float16):
                # Run batch through encoder and classifier.
                x, y = data
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)

            # Update classifier.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if verbose and (epoch == 0 or (epoch + 1) % (n_epochs // 10) == 0):
                # Record loss and accuracy.
                cumulative_loss += loss.item()
                n_correct += torch.sum(torch.argmax(logits, dim=1) == y).item()
                batches += 1
                samples += len(y)

        # Update learning rate.
        scheduler.step()

        if verbose and (epoch == 0 or (epoch + 1) % (n_epochs // 10) == 0):
            # Print and reset loss and accuracy.
            loss = cumulative_loss / batches
            acc = float(n_correct) / samples * 100
            print(f"- Epoch {epoch + 1}: train loss {loss:.4f}, train accuracy {acc:.2f}%")
            cumulative_loss = 0
            n_correct = 0
            batches = 0
            samples = 0
    return model


def test(model: Module, data: Dataset, batch_size: int, device: str, n_workers=0):
    """Test a model.

    :param model: The model comprising an encoder and classifier.
    :param data: The test data.
    :param batch_size: The batch size.
    :param device: The device used for PyTorch operations.
    :param n_workers: The number of workers (default: 0, i.e., run in the main process).
    :return: The test accuracy (percentage) and numpy array denoting whether each sample was classified correctly.
    """

    # Setup dataloader.
    loader = DataLoader(data, batch_size=batch_size, num_workers=n_workers)

    # Test loop.
    model.eval()
    correct = np.array([])
    for i, data in enumerate(loader):
        # Run batch through model and get number of correct predictions.
        x, y = data
        x = x.to(device)
        y = y.to(device)
        logits = model(x)

        correct = np.concatenate((correct, (torch.argmax(logits, dim=1) == y).numpy()))
    return np.mean(correct) * 100, correct


def summarize():
    """Summarize the classical evaluation results.

    This function computes and reports the mean and standard deviation for each model on each dataset.
    """

    for dataset in ['CIFAR10', 'CIFAR100', 'SVHN']:
        with open(f"results/{dataset}.csv", 'w') as out:
            out.write("Model,Mode,Mean,Standard deviation\n")
            for model in ['HebbNet', 'SoftHebb']:
                for mode in ['Hebbian', 'BP']:
                    accuracies = pickle.load(open(f"results/raw/accuracies_{dataset}_{mode}_{model}.pkl", 'rb'))
                    mu = statistics.mean(accuracies)
                    sigma = statistics.stdev(accuracies)
                    out.write(f"{model},{mode},{mu},{sigma}\n")


def visualize():
    """Visualize the function of multiple neural pathways.

    This method takes five samples with the best improvement ratio between HebbNet-A and the original SoftHebb network
    and generates the layer-wise class activation mapping (CAM) for that sample. For this purpose, a model is trained
    and the samples are only considered if that model gets them right as well. In addition, the CAMs are visualized
    for each layer in the original SoftHebb network.
    """

    for dataset in ['CIFAR10', 'CIFAR100', 'SVHN']:
        print(f"Visualizing HebbNet-A for {dataset}...")
        makedirs(f"results/raw", exist_ok=True)
        makedirs(f"results/visualization/{dataset}", exist_ok=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_data, test_data = load(dataset)
        test_loader = DataLoader(test_data, batch_size=len(test_data))
        data = next(iter(test_loader))

        encoder = HebbNetA()
        classifier = Classifier(encoder.out_channels * (32 // 2 ** 3) ** 2, 10, dropout=encoder.config["dropout"])
        if exists(f"results/raw/visualization_encoder_{dataset}.pt"):
            # Load checkpoint.
            encoder_state = torch.load(f"results/raw/visualization_encoder_{dataset}.pt")
            classifier_state = torch.load(f"results/raw/visualization_classifier_{dataset}.pt")
            encoder.load_state_dict(encoder_state)
            classifier.load_state_dict(classifier_state)
            model = HebbNet(encoder, classifier)
            is_correct = pickle.load(open(f"results/raw/correct_{dataset}.pkl", 'rb'))
        else:
            # Train model.
            model = train(encoder, classifier, train_data, int(encoder.config["n_epochs"]), 10, 64,
                          alpha=encoder.config["alpha"], verbose=True)
            _, is_correct = test(model, test_data, 256, device)
            torch.save(encoder.state_dict(), f"results/raw/visualization_encoder_{dataset}.pt")
            torch.save(classifier.state_dict(), f"results/raw/visualization_classifier_{dataset}.pt")
            pickle.dump(is_correct, open(f"results/raw/correct_{dataset}.pkl", 'wb'))

        correct_1 = np.load(f"results/raw/correct_{dataset}_Hebbian_HebbNet.npy")
        correct_2 = np.load(f"results/raw/correct_{dataset}_Hebbian_SoftHebb.npy")
        ratio = correct_1 / (correct_2 + 1)  # Compute improvement ratio.
        samples = np.argsort(ratio)  # Get indices sorted based on the improvement ratio.
        is_correct = is_correct[samples]  # Sort the boolean vector is_correct accordingly.
        samples = samples[is_correct]  # Filter the samples based on whether the current model gets them right.
        samples = samples[-5:]  # Get the five samples with the best improvement ratio, provided they are correct now.

        for i in range(5):
            path = f"results/visualization/{dataset}/{i}/HebbNet"
            makedirs(path, exist_ok=True)
            sample = data[samples[i]]

            with LayerCAM(model, model.initial_conv) as cam:
                filename = "initial_conv.png"
                filepath = join(path, filename)
                cam.visualize(sample[0], sample[1], filepath)

            cells = [encoder.cell_1, encoder.cell_2]
            for cell in cells:
                convs = [cell.preprocess_skip, cell.conv_skip, cell.conv_1_add, cell.conv_1_cat, cell.dil_conv_5]
                for conv in convs:
                    with LayerCAM(model, conv) as cam:
                        filename = f"{cell.name}_{conv.name}.png"
                        filepath = join(path, filename)
                        cam.visualize(sample[0], sample[1], filepath)
        print("Done!")

        print(f"Visualizing the SoftHebb network for {dataset}...")
        encoder = SoftHebbNet()
        classifier = Classifier(encoder.out_channels * (32 // 2 ** 3) ** 2, 10, dropout=encoder.config["dropout"])
        if exists(f"results/raw/visualization_softhebb_encoder_{dataset}.pt"):
            # Load checkpoint.
            encoder_state = torch.load(f"results/raw/visualization_softhebb_encoder_{dataset}.pt")
            classifier_state = torch.load(f"results/raw/visualization_softhebb_classifier_{dataset}.pt")
            encoder.load_state_dict(encoder_state)
            classifier.load_state_dict(classifier_state)
            model = HebbNet(encoder, classifier)
        else:
            # Train model.
            model = train(encoder, classifier, train_data, int(encoder.config["n_epochs"]), 10, 64,
                          alpha=encoder.config["alpha"], verbose=True)
            torch.save(encoder.state_dict(), f"results/raw/visualization_softhebb_encoder_{dataset}.pt")
            torch.save(classifier.state_dict(), f"results/raw/visualization_softhebb_classifier_{dataset}.pt")

        for i in range(5):
            path = f"results/visualization/{dataset}/{i}/SoftHebb"
            makedirs(path, exist_ok=True)
            sample = data[samples[i]]

            convs = [encoder.layer_1, encoder.layer_2, encoder.layer_3]
            for conv in convs:
                with LayerCAM(model, conv) as cam:
                    filename = f"{conv.name}.png"
                    filepath = join(path, filename)
                    cam.visualize(sample[0], sample[1], filepath)
        print("Done!")


def bayesian_analysis():
    """Run a Bayesian analysis to estimate the effect size.

    This method runs stratified 10-fold cross-validation on CIFAR-100 and SVHN for both HebbNet-A and the original
    SoftHebb network, and computes a posterior distribution over the difference between the two networks. It then
    records a 95% confidence interval.
    """

    makedirs(f"results/raw", exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for dataset in ['CIFAR100', 'SVHN']:
        print(f"Running a Bayesian analysis on {dataset}...")
        if exists(f"results/raw/cv_HebbNet_accuracies_{dataset}.pkl"):
            hebbnet_accuracies = pickle.load(open(f"results/raw/cv_HebbNet_accuracies_{dataset}.pkl", 'rb'))
            softhebb_accuracies = pickle.load(open(f"results/raw/cv_SoftHebb_accuracies_{dataset}.pkl", 'rb'))
            print("Loaded checkpoint!")
        else:
            train_data, test_data = load(dataset)
            train_data.data = train_data.data.cpu()
            train_data.targets = train_data.targets.cpu()
            test_data.data = test_data.data.cpu()
            test_data.targets = test_data.targets.cpu()
            dataset = ConcatDataset([train_data, test_data])

            folds = StratifiedKFold(n_splits=10, shuffle=True)
            hebbnet_accuracies = []
            softhebb_accuracies = []
            with open(f"results/cv_accuracies_{dataset}.csv", 'w') as out:
                out.write("HebbNet,SoftHebb\n")
                for fold, (train_ids, test_ids) in tqdm(enumerate(folds.split(dataset.data, dataset.targets)),
                                                        desc="Fold", file=sys.stdout):
                    train_data = Subset(dataset, train_ids)
                    train_data.data = train_data.dataset.data.to(device)
                    train_data.targets = train_data.dataset.targets.to(device)
                    test_data = Subset(dataset, test_ids)
                    test_data.data = test_data.dataset.data.to(device)
                    test_data.targets = test_data.dataset.targets.to(device)

                    # Train and evaluate HebbNet-A.
                    encoder = HebbNetA()
                    classifier = Classifier(encoder.out_channels * (32 // 2 ** 3) ** 2, 10,
                                            dropout=encoder.config["dropout"])
                    model = train(encoder, classifier, train_data, int(encoder.config["n_epochs"]), 10, 64,
                                  alpha=encoder.config["alpha"], verbose=True)
                    hebbnet_accuracy, _ = test(model, test_data, 256, device)
                    hebbnet_accuracies.append(hebbnet_accuracy)

                    # Train and evaluate the original SoftHebb network.
                    encoder = SoftHebbNet()
                    classifier = Classifier(encoder.out_channels * (32 // 2 ** 3) ** 2, 10,
                                            dropout=encoder.config["dropout"])
                    model = train(encoder, classifier, train_data, int(encoder.config["n_epochs"]), 10, 64,
                                  alpha=encoder.config["alpha"], verbose=True)
                    softhebb_accuracy, _ = test(model, test_data, 256, device)
                    softhebb_accuracies.append(softhebb_accuracy)

                    out.write(f"{hebbnet_accuracy},{softhebb_accuracy}\n")
            pickle.dump(hebbnet_accuracies, open(f"results/raw/cv_HebbNet_accuracies_{dataset}.pkl", 'wb'))
            pickle.dump(softhebb_accuracies, open(f"results/raw/cv_SoftHebb_accuracies_{dataset}.pkl", 'wb'))

        # Perform Bayesian analysis and record confidence intervals.
        posterior = CorrelatedTTest(hebbnet_accuracies, softhebb_accuracies)
        t_parameters = posterior.df, posterior.mean, np.sqrt(posterior.var)
        levels = [0.9, 0.95, 0.99, 0.999]
        low, high = scipy.stats.t.interval(levels, *t_parameters)
        with open(f"results/CI_{dataset}.csv", 'w') as out:
            out.write("Level,Low,High\n")
            for i in range(4):
                out.write(f"{levels[i]},{low[i]},{high[i]}\n")

        # Plot posterior (default).
        _ = posterior.plot(hebbnet_accuracies, softhebb_accuracies)
        plt.savefig(f"results/posterior_{dataset}_default.png")

        # Plot posterior (custom).
        x = np.linspace(scipy.stats.t.ppf(0.001, *t_parameters), scipy.stats.t.ppf(0.999, *t_parameters), 1000)
        y = scipy.stats.t.pdf(x, *t_parameters)
        _, ax = plt.subplots()
        lineplot(x=x, y=y, estimator=None, errorbar=None, ax=ax)
        sns.set_theme(context="paper", style="ticks")
        sns.despine()
        ax.fill_between(x, y, color=ax.lines[0].get_color(), alpha=0.5)
        ax.set_xlabel("Accuracy improvement")
        ax.set_ylabel("Density")
        plt.savefig(f"results/posterior_{dataset}.png", dpi=400)
        plt.savefig(f"results/posterior_{dataset}.eps")


if __name__ == '__main__':
    # For command-line use.
    parser = ArgumentParser()
    parser.add_argument('--model', choices=['HebbNet', 'SoftHebb', 'NoSkip'], help="The model to test.")
    parser.add_argument('--mode', choices=['Hebbian', 'BP'], help="The training mode to be applied.")
    parser.add_argument('--CIFAR10', action=BooleanOptionalAction, help="Turn on this option to test on CIFAR-10.")
    parser.add_argument('--CIFAR100', action=BooleanOptionalAction, help="Turn on this option to test on CIFAR-100.")
    parser.add_argument('--SVHN', action=BooleanOptionalAction, help="Turn on this option to test on SVHN.")
    parser.add_argument('--summarize', action=BooleanOptionalAction, help="Turn on this option to summarize results.")
    parser.add_argument('--visualize', action=BooleanOptionalAction, help="Turn on this option to visualize results.")
    parser.add_argument('--bayesian', action=BooleanOptionalAction,
                        help="Turn on this option to run a Bayesian analysis.")
    args = parser.parse_args()

    if args.CIFAR10:
        evaluate(model=args.model, mode=args.mode, dataset="CIFAR10")
    if args.CIFAR100:
        evaluate(model=args.model, mode=args.mode, dataset="CIFAR100")
    if args.SVHN:
        evaluate(model=args.model, mode=args.mode, dataset="SVHN")

    if args.summarize:
        summarize()

    if args.visualize:
        visualize()

    if args.bayesian:
        bayesian_analysis()
