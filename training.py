"""This module provides methods for model training.

Supports command-line use.

Functions
=========
- :func:`train`: Train a model.
- :func:`train_encoder`: Train a Hebbian encoder.
- :func:`train_classifier`: Train a classifier.
- :func:`test`: Test a model.
"""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import timedelta
from time import time

import torch
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from dataloader import load
from models import Classifier, HebbNet, HebbNetA, SoftHebbNet


def train(encoder: Module, classifier: Module, train_data: Dataset, val_data: Dataset, encoder_batch: int,
          classifier_batch: int, patience=10, max_epochs=100, val_batch=256, n_workers=0, verbose=False):
    """Train an encoder network and final classifier sequentially.

    Both training processes utilize automatic mixed precision in combination with gradient scaling.

    :param encoder: The Hebbian encoder network.
    :param classifier: The classifier.
    :param train_data: The training data.
    :param val_data: The validation data used for early stopping.
    :param encoder_batch: The batch size for Hebbian training of the encoder network.
    :param classifier_batch: The batch size for supervised training of the classifier.
    :param patience: The number of consecutive epochs of non-improvement required for termination (default: 10).
    :param max_epochs: The maximum number of epochs for any model (default: 100).
    :param val_batch: Batch size for testing on the validation data (default: 256).
    :param n_workers: The number of workers (default: 0, i.e., run in the main process).
    :param verbose: True if progress should be printed (default: False).
    :return: A trained network comprising the encoder and classifier.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder.to(device)
    classifier.to(device)

    # Train encoder.
    start_time = time()
    if verbose:
        print("Training encoder...")
    train_encoder(encoder, train_data, encoder_batch, device, n_workers)
    if verbose:
        print(f"Done! Elapsed time: {str(timedelta(seconds=time() - start_time)).split('.', 2)[0]}")

    # Train classifier.
    if verbose:
        print("Training classifier...")
    train_classifier(encoder, classifier, train_data, val_data, classifier_batch, device, patience, max_epochs,
                     val_batch, n_workers, verbose)
    if verbose:
        print(f"Done! Elapsed time: {str(timedelta(seconds=time() - start_time)).split('.', 2)[0]}")

    # Combine encoder and classifier.
    model = HebbNet(encoder, classifier)
    return model


def train_encoder(encoder: Module, data: Dataset, batch_size: int, device: str, n_workers=0):
    """Train the encoder using Hebbian learning.

    :param encoder: The Hebbian encoder network.
    :param data: The training data.
    :param batch_size: The batch size.
    :param device: The device used for PyTorch operations.
    :param n_workers: The number of workers (default: 0, i.e., run in the main process).
    """

    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=True)

    # Training loop.
    encoder.train()
    with autocast(device_type=device, dtype=torch.float16):
        for i, data in enumerate(loader):
            # Run data through encoder for unsupervised Hebbian learning.
            x, _ = data
            x = x.to(device)
            encoder(x)


def train_classifier(encoder: Module, classifier: Module, train_data: Dataset, val_data: Dataset, batch_size: int,
                     device: str, patience=10, max_epochs=100, val_batch=256, n_workers=0, verbose=True):
    """Train the classifier using supervised learning.

    :param encoder: The Hebbian encoder network.
    :param classifier: The classifier.
    :param train_data: The training data.
    :param val_data: The validation data used for early stopping.
    :param batch_size: The batch size.
    :param device: The device used for PyTorch operations.
    :param patience: The number of consecutive epochs of non-improvement required for termination (default: 10).
    :param max_epochs: The maximum number of epochs for any model (default: 100).
    :param val_batch: Batch size for testing on the validation data (default: 256).
    :param n_workers: The number of workers (default: 0, i.e., run in the main process).
    :param verbose: True if progress should be printed (default: False).
    """

    # Set up dataloader.
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=True)

    # Set up loss function and gradient scaler.
    loss_fn = CrossEntropyLoss()
    scaler = GradScaler()

    # Set up learning rate scheduler.
    optimizer = Adam(classifier.parameters(), lr=0.0005)

    cumulative_loss = 0  # Cumulative training loss.
    n_correct = 0  # Cumulative amount of correctly classified training samples.
    batches = 0  # Processed batches tracker.
    samples = 0  # Number of samples tracker.
    epoch = 0  # Current epoch.
    irritation = 0  # Number of consecutive epochs without improvement.
    best_acc = 0  # Best validation accuracy.
    best_model = deepcopy(classifier.state_dict())  # Model corresponding to the best validation loss.

    # Training loop.
    while epoch < max_epochs:
        encoder.eval()
        classifier.train()
        for i, train_data in enumerate(loader):
            # Reset gradients.
            optimizer.zero_grad()

            with autocast(device_type=device, dtype=torch.float16):
                # Run batch through encoder and classifier.
                x, y = train_data
                x = x.to(device)
                y = y.to(device)
                x = encoder(x)
                logits = classifier(x)
                loss = loss_fn(logits, y)

            # Update classifier.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # if verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
            if verbose:
                # Record loss and accuracy.
                cumulative_loss += loss.item()
                n_correct += torch.sum(torch.argmax(logits, dim=1) == y).item()
                batches += 1
                samples += len(y)

        val_acc = test(HebbNet(encoder, classifier), val_data, val_batch, device, n_workers)
        # if verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
        if verbose:
            # Print and reset loss and accuracy.
            loss = cumulative_loss / batches
            acc = float(n_correct) / samples * 100
            print(
                f"- Epoch {epoch + 1}: train loss {loss:.4f}, train accuracy {acc:.2f}%, validation accuracy "
                f"{val_acc:.2f}%")
            cumulative_loss = 0
            n_correct = 0
            batches = 0
            samples = 0

        # Check for early stopping.
        if val_acc > best_acc:
            irritation = 0
            best_acc = val_acc
            best_model = deepcopy(classifier.state_dict())
        else:
            irritation += 1

        if irritation >= patience:
            classifier.load_state_dict(best_model)
            break
        epoch += 1


def test(model: Module, data: Dataset, batch_size: int, device: str, n_workers=0):
    """Test a model.

    :param model: The model comprising an encoder and classifier.
    :param data: The test data.
    :param batch_size: The batch size.
    :param device: The device used for PyTorch operations.
    :param n_workers: The number of workers (default: 0, i.e., run in the main process).
    :return: The test accuracy (percentage).
    """

    # Setup dataloader.
    loader = DataLoader(data, batch_size=batch_size, num_workers=n_workers)

    # Test loop.
    model.eval()
    n_correct = 0
    samples = 0
    for i, data in enumerate(loader):
        # Run batch through model and get number of correct predictions.
        x, y = data
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        n_correct += torch.sum(torch.argmax(logits, dim=1) == y).item()
        samples += len(y)
    return float(n_correct) / samples * 100


if __name__ == '__main__':
    # For command-line use.
    parser = ArgumentParser()
    parser.add_argument('--model', choices=['SoftHebb', 'HebbNet'], help="The model to be trained.")
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'SVHN'], help="The dataset to train and test on.")
    parser.add_argument('--hebb_batch', type=int, help="The batch size for Hebbian training.")
    parser.add_argument('--sgd_batch', type=int, help="The batch size for SGD training.")
    parser.add_argument('--patience', type=int, default=10,
                        help="The number of consecutive epochs of non-improvement required for termination (default: "
                             "10).")
    parser.add_argument('--max_epochs', type=int, default=100,
                        help="The maximum number of epochs for any model (default: 100).")
    parser.add_argument('--fast', type=bool, default=True,
                        help="True if a fast version of the dataset should be used. Only works for MNIST and CIFAR10 ("
                             "default: True).")
    parser.add_argument('--n_workers', type=int, default=0,
                        help="The number of workers to use in data loading (default: 0, i.e., run in the main "
                             "process). Has to be set to zero if a fast-mode dataset is used.")
    args = parser.parse_args()

    # Load data.
    initial_size = 32
    if args.dataset == 'CIFAR10' or args.dataset == 'SVHN':
        n_classes = 10
    elif args.dataset == 'CIFAR100':
        n_classes = 100
    else:
        raise RuntimeError(f"Dataset {args.dataset} not found (internal error).")

    training_data, test_data = load(args.dataset, fast=args.fast)

    # Get setup.
    if args.model == 'SoftHebb':
        encoder_net = SoftHebbNet(config='original')
        in_features = int((initial_size / (2 ** 3)) ** 2 * 1536)
        classifier_net = Classifier(in_features=in_features, out_features=n_classes)
    elif args.model == 'HebbNet':
        encoder_net = HebbNetA()
        in_features = int((initial_size / (2 ** 3)) ** 2 * encoder_net.out_channels)
        classifier_net = Classifier(in_features=in_features, out_features=n_classes)
    else:
        raise RuntimeError(f"Model {args.model} not found (internal error).")

    # Train and test model.
    trained_model = train(encoder_net, classifier_net, training_data, test_data, args.hebb_batch, args.sgd_batch,
                          patience=args.patience, max_epochs=args.max_epochs, n_workers=args.n_workers, verbose=True)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_acc = test(trained_model, test_data, args.sgd_batch, device_str, args.n_workers)
    print(f"Test accuracy: {test_acc:.2f}")
