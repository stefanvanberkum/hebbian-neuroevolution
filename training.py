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
from datetime import timedelta
from time import time

import torch
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from dataloader import load
from models import Classifier, HebbNet, HebbNetA, SoftHebbNet


def train(encoder: Module, classifier: Module, data: Dataset, n_epochs: int, encoder_batch: int, classifier_batch: int,
          alpha=1e-3, n_workers=0, verbose=False, validation_data=None, val_batch=64, checkpoint: dict | None = None):
    """Train an encoder network and final classifier sequentially.

    Both training processes utilize automatic mixed precision in combination with gradient scaling.

    :param encoder: The Hebbian encoder network.
    :param classifier: The classifier.
    :param data: The training data.
    :param n_epochs: The number of epochs for supervised training of the classifier.
    :param encoder_batch: The batch size for Hebbian training of the encoder network.
    :param classifier_batch: The batch size for supervised training of the classifier.
    :param alpha: The learning rate for training the classifier.
    :param n_workers: The number of workers (default: 0, i.e., run in the main process).
    :param verbose: True if progress should be printed (default: False).
    :param validation_data: Validation data used for testing, only used if ``verbose`` is ``True`` (default: None).
    :param val_batch: Batch size for testing on the validation data (default: 64).
    :param checkpoint: Optional dict with {start_epoch, max_epochs, optimizer_state_dict, scheduler_state_dict,
        save_path}.
    :return: A trained network comprising the encoder and classifier.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder.to(device)
    classifier.to(device)

    # Train encoder if starting from scratch.
    start_time = time()
    if checkpoint is None or checkpoint['start_epoch'] == 0:
        if verbose:
            print("Training encoder...")
        train_encoder(encoder, data, encoder_batch, device, n_workers)
        if verbose:
            print(f"Done! Elapsed time: {str(timedelta(seconds=time() - start_time)).split('.', 2)[0]}")

    # Train classifier.
    if verbose:
        print("Training classifier...")
    train_classifier(encoder, classifier, data, n_epochs, classifier_batch, device, alpha, n_workers, verbose,
                     validation_data, val_batch, checkpoint)
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


def train_classifier(encoder: Module, classifier: Module, data: Dataset, n_epochs: int, batch_size: int, device: str,
                     alpha=1e-3, n_workers=0, verbose=True, validation_data=None, val_batch=64,
                     checkpoint: dict | None = None):
    """Train the classifier using supervised learning.

    :param encoder: The Hebbian encoder network.
    :param classifier: The classifier.
    :param data: The training data.
    :param n_epochs: The number of epochs for training.
    :param batch_size: The batch size.
    :param device: The device used for PyTorch operations.
    :param alpha: The learning rate used for Adam.
    :param n_workers: The number of workers (default: 0, i.e., run in the main process).
    :param verbose: True if progress should be printed (default: False).
    :param validation_data: Validation data used for testing, only used if ``verbose`` is ``True`` (default: None).
    :param val_batch: Batch size for testing on the validation data (default: 64).
    :param checkpoint: Optional dict with {start_epoch, max_epochs, optimizer_state_dict, scheduler_state_dict,
        save_path}.
    """

    # Set up dataloader.
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=True)

    # Set up loss function and gradient scaler.
    loss_fn = CrossEntropyLoss()
    scaler = GradScaler()

    # Set up learning rate scheduler.
    if checkpoint is None:
        optimizer = Adam(classifier.parameters(), lr=alpha)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    else:
        if n_epochs > checkpoint['max_epochs'] - checkpoint['start_epoch']:
            raise ValueError("The number of epochs is larger than the remaining number of epochs specified by the "
                             "checkpoint.")

        optimizer = Adam(classifier.parameters())
        if checkpoint['start_epoch'] > 0:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler = CosineAnnealingLR(optimizer, T_max=checkpoint['max_epochs'],
                                      last_epoch=checkpoint['start_epoch'] - 1)
        if checkpoint['start_epoch'] > 0:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Training loop.
    cumulative_loss = 0
    n_correct = 0
    batches = 0
    samples = 0
    for epoch in range(n_epochs):
        encoder.eval()
        classifier.train()
        for i, data in enumerate(loader):
            # Reset gradients.
            optimizer.zero_grad()

            with autocast(device_type=device, dtype=torch.float16):
                # Run batch through encoder and classifier.
                x, y = data
                x = x.to(device)
                y = y.to(device)
                x = encoder(x)
                logits = classifier(x)
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
            if validation_data is None:
                print(f"- Epoch {epoch + 1}: train loss {loss:.4f}, train accuracy {acc:.2f}%")
            else:
                val_acc = test(HebbNet(encoder, classifier), validation_data, val_batch, device, n_workers)
                print(f"- Epoch {epoch + 1}: train loss {loss:.4f}, train accuracy {acc:.2f}%, validation accuracy "
                      f"{val_acc:.2f}%")
            cumulative_loss = 0
            n_correct = 0
            batches = 0
            samples = 0

    # Save checkpoint if necessary.
    if checkpoint is not None:
        start_epoch = checkpoint['start_epoch'] + n_epochs
        checkpoint = {'start_epoch': start_epoch, 'max_epochs': checkpoint['max_epochs'],
                      'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                      'save_path': checkpoint['save_path']}
        torch.save(checkpoint, checkpoint['save_path'])


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
    parser.add_argument('--dataset', choices=['CIFAR10'], help="The dataset to train and test on.")
    parser.add_argument('--n_epochs', type=int, help="The number of epochs for SGD training.")
    parser.add_argument('--hebb_batch', type=int, help="The batch size for Hebbian training.")
    parser.add_argument('--sgd_batch', type=int, help="The batch size for SGD training.")
    parser.add_argument('--fast', type=bool, default=True,
                        help="True if a fast version of the dataset should be used. Only works for MNIST and CIFAR10 ("
                             "default: True).")
    parser.add_argument('--n_workers', type=int, default=0,
                        help="The number of workers to use in data loading (default: 0, i.e., run in the main "
                             "process). Has to be set to zero if a fast-mode dataset is used.")
    args = parser.parse_args()

    # Load data.
    if args.dataset == 'CIFAR10':
        initial_size = 32
        n_classes = 10
    else:
        raise RuntimeError(f"Dataset {args.dataset} not found (internal error).")

    train_data, test_data = load(args.dataset, fast=args.fast)

    # Get setup.
    if args.model == 'SoftHebb':
        encoder_net = SoftHebbNet()
        in_features = int((initial_size / (2 ** 3)) ** 2 * 1536)
        classifier_net = Classifier(in_features=in_features, out_features=n_classes)
    elif args.model == 'HebbNet':
        encoder_net = HebbNetA()
        in_features = int((initial_size / (2 ** 3)) ** 2 * encoder_net.out_channels)
        classifier_net = Classifier(in_features=in_features, out_features=n_classes)
    else:
        raise RuntimeError(f"Model {args.model} not found (internal error).")

    # Train and test model.
    trained_model = train(encoder_net, classifier_net, train_data, args.n_epochs, args.hebb_batch, args.sgd_batch,
                          n_workers=args.n_workers, verbose=True, validation_data=test_data)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_acc = test(trained_model, test_data, args.sgd_batch, device_str, args.n_workers)
    print(f"Test accuracy: {test_acc:.2f}")
