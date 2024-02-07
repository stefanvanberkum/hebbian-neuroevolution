"""This module provides methods for model training.

Allows for command-line use.

Functions
=========
- :func:`train`: Train a model.
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
from models import Classifier, HebbNet, SoftHebbSmall


def train(encoder: Module, classifier: Module, data: Dataset, n_epochs: int, encoder_batch: int, classifier_batch: int,
          n_workers=0, verbose=True, validation_data=None, val_batch=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder.to(device)
    classifier.to(device)

    # Train encoder and classifier.
    if verbose:
        print("Training encoder...")
        start_time = time()
    train_encoder(encoder, data, encoder_batch, device, n_workers)
    if verbose:
        print(f"Done! Elapsed time: {str(timedelta(seconds=time() - start_time)).split('.', 2)[0]}")
        print("Training classifier...")
    train_classifier(encoder, classifier, data, n_epochs, classifier_batch, device, n_workers, verbose, validation_data,
                     val_batch)
    if verbose:
        print(f"Done! Elapsed time: {str(timedelta(seconds=time() - start_time)).split('.', 2)[0]}")

    # Combine encoder and classifier.
    model = HebbNet(encoder, classifier)
    return model


def train_encoder(encoder: Module, data: Dataset, batch_size: int, device, n_workers=0):
    # Setup dataloader.
    loader = DataLoader(data, batch_size=batch_size, num_workers=n_workers, drop_last=True)

    encoder.train()
    with autocast(device_type=device, dtype=torch.float16):
        for i, data in enumerate(loader):
            # Run data through encoder for unsupervised Hebbian learning.
            x, _ = data
            encoder(x)


def train_classifier(encoder: Module, classifier: Module, data: Dataset, n_epochs: int, batch_size: int, device,
                     n_workers=0, verbose=True, validation_data=None, val_batch=64):
    # Setup dataloader.
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=True)

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(classifier.parameters())
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = GradScaler()

    encoder.eval()
    classifier.train()
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

        scheduler.step()
        if verbose and (epoch == 0 or (epoch + 1) % (n_epochs // 10) == 0):
            # Print and reset loss and accuracy.
            loss = cumulative_loss / batches
            acc = float(n_correct) / samples * 100
            if validation_data is None:
                print(f"- Epoch {epoch + 1}: train loss {loss:.4f}, train accuracy {acc:.2f}%")
            else:
                val_acc = test(HebbNet(encoder, classifier), validation_data, val_batch, n_workers)
                print(f"- Epoch {epoch + 1}: train loss {loss:.4f}, train accuracy {acc:.2f}%, validation accuracy "
                      f"{val_acc:.2f}%")
            cumulative_loss = 0
            n_correct = 0
            batches = 0
            samples = 0


def test(model: Module, data: Dataset, batch_size: int, n_workers=0):
    # Setup dataloader.
    loader = DataLoader(data, batch_size=batch_size, num_workers=n_workers)

    n_correct = 0
    samples = 0
    for i, data in enumerate(loader):
        # Run batch through model and get number of correct predictions.
        x, y = data
        logits = model(x)
        n_correct += torch.sum(torch.argmax(logits, dim=1) == y).item()
        samples += len(y)
    return float(n_correct) / samples * 100


if __name__ == '__main__':
    # For command-line use.
    parser = ArgumentParser()
    parser.add_argument('--model', choices=['SoftHebbSmall'], help="The model to be trained.")
    parser.add_argument('--dataset', choices=['CIFAR10'], help="The dataset to train and test on.")
    parser.add_argument('--n_epochs', type=int, help="The number of epochs for SGD training.")
    parser.add_argument('--hebb_batch', type=int, help="The batch size for Hebbian training.")
    parser.add_argument('--sgd_batch', type=int, help="The batch size for SGD training.")
    parser.add_argument('--n_workers', default=0, type=int,
                        help="The batch size for SGD training (default: 0, i.e.., run in the main process).")
    args = parser.parse_args()

    # Load data.
    if args.dataset == 'CIFAR10':
        initial_size = 32
        n_classes = 10
    else:
        raise RuntimeError(f"Dataset {args.dataset} not found (internal error).")

    train_data, test_data = load(args.dataset)

    # Get setup.
    if args.model == 'SoftHebbSmall':
        encoder_net = SoftHebbSmall()
        in_features = int((initial_size / (2 ** 3)) ** 2 * 1536)
        classifier_net = Classifier(in_features=in_features, out_features=n_classes)
    else:
        raise RuntimeError(f"Model {args.model} not found (internal error).")

    trained_model = train(encoder_net, classifier_net, train_data, args.n_epochs, args.hebb_batch, args.sgd_batch,
                          n_workers=args.n_workers, validation_data=test_data)
    test_acc = test(trained_model, test_data, args.sgd_batch, args.n_workers)
    print(f"Test accuracy: {test_acc:.2f}")
