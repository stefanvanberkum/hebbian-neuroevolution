"""This module provides methods for model training.

Allows for command-line use.

Functions
=========
- :func:`train`: Train a model.
"""

from argparse import ArgumentParser

import torch
from torch import Module, autocast
from torch.utils.data import DataLoader, Dataset

from dataloader import load
from models import Classifier, HebbNet, SoftHebbSmall


def train(encoder: Module, classifier: Module, train: Dataset, test: Dataset, n_epochs: int, encoder_batch: int,
          classifier_batch: int, num_workers=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder.to(device)
    classifier.to(device)

    with autocast(device_type=device, dtype=torch.float16):
        train_encoder(encoder, train, encoder_batch, num_workers)

    # Combine encoder and classifier and return the trained model.
    model = HebbNet(encoder, classifier)
    model.to('cpu')
    return model


def train_encoder(encoder: Module, data: Dataset, batch_size: int, num_workers=0):
    # Setup dataloader.
    loader = DataLoader(data, batch_size=batch_size, num_workers=num_workers, drop_last=True)

    encoder.train()
    for i, data in enumerate(loader):
        inputs, _ = data
        encoder.forward(inputs)
    encoder.eval()


def train_classifier(encoder: Module, classifier: Module, data: Dataset, n_epochs: int, batch_size: int, num_workers=0):
    # Setup dataloader.
    data = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


if __name__ == '__main__':
    # For command-line use.
    parser = ArgumentParser()
    parser.add_argument('--model', choices=['SoftHebbSmall'], help="The model to be trained.")
    parser.add_argument('--dataset', choices=['CIFAR10'], help="The dataset to train and test on.")
    parser.add_argument('--hebb_batch', type=int, help="The batch size for Hebbian training.")
    parser.add_argument('--sgd_batch', type=int, help="The batch size for SGD training.")
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
        in_features = (initial_size / 2 ** 3) ** 2 * 384
        classifier_net = Classifier(in_features=in_features, out_features=n_classes)
    else:
        raise RuntimeError(f"Model {args.model} not found (internal error).")

    trained_model = train(encoder_net, classifier_net, train_data, test_data, args.hebb_batch, args.sgd_batch)
