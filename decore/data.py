"""CIFAR-10 data module (native 32x32, CIFAR normalisation)."""
from __future__ import annotations

import os

import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_tf, test_tf


def build_loaders(data_root: str, batch_size: int, num_workers: int = 0):
    """Return (train_loader, test_loader). Downloads CIFAR-10 only if missing."""
    data_root = os.path.abspath(data_root)
    cached = os.path.isdir(os.path.join(data_root, "cifar-10-batches-py"))
    need_download = not cached

    train_tf, test_tf = build_transforms()
    train_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=need_download, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=need_download, transform=test_tf)

    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
