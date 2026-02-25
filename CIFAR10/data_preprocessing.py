import torch
import torch.nn as nn
import numpy as np
import os
import tarfile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from timm.data.mixup import Mixup  


def get_dataloader(data_dir=r'E:\Python\PythonProject\dataset', batch_size=128):
    norm_mean = [0.4914, 0.4822, 0.4465]
    norm_std = [0.2023, 0.1994, 0.2010]
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Fill and then trim to address the translational invariance issue
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ColorJitter(0.2, 0.2, 0.2),  # Random changes in brightness and contrast
        transforms.ToTensor(),  # 
        transforms.Normalize(norm_mean, norm_std),  # 
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))  # Random erasure
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=10
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, test_loader, mixup_fn


def add_coordinate_encoding(x):
    b, c, h, w = x.shape
    grid_h = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(b, 1, h, w)
    grid_w = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([x, grid_h.to(x.device), grid_w.to(x.device)], dim=1)
