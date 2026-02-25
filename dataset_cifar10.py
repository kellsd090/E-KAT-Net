import torch
import torch.nn as nn
import numpy as np
import os
import tarfile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from timm.data.mixup import Mixup  # 推荐使用 timm 库处理 Mixup 和 Cutmix


def get_dataloader(data_dir=r'E:\Python\PythonProject\dataset', batch_size=128):
    """
    (1) 读取、解压并加载 CIFAR-10
    (2) 标准化 (Normalization)
    (3) 数据增强 (Augmentation)
    (4) 坐标索引 (Coordinate Embedding)
    """

    # --- 1. 标准化参数 (CIFAR-10 官方统计值) ---
    # 均值和标准差控制在 0 均值左右，加速收敛
    norm_mean = [0.4914, 0.4822, 0.4465]
    norm_std = [0.2023, 0.1994, 0.2010]

    # --- 2. 完善的数据增强 Pipeline ---
    # 保持维度不变: [3, 32, 32]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 填充后裁剪，解决平移不变性
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ColorJitter(0.2, 0.2, 0.2),  # 亮度、对比度随机变换
        transforms.ToTensor(),  # 转为 [3, 32, 32] 并归一化到 [0, 1]
        transforms.Normalize(norm_mean, norm_std),  # 标准化
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))  # 随机擦除
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # --- 3. 加载数据集 ---
    # 自动处理 tar 文件解压与读取
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # --- 4. Mixup & Cutmix 配置 ---
    # 注意：Mixup 通常在训练循环中按 batch 处理，因为需要处理 Label
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=10
    )

    # --- 5. 构建 DataLoader ---
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
    """
    (4) 附加二维坐标信息 [可选]
    将 [B, 3, 32, 32] 变为 [B, 5, 32, 32]，增加 x, y 坐标通道
    """
    b, c, h, w = x.shape
    # 生成归一化坐标 (-1 到 1)
    grid_h = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(b, 1, h, w)
    grid_w = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
    # 拼接坐标通道
    return torch.cat([x, grid_h.to(x.device), grid_w.to(x.device)], dim=1)