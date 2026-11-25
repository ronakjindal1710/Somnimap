"""Epoch-level deep learning baselines for Phase 2: 1D CNN, ResNet1D, InceptionTime."""

from __future__ import annotations

import torch
import torch.nn as nn


class CNN1DEpoch(nn.Module):
    """4-block 1D CNN baseline for 30-s single-EEG epochs [B, 1, 3000]."""

    def __init__(self, in_channels: int = 1, n_classes: int = 5, base_ch: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=25, stride=2, padding=12, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(base_ch, base_ch * 2, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(base_ch * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(base_ch * 4, base_ch * 4, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(base_ch * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ResBlock1D(nn.Module):
    """1D Residual block with optional downsampling."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 9, stride: int = 2) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet1DEpoch(nn.Module):
    """1D ResNet baseline for 30-s epochs [B, 1, 3000]."""

    def __init__(self, in_channels: int = 1, n_classes: int = 5, base_ch: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=25, stride=2, padding=12, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.layer1 = ResBlock1D(base_ch, base_ch, kernel_size=9, stride=2)
        self.layer2 = ResBlock1D(base_ch, base_ch * 2, kernel_size=9, stride=2)
        self.layer3 = ResBlock1D(base_ch * 2, base_ch * 4, kernel_size=7, stride=2)
        self.layer4 = ResBlock1D(base_ch * 4, base_ch * 4, kernel_size=5, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return self.head(self.pool(h))


class InceptionModule1D(nn.Module):
    """Multi-scale 1D Inception module with parallel kernel sizes."""

    def __init__(self, in_ch: int, branch_ch: int = 24, kernels: tuple[int, ...] = (9, 19, 39)) -> None:
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, branch_ch, kernel_size=1, bias=False) if in_ch > 1 else nn.Identity()
        b_in = branch_ch if in_ch > 1 else in_ch
        self.branches = nn.ModuleList([
            nn.Conv1d(b_in, branch_ch, kernel_size=k, padding=k // 2, bias=False)
            for k in kernels
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, branch_ch, kernel_size=1, bias=False),
        )
        out_ch = branch_ch * (len(kernels) + 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        outs = [conv(b) for conv in self.branches] + [self.pool_branch(x)]
        return self.relu(self.bn(torch.cat(outs, dim=1)))


class InceptionTimeEpoch(nn.Module):
    """Multi-scale InceptionTime baseline for 30-s epochs [B, 1, 3000]."""

    def __init__(self, in_channels: int = 1, n_classes: int = 5, branch_ch: int = 24, dropout: float = 0.3) -> None:
        super().__init__()
        out_ch = branch_ch * 4  # 96
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.inc1 = InceptionModule1D(32, branch_ch=branch_ch)
        self.pool1 = nn.MaxPool1d(4)
        self.inc2 = InceptionModule1D(out_ch, branch_ch=branch_ch)
        self.pool2 = nn.MaxPool1d(4)
        self.inc3 = InceptionModule1D(out_ch, branch_ch=branch_ch)
        self.skip = nn.Sequential(
            nn.Conv1d(32, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(out_ch, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.stem(x)
        h = self.pool1(self.inc1(h0))
        h = self.pool2(self.inc2(h))
        h = self.inc3(h) + nn.functional.adaptive_avg_pool1d(self.skip(h0), h.shape[-1])
        return self.head(self.gap(h))


def build_epoch_baseline(name: str, in_channels: int = 1, n_classes: int = 5) -> nn.Module:
    if name == "CNN1D":
        return CNN1DEpoch(in_channels=in_channels, n_classes=n_classes)
    if name == "ResNet1D":
        return ResNet1DEpoch(in_channels=in_channels, n_classes=n_classes)
    if name == "InceptionTime":
        return InceptionTimeEpoch(in_channels=in_channels, n_classes=n_classes)
    raise ValueError(f"Unknown epoch baseline: {name}")
