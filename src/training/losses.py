"""Loss functions and training-only class-weight computation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights_from_records(processed_dir, train_record_ids: list[str], n_classes: int = 5) -> np.ndarray:
    """Compute inverse-sqrt balanced class weights strictly from training subjects."""
    counts = np.zeros(n_classes, dtype=np.float64)
    for rec_id in train_record_ids:
        y = np.load(processed_dir / rec_id / "labels_stage.npy")
        valid = y[y >= 0]
        counts += np.bincount(valid, minlength=n_classes)
    # Smoothed inverse frequency (sqrt) avoids extreme gradient spikes on N1 (4.3%)
    freqs = counts / np.maximum(1.0, counts.sum())
    weights = 1.0 / np.sqrt(np.maximum(freqs, 1e-6))
    weights = weights / weights.mean()
    return weights.astype(np.float32)


class MaskedFocalLoss(nn.Module):
    """Multi-class Focal Loss with ignore_index=-1 and optional class weights."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, ignore_index: int = -1) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        logits_v = logits[valid]
        targets_v = targets[valid]
        ce = F.cross_entropy(logits_v, targets_v, weight=self.weight, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits_v, targets_v, reduction="none"))
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()
