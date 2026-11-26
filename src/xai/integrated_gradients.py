"""Integrated Gradients (Captum) attribution wrapper for SomniMapNet."""

from __future__ import annotations

import numpy as np
import torch
from captum.attr import IntegratedGradients


class _StageTargetWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, ch_mask: torch.Tensor | None, epoch_idx: int) -> None:
        super().__init__()
        self.model = model
        self.ch_mask = ch_mask
        self.epoch_idx = epoch_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cm = None
        if self.ch_mask is not None:
            cm = self.ch_mask.expand(x.shape[0], -1)
        return self.model(x, ch_mask=cm)["stage_logits"][:, self.epoch_idx, :]


def compute_integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    ch_mask: torch.Tensor | None,
    epoch_idx: int,
    target_class: int,
    n_steps: int = 24,
) -> dict[str, np.ndarray]:
    """Compute IG attributions [C, 3000] for target epoch & class, plus per-channel importance."""
    model.eval()
    wrapper = _StageTargetWrapper(model, ch_mask, epoch_idx)
    ig = IntegratedGradients(wrapper)
    baseline = torch.zeros_like(x)
    attr = ig.attribute(x, baselines=baseline, target=target_class, n_steps=n_steps, internal_batch_size=4)
    ep_attr = attr[0, epoch_idx].detach().cpu().numpy()  # [C, 3000]
    ch_abs = np.abs(ep_attr).sum(axis=-1)
    ch_share = ch_abs / max(1e-9, ch_abs.sum())
    return {
        "epoch_attr": ep_attr,
        "channel_importance": ch_share,
    }
