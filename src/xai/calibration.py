"""Post-hoc temperature scaling calibration and reliability diagram plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.metrics import compute_brier, compute_ece


class TemperatureScaler(nn.Module):
    """Single scalar temperature T > 0 fitted on validation logits via NLL."""

    def __init__(self, init_temp: float = 1.0) -> None:
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([np.log(init_temp)], dtype=torch.float32))

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temp).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.exp(self.log_temp)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        mask = labels >= 0
        lg = torch.tensor(logits[mask], dtype=torch.float32)
        yb = torch.tensor(labels[mask], dtype=torch.long)

        with torch.no_grad():
            nll_before = float(F.cross_entropy(lg, yb).item())
            probs_before = torch.softmax(lg, dim=-1).numpy()

        opt = torch.optim.LBFGS([self.log_temp], lr=0.05, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(lg), yb)
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            scaled = self.forward(lg)
            nll_after = float(F.cross_entropy(scaled, yb).item())
            probs_after = torch.softmax(scaled, dim=-1).numpy()

        y_np = yb.numpy()
        return {
            "temperature": round(self.temperature, 4),
            "nll_before": round(nll_before, 4),
            "nll_after": round(nll_after, 4),
            "ece_before": round(compute_ece(probs_before, y_np), 4),
            "ece_after": round(compute_ece(probs_after, y_np), 4),
            "brier_before": round(compute_brier(probs_before, y_np), 4),
            "brier_after": round(compute_brier(probs_after, y_np), 4),
        }


def save_reliability_diagram(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    y_true: np.ndarray,
    out_path: Path,
    n_bins: int = 10,
) -> None:
    mask = y_true >= 0
    yt = y_true[mask]

    def bin_stats(probs):
        conf = np.max(probs[mask], axis=1)
        pred = np.argmax(probs[mask], axis=1)
        corr = (pred == yt).astype(np.float64)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        accs, confs = [], []
        for i in range(n_bins):
            m = (conf > bins[i]) & (conf <= bins[i + 1]) if i > 0 else (conf >= bins[i]) & (conf <= bins[i + 1])
            accs.append(float(corr[m].mean()) if m.any() else np.nan)
            confs.append(float(conf[m].mean()) if m.any() else (bins[i] + bins[i + 1]) / 2.0)
        return np.array(confs), np.array(accs)

    c1, a1 = bin_stats(probs_before)
    c2, a2 = bin_stats(probs_after)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(c1, a1, "o-", label=f"Uncalibrated (ECE={compute_ece(probs_before[mask], yt):.4f})")
    ax.plot(c2, a2, "s-", label=f"Temp-Scaled (ECE={compute_ece(probs_after[mask], yt):.4f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram (Validation Split)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
