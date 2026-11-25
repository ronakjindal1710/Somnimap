"""MLflow logging and confusion matrix plotting helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from src.training.metrics import STAGE_NAMES


def save_confusion_matrix_plot(cm: list[list[int]], title: str, out_path: Path) -> None:
    """Save normalised + raw count confusion matrix PNG."""
    cm_arr = np.array(cm, dtype=np.float64)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_arr, np.maximum(1.0, row_sums))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(STAGE_NAMES)
    ax.set_yticklabels(STAGE_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(5):
        for j in range(5):
            val = cm_norm[i, j]
            cnt = int(cm_arr[i, j])
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val*100:.1f}%\n({cnt})", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def log_run_to_mlflow(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    cm_path: Path | None = None,
) -> None:
    """Log experiment run, metrics, and confusion matrix artifact to MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        scalar_metrics = {
            "val_macro_f1": metrics["macro_f1"],
            "val_balanced_acc": metrics["balanced_accuracy"],
            "val_cohen_kappa": metrics["cohen_kappa"],
            "val_weighted_f1": metrics["weighted_f1"],
            "val_accuracy": metrics["accuracy"],
        }
        if "ece" in metrics:
            scalar_metrics["val_ece"] = metrics["ece"]
            scalar_metrics["val_brier"] = metrics["brier_score"]
        for st in STAGE_NAMES:
            scalar_metrics[f"val_{st}_f1"] = metrics["per_class"][st]["f1"]
            scalar_metrics[f"val_{st}_sens"] = metrics["per_class"][st]["sensitivity"]
        mlflow.log_metrics(scalar_metrics)
        if cm_path is not None and cm_path.exists():
            mlflow.log_artifact(str(cm_path))
