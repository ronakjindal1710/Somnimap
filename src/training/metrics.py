"""Evaluation metrics for five-class sleep-stage classification (Blueprint Sec. 8)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) for multi-class probabilities."""
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_brier(probs: np.ndarray, y_true: np.ndarray, n_classes: int = 5) -> float:
    """Multi-class Brier score."""
    one_hot = np.eye(n_classes, dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_sleep_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute all blueprint-required sleep-staging metrics."""
    mask = y_true >= 0
    yt = y_true[mask]
    yp = y_pred[mask]
    labels = list(range(5))

    prec, rec, f1, support = precision_recall_fscore_support(
        yt, yp, labels=labels, zero_division=0
    )
    cm = confusion_matrix(yt, yp, labels=labels)

    # Per-class specificity
    total = cm.sum()
    spec = []
    for i in range(5):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        spec.append(float(tn / max(1, tn + fp)))

    res: dict[str, Any] = {
        "n_samples": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(yt, yp)),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            STAGE_NAMES[i]: {
                "precision": float(prec[i]),
                "sensitivity": float(rec[i]),
                "specificity": spec[i],
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(5)
        },
    }

    if probs is not None:
        pr = probs[mask]
        res["ece"] = compute_ece(pr, yt)
        res["brier_score"] = compute_brier(pr, yt)

    return res


def flatten_metrics_row(model_name: str, split: str, metrics: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten metrics dict into a single CSV-friendly row."""
    row: dict[str, Any] = {
        "model": model_name,
        "split": split,
        "n_samples": metrics["n_samples"],
        "macro_f1": round(metrics["macro_f1"], 4),
        "balanced_acc": round(metrics["balanced_accuracy"], 4),
        "cohen_kappa": round(metrics["cohen_kappa"], 4),
        "weighted_f1": round(metrics["weighted_f1"], 4),
        "accuracy": round(metrics["accuracy"], 4),
        "ece": round(metrics["ece"], 4) if "ece" in metrics else None,
        "brier": round(metrics["brier_score"], 4) if "brier_score" in metrics else None,
    }
    for st in STAGE_NAMES:
        pc = metrics["per_class"][st]
        row[f"{st}_f1"] = round(pc["f1"], 4)
        row[f"{st}_sens"] = round(pc["sensitivity"], 4)
    if extra:
        row.update(extra)
    return row
