"""Classical ML baselines (Random Forest & RBF-Nystroem SVM) on single-EEG features.

Extracts vectorized time- and frequency-domain features per 30-s epoch for train
and val subjects only (never touching test_locked), fits scalers strictly on
training data, logs to MLflow, and exports results to CSV.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.metrics import compute_sleep_metrics, flatten_metrics_row
from src.training.mlflow_utils import log_run_to_mlflow, save_confusion_matrix_plot


BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("sigma", 12.0, 15.0),
    ("beta", 13.0, 30.0),
]


def extract_epoch_features(epochs_1ch: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Extract 24 time + frequency features from [N, 3000] single-channel epochs."""
    x = epochs_1ch.astype(np.float64, copy=False)
    mean = np.mean(x, axis=1)
    std = np.std(x, axis=1) + 1e-8
    vmin = np.min(x, axis=1)
    vmax = np.max(x, axis=1)
    rms = np.sqrt(np.mean(x ** 2, axis=1))
    skew = np.nan_to_num(stats.skew(x, axis=1))
    kurt = np.nan_to_num(stats.kurtosis(x, axis=1))
    zcr = np.mean((x[:, 1:] * x[:, :-1]) < 0, axis=1)

    # Hjorth parameters
    dx = np.diff(x, axis=1)
    ddx = np.diff(dx, axis=1)
    var_x = np.var(x, axis=1) + 1e-8
    var_dx = np.var(dx, axis=1) + 1e-8
    var_ddx = np.var(ddx, axis=1) + 1e-8
    hj_mob = np.sqrt(var_dx / var_x)
    hj_comp = np.sqrt(var_ddx / var_dx) / (hj_mob + 1e-8)

    # Welch PSD
    freqs, psd = signal.welch(x, fs=fs, nperseg=256, noverlap=128, axis=1)
    df = freqs[1] - freqs[0]
    total_pow = np.sum(psd, axis=1) * df + 1e-12

    band_feats = []
    for _, f_lo, f_hi in BANDS:
        idx = (freqs >= f_lo) & (freqs < f_hi)
        bp = np.sum(psd[:, idx], axis=1) * df
        band_feats.append(np.log1p(bp))
        band_feats.append(bp / total_pow)

    # Spectral entropy & 95% spectral edge frequency
    p_norm = psd / (np.sum(psd, axis=1, keepdims=True) + 1e-12)
    spec_ent = -np.sum(p_norm * np.log(p_norm + 1e-12), axis=1) / np.log(psd.shape[1])
    cdf = np.cumsum(p_norm, axis=1)
    sef95 = freqs[np.argmax(cdf >= 0.95, axis=1)]

    feats = np.column_stack([
        mean, std, vmin, vmax, rms, skew, kurt, zcr,
        var_x, hj_mob, hj_comp,
        *band_feats,
        np.log1p(total_pow), spec_ent, sef95,
    ])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_or_extract_split(
    processed_dir: Path,
    record_ids: list[str],
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists():
        data = np.load(cache_path)
        return data["X"], data["y"]

    xs, ys = [], []
    for rec_id in record_ids:
        rec_dir = processed_dir / rec_id
        meta = json.loads((rec_dir / "metadata.json").read_text(encoding="utf-8"))
        y = np.load(rec_dir / "labels_stage.npy")
        valid = y >= 0
        if not valid.any():
            continue
        ch_idx = 0 if meta["channel_mask"][0] == 1 else 6
        epochs = np.load(rec_dir / "epochs.npy", mmap_mode="r")
        x_valid = np.array(epochs[valid, :, ch_idx])
        xs.append(extract_epoch_features(x_valid))
        ys.append(y[valid])

    X = np.vstack(xs)
    y_all = np.concatenate(ys)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y_all)
    return X, y_all


def run_classical_baselines(config_path: Path) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    proc_dir = Path(cfg["paths"]["processed_dir"])
    proj_root = Path(cfg["paths"]["project_root"])
    split_data = json.loads(Path(cfg["paths"]["split_manifest_json"]).read_text(encoding="utf-8"))
    feat_dir = proj_root / "data" / "features"
    res_dir = proj_root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    X_tr, y_tr = load_or_extract_split(
        proc_dir, split_data["partitions"]["train"]["record_ids"], feat_dir / "train_single_eeg.npz"
    )
    X_va, y_va = load_or_extract_split(
        proc_dir, split_data["partitions"]["val"]["record_ids"], feat_dir / "val_single_eeg.npz"
    )
    print(f"Features ready in {time.time() - t0:.1f}s | Train={X_tr.shape}, Val={X_va.shape}", flush=True)

    models = {
        "RandomForest_300": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "SVM_RBF_Nystroem": Pipeline([
            ("scaler", StandardScaler()),
            ("nystroem", Nystroem(kernel="rbf", n_components=400, random_state=42)),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=500,
                random_state=42,
            )),
        ]),
    }

    rows = []
    for name, pipe in models.items():
        m_t0 = time.time()
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_va)
        probs = pipe.predict_proba(X_va)
        metrics = compute_sleep_metrics(y_va, y_pred, probs=probs)
        elapsed = time.time() - m_t0

        cm_path = res_dir / "confusion_matrices" / f"{name}_val.png"
        save_confusion_matrix_plot(metrics["confusion_matrix"], f"{name} (Val)", cm_path)
        log_run_to_mlflow(
            tracking_uri=cfg["paths"]["mlflow_tracking_uri"],
            experiment_name="SomniMap_Phase2_Baselines",
            run_name=name,
            params={"model": name, "features": "24_time_freq", "mode": "single_eeg"},
            metrics=metrics,
            cm_path=cm_path,
        )

        row = flatten_metrics_row(name, "val", metrics, extra={"train_time_s": round(elapsed, 1), "params": "classical"})
        rows.append(row)
        print(
            f"[{name}] val macro_f1={metrics['macro_f1']:.4f} | bal_acc={metrics['balanced_accuracy']:.4f} "
            f"| kappa={metrics['cohen_kappa']:.4f} | N1_f1={metrics['per_class']['N1']['f1']:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

    csv_path = res_dir / "phase2_baselines.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved classical baseline results to {csv_path}")


if __name__ == "__main__":
    cfg_path = Path("C:/Users/prave/.gemini/antigravity/scratch/SomniMap/experiments/configs/base_config.yaml")
    run_classical_baselines(cfg_path)
