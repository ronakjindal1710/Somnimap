"""Training & evaluation runner for Phase 3 temporal models (20-epoch windows)."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import SomniMapSequenceDataset
from src.models.temporal import build_temporal_model
from src.training.losses import compute_class_weights_from_records
from src.training.metrics import compute_sleep_metrics, flatten_metrics_row
from src.training.mlflow_utils import log_run_to_mlflow, save_confusion_matrix_plot


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate_temporal_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y_stage"].numpy().reshape(-1)
        logits = model(x)  # [B, T, 5]
        probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1, 5)
        preds = np.argmax(probs, axis=-1)
        all_y.append(y)
        all_pred.append(preds)
        all_prob.append(probs)
    yt = np.concatenate(all_y)
    yp = np.concatenate(all_pred)
    pr = np.vstack(all_prob)
    return compute_sleep_metrics(yt, yp, probs=pr)


def train_one_temporal_model(
    model_name: str,
    cfg: dict,
    window_epochs: int = 20,
    train_stride: int = 10,
    val_stride: int = 20,
    max_epochs: int = 18,
    batch_size: int = 16,
    lr: float = 5e-4,
    patience: int = 5,
) -> dict:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = Path(cfg["paths"]["processed_dir"])
    proj_root = Path(cfg["paths"]["project_root"])
    split_data = json.loads(Path(cfg["paths"]["split_manifest_json"]).read_text(encoding="utf-8"))

    train_ids = split_data["partitions"]["train"]["record_ids"]
    val_ids = split_data["partitions"]["val"]["record_ids"]

    train_ds = SomniMapSequenceDataset(proc_dir, train_ids, window_epochs=window_epochs, stride_epochs=train_stride, mode="single_eeg")
    val_ds = SomniMapSequenceDataset(proc_dir, val_ids, window_epochs=window_epochs, stride_epochs=val_stride, mode="single_eeg")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = torch.tensor(
        compute_class_weights_from_records(proc_dir, train_ids), dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    model = build_temporal_model(model_name, in_channels=1, n_classes=5).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    ckpt_dir = proj_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"phase3_{model_name}_w{window_epochs}_best.pt"

    best_f1 = -1.0
    best_metrics: dict = {}
    no_improve = 0
    t0 = time.time()

    print(f"\n=== Training {model_name} (win={window_epochs}) | params={n_params:,} | train_wins={len(train_ds)} | val_wins={len(val_ds)} ===", flush=True)
    for ep in range(1, max_epochs + 1):
        ep_t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y_stage"].to(device)  # [B, T]
            optimizer.zero_grad()
            logits = model(x)  # [B, T, 5]
            loss = criterion(logits.reshape(-1, 5), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()

        val_metrics = evaluate_temporal_model(model, val_loader, device)
        val_f1 = val_metrics["macro_f1"]
        print(
            f"  Ep {ep:02d}/{max_epochs} | loss={total_loss/max(1,n_batches):.4f} "
            f"| val_macro_f1={val_f1:.4f} | kappa={val_metrics['cohen_kappa']:.4f} "
            f"| N1_f1={val_metrics['per_class']['N1']['f1']:.4f} ({time.time()-ep_t0:.1f}s)",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = val_metrics
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "val_metrics": val_metrics, "epoch": ep}, best_ckpt)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {ep} (best val_macro_f1={best_f1:.4f})", flush=True)
                break

    elapsed = time.time() - t0
    cm_path = proj_root / "results" / "confusion_matrices" / f"{model_name}_w{window_epochs}_val.png"
    save_confusion_matrix_plot(best_metrics["confusion_matrix"], f"{model_name} (win={window_epochs}, Val)", cm_path)
    log_run_to_mlflow(
        tracking_uri=cfg["paths"]["mlflow_tracking_uri"],
        experiment_name="SomniMap_Phase3_Temporal",
        run_name=f"{model_name}_w{window_epochs}",
        params={
            "model": model_name,
            "mode": "single_eeg",
            "window_epochs": window_epochs,
            "train_stride": train_stride,
            "lr": lr,
            "batch_size": batch_size,
            "n_params": n_params,
        },
        metrics=best_metrics,
        cm_path=cm_path,
    )
    return flatten_metrics_row(
        model_name,
        "val",
        best_metrics,
        extra={"window_epochs": window_epochs, "train_time_s": round(elapsed, 1), "params": n_params},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["CNN_BiLSTM", "CNN_TCN", "CNN_Transformer", "TemporalUNet"])
    parser.add_argument("--window_epochs", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=18)
    args = parser.parse_args()

    from src.config import load_config
    cfg = load_config()

    res_csv = Path(cfg["paths"]["project_root"]) / "results" / "phase3_temporal.csv"
    existing = pd.read_csv(res_csv) if res_csv.exists() else pd.DataFrame()

    rows = []
    for m in args.models:
        row = train_one_temporal_model(m, cfg, window_epochs=args.window_epochs, max_epochs=args.max_epochs)
        rows.append(row)

    new_df = pd.DataFrame(rows)
    if not existing.empty:
        existing = existing[~((existing["model"].isin(new_df["model"])) & (existing["window_epochs"] == args.window_epochs))]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(res_csv, index=False)
    print("\n=== Phase 3 Temporal Models Table ===")
    print(combined[["model", "window_epochs", "macro_f1", "balanced_acc", "cohen_kappa", "N1_f1", "Wake_f1", "N2_f1", "N3_f1", "REM_f1", "params"]].to_string(index=False))


if __name__ == "__main__":
    main()
