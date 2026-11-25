"""Phase 4 — SomniMapNet training script.

Ablations supported via CLI flags:
  --context_type   bilstm | tcn | transformer | none   (E3: bottleneck choice)
  --use_cap_head   store_true                           (E8: stage-only vs dual-head)
  --multi_channel  store_true                           (E5: multi-EEG mode)
  --loss_type      ce | focal | weighted_ce             (E6: loss ablation)
  --no_chan_attn   store_true                           (E7: without channel attention)
  --window_epochs  int (default 20)                    (E3-window: context size ablation)

MLflow experiment: SomniMap_Phase4_Flagship
Checkpoints: checkpoints/phase4_SomniMapNet_<context>_<suffix>_best.pt
"""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import SomniMapSequenceDataset
from src.models.somnimap_net import SomniMapNet
from src.training.losses import compute_class_weights_from_records
from src.training.metrics import compute_sleep_metrics, flatten_metrics_row
from src.training.mlflow_utils import log_run_to_mlflow, save_confusion_matrix_plot


# ---------------------------------------------------------------------------
# Focal loss helper
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, ignore_index: int = -1) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        mask = targets != self.ignore_index
        return loss[mask].mean() if mask.any() else loss.mean()


# ---------------------------------------------------------------------------
# CAP loss (masked BCE over NREM frames only)
# ---------------------------------------------------------------------------

NREM_STAGES = {1, 2, 3}  # stage indices: 0=Wake,1=N1,2=N2,3=N3,4=REM


def cap_loss_masked(cap_logits: torch.Tensor, cap_labels: torch.Tensor, stage_labels: torch.Tensor) -> torch.Tensor:
    """Binary multi-class CAP loss, masked to NREM epochs only.

    cap_logits: [B, T_ep, 300, 4]
    cap_labels: [B, T_ep, 300]  (0=none, 1=A1, 2=A2, 3=A3)
    stage_labels: [B, T_ep]     (stage indices)
    """
    b, t, f, nc = cap_logits.shape
    # NREM mask: [B, T_ep] -> [B, T_ep, 1, 1] broadcast
    nrem_mask = sum((stage_labels == s) for s in NREM_STAGES)  # [B, T_ep]
    nrem_mask = nrem_mask.bool().unsqueeze(-1).unsqueeze(-1)  # [B, T_ep, 1, 1]

    # One-hot encode cap labels
    cap_oh = F.one_hot(cap_labels.clamp(0), num_classes=nc).float()  # [B, T, 300, 4]

    loss = F.binary_cross_entropy_with_logits(cap_logits, cap_oh, reduction="none")  # [B, T, 300, 4]
    # Zero out non-NREM epochs
    loss = loss * nrem_mask.float()
    n_valid = nrem_mask.float().sum() * f * nc
    return loss.sum() / (n_valid + 1e-8)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_somnimap(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        ch_mask = batch["ch_mask"].to(device)
        y = batch["y_stage"].numpy().reshape(-1)
        out = model(x, ch_mask=ch_mask)
        logits = out["stage_logits"]  # [B, T, 5]
        probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1, 5)
        preds = np.argmax(probs, axis=-1)
        all_y.append(y)
        all_pred.append(preds)
        all_prob.append(probs)
    yt = np.concatenate(all_y)
    yp = np.concatenate(all_pred)
    pr = np.vstack(all_prob)
    return compute_sleep_metrics(yt, yp, probs=pr)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_somnimap(
    context_type: str = "bilstm",
    use_cap_head: bool = True,
    multi_channel: bool = False,
    loss_type: str = "weighted_ce",
    no_chan_attn: bool = False,
    window_epochs: int = 20,
    max_epochs: int = 20,
    batch_size: int = 8,
    lr: float = 3e-4,
    patience: int = 5,
    cap_weight: float = 0.3,
    run_suffix: str = "",
    cfg: dict | None = None,
) -> dict:
    if cfg is None:
        cfg_path = Path("C:/Users/prave/.gemini/antigravity/scratch/SomniMap/experiments/configs/base_config.yaml")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proc_dir = Path(cfg["paths"]["processed_dir"])
    proj_root = Path(cfg["paths"]["project_root"])
    split_data = json.loads(Path(cfg["paths"]["split_manifest_json"]).read_text(encoding="utf-8"))
    train_ids = split_data["partitions"]["train"]["record_ids"]
    val_ids = split_data["partitions"]["val"]["record_ids"]

    mode = "multi_channel" if multi_channel else "single_eeg"
    train_ds = SomniMapSequenceDataset(
        proc_dir, train_ids, window_epochs=window_epochs, stride_epochs=window_epochs // 2, mode=mode
    )
    val_ds = SomniMapSequenceDataset(
        proc_dir, val_ids, window_epochs=window_epochs, stride_epochs=window_epochs, mode=mode
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Determine in_channels
    sample = train_ds[0]
    in_ch = sample["x"].shape[1]  # [T_ep, C, 3000] -> C

    # Build model (optionally disable channel attention for E7)
    model = SomniMapNet(
        in_channels=in_ch,
        context_type=context_type,
        use_cap_head=use_cap_head,
        dropout=0.25,
    )
    if no_chan_attn and model.chan_attn is not None:
        model.chan_attn = None  # ablate channel attention
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Build loss
    cw = torch.tensor(compute_class_weights_from_records(proc_dir, train_ids), dtype=torch.float32, device=device)
    if loss_type == "focal":
        stage_criterion = FocalLoss(gamma=2.0, weight=cw, ignore_index=-1)
    else:  # weighted_ce or ce
        stage_criterion = nn.CrossEntropyLoss(
            weight=cw if loss_type == "weighted_ce" else None,
            ignore_index=-1,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    run_name = f"SomniMapNet_{context_type}_w{window_epochs}"
    if run_suffix:
        run_name += f"_{run_suffix}"
    ckpt_path = proj_root / "checkpoints" / f"phase4_{run_name}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1, best_metrics, no_improve = -1.0, {}, 0
    t0 = time.time()
    has_cap = use_cap_head and "y_cap" in train_ds[0]

    print(
        f"\n=== SomniMapNet | ctx={context_type} | cap={use_cap_head} | mode={mode} "
        f"| loss={loss_type} | chan_attn={not no_chan_attn} | w={window_epochs} "
        f"| params={n_params:,} | train={len(train_ds)} val={len(val_ds)} ===",
        flush=True,
    )

    for ep in range(1, max_epochs + 1):
        ep_t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch in train_loader:
            x = batch["x"].to(device)              # [B, T, C, 3000]
            ch_mask = batch["ch_mask"].to(device)  # [B, C]
            y_stage = batch["y_stage"].to(device)  # [B, T]

            optimizer.zero_grad()
            out = model(x, ch_mask=ch_mask)
            stage_logits = out["stage_logits"]     # [B, T, 5]

            loss = stage_criterion(stage_logits.reshape(-1, 5), y_stage.reshape(-1))

            if has_cap and "cap_logits" in out and "y_cap" in batch:
                y_cap = batch["y_cap"].to(device)  # [B, T, 300]
                c_loss = cap_loss_masked(out["cap_logits"], y_cap, y_stage)
                loss = loss + cap_weight * c_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        scheduler.step()

        val_metrics = evaluate_somnimap(model, val_loader, device)
        val_f1 = val_metrics["macro_f1"]
        print(
            f"  Ep {ep:02d}/{max_epochs} | loss={total_loss/max(1,n_batches):.4f} "
            f"| val_f1={val_f1:.4f} | kappa={val_metrics['cohen_kappa']:.4f} "
            f"| N1_f1={val_metrics['per_class']['N1']['f1']:.4f} ({time.time()-ep_t0:.1f}s)",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = val_metrics
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "val_metrics": val_metrics, "epoch": ep}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at ep {ep} (best val_f1={best_f1:.4f})", flush=True)
                break

    elapsed = time.time() - t0
    cm_path = proj_root / "results" / "confusion_matrices" / f"{run_name}_val.png"
    save_confusion_matrix_plot(best_metrics.get("confusion_matrix"), f"{run_name} Val", cm_path)
    log_run_to_mlflow(
        tracking_uri=cfg["paths"]["mlflow_tracking_uri"],
        experiment_name="SomniMap_Phase4_Flagship",
        run_name=run_name,
        params={
            "model": "SomniMapNet",
            "context_type": context_type,
            "use_cap_head": use_cap_head,
            "mode": mode,
            "loss_type": loss_type,
            "no_chan_attn": no_chan_attn,
            "window_epochs": window_epochs,
            "in_channels": in_ch,
            "n_params": n_params,
            "lr": lr,
            "batch_size": batch_size,
        },
        metrics=best_metrics,
        cm_path=cm_path,
    )
    return flatten_metrics_row(
        f"SomniMapNet_{context_type}",
        "val",
        best_metrics,
        extra={
            "window_epochs": window_epochs,
            "train_time_s": round(elapsed, 1),
            "params": n_params,
            "loss_type": loss_type,
            "chan_attn": not no_chan_attn,
            "use_cap_head": use_cap_head,
            "mode": mode,
            "run_name": run_name,
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SomniMapNet (Phase 4)")
    parser.add_argument("--context_type", default="bilstm", choices=["bilstm", "tcn", "transformer", "none"])
    parser.add_argument("--use_cap_head", action="store_true", default=False)
    parser.add_argument("--multi_channel", action="store_true", default=False)
    parser.add_argument("--loss_type", default="weighted_ce", choices=["ce", "focal", "weighted_ce"])
    parser.add_argument("--no_chan_attn", action="store_true", default=False)
    parser.add_argument("--window_epochs", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cap_weight", type=float, default=0.3)
    parser.add_argument("--run_suffix", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="")
    args = parser.parse_args()

    cfg_path = Path("C:/Users/prave/.gemini/antigravity/scratch/SomniMap/experiments/configs/base_config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    row = train_somnimap(
        context_type=args.context_type,
        use_cap_head=args.use_cap_head,
        multi_channel=args.multi_channel,
        loss_type=args.loss_type,
        no_chan_attn=args.no_chan_attn,
        window_epochs=args.window_epochs,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        cap_weight=args.cap_weight,
        run_suffix=args.run_suffix,
        cfg=cfg,
    )

    out_csv = args.out_csv or str(Path(cfg["paths"]["project_root"]) / "results" / "phase4_somnimap.csv")
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([row])
    if out_path.exists():
        existing = pd.read_csv(out_path)
        # Remove any existing row with same run_name
        if "run_name" in existing.columns and "run_name" in new_df.columns:
            existing = existing[existing["run_name"] != row.get("run_name", "")]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(out_path, index=False)

    print(f"\n=== Saved to {out_path} ===")
    disp_cols = ["model", "run_name", "macro_f1", "cohen_kappa", "N1_f1", "params", "loss_type", "chan_attn", "use_cap_head"]
    avail = [c for c in disp_cols if c in combined.columns]
    print(combined[avail].to_string(index=False))


if __name__ == "__main__":
    main()
