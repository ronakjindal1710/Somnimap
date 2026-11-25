"""Phase 1 spot-check and zero-leakage verification script.

Verifies:
1. All 108 records in split_manifest.json have valid processed arrays without NaN/Inf.
2. Zero subject/record overlap across train / val / test_locked.
3. Saves visual spot-check plots (hypnogram + 30-s EEG/EOG/EMG + CAP overlay)
   for 5 randomly sampled recordings into results/spot_checks/.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


STAGE_LABELS = ["Wake", "N1", "N2", "N3", "REM"]


def run_spot_checks(config_path: Path, n_plots: int = 5, seed: int = 42) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    proc_dir = Path(cfg["paths"]["processed_dir"])
    split_data = json.loads(Path(cfg["paths"]["split_manifest_json"]).read_text(encoding="utf-8"))
    out_plots = Path(cfg["paths"]["project_root"]) / "results" / "spot_checks"
    out_plots.mkdir(parents=True, exist_ok=True)

    # 1. Verify zero overlap and completeness
    seen: dict[str, str] = {}
    all_records: list[str] = []
    nan_errors: list[str] = []
    shape_errors: list[str] = []

    for part_name, p_info in split_data["partitions"].items():
        for rec_id in p_info["record_ids"]:
            assert rec_id not in seen, f"LEAKAGE DETECTED: {rec_id} in {seen[rec_id]} and {part_name}!"
            seen[rec_id] = part_name
            all_records.append(rec_id)

            rec_dir = proc_dir / rec_id
            meta = json.loads((rec_dir / "metadata.json").read_text(encoding="utf-8"))
            assert meta["split"] == part_name, f"Split mismatch for {rec_id}: {meta['split']} != {part_name}"

            epochs = np.load(rec_dir / "epochs.npy", mmap_mode="r")
            y_stage = np.load(rec_dir / "labels_stage.npy")
            y_cap = np.load(rec_dir / "labels_cap.npy")

            if epochs.ndim != 3 or epochs.shape[1:] != (3000, 7):
                shape_errors.append(f"{rec_id}: epochs shape {epochs.shape}")
            if len(y_stage) != epochs.shape[0] or y_cap.shape != (epochs.shape[0], 300):
                shape_errors.append(f"{rec_id}: label shape mismatch")
            if not np.isfinite(epochs).all():
                nan_errors.append(rec_id)

    print(f"Verified {len(all_records)}/108 records | Zero split leakage confirmed.")
    print(f"Shape errors: {len(shape_errors)} | NaN/Inf errors: {len(nan_errors)}")
    if shape_errors:
        print("  Shape errors:", shape_errors[:5])
    if nan_errors:
        print("  NaN errors:", nan_errors[:5])

    # 2. Visual spot-checks on 5 random subjects
    rng = random.Random(seed)
    sample_ids = rng.sample(all_records, min(n_plots, len(all_records)))
    for rec_id in sample_ids:
        rec_dir = proc_dir / rec_id
        epochs = np.load(rec_dir / "epochs.npy", mmap_mode="r")
        y_stage = np.load(rec_dir / "labels_stage.npy")
        y_cap = np.load(rec_dir / "labels_cap.npy")
        meta = json.loads((rec_dir / "metadata.json").read_text(encoding="utf-8"))

        # Pick primary EEG channel (C4-A1 if present, else C3-A2)
        ch_idx = 0 if meta["channel_mask"][0] == 1 else 6
        ch_name = meta["canonical_channels"][ch_idx]

        # Find an N2/N3 epoch with CAP activity if available
        cap_ep_indices = np.where((y_cap > 0).any(axis=1))[0]
        ep_idx = int(cap_ep_indices[len(cap_ep_indices) // 2]) if len(cap_ep_indices) > 0 else len(y_stage) // 2

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 1.4]})

        # Top: Whole-night hypnogram + CAP density
        t_hours = np.arange(len(y_stage)) * 30.0 / 3600.0
        y_plot = np.where(y_stage >= 0, y_stage, np.nan)
        axes[0].step(t_hours, y_plot, where="post", color="#1f77b4", linewidth=1.2)
        axes[0].set_yticks([0, 1, 2, 3, 4])
        axes[0].set_yticklabels(STAGE_LABELS)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Time (hours)")
        axes[0].set_title(f"{rec_id} ({meta['split']}) — Hypnogram (valid={meta['n_valid_stage_epochs']}/{meta['n_epochs']})")
        axes[0].grid(alpha=0.3)

        # Bottom: 30-s EEG waveform + CAP overlay
        t_sec = np.arange(3000) / 100.0
        sig = np.array(epochs[ep_idx, :, ch_idx])
        axes[1].plot(t_sec, sig, color="#222222", linewidth=0.7, label=ch_name)
        cap_mask_10hz = y_cap[ep_idx]
        if (cap_mask_10hz > 0).any():
            cap_up = np.repeat(cap_mask_10hz, 10)
            axes[1].fill_between(t_sec, sig.min(), sig.max(), where=cap_up > 0, color="orange", alpha=0.3, label="CAP A-phase")
        st_lbl = STAGE_LABELS[y_stage[ep_idx]] if y_stage[ep_idx] >= 0 else "Masked"
        axes[1].set_title(f"Epoch {ep_idx} (Stage={st_lbl}) — Normalized {ch_name}")
        axes[1].set_xlabel("Seconds")
        axes[1].set_ylabel("Robust z-score")
        axes[1].legend(loc="upper right")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_plots / f"spot_check_{rec_id}.png", dpi=150)
        plt.close(fig)

    print(f"Saved {len(sample_ids)} spot-check plots to {out_plots}: {sample_ids}")


if __name__ == "__main__":
    cfg_path = Path("C:/Users/prave/.gemini/antigravity/scratch/SomniMap/experiments/configs/base_config.yaml")
    run_spot_checks(cfg_path)
