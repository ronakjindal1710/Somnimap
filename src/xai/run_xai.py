"""Phase 6 — XAI (Grad-CAM 1D, Integrated Gradients) + Temperature Scaling Calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import SomniMapSequenceDataset
from src.models.somnimap_net import SomniMapNet
from src.training.metrics import STAGE_NAMES
from src.xai.calibration import TemperatureScaler, save_reliability_diagram
from src.xai.gradcam_1d import GradCAM1D
from src.xai.integrated_gradients import compute_integrated_gradients


CH_NAMES = ["EEG C4-A1", "EEG F4-C4", "EEG C4-P4", "EEG Fp2-F4", "EOG", "EMG Chin"]


def plot_attribution_case(
    x_ep: np.ndarray,
    cam: np.ndarray,
    ig_attr: np.ndarray,
    ch_imp: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    t = np.arange(x_ep.shape[-1]) / 100.0
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), gridspec_kw={"height_ratios": [2, 1.5, 1]})

    # Top: Primary EEG + Grad-CAM overlay
    eeg = x_ep[0]
    axes[0].plot(t, eeg, color="#1f77b4", linewidth=0.7, label="Primary EEG (norm)")
    ax_cam = axes[0].twinx()
    ax_cam.fill_between(t, 0, cam, color="crimson", alpha=0.30, label="Grad-CAM 1D")
    ax_cam.set_ylim(0, 1.1)
    ax_cam.set_ylabel("Grad-CAM", color="crimson")
    axes[0].set_title(title)
    axes[0].set_ylabel("Z-score")

    # Middle: Integrated Gradients on primary EEG
    ig_abs = np.abs(ig_attr[0])
    axes[1].plot(t, ig_attr[0], color="#2ca02c", linewidth=0.7, label="IG Attribution (Ch 0)")
    axes[1].fill_between(t, 0, ig_abs, color="#2ca02c", alpha=0.2)
    axes[1].set_ylabel("IG Score")
    axes[1].set_xlabel("Time within 30s epoch (s)")

    # Bottom: Channel importance bar chart
    names = CH_NAMES[:len(ch_imp)] if len(ch_imp) > 1 else ["Primary EEG"]
    axes[2].barh(names, ch_imp, color="#4c72b0")
    axes[2].set_xlim(0, 1.0)
    axes[2].set_xlabel("Channel Importance Share (|IG|)")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 XAI & Calibration")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--context_type", type=str, default="bilstm")
    parser.add_argument("--multi_channel", action="store_true", default=False)
    parser.add_argument("--use_cap_head", action="store_true", default=False)
    parser.add_argument("--window_epochs", type=int, default=20)
    args = parser.parse_args()

    cfg_path = Path("C:/Users/admin/Desktop/Somnimap/experiments/configs/base_config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    proc_dir = Path(cfg["paths"]["processed_dir"])
    proj_root = Path(cfg["paths"]["project_root"])
    split_data = json.loads(Path(cfg["paths"]["split_manifest_json"]).read_text(encoding="utf-8"))
    val_ids = split_data["partitions"]["val"]["record_ids"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "multi_channel" if args.multi_channel else "single_eeg"
    in_ch = 6 if args.multi_channel else 1

    model = SomniMapNet(
        in_channels=in_ch, context_type=args.context_type, use_cap_head=args.use_cap_head
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_ds = SomniMapSequenceDataset(
        proc_dir, val_ids, window_epochs=args.window_epochs, stride_epochs=args.window_epochs, mode=mode
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # 1. Collect validation logits for temperature scaling
    all_logits, all_y = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            cm = batch["ch_mask"].to(device)
            lg = model(x, ch_mask=cm)["stage_logits"].cpu().numpy().reshape(-1, 5)
            all_logits.append(lg)
            all_y.append(batch["y_stage"].numpy().reshape(-1))
    logits_np = np.vstack(all_logits)
    y_np = np.concatenate(all_y)

    scaler = TemperatureScaler()
    cal_summary = scaler.fit(logits_np, y_np)
    probs_before = torch.softmax(torch.tensor(logits_np), dim=-1).numpy()
    probs_after = torch.softmax(scaler(torch.tensor(logits_np)), dim=-1).detach().numpy()

    xai_dir = proj_root / "results" / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)
    save_reliability_diagram(probs_before, probs_after, y_np, xai_dir / "reliability_diagram.png")

    # 2. Generate Grad-CAM 1D + IG plots for 2 correct cases (e.g., N3, REM) and 2 failure cases
    gradcam = GradCAM1D(model, model.dec2)
    correct_saved, failure_saved = 0, 0
    case_records = []

    with torch.backends.cudnn.flags(enabled=False):
        for w_idx in range(0, len(val_ds), max(1, len(val_ds) // 40)):
            if correct_saved >= 2 and failure_saved >= 2:
                break
            sample = val_ds[w_idx]
            x = sample["x"].unsqueeze(0).to(device)
            cm = sample["ch_mask"].unsqueeze(0).to(device)
            y_st = sample["y_stage"].numpy()
            with torch.no_grad():
                pred_st = model(x, ch_mask=cm)["stage_logits"][0].argmax(dim=-1).cpu().numpy()

            for ep_i in range(len(y_st)):
                yt, yp = int(y_st[ep_i]), int(pred_st[ep_i])
                if yt < 0:
                    continue
                is_corr = (yt == yp)
                if is_corr and correct_saved < 2 and yt in (3, 4):  # N3 or REM
                    tag = f"correct_{STAGE_NAMES[yt]}_{correct_saved+1}"
                    correct_saved += 1
                elif (not is_corr) and failure_saved < 2 and yt in (1, 2):  # N1/N2 failure
                    tag = f"failure_true_{STAGE_NAMES[yt]}_pred_{STAGE_NAMES[yp]}_{failure_saved+1}"
                    failure_saved += 1
                else:
                    continue

                cam = gradcam.explain_stage(x, cm, ep_i, yp).numpy()
                ig_res = compute_integrated_gradients(model, x, cm, ep_i, yp, n_steps=16)
                plot_attribution_case(
                    sample["x"][ep_i].numpy(),
                    cam,
                    ig_res["epoch_attr"],
                    ig_res["channel_importance"],
                    f"{tag} (True={STAGE_NAMES[yt]}, Pred={STAGE_NAMES[yp]})",
                    xai_dir / f"{tag}.png",
                )
                case_records.append({
                    "tag": tag,
                    "true": STAGE_NAMES[yt],
                    "pred": STAGE_NAMES[yp],
                    "channel_importance": [round(float(v), 4) for v in ig_res["channel_importance"]],
                })
                break

    gradcam.remove()

    summary = {"calibration": cal_summary, "xai_cases": case_records}
    (xai_dir / "xai_calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=== Phase 6 XAI & Calibration Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
