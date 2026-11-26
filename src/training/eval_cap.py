"""Phase 5 — CAP Event Detection Evaluation (A1/A2/A3, IoU >= 0.5, timing error)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import SomniMapSequenceDataset
from src.models.somnimap_net import SomniMapNet


CAP_NAMES = {1: "A1", 2: "A2", 3: "A3"}


def extract_events_from_frames(
    frames: np.ndarray,
    sfreq_hz: float = 10.0,
    min_dur_s: float = 0.5,
    merge_gap_s: float = 0.5,
) -> list[dict[str, Any]]:
    """Convert 1D frame-level CAP labels (0=none, 1=A1, 2=A2, 3=A3) into event dicts."""
    events: list[dict[str, Any]] = []
    n = len(frames)
    i = 0
    while i < n:
        cls = int(frames[i])
        if cls <= 0:
            i += 1
            continue
        j = i + 1
        while j < n and int(frames[j]) == cls:
            j += 1
        events.append({
            "subtype": cls,
            "start_s": i / sfreq_hz,
            "end_s": j / sfreq_hz,
        })
        i = j

    # Merge tiny gaps of same subtype
    merged: list[dict[str, Any]] = []
    for ev in events:
        if merged and merged[-1]["subtype"] == ev["subtype"] and (ev["start_s"] - merged[-1]["end_s"]) <= merge_gap_s:
            merged[-1]["end_s"] = ev["end_s"]
        else:
            merged.append(ev)

    return [ev for ev in merged if (ev["end_s"] - ev["start_s"]) >= min_dur_s]


def compute_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0.0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / max(1e-9, union)


def match_events(
    gt_events: list[dict[str, Any]],
    pred_events: list[dict[str, Any]],
    iou_thresh: float = 0.5,
    subtype: int | None = None,
) -> dict[str, Any]:
    """Greedy 1-to-1 event matching by highest IoU >= iou_thresh."""
    gt = [e for e in gt_events if subtype is None or e["subtype"] == subtype]
    pr = [e for e in pred_events if subtype is None or e["subtype"] == subtype]

    if not gt and not pr:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "mean_iou": 0.0, "onset_err_s": 0.0, "offset_err_s": 0.0, "dur_err_s": 0.0}

    pairs = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pr):
            if subtype is None and g["subtype"] != p["subtype"]:
                continue
            iou = compute_iou(g["start_s"], g["end_s"], p["start_s"], p["end_s"])
            if iou >= iou_thresh:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_g, matched_p = set(), set()
    ious, onset_errs, offset_errs, dur_errs = [], [], [], []
    for iou, gi, pi in pairs:
        if gi in matched_g or pi in matched_p:
            continue
        matched_g.add(gi)
        matched_p.add(pi)
        g, p = gt[gi], pr[pi]
        ious.append(iou)
        onset_errs.append(abs(p["start_s"] - g["start_s"]))
        offset_errs.append(abs(p["end_s"] - g["end_s"]))
        dur_errs.append(abs((p["end_s"] - p["start_s"]) - (g["end_s"] - g["start_s"])))

    tp = len(matched_g)
    fp = len(pr) - tp
    fn = len(gt) - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec) / max(1e-9, prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "onset_err_s": round(float(np.mean(onset_errs)), 4) if onset_errs else 0.0,
        "offset_err_s": round(float(np.mean(offset_errs)), 4) if offset_errs else 0.0,
        "dur_err_s": round(float(np.mean(dur_errs)), 4) if dur_errs else 0.0,
    }


@torch.no_grad()
def evaluate_cap_on_records(
    model: torch.nn.Module,
    proc_dir: Path,
    record_ids: list[str],
    device: torch.device,
    window_epochs: int = 20,
    mode: str = "multi_channel",
    iou_thresh: float = 0.5,
    bg_prob_thresh: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    """Run non-overlapping window inference per recording and compute CAP event metrics."""
    model.eval()
    all_gt_events: list[dict[str, Any]] = []
    all_pred_events: list[dict[str, Any]] = []
    rec_rows = []
    examples = []

    for rec_id in record_ids:
        rec_dir = proc_dir / rec_id
        if not (rec_dir / "metadata.json").exists():
            continue
        ds = SomniMapSequenceDataset(
            proc_dir, [rec_id], window_epochs=window_epochs, stride_epochs=window_epochs, mode=mode
        )
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

        pred_frames_list = []
        gt_frames_list = []
        nrem_mask_list = []
        for batch in loader:
            x = batch["x"].to(device)
            ch_mask = batch["ch_mask"].to(device)
            out = model(x, ch_mask=ch_mask)
            probs = torch.softmax(out["cap_logits"], dim=-1).cpu().numpy()  # [B, T, 300, 4]
            if bg_prob_thresh is not None:
                # Predict non-zero subtype if 1 - P(none) >= thresh
                a_probs = probs[..., 1:]
                best_sub = np.argmax(a_probs, axis=-1) + 1
                is_cap = (1.0 - probs[..., 0]) >= bg_prob_thresh
                preds = np.where(is_cap, best_sub, 0)
            else:
                preds = np.argmax(probs, axis=-1)

            y_cap = batch["y_cap"].numpy()
            y_stage = batch["y_stage"].numpy()
            nrem = np.isin(y_stage, [1, 2, 3])  # [B, T]
            nrem_frames = np.repeat(nrem[..., None], 300, axis=-1)

            pred_frames_list.append(preds.reshape(-1))
            gt_frames_list.append(y_cap.reshape(-1))
            nrem_mask_list.append(nrem_frames.reshape(-1))

        pred_f = np.concatenate(pred_frames_list)
        gt_f = np.concatenate(gt_frames_list)
        nrem_f = np.concatenate(nrem_mask_list)

        # Mask CAP predictions to NREM frames (CAP is defined in NREM)
        pred_f_nrem = np.where(nrem_f, pred_f, 0)
        gt_f_nrem = np.where(nrem_f, gt_f, 0)

        gt_ev = extract_events_from_frames(gt_f_nrem)
        pr_ev = extract_events_from_frames(pred_f_nrem)

        # Offset timestamps by recording index to avoid cross-recording collisions
        offset = len(rec_rows) * 100000.0
        for e in gt_ev:
            all_gt_events.append({**e, "start_s": e["start_s"] + offset, "end_s": e["end_s"] + offset})
        for e in pr_ev:
            all_pred_events.append({**e, "start_s": e["start_s"] + offset, "end_s": e["end_s"] + offset})

        gt_rate = float((gt_f_nrem > 0).sum() / max(1, nrem_f.sum()))
        pr_rate = float((pred_f_nrem > 0).sum() / max(1, nrem_f.sum()))
        rec_rows.append({
            "record_id": rec_id,
            "n_gt_events": len(gt_ev),
            "n_pred_events": len(pr_ev),
            "gt_cap_rate": round(gt_rate, 4),
            "pred_cap_rate": round(pr_rate, 4),
        })
        if len(examples) < 3 and len(gt_ev) >= 5:
            examples.append({"record_id": rec_id, "gt_f": gt_f_nrem, "pr_f": pred_f_nrem})

    overall = match_events(all_gt_events, all_pred_events, iou_thresh=iou_thresh, subtype=None)
    per_subtype = {
        CAP_NAMES[s]: match_events(all_gt_events, all_pred_events, iou_thresh=iou_thresh, subtype=s)
        for s in (1, 2, 3)
    }

    rec_df = pd.DataFrame(rec_rows)
    cap_rate_corr = float(rec_df["gt_cap_rate"].corr(rec_df["pred_cap_rate"])) if len(rec_df) > 1 else 0.0
    summary = {
        "iou_threshold": iou_thresh,
        "overall": overall,
        "per_subtype": per_subtype,
        "cap_rate_pearson_r": round(cap_rate_corr, 4),
        "n_records": len(rec_df),
    }
    return summary, rec_df, examples


def plot_cap_examples(examples: list[dict[str, Any]], out_path: Path) -> None:
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 3.0 * len(examples)), sharex=False)
    if len(examples) == 1:
        axes = [axes]
    for ax, ex in zip(axes, examples):
        # Show a 5-minute (300s = 3000 frames) segment around first GT CAP event
        gt = ex["gt_f"]
        pr = ex["pr_f"]
        pos = np.where(gt > 0)[0]
        start = max(0, int(pos[0]) - 300) if len(pos) else 0
        end = min(len(gt), start + 3000)
        t = np.arange(start, end) / 10.0
        ax.step(t, gt[start:end], where="post", label="Ground Truth (A1=1, A2=2, A3=3)", linewidth=1.6)
        ax.step(t, pr[start:end] + 0.1, where="post", label="Predicted", linestyle="--", alpha=0.85, linewidth=1.4)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["None", "A1", "A2", "A3"])
        ax.set_title(f"CAP Event Detection Timeline — {ex['record_id']}")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 CAP Event Detection Evaluation")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test_locked"])
    parser.add_argument("--context_type", type=str, default="bilstm")
    parser.add_argument("--multi_channel", action="store_true", default=False)
    parser.add_argument("--window_epochs", type=int, default=20)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--bg_prob_thresh", type=float, default=None)
    args = parser.parse_args()

    cfg_path = Path("C:/Users/admin/Desktop/Somnimap/experiments/configs/base_config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    proc_dir = Path(cfg["paths"]["processed_dir"])
    proj_root = Path(cfg["paths"]["project_root"])
    split_data = json.loads(Path(cfg["paths"]["split_manifest_json"]).read_text(encoding="utf-8"))
    record_ids = split_data["partitions"][args.split]["record_ids"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "multi_channel" if args.multi_channel else "single_eeg"
    in_ch = 6 if args.multi_channel else 1

    model = SomniMapNet(in_channels=in_ch, context_type=args.context_type, use_cap_head=True).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    summary, rec_df, examples = evaluate_cap_on_records(
        model, proc_dir, record_ids, device,
        window_epochs=args.window_epochs, mode=mode,
        iou_thresh=args.iou_thresh, bg_prob_thresh=args.bg_prob_thresh,
    )

    out_dir = proj_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"phase5_cap_eval_{args.split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rec_df.to_csv(out_dir / f"phase5_cap_records_{args.split}.csv", index=False)
    plot_cap_examples(examples, out_dir / f"phase5_cap_examples_{args.split}.png")

    print(f"\n=== Phase 5 CAP Event Evaluation ({args.split}, IoU >= {args.iou_thresh}) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
