"""Stage 2 — Zero-Retrain Post-Processing Benchmarks on Validation Split Only.

Evaluates on `val` (never touching `test_locked`):
1. Sleep staging:
   - Baseline (stride=20, raw argmax)
   - Overlapping window probability averaging (stride=5)
   - Viterbi / HMM transition smoothing (5x5 transition matrix learned on `train`)
2. CAP event detection:
   - Baseline (argmax, min_dur=0.5s, merge_gap=0.5s, GT-NREM)
   - Temporal smoothing + physiological min_dur_s (2.0s) + merge_gap_s (1.0s)
   - Subtype-aware probability thresholding (especially A2)
   - Both GT-NREM and Predicted-NREM masking
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import SomniMapSequenceDataset
from src.models.somnimap_net import SomniMapNet
from src.training.eval_cap import extract_events_from_frames, match_events
from src.training.metrics import compute_sleep_metrics


def learn_train_transition_logprobs(proc_dir: Path, train_ids: list[str], smoothing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Estimate initial-state and 5x5 transition log-probabilities strictly from `train`."""
    init_counts = np.full(5, smoothing, dtype=np.float64)
    trans_counts = np.full((5, 5), smoothing, dtype=np.float64)
    for rid in train_ids:
        p = proc_dir / rid / "labels_stage.npy"
        if not p.exists():
            continue
        y = np.load(p)
        valid = y[y >= 0]
        if len(valid) > 0:
            init_counts[int(valid[0])] += 1.0
        for a, b in zip(y[:-1], y[1:]):
            if a >= 0 and b >= 0:
                trans_counts[int(a), int(b)] += 1.0
    init_logp = np.log(init_counts / init_counts.sum())
    trans_logp = np.log(trans_counts / trans_counts.sum(axis=1, keepdims=True))
    return init_logp, trans_logp


def viterbi_decode(probs: np.ndarray, init_logp: np.ndarray, trans_logp: np.ndarray, trans_weight: float = 0.5) -> np.ndarray:
    """Viterbi decoding over [N, 5] calibrated probabilities."""
    n, k = probs.shape
    emit = np.log(np.clip(probs, 1e-8, 1.0))
    dp = np.zeros((n, k), dtype=np.float64)
    back = np.zeros((n, k), dtype=np.int64)
    dp[0] = emit[0] + trans_weight * init_logp
    for t in range(1, n):
        scores = dp[t - 1][:, None] + trans_weight * trans_logp  # [5, 5]
        back[t] = np.argmax(scores, axis=0)
        dp[t] = np.max(scores, axis=0) + emit[t]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


@torch.no_grad()
def predict_recording_overlapping(
    model: torch.nn.Module,
    proc_dir: Path,
    rec_id: str,
    device: torch.device,
    window_epochs: int = 20,
    stride_epochs: int = 5,
    mode: str = "multi_channel",
    temperature: float = 1.1757,
) -> dict[str, np.ndarray]:
    """Run sliding-window inference and average stage & CAP probabilities per epoch/frame."""
    rec_dir = proc_dir / rec_id
    y_stage = np.load(rec_dir / "labels_stage.npy")
    y_cap = np.load(rec_dir / "labels_cap.npy")
    n_ep = len(y_stage)

    ds = SomniMapSequenceDataset(
        proc_dir, [rec_id], window_epochs=window_epochs, stride_epochs=stride_epochs, mode=mode
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    stage_prob_sum = np.zeros((n_ep, 5), dtype=np.float64)
    stage_count = np.zeros(n_ep, dtype=np.float64)
    cap_prob_sum = np.zeros((n_ep, 300, 4), dtype=np.float64)
    cap_count = np.zeros(n_ep, dtype=np.float64)

    win_idx = 0
    for batch in loader:
        x = batch["x"].to(device)
        cm = batch["ch_mask"].to(device)
        out = model(x, ch_mask=cm)
        st_p = torch.softmax(out["stage_logits"] / temperature, dim=-1).cpu().numpy()
        cp_p = torch.softmax(out["cap_logits"], dim=-1).cpu().numpy()
        bsz = st_p.shape[0]
        for b in range(bsz):
            _, start = ds.windows[win_idx]
            end = start + window_epochs
            stage_prob_sum[start:end] += st_p[b]
            stage_count[start:end] += 1.0
            cap_prob_sum[start:end] += cp_p[b]
            cap_count[start:end] += 1.0
            win_idx += 1

    stage_probs = stage_prob_sum / np.maximum(stage_count[:, None], 1.0)
    cap_probs = cap_prob_sum / np.maximum(cap_count[:, None, None], 1.0)
    return {
        "y_stage": y_stage,
        "y_cap": y_cap,
        "stage_probs": stage_probs,
        "cap_probs": cap_probs,
    }


def decode_cap_frames(
    cap_probs: np.ndarray,
    smooth_frames: int = 1,
    bg_thresh: float | None = None,
    a2_mult: float = 1.0,
) -> np.ndarray:
    """Decode [N_frames, 4] CAP probabilities into discrete frame labels."""
    p = cap_probs.copy()
    if smooth_frames > 1:
        p = uniform_filter1d(p, size=smooth_frames, axis=0)
    if bg_thresh is None and a2_mult == 1.0:
        return np.argmax(p, axis=-1)

    a_probs = p[..., 1:].copy()
    a_probs[..., 1] *= a2_mult  # boost A2 relative to A1/A3
    best_sub = np.argmax(a_probs, axis=-1) + 1
    if bg_thresh is None:
        is_cap = np.max(a_probs, axis=-1) > p[..., 0]
    else:
        is_cap = (1.0 - p[..., 0]) >= bg_thresh
    return np.where(is_cap, best_sub, 0)


def eval_cap_configs(
    rec_preds: dict[str, dict[str, np.ndarray]],
    stage_preds: dict[str, np.ndarray],
    use_pred_nrem: bool,
    smooth_frames: int,
    bg_thresh: float | None,
    a2_mult: float,
    min_dur_s: float,
    merge_gap_s: float,
) -> dict[str, Any]:
    all_gt, all_pr = [], []
    rec_rows = []
    for idx, (rid, d) in enumerate(rec_preds.items()):
        y_st = d["y_stage"]
        y_cap = d["y_cap"].reshape(-1)
        st_use = stage_preds[rid] if use_pred_nrem else y_st
        nrem_ep = np.isin(st_use, [1, 2, 3])
        nrem_gt = np.isin(y_st, [1, 2, 3])
        nrem_f_pr = np.repeat(nrem_ep, 300)
        nrem_f_gt = np.repeat(nrem_gt, 300)

        cap_p = d["cap_probs"].reshape(-1, 4)
        pr_raw = decode_cap_frames(cap_p, smooth_frames=smooth_frames, bg_thresh=bg_thresh, a2_mult=a2_mult)
        pr_f = np.where(nrem_f_pr, pr_raw, 0)
        gt_f = np.where(nrem_f_gt, y_cap, 0)

        gt_ev = extract_events_from_frames(gt_f, min_dur_s=0.5, merge_gap_s=0.5)
        pr_ev = extract_events_from_frames(pr_f, min_dur_s=min_dur_s, merge_gap_s=merge_gap_s)

        off = idx * 100000.0
        for e in gt_ev:
            all_gt.append({**e, "start_s": e["start_s"] + off, "end_s": e["end_s"] + off})
        for e in pr_ev:
            all_pr.append({**e, "start_s": e["start_s"] + off, "end_s": e["end_s"] + off})

        rec_rows.append((
            float((gt_f > 0).sum() / max(1, nrem_f_gt.sum())),
            float((pr_f > 0).sum() / max(1, nrem_f_pr.sum())),
        ))

    ov = match_events(all_gt, all_pr, iou_thresh=0.5, subtype=None)
    sub = {name: match_events(all_gt, all_pr, iou_thresh=0.5, subtype=k) for k, name in [(1, "A1"), (2, "A2"), (3, "A3")]}
    gt_r = np.array([r[0] for r in rec_rows])
    pr_r = np.array([r[1] for r in rec_rows])
    r_corr = float(np.corrcoef(gt_r, pr_r)[0, 1]) if np.std(pr_r) > 0 and np.std(gt_r) > 0 else 0.0
    return {"overall": ov, "per_subtype": sub, "cap_rate_pearson_r": round(r_corr, 4)}


def main() -> None:
    cfg = load_config()
    proc_dir = Path(cfg["paths"]["processed_dir"])
    proj_root = Path(cfg["paths"]["project_root"])
    splits = json.loads(Path(cfg["paths"]["split_manifest_json"]).read_text(encoding="utf-8"))
    train_ids = splits["partitions"]["train"]["record_ids"]
    val_ids = splits["partitions"]["val"]["record_ids"]

    ckpt_path = proj_root / "checkpoints" / "phase4_SomniMapNet_bilstm_w20_E8_dualhead_best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SomniMapNet(in_channels=6, context_type="bilstm", use_cap_head=True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    init_logp, trans_logp = learn_train_transition_logprobs(proc_dir, train_ids)

    print("Running stride=20 baseline and stride=5 overlap inference on val...", flush=True)
    rec_s20 = {
        rid: predict_recording_overlapping(model, proc_dir, rid, device, stride_epochs=20, temperature=1.1757)
        for rid in val_ids
    }
    rec_s5 = {
        rid: predict_recording_overlapping(model, proc_dir, rid, device, stride_epochs=5, temperature=1.1757)
        for rid in val_ids
    }

    # --- Sleep stage benchmarks ---
    stage_results = {}
    for label, preds_map in [
        ("baseline_s20_argmax", {r: np.argmax(d["stage_probs"], axis=-1) for r, d in rec_s20.items()}),
        ("overlap_s5_argmax", {r: np.argmax(d["stage_probs"], axis=-1) for r, d in rec_s5.items()}),
        ("overlap_s5_viterbi_0.25", {r: viterbi_decode(d["stage_probs"], init_logp, trans_logp, 0.25) for r, d in rec_s5.items()}),
        ("overlap_s5_viterbi_0.50", {r: viterbi_decode(d["stage_probs"], init_logp, trans_logp, 0.50) for r, d in rec_s5.items()}),
        ("overlap_s5_viterbi_1.00", {r: viterbi_decode(d["stage_probs"], init_logp, trans_logp, 1.00) for r, d in rec_s5.items()}),
    ]:
        yt = np.concatenate([rec_s5[r]["y_stage"] for r in val_ids])
        yp = np.concatenate([preds_map[r] for r in val_ids])
        m = compute_sleep_metrics(yt, yp)
        stage_results[label] = {
            "macro_f1": round(m["macro_f1"], 4),
            "balanced_acc": round(m["balanced_accuracy"], 4),
            "cohen_kappa": round(m["cohen_kappa"], 4),
            "accuracy": round(m["accuracy"], 4),
            "N1_f1": round(m["per_class"]["N1"]["f1"], 4),
            "N1_prec": round(m["per_class"]["N1"]["precision"], 4),
            "N1_rec": round(m["per_class"]["N1"]["sensitivity"], 4),
        }
        print(f"Stage [{label:24s}]: MacroF1={stage_results[label]['macro_f1']} | Kappa={stage_results[label]['cohen_kappa']} | N1_F1={stage_results[label]['N1_f1']}", flush=True)

    # Best stage predictions for predicted-NREM CAP masking
    best_st_preds = {r: viterbi_decode(d["stage_probs"], init_logp, trans_logp, 0.50) for r, d in rec_s5.items()}

    # --- CAP post-processing benchmarks ---
    cap_configs = [
        ("baseline_s20_gtNREM", rec_s20, False, 1, None, 1.0, 0.5, 0.5),
        ("s5_smooth5_dur2.0_gap1.0_gtNREM", rec_s5, False, 5, None, 1.0, 2.0, 1.0),
        ("s5_smooth5_bg0.35_a2x1.5_dur2.0_gtNREM", rec_s5, False, 5, 0.35, 1.5, 2.0, 1.0),
        ("s5_smooth5_bg0.45_a2x1.5_dur2.0_gtNREM", rec_s5, False, 5, 0.45, 1.5, 2.0, 1.0),
        ("s5_smooth7_bg0.50_a2x2.0_dur2.0_gtNREM", rec_s5, False, 7, 0.50, 2.0, 2.0, 1.0),
        ("s5_smooth7_bg0.55_a2x2.0_dur2.0_gtNREM", rec_s5, False, 7, 0.55, 2.0, 2.0, 1.0),
        ("s5_smooth7_bg0.50_a2x2.0_dur2.0_predNREM", rec_s5, True, 7, 0.50, 2.0, 2.0, 1.0),
    ]
    cap_results = {}
    for name, rmap, pred_nrem, sm, bg, a2m, md, mg in cap_configs:
        res = eval_cap_configs(rmap, best_st_preds, pred_nrem, sm, bg, a2m, md, mg)
        cap_results[name] = res
        ov = res["overall"]
        a2 = res["per_subtype"]["A2"]
        print(
            f"CAP [{name:40s}]: P={ov['precision']} R={ov['recall']} F1={ov['f1']} (FP={ov['fp']}) "
            f"| A1_F1={res['per_subtype']['A1']['f1']} A2_F1={a2['f1']} A3_F1={res['per_subtype']['A3']['f1']} | r={res['cap_rate_pearson_r']}",
            flush=True,
        )

    out_path = proj_root / "results" / "stage2_postproc_val.json"
    out_path.write_text(json.dumps({"stage": stage_results, "cap": cap_results}, indent=2), encoding="utf-8")
    print(f"Saved Stage 2 validation benchmark to {out_path}")


if __name__ == "__main__":
    main()
