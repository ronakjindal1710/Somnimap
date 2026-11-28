"""Structured Markdown & JSON research report generator for NIDRA EDF analyses."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data.annotation_parser import parse_txt_annotations
from src.data.preprocessing import align_annotations_to_epochs
from src.training.eval_cap import extract_events_from_frames, match_events
from src.training.metrics import STAGE_NAMES


CANONICAL_EEG_CHANNELS = ("EEG_C4_A1", "EEG_C3_A2", "EEG_F4_C4", "EEG_C4_P4", "EEG_FP2_F4")


def assess_domain_shift(edf_info: dict[str, Any], predictions: dict[str, Any]) -> dict[str, Any]:
    """Detect missing canonical EEG channels, fallback montage mapping, or OOD stage shares."""
    mapping = edf_info.get("canonical_mapping", {})
    missing_eeg = [ch for ch in CANONICAL_EEG_CHANNELS if ch not in mapping]
    warnings_list = edf_info.get("warnings", []) or []
    fallback_used = any("absent; mapped" in w.lower() or "fallback" in w.lower() for w in warnings_list)

    pcts = predictions.get("stage_percentages", {})
    rem_pct = float(pcts.get("REM", 0.0))
    n2_pct = float(pcts.get("N2", 0.0))

    reasons: list[str] = []
    if len(missing_eeg) > 2:
        reasons.append(
            f"{len(missing_eeg)}/{len(CANONICAL_EEG_CHANNELS)} canonical EEG channels missing/masked to zero ({missing_eeg})"
        )
    if fallback_used:
        reasons.append("fallback EEG montage mapping used (recording montage differs from CAP Sleep Database)")
    if rem_pct > 50.0:
        reasons.append(f"out-of-distribution predicted REM share ({rem_pct:.1f}% > 50%)")
    if n2_pct < 10.0:
        reasons.append(f"out-of-distribution predicted N2 share ({n2_pct:.1f}% < 10%)")

    high_risk = len(reasons) > 0
    summary = (
        "HIGH DOMAIN-SHIFT / UNCALIBRATED MONTAGE WARNING: "
        + "; ".join(reasons)
        + ". High model confidence reflects in-distribution softmax calibration (T=1.1757) and must NOT be interpreted as in-domain physiological reliability under missing/mismatched montages."
        if high_risk
        else "In-domain channel coverage and physiological stage distribution appear nominal."
    )
    return {
        "high_domain_shift": high_risk,
        "n_missing_canonical_eeg": len(missing_eeg),
        "missing_canonical_eeg": missing_eeg,
        "fallback_montage_used": fallback_used,
        "rem_pct": round(rem_pct, 2),
        "n2_pct": round(n2_pct, 2),
        "reasons": reasons,
        "summary": summary,
    }


def evaluate_against_reference_txt(
    txt_path: Path | str | None,
    edf_start_datetime: str | None,
    predictions: dict[str, Any],
) -> dict[str, Any] | None:
    """Compute post-hoc stage and CAP metrics against an uploaded RemLogic .txt file.

    Returns None when no .txt file is provided or alignment cannot be performed.
    """
    if txt_path is None or not edf_start_datetime:
        return None
    p = Path(txt_path)
    if not p.exists():
        return None

    parsed_ref = parse_txt_annotations(p)
    if not parsed_ref.get("parse_ok"):
        return None

    n_epochs = int(predictions["n_epochs"])
    gt_st, gt_cap = align_annotations_to_epochs(parsed_ref, edf_start_datetime, n_epochs)
    valid_stage = gt_st >= 0
    if not valid_stage.any():
        return None

    y_true = gt_st[valid_stage]
    y_pred = np.asarray(predictions["stage_preds"], dtype=np.int64)[valid_stage]

    acc = float(accuracy_score(y_true, y_pred))
    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0
    )
    p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0
    )
    per_stage = [
        {
            "Stage": STAGE_NAMES[i],
            "Precision": round(float(p_cls[i]), 4),
            "Recall": round(float(r_cls[i]), 4),
            "F1": round(float(f_cls[i]), 4),
            "Support": int(s_cls[i]),
        }
        for i in range(5)
    ]

    cap_frame_metrics: dict[str, Any] | None = None
    cap_event_metrics: dict[str, Any] | None = None
    nrem_mask = np.isin(gt_st, [1, 2, 3])
    if nrem_mask.any() and "cap_frames" in predictions:
        gt_bin = (gt_cap[nrem_mask] > 0).ravel()
        pr_bin = (np.asarray(predictions["cap_frames"])[nrem_mask] > 0).ravel()
        fp_prec, fp_rec, fp_f1, _ = precision_recall_fscore_support(
            gt_bin, pr_bin, average="binary", zero_division=0
        )
        cap_frame_metrics = {
            "n_nrem_epochs": int(nrem_mask.sum()),
            "precision": round(float(fp_prec), 4),
            "recall": round(float(fp_rec), 4),
            "f1": round(float(fp_f1), 4),
        }

        if parsed_ref.get("cap_events") or gt_cap.max() > 0:
            gt_events = extract_events_from_frames(
                gt_cap.ravel(), sfreq_hz=10.0, min_dur_s=0.5, merge_gap_s=0.5
            )
            ev_res = match_events(gt_events, predictions.get("cap_events", []), iou_thresh=0.5)
            cap_event_metrics = {
                "n_gt_events": len(gt_events),
                "n_pred_events": len(predictions.get("cap_events", [])),
                "precision": ev_res["precision"],
                "recall": ev_res["recall"],
                "f1": ev_res["f1"],
                "tp": ev_res["tp"],
                "fp": ev_res["fp"],
                "fn": ev_res["fn"],
            }

    return {
        "n_scored_epochs": int(valid_stage.sum()),
        "stage_accuracy": round(acc, 4),
        "stage_macro_f1": round(float(m_f1), 4),
        "stage_macro_precision": round(float(m_prec), 4),
        "stage_macro_recall": round(float(m_rec), 4),
        "per_stage": per_stage,
        "cap_frame_metrics": cap_frame_metrics,
        "cap_event_metrics": cap_event_metrics,
        "gt_stage_array": gt_st,
    }


def build_analysis_report_markdown(
    result: dict[str, Any],
    physiology: dict[str, Any],
    recording_name: str | None = None,
    gt_eval: dict[str, Any] | None = None,
) -> str:
    """Generate a self-contained Markdown research report with provenance and caveats."""
    info = result["edf_info"]
    display_name = recording_name or info["file_name"]
    pred = result["predictions"]
    prov = pred["provenance"]
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"# NIDRA Research Analysis Report — `{display_name}`",
        "",
        f"**Generated:** {now_str}  ",
        "**Platform:** NIDRA (*Neural Inference for Deep, REM & Arousal Architecture*)  ",
        "**Research Disclaimer:** *NIDRA is an exploratory sleep-signal research tool, not a validated clinical diagnostic device. Predictive outputs must not be interpreted as medical diagnoses.*",
        "",
        "---",
        "",
        "## 1. Recording & Channel Provenance",
        f"- **File:** `{display_name}` | **Duration:** {info['duration_sec']} s ({pred['n_epochs']} × 30-s epochs)",
        f"- **Header Sampling Rate:** {info['header_sfreq_hz']} Hz | **Loaded Subset Rate:** {result.get('loaded_sfreq', 'N/A')} Hz",
        f"- **Model Checkpoint:** `{prov['checkpoint']}` (`{prov['context_type']}`, `in_channels={prov['in_channels']}`, temperature $T={prov['temperature']}$, stride={prov['stride_epochs']})",
        f"- **Canonical Channels Mapped:** `{list(info['canonical_mapping'].keys())}`",
        f"- **Optional Multimodal Channels:** `{info.get('multimodal_extras', {})}`",
    ]
    if info.get("warnings"):
        lines.append("- **Compatibility / Channel Warnings:**")
        for w in info["warnings"]:
            lines.append(f"  - {w}")

    domain_shift = assess_domain_shift(info, pred)
    if domain_shift["high_domain_shift"]:
        lines += [
            "",
            "> [!WARNING]",
            f"> **{domain_shift['summary']}**",
        ]

    lines += [
        "",
        "## 2. Predictive Outputs — Sleep-Stage Architecture",
        "| Stage | Epochs | Percentage (%) |",
        "|---|---:|---:|",
    ]
    for st, cnt in pred["stage_counts"].items():
        pct = pred["stage_percentages"][st]
        lines.append(f"| {st} | {cnt} | {pct:.2f}% |")

    mean_conf = float(np.mean(pred["stage_confidence"]))
    mean_ent = float(np.mean(pred["stage_entropy"]))
    n_low = len(pred["low_confidence_epochs"])
    lines += [
        "",
        f"- **Mean Calibrated Confidence:** {mean_conf:.3f} | **Mean Normalized Entropy:** {mean_ent:.3f}",
        f"- **Low-Confidence Epochs (< 0.55):** {n_low} / {pred['n_epochs']} ({100.0 * n_low / max(1, pred['n_epochs']):.1f}%)",
    ]

    if "cap_events" in pred:
        counts = pred["cap_event_counts"]
        lines += [
            "",
            "## 3. Predictive Outputs — CAP Event Analysis (Predicted NREM)",
            f"- **Predicted NREM CAP Rate:** {pred['cap_rate_nrem'] * 100:.2f}% (macro CAP rate Pearson $r = 0.9319$ on `val`)",
            f"- **Detected Events (>= 2.5 s, smoothed, merge gap <= 1.5 s):** Total = {len(pred['cap_events'])} (`A1={counts['A1']}`, `A2={counts['A2']}`, `A3={counts['A3']}`)",
            "- *Note:* Macro NREM CAP rate correlates strongly with ground truth on in-domain `val` (`r = 0.9319`), but per-event precision (`0.4558` overall) and `A2` subtype sensitivity (`F1 ~0.27–0.29` on `val`) remain exploratory; do not treat individual `A2` counts or exact event boundaries as clinical truth.",
        ]

    lines += [
        "",
        "## 4. Exploratory Signal-Derived Physiology (Brain–Heart–Respiration)",
        f"*{physiology.get('disclaimer', '')}*",
        "",
    ]
    ecg = physiology.get("ecg", {})
    if ecg.get("available"):
        lines.append(
            f"- **Cardiac (ECG):** Mean HR = **{ecg['mean_hr_bpm']} bpm** | SDNN = **{ecg['overall_sdnn_ms']} ms** | RMSSD = **{ecg['overall_rmssd_ms']} ms** | Beats = {ecg['n_beats']}"
        )
        if ecg.get("mean_cap_hr_delta_bpm") is not None:
            lines.append(f"  - Mean HR change during predicted CAP events vs pre-event baseline: **{ecg['mean_cap_hr_delta_bpm']:+.2f} bpm**")
    else:
        lines.append(f"- **Cardiac (ECG):** Unavailable ({ecg.get('reason', 'not present')})")

    resp = physiology.get("respiratory", {})
    if resp.get("available"):
        lines.append(f"- **Respiratory Rate:** Mean = **{resp['mean_br_bpm']} breaths/min**")
    else:
        lines.append(f"- **Respiratory:** Unavailable ({resp.get('reason', 'not present')})")

    spo2 = physiology.get("spo2", {})
    if spo2.get("available"):
        lines.append(f"- **SpO2:** Mean = **{spo2['mean_pct']}%**, Min = **{spo2['min_pct']}%**, Time < 90% = **{spo2['pct_below_90']}%**")
    else:
        lines.append(f"- **SpO2:** Unavailable ({spo2.get('reason', 'not present')})")

    if gt_eval is not None:
        lines += [
            "",
            "## 5. Ground-Truth Evaluation (vs. Uploaded Reference `.txt`)",
            "*Computed post-hoc against uploaded reference annotations (zero model leakage).*",
            f"- **Sleep-Stage (5-Class, {gt_eval['n_scored_epochs']} scored epochs):** "
            f"Accuracy = **{gt_eval['stage_accuracy']*100:.2f}%** | "
            f"Macro F1 = **{gt_eval['stage_macro_f1']:.4f}** | "
            f"Macro Precision = **{gt_eval['stage_macro_precision']:.4f}** | "
            f"Macro Recall = **{gt_eval['stage_macro_recall']:.4f}**",
            "",
            "| Stage | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in gt_eval["per_stage"]:
            lines.append(
                f"| {row['Stage']} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['Support']} |"
            )
        if gt_eval.get("cap_frame_metrics"):
            cf = gt_eval["cap_frame_metrics"]
            lines.append(
                f"- **CAP Frame Metrics (Valid NREM, {cf['n_nrem_epochs']} epochs):** "
                f"Precision = **{cf['precision']:.4f}** | Recall = **{cf['recall']:.4f}** | F1 = **{cf['f1']:.4f}**"
            )
        if gt_eval.get("cap_event_metrics"):
            ce = gt_eval["cap_event_metrics"]
            lines.append(
                f"- **CAP Event Metrics (IoU >= 0.5, GT={ce['n_gt_events']}, Pred={ce['n_pred_events']}):** "
                f"Precision = **{ce['precision']:.4f}** | Recall = **{ce['recall']:.4f}** | F1 = **{ce['f1']:.4f}**"
            )

    return "\n".join(lines)


def build_analysis_summary_json(
    result: dict[str, Any],
    physiology: dict[str, Any],
    recording_name: str | None = None,
    gt_eval: dict[str, Any] | None = None,
) -> str:
    """Serialize lightweight JSON summary (without large raw signal arrays)."""
    pred = result["predictions"]
    edf_info = dict(result["edf_info"])
    if recording_name:
        edf_info["file_name"] = recording_name
    domain_shift = assess_domain_shift(edf_info, pred)
    payload: dict[str, Any] = {
        "platform": "NIDRA (Neural Inference for Deep, REM & Arousal Architecture)",
        "edf_info": edf_info,
        "provenance": pred["provenance"],
        "domain_shift_assessment": domain_shift,
        "n_epochs": pred["n_epochs"],
        "stage_counts": pred["stage_counts"],
        "stage_percentages": pred["stage_percentages"],
        "mean_confidence": round(float(np.mean(pred["stage_confidence"])), 4),
        "mean_entropy": round(float(np.mean(pred["stage_entropy"])), 4),
        "n_low_confidence_epochs": len(pred["low_confidence_epochs"]),
        "cap_rate_nrem": pred.get("cap_rate_nrem"),
        "cap_event_counts": pred.get("cap_event_counts"),
        "physiology": {
            k: ({sk: sv for sk, sv in v.items() if not isinstance(sv, np.ndarray)} if isinstance(v, dict) else v)
            for k, v in physiology.items()
        },
    }
    if gt_eval is not None:
        payload["ground_truth_evaluation"] = {
            k: v for k, v in gt_eval.items() if k != "gt_stage_array"
        }
    return json.dumps(payload, indent=2)
