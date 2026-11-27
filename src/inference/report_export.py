"""Structured Markdown & JSON research report generator for SomniMap EDF analyses."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np


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


def build_analysis_report_markdown(result: dict[str, Any], physiology: dict[str, Any]) -> str:
    """Generate a self-contained Markdown research report with provenance and caveats."""
    info = result["edf_info"]
    pred = result["predictions"]
    prov = pred["provenance"]
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"# SomniMap Research Analysis Report — `{info['file_name']}`",
        "",
        f"**Generated:** {now_str}  ",
        f"**Research Disclaimer:** *SomniMap is an exploratory sleep-signal research tool, not a validated clinical diagnostic device. Predictive outputs must not be interpreted as medical diagnoses.*",
        "",
        "---",
        "",
        "## 1. Recording & Channel Provenance",
        f"- **File:** `{info['file_name']}` | **Duration:** {info['duration_sec']} s ({pred['n_epochs']} × 30-s epochs)",
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

    return "\n".join(lines)


def build_analysis_summary_json(result: dict[str, Any], physiology: dict[str, Any]) -> str:
    """Serialize lightweight JSON summary (without large raw signal arrays)."""
    pred = result["predictions"]
    domain_shift = assess_domain_shift(result["edf_info"], pred)
    payload = {
        "edf_info": result["edf_info"],
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
    return json.dumps(payload, indent=2)
