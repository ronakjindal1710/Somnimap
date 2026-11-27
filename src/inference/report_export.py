"""Structured Markdown & JSON research report generator for SomniMap EDF analyses."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np


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
            f"- **Predicted NREM CAP Rate:** {pred['cap_rate_nrem'] * 100:.2f}%",
            f"- **Detected Events (>= 2.0 s, smoothed):** Total = {len(pred['cap_events'])} (`A1={counts['A1']}`, `A2={counts['A2']}`, `A3={counts['A3']}`)",
            "- *Note:* CAP A2 detection has lower sensitivity in automated scoring; event counts should be treated as exploratory.",
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
    payload = {
        "edf_info": result["edf_info"],
        "provenance": pred["provenance"],
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
