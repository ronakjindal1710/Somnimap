"""Multimodal physiological signal analytics (ECG, Respiratory, SpO2).

Computes clearly defined signal-derived measurements aligned to sleep stages and
CAP events when those channels are present in the raw EDF. Never fabricates missing
channels and never makes clinical diagnostic claims.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from src.training.metrics import STAGE_NAMES


def analyze_ecg_hrv(
    ecg_raw: np.ndarray,
    sfreq: float,
    stage_preds: np.ndarray,
    cap_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Detect R-peaks and compute per-epoch HR (bpm), SDNN, and RMSSD."""
    if ecg_raw.size < int(sfreq * 30):
        return {"available": False, "reason": "ECG recording shorter than 30 s"}

    # Bandpass 5-25 Hz to emphasize QRS complexes
    nyq = 0.5 * sfreq
    low = min(5.0 / nyq, 0.95)
    high = min(25.0 / nyq, 0.99)
    if low >= high or low <= 0:
        return {"available": False, "reason": f"ECG sampling rate ({sfreq} Hz) too low for QRS detection"}

    sos = signal.butter(3, [low, high], btype="bandpass", output="sos")
    filt = signal.sosfiltfilt(sos, ecg_raw.astype(np.float64))
    # Polarity-robust squared envelope
    env = filt ** 2
    win = max(1, int(0.12 * sfreq))
    smooth = np.convolve(env, np.ones(win) / win, mode="same")
    thresh = float(np.percentile(smooth, 75)) + 0.5 * float(np.std(smooth))
    min_dist = max(1, int(0.35 * sfreq))  # max ~171 bpm
    peaks, _ = signal.find_peaks(smooth, height=thresh, distance=min_dist)

    if len(peaks) < 10:
        return {"available": False, "reason": "Insufficient R-peaks detected"}

    peak_times = peaks / sfreq
    rr_ms = np.diff(peak_times) * 1000.0
    valid_rr = (rr_ms >= 300.0) & (rr_ms <= 2000.0)
    rr_ms_clean = rr_ms[valid_rr]
    rr_times = peak_times[1:][valid_rr]

    if len(rr_ms_clean) < 5:
        return {"available": False, "reason": "Insufficient physiological RR intervals (300-2000 ms)"}

    n_ep = len(stage_preds)
    epoch_hr = np.full(n_ep, np.nan, dtype=np.float64)
    for ep in range(n_ep):
        t0, t1 = ep * 30.0, (ep + 1) * 30.0
        m = (rr_times >= t0) & (rr_times < t1)
        if m.sum() >= 3:
            epoch_hr[ep] = 60000.0 / float(np.median(rr_ms_clean[m]))

    by_stage: dict[str, dict[str, float]] = {}
    for st_id, st_name in enumerate(STAGE_NAMES):
        ep_mask = stage_preds == st_id
        vals = epoch_hr[ep_mask]
        vals = vals[~np.isnan(vals)]
        # Epoch-wise RR intervals for stage HRV
        st_rr = []
        for ep_i in np.where(ep_mask)[0]:
            t0, t1 = ep_i * 30.0, (ep_i + 1) * 30.0
            m = (rr_times >= t0) & (rr_times < t1)
            if m.any():
                st_rr.append(rr_ms_clean[m])
        if vals.size > 0 and st_rr:
            rr_cat = np.concatenate(st_rr)
            rmssd = float(np.sqrt(np.mean(np.diff(rr_cat) ** 2))) if len(rr_cat) > 2 else 0.0
            by_stage[st_name] = {
                "mean_hr_bpm": round(float(np.mean(vals)), 2),
                "sdnn_ms": round(float(np.std(rr_cat)), 2),
                "rmssd_ms": round(rmssd, 2),
            }

    cap_hr_delta = None
    if cap_events:
        deltas = []
        for ev in cap_events[:200]:
            pre_m = (rr_times >= ev["start_s"] - 10.0) & (rr_times < ev["start_s"])
            dur_m = (rr_times >= ev["start_s"]) & (rr_times <= ev["end_s"] + 5.0)
            if pre_m.sum() >= 2 and dur_m.sum() >= 2:
                hr_pre = 60000.0 / float(np.median(rr_ms_clean[pre_m]))
                hr_dur = 60000.0 / float(np.median(rr_ms_clean[dur_m]))
                deltas.append(hr_dur - hr_pre)
        if deltas:
            cap_hr_delta = round(float(np.mean(deltas)), 2)

    return {
        "available": True,
        "n_beats": int(len(peaks)),
        "mean_hr_bpm": round(float(60000.0 / np.median(rr_ms_clean)), 2),
        "overall_sdnn_ms": round(float(np.std(rr_ms_clean)), 2),
        "overall_rmssd_ms": round(float(np.sqrt(np.mean(np.diff(rr_ms_clean) ** 2))), 2),
        "epoch_hr_bpm": epoch_hr,
        "by_stage": by_stage,
        "mean_cap_hr_delta_bpm": cap_hr_delta,
    }


def analyze_respiratory(
    resp_raw: np.ndarray,
    sfreq: float,
    stage_preds: np.ndarray,
) -> dict[str, Any]:
    """Estimate per-epoch respiratory rate (breaths/min) from flow or effort trace."""
    if resp_raw.size < int(sfreq * 30) or sfreq < 2.0:
        return {"available": False, "reason": "Respiratory signal too short or low rate"}

    nyq = 0.5 * sfreq
    low = min(0.08 / nyq, 0.90)
    high = min(0.8 / nyq, 0.98)
    if low >= high or low <= 0:
        return {"available": False, "reason": "Invalid respiratory filter band"}

    sos = signal.butter(2, [low, high], btype="bandpass", output="sos")
    filt = signal.sosfiltfilt(sos, resp_raw.astype(np.float64))

    n_ep = len(stage_preds)
    ep_samples = int(round(30.0 * sfreq))
    epoch_br = np.full(n_ep, np.nan, dtype=np.float64)
    for ep in range(min(n_ep, len(filt) // ep_samples)):
        seg = filt[ep * ep_samples : (ep + 1) * ep_samples]
        if np.std(seg) < 1e-8:
            continue
        pks, _ = signal.find_peaks(seg, distance=max(1, int(1.5 * sfreq)), prominence=0.25 * np.std(seg))
        if len(pks) >= 2:
            epoch_br[ep] = len(pks) * 2.0  # 30s -> breaths/min

    by_stage = {}
    for st_id, st_name in enumerate(STAGE_NAMES):
        vals = epoch_br[stage_preds == st_id]
        vals = vals[~np.isnan(vals)]
        if vals.size > 0:
            by_stage[st_name] = {"mean_br_bpm": round(float(np.mean(vals)), 2)}

    valid_br = epoch_br[~np.isnan(epoch_br)]
    return {
        "available": bool(valid_br.size > 0),
        "mean_br_bpm": round(float(np.mean(valid_br)), 2) if valid_br.size > 0 else None,
        "epoch_br_bpm": epoch_br,
        "by_stage": by_stage,
    }


def analyze_multimodal_physiology(
    extras_raw: dict[str, np.ndarray],
    sfreq: float,
    stage_preds: np.ndarray,
    cap_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute brain-heart-respiration summary from available non-EEG EDF channels."""
    out: dict[str, Any] = {
        "disclaimer": "Exploratory signal-derived physiological measurements only; not a clinical diagnosis.",
        "ecg": {"available": False, "reason": "ECG channel not present in EDF"},
        "respiratory": {"available": False, "reason": "Respiratory channel not present in EDF"},
        "spo2": {"available": False, "reason": "SpO2 channel not present in EDF"},
    }
    if "ECG" in extras_raw:
        out["ecg"] = analyze_ecg_hrv(extras_raw["ECG"], sfreq, stage_preds, cap_events)
    resp_key = "RESP_FLOW" if "RESP_FLOW" in extras_raw else ("RESP_EFFORT" if "RESP_EFFORT" in extras_raw else None)
    if resp_key is not None:
        out["respiratory"] = analyze_respiratory(extras_raw[resp_key], sfreq, stage_preds)
    if "SPO2" in extras_raw:
        sp = extras_raw["SPO2"].astype(np.float64)
        # Filter plausible SpO2 % values
        valid = sp[(sp >= 50.0) & (sp <= 100.0)]
        if valid.size > 0:
            out["spo2"] = {
                "available": True,
                "mean_pct": round(float(np.mean(valid)), 2),
                "min_pct": round(float(np.min(valid)), 2),
                "pct_below_90": round(float((valid < 90.0).mean() * 100.0), 2),
            }
    return out
