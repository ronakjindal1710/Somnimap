"""EDF inspection and explicit channel mapping (no channel fabrication)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import mne
import numpy as np

from src.data.preprocessing import CANONICAL_CHANNELS, resolve_channel_sources

# Common fallback aliases for external PSGs (e.g., Sleep-EDF) when CAP bipolar names are absent
EXTERNAL_EEG_FALLBACKS = [
    "EEGFPZCZ", "FPZCZ", "EEGCF4A1", "C4M1", "C3M2", "EEGC3A2", "C3A2", "EEGPZOZ", "PZOZ", "F4M1", "F3M2"
]
EXTERNAL_EOG_FALLBACKS = ["EOGHORIZONTAL", "EOGL", "EOGR", "E1M2", "E2M1", "LOC", "ROC"]
EXTERNAL_EMG_FALLBACKS = ["EMGSUBMENTAL", "CHIN", "EMGCHIN"]


def _norm(lbl: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", lbl.upper())


def _find_first_match(all_labels: list[str], patterns: list[str]) -> str | None:
    by_norm = {_norm(lbl): lbl for lbl in all_labels}
    for pat in patterns:
        if pat in by_norm:
            return by_norm[pat]
    for lbl in all_labels:
        n = _norm(lbl)
        for pat in patterns:
            if pat in n:
                return lbl
    return None


def resolve_multimodal_extras(all_labels: list[str]) -> dict[str, str]:
    """Identify optional ECG, Respiratory, and SpO2 channels if present."""
    extras: dict[str, str] = {}
    ecg = _find_first_match(all_labels, ["ECG1ECG2", "ECG1", "ECG", "EKG"])
    if ecg:
        extras["ECG"] = ecg
    resp_flow = _find_first_match(all_labels, ["FLUSSO", "FLOW", "CANNULA", "ORONASAL", "RESP", "TERMISTORE"])
    if resp_flow:
        extras["RESP_FLOW"] = resp_flow
    resp_effort = _find_first_match(all_labels, ["TORACE", "THORAX", "THOR", "ADDOME", "ABDOMEN", "ABDO"])
    if resp_effort:
        extras["RESP_EFFORT"] = resp_effort
    spo2 = _find_first_match(all_labels, ["SAO2", "SPO2"])
    if spo2:
        extras["SPO2"] = spo2
    return extras


def resolve_channels_with_fallbacks(all_labels: list[str]) -> tuple[dict[str, tuple[str, str | None]], list[str]]:
    """Resolve CAP canonical channels first, then safe external PSG aliases if missing."""
    warnings: list[str] = []
    ch_map = resolve_channel_sources(all_labels)

    if "EEG_C4_A1" not in ch_map and "EEG_C3_A2" not in ch_map:
        fb_eeg = _find_first_match(all_labels, EXTERNAL_EEG_FALLBACKS)
        if fb_eeg:
            ch_map["EEG_C4_A1"] = (fb_eeg, None)
            warnings.append(f"Primary CAP EEG (C4-A1/C3-A2) absent; mapped '{fb_eeg}' as primary EEG.")
    if "EOG" not in ch_map:
        fb_eog = _find_first_match(all_labels, EXTERNAL_EOG_FALLBACKS)
        if fb_eog:
            ch_map["EOG"] = (fb_eog, None)
            warnings.append(f"CAP EOG absent; mapped '{fb_eog}' to EOG.")
    if "EMG_CHIN" not in ch_map:
        fb_emg = _find_first_match(all_labels, EXTERNAL_EMG_FALLBACKS)
        if fb_emg:
            ch_map["EMG_CHIN"] = (fb_emg, None)
            warnings.append(f"CAP Chin EMG absent; mapped '{fb_emg}' to EMG_CHIN.")

    return ch_map, warnings


def inspect_edf(edf_path: Path | str) -> dict[str, Any]:
    """Read EDF header without loading full data into RAM and validate compatibility."""
    p = Path(edf_path)
    if not p.exists():
        raise FileNotFoundError(f"EDF file not found: {p}")

    raw = mne.io.read_raw_edf(str(p), preload=False, verbose="ERROR")
    try:
        all_labels = [c.strip() for c in raw.ch_names]
        sfreq = float(raw.info["sfreq"])
        duration_s = float(raw.n_times / max(sfreq, 1e-6))
        meas_date = raw.info.get("meas_date")
        start_iso = meas_date.isoformat() if meas_date is not None else None
        ch_units = {ch["ch_name"].strip(): str(ch.get("unit", "")) for ch in raw.info["chs"]}
    finally:
        raw.close()

    ch_map, warnings = resolve_channels_with_fallbacks(all_labels)
    extras = resolve_multimodal_extras(all_labels)

    has_primary_eeg = ("EEG_C4_A1" in ch_map) or ("EEG_C3_A2" in ch_map)
    n_epochs_30s = int(duration_s // 30.0)
    compatible = has_primary_eeg and n_epochs_30s >= 20

    if not has_primary_eeg:
        warnings.append("Incompatible EDF: no recognized primary EEG channel (e.g., C4-A1, C3-A2, Fpz-Cz).")
    if n_epochs_30s < 20:
        warnings.append(f"Recording too short ({n_epochs_30s} epochs < 20 epochs required for 10-min context).")

    missing_canonical = [c for c in CANONICAL_CHANNELS[:6] if c not in ch_map]
    if missing_canonical:
        warnings.append(f"Missing canonical channels (masked to 0, not fabricated): {missing_canonical}")

    return {
        "file_name": p.name,
        "duration_sec": round(duration_s, 2),
        "n_epochs_30s": n_epochs_30s,
        "header_sfreq_hz": sfreq,
        "start_datetime": start_iso,
        "all_labels": all_labels,
        "channel_units": ch_units,
        "canonical_mapping": {k: list(v) for k, v in ch_map.items()},
        "multimodal_extras": extras,
        "compatible": compatible,
        "warnings": warnings,
    }


def load_edf_channels(edf_path: Path | str) -> dict[str, Any]:
    """Load mapped canonical PSG channels + optional multimodal extras from raw EDF."""
    info = inspect_edf(edf_path)
    if not info["compatible"]:
        raise ValueError("EDF is not compatible with SomniMap: " + " | ".join(info["warnings"]))

    ch_map = {k: tuple(v) for k, v in info["canonical_mapping"].items()}
    extras = info["multimodal_extras"]

    needed = {lbl for pair in ch_map.values() for lbl in pair if lbl is not None}
    needed.update(extras.values())
    include_chs = sorted({c for ch in needed for c in (ch, ch.rsplit("-", 1)[0])})

    raw = mne.io.read_raw_edf(str(edf_path), include=include_chs, preload=True, verbose="ERROR")
    try:
        loaded_sfreq = float(raw.info["sfreq"])
        data_dict = {ch.strip(): raw.get_data(picks=[ch])[0].astype(np.float64) for ch in raw.ch_names}
    finally:
        raw.close()

    return {
        "info": info,
        "loaded_sfreq": loaded_sfreq,
        "data_dict": data_dict,
        "ch_map": ch_map,
        "extras": extras,
    }
