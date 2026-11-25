"""Leakage-safe PSG preprocessing pipeline for SomniMap (Phase 1).

Reads raw EDF files and RemLogic .txt annotations, harmonises channels into a
7-channel canonical montage with explicit channel-presence masks, applies 50 Hz
notch + band-pass filtering, resamples to 100 Hz, computes per-epoch QC flags,
applies per-recording robust z-score normalisation, aligns 30-s sleep stages and
CAP A1/A2/A3 events (handling midnight rollover), and writes per-record arrays.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import yaml
from scipy import signal

from src.data.annotation_parser import parse_txt_annotations


CANONICAL_CHANNELS = [
    "EEG_C4_A1",
    "EEG_F4_C4",
    "EEG_C4_P4",
    "EEG_FP2_F4",
    "EOG",
    "EMG_CHIN",
    "EEG_C3_A2",
]


def _norm_label(lbl: str) -> str:
    return lbl.upper().replace("-", "").replace(" ", "").replace("EEG", "")


def resolve_channel_sources(all_labels: list[str]) -> dict[str, tuple[str, str | None]]:
    """Map canonical channel names to (pos_label, neg_label_or_None) in an EDF."""
    by_norm = {_norm_label(lbl): lbl for lbl in all_labels}
    by_upper = {lbl.upper().strip(): lbl for lbl in all_labels}

    def find_bipolar_or_diff(bipolar_cands: list[str], pos_cands: list[str], neg_cands: list[str]) -> tuple[str, str | None] | None:
        for cand in bipolar_cands:
            if cand in by_norm:
                return (by_norm[cand], None)
        pos = next((by_norm[p] for p in pos_cands if p in by_norm), None)
        neg = next((by_norm[n] for n in neg_cands if n in by_norm), None)
        if pos and neg:
            return (pos, neg)
        return None

    mapping: dict[str, tuple[str, str | None]] = {}
    specs = {
        "EEG_C4_A1": (["C4A1"], ["C4"], ["A1"]),
        "EEG_F4_C4": (["F4C4"], ["F4"], ["C4"]),
        "EEG_C4_P4": (["C4P4"], ["C4"], ["P4"]),
        "EEG_FP2_F4": (["FP2F4"], ["FP2"], ["F4"]),
        "EEG_C3_A2": (["C3A2"], ["C3"], ["A2"]),
    }
    for cname, (bip, pos, neg) in specs.items():
        res = find_bipolar_or_diff(bip, pos, neg)
        if res is not None:
            mapping[cname] = res

    # EOG
    eog_res = find_bipolar_or_diff(
        ["ROCLOC", "LOCROC", "LOCA1", "ROCA2", "EOG", "EOGL", "EOGR"],
        ["ROC", "EOGL"],
        ["LOC", "EOGR"],
    )
    if eog_res is None:
        for up, orig in by_upper.items():
            if "EOG" in up or "LOC" in up or "ROC" in up:
                eog_res = (orig, None)
                break
    if eog_res is not None:
        mapping["EOG"] = eog_res

    # Chin EMG
    emg_res = find_bipolar_or_diff(
        ["EMG1EMG2", "EMGEMG", "CHIN1CHIN2", "CHIN"],
        ["CHIN1", "EMG1"],
        ["CHIN2", "EMG2"],
    )
    if emg_res is None:
        for up, orig in by_upper.items():
            if ("EMG" in up or "CHIN" in up or "MENT" in up) and "TIB" not in up and "DX" not in up and "SX" not in up:
                emg_res = (orig, None)
                break
    if emg_res is not None:
        mapping["EMG_CHIN"] = emg_res

    return mapping


def filter_and_resample(
    sig: np.ndarray,
    orig_sfreq: float,
    target_sfreq: float,
    low_hz: float,
    high_hz: float,
    notch_hz: float,
) -> np.ndarray:
    """Apply 50 Hz notch, band-pass filter, and polyphase resample to target_sfreq."""
    x = sig.astype(np.float64, copy=False)
    nyq = 0.5 * orig_sfreq

    if notch_hz < nyq - 1.0:
        b_n, a_n = signal.iirnotch(notch_hz, Q=30.0, fs=orig_sfreq)
        x = signal.filtfilt(b_n, a_n, x)

    hi = min(high_hz, nyq * 0.95)
    if low_hz < hi:
        sos = signal.butter(4, [low_hz, hi], btype="bandpass", fs=orig_sfreq, output="sos")
        x = signal.sosfiltfilt(sos, x)

    if abs(orig_sfreq - target_sfreq) > 1e-3:
        up = int(round(target_sfreq))
        down = int(round(orig_sfreq))
        g = math.gcd(up, down)
        x = signal.resample_poly(x, up // g, down // g)

    return x.astype(np.float32)


def _hms_to_sec(time_str: str) -> float:
    parts = time_str.replace(".", ":").split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600.0 + m * 60.0 + s


def compute_event_offsets_sec(edf_start_iso: str, events: list[dict[str, Any]]) -> list[float]:
    """Compute seconds since EDF start for each event, handling midnight rollover."""
    dt = datetime.fromisoformat(edf_start_iso)
    edf_hms_sec = dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond / 1e6

    offsets: list[float] = []
    prev_off = -1e9
    for ev in events:
        t_sec = _hms_to_sec(ev["time_str"])
        delta = t_sec - edf_hms_sec
        if delta < -3600.0:
            delta += 86400.0
        while delta < prev_off - 43200.0:
            delta += 86400.0
        offsets.append(delta)
        prev_off = delta
    return offsets


def process_single_record(
    rec_id: str,
    edf_path: Path,
    txt_path: Path,
    out_dir: Path,
    split_name: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Preprocess one PSG recording and save epochs, labels, QC, and metadata."""
    pre_cfg = cfg["preprocessing"]
    target_sfreq = float(pre_cfg["target_sfreq_hz"])
    epoch_sec = int(pre_cfg["epoch_duration_sec"])
    samples_per_epoch = int(round(target_sfreq * epoch_sec))
    cap_hz = 10  # 0.1 s resolution -> 300 steps per 30-s epoch
    cap_steps_per_epoch = cap_hz * epoch_sec

    raw_hdr = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    all_labels = [c.strip() for c in raw_hdr.ch_names]
    meas_date = raw_hdr.info.get("meas_date")
    edf_start_iso = meas_date.isoformat() if meas_date is not None else "2000-01-01T00:00:00+00:00"
    common_sfreq = float(raw_hdr.info["sfreq"])
    raw_hdr.close()

    ch_map = resolve_channel_sources(all_labels)
    needed_raw_chs = sorted({lbl for pair in ch_map.values() for lbl in pair if lbl is not None})
    # MNE appends -0/-1 to duplicate EDF channel names; include base names so MNE loads them
    include_chs = sorted({c for ch in needed_raw_chs for c in (ch, ch.rsplit("-", 1)[0])})

    raw = mne.io.read_raw_edf(str(edf_path), include=include_chs, preload=True, verbose="ERROR")
    data_dict = {ch.strip(): raw.get_data(picks=[ch])[0] for ch in raw.ch_names}
    raw.close()

    n_target_samples: int | None = None
    processed_channels: list[np.ndarray | None] = [None] * len(CANONICAL_CHANNELS)
    channel_mask = np.zeros(len(CANONICAL_CHANNELS), dtype=np.uint8)

    for idx, cname in enumerate(CANONICAL_CHANNELS):
        if cname not in ch_map:
            continue
        pos_lbl, neg_lbl = ch_map[cname]
        if pos_lbl not in data_dict or (neg_lbl is not None and neg_lbl not in data_dict):
            continue
        sig = data_dict[pos_lbl] if neg_lbl is None else (data_dict[pos_lbl] - data_dict[neg_lbl])
        low_hz = 10.0 if cname == "EMG_CHIN" else float(pre_cfg["bandpass_low_hz"])
        high_hz = 45.0 if cname == "EMG_CHIN" else float(pre_cfg["bandpass_high_hz"])
        resamp = filter_and_resample(
            sig,
            orig_sfreq=common_sfreq,
            target_sfreq=target_sfreq,
            low_hz=low_hz,
            high_hz=high_hz,
            notch_hz=float(pre_cfg["notch_hz"]),
        )
        processed_channels[idx] = resamp
        channel_mask[idx] = 1
        if n_target_samples is None or len(resamp) < n_target_samples:
            n_target_samples = len(resamp)

    if n_target_samples is None:
        raise RuntimeError(f"No canonical channels resolved for {rec_id}")

    n_epochs = n_target_samples // samples_per_epoch
    usable_samples = n_epochs * samples_per_epoch

    epochs_arr = np.zeros((n_epochs, samples_per_epoch, len(CANONICAL_CHANNELS)), dtype=np.float32)
    qc_flags = np.zeros((n_epochs, len(CANONICAL_CHANNELS)), dtype=np.uint8)
    norm_stats: dict[str, dict[str, float]] = {}

    for c_idx, cname in enumerate(CANONICAL_CHANNELS):
        arr = processed_channels[c_idx]
        if arr is None:
            qc_flags[:, c_idx] = 4  # missing channel
            continue

        arr_trim = arr[:usable_samples].reshape(n_epochs, samples_per_epoch)
        ep_std = np.std(arr_trim, axis=1)
        flat_mask = ep_std < 1e-8
        qc_flags[flat_mask, c_idx] |= 1

        valid_vals = arr_trim[~flat_mask]
        if valid_vals.size == 0:
            valid_vals = arr_trim.ravel()

        med = float(np.median(valid_vals))
        q75, q25 = np.percentile(valid_vals, [75, 25])
        iqr = float(q75 - q25)
        scale = iqr if iqr > 1e-8 else (float(np.std(valid_vals)) + 1e-6)

        ep_max_abs = np.max(np.abs(arr_trim - med), axis=1)
        art_mask = ep_max_abs > (15.0 * scale)
        qc_flags[art_mask, c_idx] |= 2

        normed = np.clip((arr_trim - med) / (scale + 1e-6), -20.0, 20.0).astype(np.float32)
        epochs_arr[:, :, c_idx] = normed
        norm_stats[cname] = {"median": med, "iqr": scale}

    # Align sleep-stage & CAP annotations
    parsed = parse_txt_annotations(txt_path)
    labels_stage = np.full(n_epochs, -1, dtype=np.int8)
    sleep_offsets = compute_event_offsets_sec(edf_start_iso, parsed["sleep_events"])
    out_of_bounds_stage = 0
    for ev, off in zip(parsed["sleep_events"], sleep_offsets):
        ep_idx = int(round(off / float(epoch_sec)))
        if 0 <= ep_idx < n_epochs:
            labels_stage[ep_idx] = int(ev["stage_id"])
        else:
            out_of_bounds_stage += 1

    labels_cap = np.zeros((n_epochs * cap_steps_per_epoch,), dtype=np.int8)
    cap_subtype_map = {"A1": 1, "A2": 2, "A3": 3}
    cap_offsets = compute_event_offsets_sec(edf_start_iso, parsed["cap_events"])
    out_of_bounds_cap = 0
    for ev, off in zip(parsed["cap_events"], cap_offsets):
        s_idx = int(round(off * cap_hz))
        e_idx = int(round((off + float(ev["duration_s"])) * cap_hz))
        if e_idx <= 0 or s_idx >= len(labels_cap):
            out_of_bounds_cap += 1
            continue
        s_clip = max(0, s_idx)
        e_clip = min(len(labels_cap), max(s_clip + 1, e_idx))
        labels_cap[s_clip:e_clip] = cap_subtype_map.get(ev["subtype"], 0)
    labels_cap = labels_cap.reshape(n_epochs, cap_steps_per_epoch)

    rec_out = out_dir / rec_id
    rec_out.mkdir(parents=True, exist_ok=True)
    np.save(rec_out / "epochs.npy", epochs_arr)
    np.save(rec_out / "labels_stage.npy", labels_stage)
    np.save(rec_out / "labels_cap.npy", labels_cap)
    np.save(rec_out / "qc_flags.npy", qc_flags)

    meta = {
        "record_id": rec_id,
        "split": split_name,
        "n_epochs": int(n_epochs),
        "target_sfreq_hz": target_sfreq,
        "cap_label_hz": cap_hz,
        "canonical_channels": CANONICAL_CHANNELS,
        "channel_mask": channel_mask.tolist(),
        "resolved_sources": {k: list(v) for k, v in ch_map.items()},
        "n_valid_stage_epochs": int((labels_stage >= 0).sum()),
        "n_masked_stage_epochs": int((labels_stage < 0).sum()),
        "n_cap_positive_frames": int((labels_cap > 0).sum()),
        "out_of_bounds_stage_events": out_of_bounds_stage,
        "out_of_bounds_cap_events": out_of_bounds_cap,
        "norm_stats": norm_stats,
    }
    (rec_out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def run_preprocessing(config_path: Path, limit: int | None = None) -> None:
    """Run preprocessing across all eligible records in split_manifest.json."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dataset_dir"])
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    split_data = json.loads(Path(cfg["paths"]["split_manifest_json"]).read_text(encoding="utf-8"))

    tasks: list[tuple[str, str]] = []
    for split_name, p_info in split_data["partitions"].items():
        for rec_id in p_info["record_ids"]:
            tasks.append((rec_id, split_name))
    if limit is not None:
        tasks = tasks[:limit]

    t0 = time.time()
    summaries = []
    for i, (rec_id, split_name) in enumerate(tasks, 1):
        meta_path = out_dir / rec_id / "metadata.json"
        if meta_path.exists():
            summaries.append(json.loads(meta_path.read_text(encoding="utf-8")))
            continue
        rec_t0 = time.time()
        meta = process_single_record(
            rec_id=rec_id,
            edf_path=raw_dir / f"{rec_id}.edf",
            txt_path=raw_dir / f"{rec_id}.txt",
            out_dir=out_dir,
            split_name=split_name,
            cfg=cfg,
        )
        summaries.append(meta)
        print(
            f"[{i:3d}/{len(tasks)}] {rec_id:8s} ({split_name:11s}) | "
            f"epochs={meta['n_epochs']:4d} (valid={meta['n_valid_stage_epochs']:4d}) | "
            f"ch_mask={meta['channel_mask']} | {time.time() - rec_t0:.1f}s",
            flush=True,
        )

    summary_df = pd.DataFrame([
        {
            "record_id": m["record_id"],
            "split": m["split"],
            "n_epochs": m["n_epochs"],
            "n_valid_stage_epochs": m["n_valid_stage_epochs"],
            "n_masked_stage_epochs": m["n_masked_stage_epochs"],
            "n_cap_positive_frames": m["n_cap_positive_frames"],
            "oob_stage": m["out_of_bounds_stage_events"],
            "oob_cap": m["out_of_bounds_cap_events"],
            "has_c4_a1": m["channel_mask"][0],
            "has_c3_a2": m["channel_mask"][6],
            "n_channels_present": sum(m["channel_mask"]),
        }
        for m in summaries
    ])
    summary_df.to_csv(out_dir / "preprocessing_summary.csv", index=False)
    print(f"Preprocessing complete for {len(summaries)} records in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N records for testing")
    args = parser.parse_args()
    cfg_path = Path("C:/Users/prave/.gemini/antigravity/scratch/SomniMap/experiments/configs/base_config.yaml")
    run_preprocessing(cfg_path, limit=args.limit)
