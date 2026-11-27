"""Exact Phase-1-parity preprocessing for raw EDF inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.data.preprocessing import CANONICAL_CHANNELS, filter_and_resample
from src.inference.edf_loader import load_edf_channels


def preprocess_raw_edf(edf_path: str | Any, cfg: dict[str, Any]) -> dict[str, Any]:
    """Preprocess a raw EDF into [N_epochs, 3000, 7] normalized arrays + QC flags."""
    loaded = load_edf_channels(edf_path)
    info = loaded["info"]
    loaded_sfreq = loaded["loaded_sfreq"]
    data_dict = loaded["data_dict"]
    ch_map = loaded["ch_map"]

    pre_cfg = cfg["preprocessing"]
    target_sfreq = float(pre_cfg["target_sfreq_hz"])
    epoch_sec = int(pre_cfg["epoch_duration_sec"])
    samples_per_epoch = int(round(target_sfreq * epoch_sec))

    processed_channels: list[np.ndarray | None] = [None] * len(CANONICAL_CHANNELS)
    channel_mask = np.zeros(len(CANONICAL_CHANNELS), dtype=np.uint8)
    n_target_samples: int | None = None

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
            orig_sfreq=loaded_sfreq,
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
        raise RuntimeError("No canonical PSG channels could be preprocessed.")

    n_epochs = n_target_samples // samples_per_epoch
    usable_samples = n_epochs * samples_per_epoch

    epochs_arr = np.zeros((n_epochs, samples_per_epoch, len(CANONICAL_CHANNELS)), dtype=np.float32)
    qc_flags = np.zeros((n_epochs, len(CANONICAL_CHANNELS)), dtype=np.uint8)

    for c_idx, _ in enumerate(CANONICAL_CHANNELS):
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
        qc_flags[ep_max_abs > (15.0 * scale), c_idx] |= 2

        normed = np.clip((arr_trim - med) / scale, -20.0, 20.0).astype(np.float32)
        normed[flat_mask] = 0.0
        epochs_arr[:, :, c_idx] = normed

    # If C4-A1 (slot 0) is absent and C3-A2 (slot 6) is present, route C3-A2 into slot 0 for 6-ch model input
    model_epochs = epochs_arr[:, :, :6].copy()
    model_ch_mask = channel_mask[:6].copy()
    if model_ch_mask[0] == 0 and channel_mask[6] == 1:
        model_epochs[:, :, 0] = epochs_arr[:, :, 6]
        model_ch_mask[0] = 1

    return {
        "info": info,
        "epochs": epochs_arr,
        "qc_flags": qc_flags,
        "channel_mask": channel_mask,
        "model_epochs": model_epochs,
        "model_ch_mask": model_ch_mask,
        "n_epochs": n_epochs,
        "loaded_sfreq": loaded_sfreq,
        "extras_raw": {k: data_dict[v] for k, v in loaded["extras"].items() if v in data_dict},
    }
