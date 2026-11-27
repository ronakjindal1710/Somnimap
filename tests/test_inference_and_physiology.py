"""Tests for standalone raw-EDF inference, preprocessing parity, CAP decoding, and physiology."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.inference.edf_loader import inspect_edf, resolve_channels_with_fallbacks
from src.inference.predictor import decode_cap_probabilities, predict_from_edf
from src.inference.preprocess_edf import preprocess_raw_edf
from src.physiology.multimodal import analyze_multimodal_physiology


def test_channel_mapping_rejects_unknown_and_never_fabricates() -> None:
    ch_map, warnings = resolve_channels_with_fallbacks(["TEMP", "BODY_POS", "MIC"])
    assert "EEG_C4_A1" not in ch_map
    assert "EEG_C3_A2" not in ch_map
    assert len(ch_map) == 0


def test_missing_edf_raises() -> None:
    with pytest.raises(FileNotFoundError):
        inspect_edf("non_existent_recording_12345.edf")


def test_cap_decoder_enforces_nrem_and_min_duration() -> None:
    n_ep = 4
    cap_probs = np.zeros((n_ep, 300, 4), dtype=np.float32)
    cap_probs[..., 0] = 0.9
    # Put a 3.0s A2 burst in epoch 0 (Wake -> must be masked out!)
    cap_probs[0, 50:80, 0] = 0.1
    cap_probs[0, 50:80, 2] = 0.9
    # Put a 0.6s short spike in epoch 1 (N2 -> too short < 2.0s, must be filtered out!)
    cap_probs[1, 10:16, 0] = 0.1
    cap_probs[1, 10:16, 1] = 0.9
    # Put a valid 3.0s A2 burst in epoch 2 (N2 -> must be kept!)
    cap_probs[2, 100:130, 0] = 0.1
    cap_probs[2, 100:130, 2] = 0.9

    stage_preds = np.array([0, 2, 2, 4], dtype=np.int64)  # Wake, N2, N2, REM
    frames, events = decode_cap_probabilities(
        cap_probs, stage_preds, smooth_frames=1, bg_thresh=0.35, a2_mult=1.5, min_dur_s=2.0, merge_gap_s=1.0
    )
    assert (frames[0] == 0).all()
    assert (frames[3] == 0).all()
    assert len(events) == 1
    assert events[0]["subtype_name"] == "A2"
    assert events[0]["duration_s"] >= 2.0


def test_multimodal_physiology_synthetic_ecg_and_missing_resp() -> None:
    sfreq = 100.0
    t = np.arange(0, 120, 1.0 / sfreq)
    # 60 bpm synthetic QRS spikes
    ecg = np.zeros_like(t)
    for sec in range(1, 120):
        idx = int(sec * sfreq)
        ecg[idx - 1 : idx + 2] = [0.5, 1.5, 0.5]
    stages = np.array([2, 2, 2, 2], dtype=np.int64)
    res = analyze_multimodal_physiology({"ECG": ecg}, sfreq=sfreq, stage_preds=stages)
    assert res["ecg"]["available"] is True
    assert 50.0 <= res["ecg"]["mean_hr_bpm"] <= 70.0
    assert res["respiratory"]["available"] is False
    assert res["spo2"]["available"] is False


def test_raw_edf_preprocessing_parity_and_inference_on_val_record() -> None:
    cfg = load_config()
    raw_edf = Path(cfg["paths"]["raw_dataset_dir"]) / "n10.edf"
    proc_epochs_p = Path(cfg["paths"]["processed_dir"]) / "n10" / "epochs.npy"
    if not raw_edf.exists() or not proc_epochs_p.exists():
        pytest.skip("Local n10.edf or preprocessed n10 not available")

    pre = preprocess_raw_edf(raw_edf, cfg)
    ref = np.load(proc_epochs_p)
    assert pre["epochs"].shape == ref.shape
    for c_idx in range(6):
        corr = float(np.corrcoef(pre["epochs"][:, :, c_idx].ravel(), ref[:, :, c_idx].ravel())[0, 1])
        assert corr > 0.98, f"Channel {c_idx} correlation too low: {corr}"

    out = predict_from_edf(raw_edf, stride_epochs=20, cfg=cfg)
    pred = out["predictions"]
    assert pred["n_epochs"] == ref.shape[0]
    assert len(pred["stage_preds"]) == ref.shape[0]
    assert "provenance" in pred and pred["provenance"]["checkpoint"].endswith(".pt")
    phys = analyze_multimodal_physiology(
        out["extras_raw"], out["loaded_sfreq"], pred["stage_preds"], pred.get("cap_events")
    )
    assert phys["ecg"]["available"] is True


def test_edf_loader_external_eeg_fallback() -> None:
    ch_map, warnings = resolve_channels_with_fallbacks(
        ["Fpz-Cz", "Pz-Oz", "EOG horizontal", "EMG submental"]
    )
    assert "EEG_C4_A1" in ch_map
    assert len(warnings) >= 1


def test_edf_loader_missing_emg_not_fabricated() -> None:
    ch_map, _ = resolve_channels_with_fallbacks(["Fpz-Cz"])
    assert "EMG_CHIN" not in ch_map
    assert "EOG" not in ch_map


def test_physiology_no_ecg_no_resp_no_spo2() -> None:
    res = analyze_multimodal_physiology({}, 100.0, np.array([2, 2, 2, 2], dtype=np.int64))
    assert res["ecg"]["available"] is False
    assert res["respiratory"]["available"] is False
    assert res["spo2"]["available"] is False


def test_cap_decoder_long_burst_not_split() -> None:
    n_ep = 3
    cap_probs = np.zeros((n_ep, 300, 4), dtype=np.float32)
    cap_probs[..., 0] = 0.9
    # 40s continuous A1 burst spanning epoch 0..1 (N2)
    cap_probs[0, 100:300, 0] = 0.1
    cap_probs[0, 100:300, 1] = 0.9
    cap_probs[1, 0:200, 0] = 0.1
    cap_probs[1, 0:200, 1] = 0.9

    stage_preds = np.array([2, 2, 2], dtype=np.int64)
    _, events = decode_cap_probabilities(
        cap_probs, stage_preds, smooth_frames=1, bg_thresh=0.35, a2_mult=1.5, min_dur_s=2.0, merge_gap_s=1.0
    )
    assert len(events) == 1
    assert events[0]["subtype_name"] == "A1"
    assert events[0]["duration_s"] >= 39.0


def test_align_annotations_to_epochs_helper() -> None:
    from src.data.preprocessing import align_annotations_to_epochs

    parsed = {
        "sleep_events": [
            {"time_str": "23:59:30", "stage_id": 0},
            {"time_str": "00:00:00", "stage_id": 2},
            {"time_str": "00:00:30", "stage_id": 3},
        ],
        "cap_events": [
            {"time_str": "00:00:05", "duration_s": 4.0, "subtype": "A2"},
        ],
    }
    st, cap = align_annotations_to_epochs(parsed, "2000-01-01T23:59:30+00:00", n_epochs=3)
    assert list(st) == [0, 2, 3]
    assert cap.shape == (3, 300)
    assert int((cap[1] == 2).sum()) == 40
