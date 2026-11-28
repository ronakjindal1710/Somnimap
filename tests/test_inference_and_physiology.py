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


def test_domain_shift_warning_triggered_on_ood_sleep_edf() -> None:
    from src.inference.report_export import assess_domain_shift, build_analysis_report_markdown, build_analysis_summary_json

    edf_info = {
        "file_name": "SC4001E0-PSG.edf",
        "duration_sec": 79500.0,
        "header_sfreq_hz": 100.0,
        "canonical_mapping": {"EEG_C4_A1": ["EEG Fpz-Cz", None], "EOG": ["EOG horizontal", None]},
        "multimodal_extras": {},
        "warnings": ["Primary CAP EEG (C4-A1/C3-A2) absent; mapped 'EEG Fpz-Cz' as primary EEG."],
    }
    predictions = {
        "n_epochs": 2650,
        "stage_counts": {"Wake": 500, "N1": 16, "N2": 190, "N3": 246, "REM": 1698},
        "stage_percentages": {"Wake": 18.87, "N1": 0.60, "N2": 7.17, "N3": 9.28, "REM": 64.08},
        "stage_confidence": np.full(2650, 0.898, dtype=np.float32),
        "stage_entropy": np.full(2650, 0.15, dtype=np.float32),
        "low_confidence_epochs": [],
        "cap_rate_nrem": 0.12,
        "cap_events": [],
        "cap_event_counts": {"A1": 0, "A2": 0, "A3": 0},
        "provenance": {"checkpoint": "test.pt", "context_type": "bilstm", "in_channels": 6, "temperature": 1.1757, "stride_epochs": 5},
    }
    ds = assess_domain_shift(edf_info, predictions)
    assert ds["high_domain_shift"] is True
    assert ds["n_missing_canonical_eeg"] == 4
    assert ds["fallback_montage_used"] is True
    assert any("REM share" in r for r in ds["reasons"])
    assert any("N2 share" in r for r in ds["reasons"])

    md = build_analysis_report_markdown(
        {"edf_info": edf_info, "predictions": predictions, "loaded_sfreq": 100.0},
        {"disclaimer": "test", "ecg": {"available": False}, "respiratory": {"available": False}, "spo2": {"available": False}},
        recording_name="custom_upload.edf",
    )
    assert "HIGH DOMAIN-SHIFT / UNCALIBRATED MONTAGE WARNING" in md
    assert "r = 0.9319" in md
    assert "custom_upload.edf" in md

    import json
    js = json.loads(
        build_analysis_summary_json(
            {"edf_info": edf_info, "predictions": predictions},
            {"ecg": {"available": False}},
            recording_name="custom_upload.edf",
        )
    )
    assert js["domain_shift_assessment"]["high_domain_shift"] is True
    assert js["edf_info"]["file_name"] == "custom_upload.edf"


def test_domain_shift_nominal_on_full_channel_cap() -> None:
    from src.inference.report_export import assess_domain_shift

    edf_info = {
        "canonical_mapping": {
            "EEG_C4_A1": ["C4-A1", None],
            "EEG_F4_C4": ["F4-C4", None],
            "EEG_C4_P4": ["C4-P4", None],
            "EEG_FP2_F4": ["FP2-F4", None],
            "EOG": ["ROC-LOC", None],
            "EMG_CHIN": ["EMG1-EMG2", None],
        },
        "warnings": [],
    }
    predictions = {
        "stage_percentages": {"Wake": 15.0, "N1": 5.0, "N2": 45.0, "N3": 20.0, "REM": 15.0},
    }
    ds = assess_domain_shift(edf_info, predictions)
    assert ds["high_domain_shift"] is False
    assert ds["reasons"] == []


def test_predictor_loads_unregistered_no_chan_attn_w10_checkpoint() -> None:
    from src.inference.predictor import SomniMapPredictor

    cfg = load_config()
    ckpt_p = Path(cfg["paths"]["project_root"]) / "checkpoints" / "phase4_SomniMapNet_bilstm_w10_E3_w10_best.pt"
    if not ckpt_p.exists():
        pytest.skip("phase4_SomniMapNet_bilstm_w10_E3_w10_best.pt not present")

    pred_wrapper = SomniMapPredictor(ckpt_name="phase4_SomniMapNet_bilstm_w10_E3_w10_best.pt", device="cpu", cfg=cfg)
    assert pred_wrapper.in_channels == 1
    assert pred_wrapper.chan_attn is False
    assert pred_wrapper.window_epochs == 10
    dummy_x = np.zeros((12, 3000, 6), dtype=np.float32)
    dummy_mask = np.ones(6, dtype=np.float32)
    out = pred_wrapper.predict_arrays(dummy_x, dummy_mask, stride_epochs=2)
    assert out["n_epochs"] == 12


def test_conditional_gt_evaluation_absent_vs_present(tmp_path: Path) -> None:
    from src.inference.report_export import (
        build_analysis_report_markdown,
        build_analysis_summary_json,
        evaluate_against_reference_txt,
    )

    # Synthetic 3-epoch predictions
    cap_frames = np.zeros((3, 300), dtype=np.int64)
    cap_frames[1, 50:90] = 2  # 4.0s A2 at t=35..39s
    predictions = {
        "n_epochs": 3,
        "stage_preds": np.array([0, 2, 3], dtype=np.int64),
        "stage_confidence": np.array([0.9, 0.85, 0.88], dtype=np.float32),
        "stage_entropy": np.array([0.1, 0.15, 0.12], dtype=np.float32),
        "low_confidence_epochs": [],
        "stage_counts": {"Wake": 1, "N1": 0, "N2": 1, "N3": 1, "REM": 0},
        "stage_percentages": {"Wake": 33.33, "N1": 0.0, "N2": 33.33, "N3": 33.33, "REM": 0.0},
        "cap_frames": cap_frames,
        "cap_events": [{"subtype": 2, "subtype_name": "A2", "start_s": 35.0, "end_s": 39.0, "duration_s": 4.0, "epoch_idx": 1}],
        "cap_rate_nrem": 0.0667,
        "cap_event_counts": {"A1": 0, "A2": 1, "A3": 0},
        "provenance": {"checkpoint": "test.pt", "context_type": "bilstm", "in_channels": 6, "temperature": 1.1757, "stride_epochs": 5},
    }
    edf_info = {
        "file_name": "demo.edf",
        "duration_sec": 90.0,
        "header_sfreq_hz": 100.0,
        "canonical_mapping": {"EEG_C4_A1": ["C4-A1", None], "EEG_F4_C4": ["F4-C4", None], "EEG_C4_P4": ["C4-P4", None], "EEG_FP2_F4": ["FP2-F4", None]},
        "multimodal_extras": {},
        "warnings": [],
    }
    phys = {"disclaimer": "test", "ecg": {"available": False}, "respiratory": {"available": False}, "spo2": {"available": False}}

    # 1. Absent .txt -> None and omitted from reports
    assert evaluate_against_reference_txt(None, "2000-01-01T23:59:30+00:00", predictions) is None
    md_blind = build_analysis_report_markdown({"edf_info": edf_info, "predictions": predictions}, phys, gt_eval=None)
    assert "NIDRA Research Analysis Report" in md_blind
    assert "Ground-Truth Evaluation" not in md_blind

    # 2. Present .txt -> exact metrics computed and included in reports
    txt_file = tmp_path / "ref.txt"
    txt_file.write_text(
        "Sleep Stage\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation\n"
        "W\t23:59:30\tSLEEP-S0\t30\tEEG\n"
        "N2\t00:00:00\tSLEEP-S2\t30\tEEG\n"
        "N2\t00:00:05\tMCAP-A2\t4\tEEG\n"
        "N3\t00:00:30\tSLEEP-S3\t30\tEEG\n",
        encoding="utf-8",
    )
    gt_eval = evaluate_against_reference_txt(txt_file, "2000-01-01T23:59:30+00:00", predictions)
    assert gt_eval is not None
    assert gt_eval["n_scored_epochs"] == 3
    assert gt_eval["stage_accuracy"] == 1.0
    assert gt_eval["cap_frame_metrics"]["f1"] == 1.0
    assert gt_eval["cap_event_metrics"]["f1"] == 1.0

    md_gt = build_analysis_report_markdown({"edf_info": edf_info, "predictions": predictions}, phys, gt_eval=gt_eval)
    assert "Ground-Truth Evaluation" in md_gt
    import json
    js_gt = json.loads(build_analysis_summary_json({"edf_info": edf_info, "predictions": predictions}, phys, gt_eval=gt_eval))
    assert "ground_truth_evaluation" in js_gt
    assert js_gt["ground_truth_evaluation"]["stage_accuracy"] == 1.0
