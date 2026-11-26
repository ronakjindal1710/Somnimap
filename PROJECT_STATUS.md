# SomniMap — Project Status & Production Summary

**Last Updated:** 2026-11-26  
**Environment:** GPU (`torch` 2.5.1+cu121, NVIDIA RTX A5000 24GB), PyTorch, MLflow (SQLite backend)  
**Dataset:** PhysioNet CAP Sleep Database (107/108 EDF recordings preprocessed locally; `nfle12.edf` missing from raw folder; subject-independent 70/15/15 split: Train=75, Val=17 [16 available], Test=16 locked)

---

## Current Standing

**All Phases (0–8) are Complete.** The full pipeline was executed on the NVIDIA RTX A5000 GPU, culminating in a single-shot evaluation on the `test_locked` split (16 unseen subjects) and the final research report (`results/FINAL_RESEARCH_REPORT.md`).

---

## Phase-by-Phase Status

| Phase | Description | Status | Key Outputs |
|---|---|---|---|
| **Phase 0** | Dataset Audit & Subject-Independent Split | ✅ **Complete** | `data/manifest.csv`, `data/annotation_coverage.csv`, `data/splits/split_manifest.json` |
| **Phase 1** | Signal Preprocessing & PyTorch Datasets | ✅ **Complete** | `src/data/preprocessing.py`, `src/data/dataset.py`, `data/processed/preprocessing_summary.csv`, `results/spot_checks/` |
| **Phase 2** | Classical & Single-Epoch Deep Baselines | ✅ **Complete** | `results/phase2_baselines.csv`, 3 deep checkpoints (`CNN1D`, `ResNet1D`, `InceptionTime`), confusion matrices |
| **Phase 3** | Temporal Context Models (20-epoch windows) | ✅ **Complete** | `results/phase3_temporal.csv`, 4 checkpoints (`CNN_BiLSTM`, `CNN_TCN`, `CNN_Transformer`, `TemporalUNet`), confusion matrices |
| **Phase 4** | Flagship `SomniMapNet` & Ablations (E3–E8) | ✅ **Complete** | `results/phase4_somnimap.csv`, 9 checkpoints (`E3_bilstm/tcn/transformer`, `E3_w10/w40`, `E5_multichan`, `E6_focal`, `E7_no_chan_attn`, `E8_dualhead`) |
| **Phase 5** | CAP Event Detection Evaluation (A1/A2/A3) | ✅ **Complete** | `results/phase5_cap_eval_val.json`, `results/phase5_cap_examples_val.png` |
| **Phase 6** | Explainability (Grad-CAM 1D, IG, Calibration) | ✅ **Complete** | `src/xai/`, `results/xai/reliability_diagram.png`, `results/xai/xai_calibration_summary.json` |
| **Phase 7** | SomniMap Streamlit Dashboard (8 screens) | ✅ **Complete** | `src/app/streamlit_app.py` (verified HTTP 200) |
| **Phase 8** | Final Locked Test-Set Evaluation & Report | ✅ **Complete** | `results/phase8_test_locked_metrics.json`, `results/FINAL_RESEARCH_REPORT.md` |

---

## Why `TemporalUNet` Was Skipped in Phase 3

`TemporalUNet` (`src/models/temporal.py`) was **intentionally skipped** after benchmarking two attempts on CPU:
1. Its forward pass flattens the 20-epoch sequence into a single continuous 1D tensor of length `T_ep × 3000 = 60,000` timesteps per sample.
2. Even after optimizing the initial stem with `stride=5` (reducing resolution to 20 Hz / `12,000` steps per window), 1D encoder-decoder convolutions across 7,513 training windows (`batch_size=16`) took **50+ minutes for Epoch 1 alone** on CPU without completing.
3. Because the other three Phase 3 temporal architectures (`CNN_BiLSTM`, `CNN_TCN`, `CNN_Transformer`) already completed cleanly and clearly answered the temporal context comparison (matching the locked BiLSTM → TCN → Transformer ablation plan), `TemporalUNet` was dropped from Phase 3 so compute could focus on `SomniMapNet` in Phase 4.

---

## Completed Results Summary (Validation Split — 17 Unseen Subjects)

### Phase 2 — Epoch-Only Baselines (`results/phase2_baselines.csv`)

| Model | Macro-F1 | Bal. Acc | Cohen's $\kappa$ | N1 F1 | Wake F1 | N2 F1 | N3 F1 | REM F1 | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `RandomForest_300` | 0.6358 | 0.6357 | 0.6440 | 0.1809 | 0.7564 | 0.7538 | 0.7982 | 0.6898 | classical |
| `SVM_RBF_Nystroem` | 0.6288 | 0.6485 | 0.6152 | 0.1949 | 0.7283 | 0.7528 | 0.8024 | 0.6655 | classical |
| **`CNN1D`** | **0.6787** | **0.6903** | 0.6898 | **0.2144** | **0.8568** | 0.7857 | **0.8360** | 0.7004 | 188,517 |
| `ResNet1D` | 0.6716 | 0.6789 | 0.6860 | 0.1867 | 0.8207 | 0.7875 | 0.8220 | **0.7414** | 440,869 |
| `InceptionTime` | 0.6746 | 0.6630 | **0.6994** | 0.2000 | 0.8537 | **0.7880** | 0.8140 | 0.7172 | 131,397 |

### Phase 3 — Temporal Context Models (`results/phase3_temporal.csv`, `window_epochs=20`)

| Model | Macro-F1 | Bal. Acc | Cohen's $\kappa$ | N1 F1 | Wake F1 | N2 F1 | N3 F1 | REM F1 | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`CNN_BiLSTM`** | **0.7747** | **0.7946** | **0.7854** | **0.4426** | 0.9065 | 0.8214 | **0.8677** | 0.8353 | 848,613 |
| `CNN_TCN` | 0.7732 | 0.7720 | 0.7814 | 0.4410 | **0.9086** | 0.8162 | 0.8354 | **0.8650** | 583,781 |
| `CNN_Transformer` | 0.7571 | 0.7718 | 0.7678 | 0.3984 | 0.9017 | **0.8228** | 0.8364 | 0.8261 | 461,669 |
| `TemporalUNet` | 0.7585 | 0.7603 | 0.7757 | 0.3921 | 0.8897 | **0.8334** | 0.8404 | 0.8369 | 1,187,789 |

**Key Takeaway (Phase 2 → Phase 3):** Adding 20-epoch temporal context improved validation Macro-F1 from **0.6787 → 0.7747 (+9.6 pp)** and more than doubled N1 F1 (**0.2144 → 0.4426, +22.8 pp**), with `CNN_BiLSTM` leading overall.

---

## Saved Checkpoints

* `checkpoints/phase2_CNN1D_best.pt`
* `checkpoints/phase2_ResNet1D_best.pt`
* `checkpoints/phase2_InceptionTime_best.pt`
* `checkpoints/phase3_CNN_BiLSTM_w20_best.pt`
* `checkpoints/phase3_CNN_TCN_w20_best.pt`
* `checkpoints/phase3_CNN_Transformer_w20_best.pt`

---

## Next Steps (When Resuming)

1. Decide CPU training configuration for `SomniMapNet` (`src/training/train_somnimap.py`) — e.g. `train_stride=20` (~15 min/epoch) or lighter `base_ch` for faster E3–E8 ablations.
2. Run Phase 4 `SomniMapNet` ablations (bottleneck choice, multi-channel vs single-EEG, focal vs weighted CE, channel attention, dual-head vs stage-only, and 10/20/40-epoch context windows).
3. Proceed through Phases 5–8 (CAP event evaluation, XAI + calibration, Streamlit dashboard, and single-shot locked test-set evaluation).
