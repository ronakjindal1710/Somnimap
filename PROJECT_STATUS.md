# NIDRA (*Neural Inference for Deep, REM & Arousal Architecture*) — Project Status & Production Summary

**Last Updated:** 2026-11-27  
**Environment:** PyTorch (GPU RTX A5000 for Phases 3–8 training; portable CPU/GPU inference on laptop), MLflow (SQLite backend)  
**Dataset:** PhysioNet CAP Sleep Database (108 EDF recordings; subject-independent 70/15/15 split: Train=75, Val=17, Test=16 locked)

---

## Current Standing

**Phases 0–8 AND Post-Phase-8 Platform & Reliability Upgrades (Stages 1, 2, 4, 5, 6) are Complete.**  
The project now includes:
- Complete Phase 0–8 research results and locked test-set evaluation (`1f9f263`).
- Portable configuration (`src/config.py`) and checkpoint registry (`checkpoints/registry.json`).
- Fixed `common_sfreq` resampling bug in `src/data/preprocessing.py` and reprocessed the 3 affected records (`n4`, `n8`, `nfle25` — all dropped to `oob_stage=0, oob_cap=0`).
- Zero-retrain `val` benchmarks (`results/stage2_postproc_val.json`): overlapping `stride=5` inference + physiological CAP post-processing (`min_dur_s=2.0`, smoothing, A2 calibration) boosted `val` CAP F1 from `0.2986 → 0.4027` (+10.4 pp) and A2 F1 `0.0560 → 0.1987 / 0.2319` (3.5–4.1×), and lifted sleep-stage Macro-F1 from `0.7798 → 0.7850`.
- Standalone raw-EDF inference engine (`src/inference/`), multimodal ECG/respiratory/$\text{SpO}_2$ analytics (`src/physiology/`), Markdown/JSON report export (`src/inference/report_export.py`), and `pytest` suite (`tests/test_inference_and_physiology.py`, 5/5 passing).
- Upgraded 9-screen Streamlit platform (`src/app/streamlit_app.py`) with raw `.edf` upload, channel/QC validation, uncertainty/entropy, multimodal physiology, and downloadable reports.
- External unseen generalization demo on 3 PhysioNet Sleep-EDF Expanded recordings (`results/external_sleep_edf_demo.json`, `reports/sample_unseen_demo_report.md`), kept strictly separate from CAP `test_locked`.

---

## Phase-by-Phase Status

| Phase / Stage | Description | Status | Key Outputs |
|---|---|---|---|
| **Phase 0–3** | Audit, Preprocessing, Baselines, Temporal Models | ✅ **Complete** | `data/splits/split_manifest.json`, `results/phase2_baselines.csv`, `results/phase3_temporal.csv` |
| **Phase 4** | Flagship `SomniMapNet` & Ablations (E3–E8) | ✅ **Complete** | `results/phase4_somnimap.csv`, 9 checkpoints |
| **Phase 5–6** | CAP Eval, 1D Grad-CAM, IG, Temperature Scaling | ✅ **Complete** | `results/phase5_cap_eval_val.json`, `results/xai/` |
| **Phase 7–8** | Dashboard & Single-Shot Locked Test Eval | ✅ **Complete** | `results/phase8_test_locked_metrics.json`, `results/FINAL_RESEARCH_REPORT.md` |
| **Post-P8 Stage 1–2** | Portable Config, `nfle25` Fix, Zero-Retrain `val` Benchmarks | ✅ **Complete** | `src/config.py`, `checkpoints/registry.json`, `results/stage2_postproc_val.json` |
| **Post-P8 Stage 4–5** | Standalone EDF Inference, Multimodal Physiology, `pytest`, UI | ✅ **Complete** | `src/inference/`, `src/physiology/`, `tests/`, `src/app/streamlit_app.py` |
| **Post-P8 Stage 6** | External Unseen Demo (Sleep-EDF Expanded, 3 subjects) | ✅ **Complete** | `data/splits/external_demo_manifest.json`, `results/external_sleep_edf_demo.json`, `reports/sample_unseen_demo_report.md` |

---

## Optional Future Work (Stage 3 — Requires Explicit Approval)
- GPU retraining of `SomniMapNet` with class-weighted/focal+dice CAP loss (to further boost `A2` sensitivity at model level) and N1-balanced sampling, followed by an authorized versioned evaluation.

---

## Demo vs. Stage 3 Readiness Check

### Ready Right Now for Live Demo / Review
- **Raw `.edf` Upload & Inference (Screen 9):** Works end-to-end with `SomniMapPredictor` (`stride=5`, `T=1.1757`, frozen `val` CAP sweep defaults `smooth_frames=7, bg_thresh=0.35, a2_mult=2.0, min_dur_s=2.5, merge_gap_s=1.5` -> `val` CAP F1 `0.4326`, CAP Rate Pearson `r = 0.9319`).
- **Optional Ground-Truth Overlay:** Supports optional RemLogic `.txt` annotation upload (`parse_txt_annotations` + `align_annotations_to_epochs`) for side-by-side hypnogram comparison.
- **Domain-Shift & `A2` Guardrails:** Both Screen 9 and exported `.md`/`.json` reports automatically surface a `HIGH DOMAIN-SHIFT / UNCALIBRATED MONTAGE WARNING` whenever `> 2` canonical EEG channels are missing, fallback montage mapping is used, or predicted stage shares look out-of-distribution (`REM > 50%` or `N2 < 10%`), preventing high softmax confidence (`~0.90`) from being misread as in-domain physiological reliability. Macro CAP rate (`r = 0.9319` on `val`) is paired with explicit exploratory caveats on per-event precision (`0.4558`) and `A2` sensitivity (`F1 ~0.27–0.29`).
- **Honest Multimodal Physiology & Export:** Extracts ECG HR/SDNN/RMSSD, respiratory rate, and $\text{SpO}_2$ only when present (explicitly marking missing channels unavailable) and exports self-contained `.md` and `.json` reports.
- **Demo Recordings:**
  - *Full-channel non-training CAP `val` candidates* (note: all 108 CAP recordings are partitioned into `train(75)/val(17)/test_locked(16)`, with `test_locked` untouched): `n10`, `ins5`, `rbd17`, `sdb3` (e.g. `n10` achieves balanced sleep stages, `domain_shift=False`, `ecg_available=True`, `cap_rate_nrem=11.13%` — see `results/heldout_cap_vs_sleepedf_comparison.json`).
  - *External unseen partial-channel demo:* Sleep-EDF Expanded (`SC4001E0-PSG.edf`, `SC4011E0-PSG.edf`, `SC4021E0-PSG.edf`), which cleanly triggers the domain-shift warning (`4/5` canonical EEG missing, fallback montage, `REM=64.08%`, `N2=7.17%`).

### Requires Stage 3 GPU Retraining Later (Not Run Automatically)
- Pushing `A2` subtype F1 beyond the post-processing ceiling (`~0.27–0.29` on `val`) and improving fine-grained per-event precision (`0.4558` on `val`) requires GPU retraining of `SomniMapNet` with class-weighted / Focal + Dice CAP loss (`cap_weight` increased from `0.3` to `1.0`) and N1-balanced sampling.
- Improving cross-montage generalization to single-EEG external datasets (such as Sleep-EDF `Fpz-Cz`) without triggering domain-shift warnings requires montage-agnostic / channel-dropout augmentation during training.
