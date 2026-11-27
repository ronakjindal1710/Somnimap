# SomniMap — Project Status & Production Summary

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
