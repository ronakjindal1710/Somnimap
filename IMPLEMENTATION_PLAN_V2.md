# SomniMap — Post-Phase-8 Implementation & Model-Improvement Plan

**Created:** 2026-11-27  
**Baseline Commit:** `1f9f263` (`SomniMapNet E8_dualhead` — Test Macro-F1 `0.7382`, N1 F1 `0.4930`, CAP Event F1 `0.3780`, A2 F1 `0.0694`)  
**Goal:** Evolve SomniMap from an offline Phase 0–8 research pipeline into a reproducible, explainable, multimodal EDF analysis platform while diagnosing and improving N1 sleep staging and CAP A2/event detection.

---

## Part 1 — Model Improvements (Targeted, Evidence-Based)

### 1A. Fix the `nfle25` Alignment Bug & Audit Weak Recordings
- **Observed Evidence:** In `data/processed/preprocessing_summary.csv`, `nfle25` has `oob_stage = 518` and `oob_cap = 163` out of `522` epochs, explaining its test outlier score (`accuracy = 0.1776`, `kappa = -0.0046`). Meanwhile `rbd5` (`0.3404` macro-F1) and `rbd16` (`0.4654` macro-F1) have `oob_stage = 0` and 6/7 channels present.
- **Actions:**
  - Inspect `nfle25.txt` vs `nfle25.edf` start timestamp and midnight-rollover logic in `src/data/preprocessing.py`.
  - Audit per-subject confusion matrices, signal quality, and channel availability on `val` (keeping `test_locked` untouched).

### 1B. Improve Sleep-Stage N1 & Transition Accuracy (Validation-Driven)
Currently `E8_dualhead` achieves `N1 F1 = 0.4690` on `val` (`0.4930` on `test_locked`), while `Wake/N2/N3/REM` are `0.74–0.82`.
1. **Zero-Retrain Inference & Post-Processing Upgrades (Benchmark on `val` first):**
   - **Overlapping Sliding-Window Averaging:** Switch evaluation from non-overlapping `stride_epochs=20` to `stride_epochs=5` or `10`, averaging temperature-scaled (`T = 1.1757`) softmax probabilities across overlapping windows to eliminate boundary degradation.
   - **Viterbi / HMM Transition Smoothing:** Learn empirical 5×5 sleep-stage transition priors from `train` and apply Viterbi decoding / transition smoothing on `val` to suppress physiologically implausible 1-epoch flickers (e.g., `N2 → N1 → N2`).
2. **Retraining Upgrades (Only If Approved After Zero-Retrain Benchmark):**
   - **N1-Aware Sampling & Capped Class Weights:** Oversample windows containing $\ge 2$ N1 epochs in `SomniMapSequenceDataset` and cap extreme inverse-frequency weights with mild label smoothing.

### 1C. Improve CAP Event Detection (Fix Precision `0.3340` & `A2` F1 `0.0694`)
Why CAP event F1 is `0.3780` on `test_locked` (`0.2984` on `val`) and `A2` is `0.0694`:
1. **Zero-Retrain Post-Processing & Threshold Calibration (Tune on `val` only):**
   - Currently `src/training/eval_cap.py` uses raw `argmax` (`bg_prob_thresh=None`), `min_dur_s=0.5`, and `merge_gap_s=0.5`.
   - **Upgrades to benchmark on `val`:**
     - 1D probability smoothing at 10 Hz + **per-subtype probability thresholds** (`P(A1)`, `P(A2)`, `P(A3)`) so rare `A2` events are not suppressed by `0=none` or `A1/A3`.
     - **Physiological CAP Duration & Gap Constraints:** Enforce CAP A-phase duration bounds (e.g., `min_dur_s = 1.5–2.0 s`, max `60 s`, tuned on `val`) and subtype-consistent `merge_gap_s` to eliminate sub-second false positives (`7,117` FP on test).
     - **Predicted-NREM Masking Mode:** Evaluate both GT-NREM masking and predicted-NREM masking so CAP detection works during annotation-free EDF inference.
2. **Retraining Upgrades (Only If Approved):**
   - Replace unweighted `binary_cross_entropy_with_logits` in `src/training/train_somnimap.py` with **Class-Weighted Cross-Entropy + Dice/Focal Loss** over `[0=none, 1=A1, 2=A2, 3=A3]` and increase `cap_weight` (e.g., `0.3 → 1.0`).

---

## Part 2 — New Engineering & Platform Additions

### 2A. Portable Configuration & Checkpoint Registry
- Remove all 13 hardcoded `C:/Users/...` paths across `src/` by introducing a shared `src/config.py` resolver (`REPO_ROOT` / `SOMNIMAP_CONFIG` env var).
- Update `experiments/configs/base_config.yaml` and refresh `PROJECT_STATUS.md`.
- Add `checkpoints/registry.json` documenting every checkpoint's architecture, `in_channels`, `context_type`, `use_cap_head`, `temperature`, CAP thresholds, and preprocessing version.

### 2B. Standalone Raw-EDF Inference Engine (`src/inference/`)
Decoupled from Streamlit:
- `edf_loader.py`: Reads `.edf` via MNE; inspects channel labels, sampling rates, units, and duration; maps canonical EEG/EOG/EMG + optional **ECG** and **Respiratory/SpO2** channels without fabricating missing channels.
- `preprocess_edf.py`: Exact Phase 1 preprocessing parity (50 Hz notch, 0.3–35 Hz / 10–45 Hz bandpass, 100 Hz resample, 30 s epochs, per-recording robust z-score) + per-epoch QC flags (flatline, clipping, extreme amplitude).
- `predictor.py`: Overlapping-window `SomniMapNet` inference (CPU + optional CUDA), temperature scaling, calibrated CAP post-processing, predictive entropy/confidence, and structured JSON/DataFrame output.
- Optional `.edf.st` / `.txt` parser strictly separated as ground-truth reference overlay (never required for inference).

### 2C. Multimodal Brain–Heart–Respiration Analytics (`src/physiology/`)
Leverages ECG (present in 107/108 CAP EDFs) and Respiratory/SpO2 channels (present in 94/108 CAP EDFs) when available:
- **Cardiac (ECG):** R-peak detection, per-epoch Heart Rate (bpm), SDNN / RMSSD HRV aligned to predicted sleep stages and CAP A1/A2/A3 events.
- **Respiratory / $\text{SpO}_2$:** Breathing rate, effort/flow envelope variability, and $\text{SpO}_2$ summary when present.
- **Coupling Summary:** Stage-stratified brain–heart–respiration measurements explicitly labeled as exploratory physiological features, not clinical diagnoses.

### 2D. Upgraded Streamlit Platform (`src/app/streamlit_app.py`)
Extend the existing 8-screen dashboard:
- **Raw EDF Upload & Compatibility Screen:** Upload `.edf` (+ optional `.edf.st`/`.txt`), channel inventory, sampling-rate/unit validation, and missing-channel warnings.
- **Synchronized Multimodal Explorer:** EEG + EOG + EMG + ECG + Respiratory traces with predicted hypnogram and CAP timeline.
- **Uncertainty & Quality Panel:** Predictive entropy, low-confidence epoch flags, QC indicators, and model caveats.
- **Report Export:** Downloadable self-contained HTML/Markdown + CSV/JSON report with checkpoint provenance, plots, predictions, and research disclaimers.

### 2E. Automated Tests & Unseen External Generalization Demo
- `tests/` (`pytest`): Raw-EDF vs `data/processed/` parity test, missing-channel masking test, malformed EDF rejection, CAP post-processing invariants.
- **Unseen External Demo (3–4 Recordings):** Select 3–4 recordings from an open-access external dataset (e.g., **Sleep-EDF Expanded** on PhysioNet) since all 108 CAP Sleep recordings are already assigned in `data/splits/split_manifest.json`. Report external results strictly separate from CAP `test_locked`.

---

## Part 3 — Execution Sequence & Approval Gates

| Stage | Scope | Requires Approval? |
|---|---|---|
| **Stage 1** | Portable paths (`src/config.py`), `base_config.yaml`, checkpoint registry, `nfle25` & CAP duration diagnostics | Safe to start once approved |
| **Stage 2** | Zero-retrain N1 (overlap + Viterbi) & CAP (`val`-tuned thresholds/duration/smoothing) benchmarks on `val` | Needs approval for `val` tuning |
| **Stage 3** | Optional `SomniMapNet` retraining (N1-balanced + Weighted/Focal+Dice CAP loss, ideally on GPU) | Explicit approval required |
| **Stage 4** | Standalone `src/inference/` + `src/physiology/` + `pytest` parity & edge-case tests | Safe after Stage 1 |
| **Stage 5** | Upgrade `src/app/streamlit_app.py` (EDF upload, multimodal plots, uncertainty, report export) | Safe after Stage 4 |
| **Stage 6** | 3–4 external unseen EDFs (Sleep-EDF Expanded) + final docs & sample report | Explicit approval on external dataset choice |

---

## Post-Processing A2 Ceiling & Stage 3 Retraining Recommendation

- **Observed Post-Processing Ceiling (`val`):** Across the 72-config `val` sweep (`results/stage2_cap_sweep_val.json`), zero-retrain post-processing (`stride=5`, `smooth=7`, `bg_thresh=0.35`, `min_dur_s=2.5 s`, `merge_gap_s=1.5 s`) raises overall CAP event F1 from `0.2986` to `0.4326` (`a2_mult=2.0`, `A2` F1 `0.2662`, `predNREM` CAP-rate Pearson `r = 0.9319`) and reaches up to `A2` F1 `0.2938` (`a2_mult=2.5`, Overall F1 `0.4309`, `r = 0.9319`), where `A2` F1 plateaus around `~0.27–0.29`.
- **Root Cause:** `E8_dualhead` was trained with unweighted `binary_cross_entropy_with_logits` (or unweighted frame loss) where `0 = none` accounts for >90% of 10 Hz frames and `A2` is the rarest, morphologically intermediate subtype (between slow-wave `A1` and fast-EEG `A3`). Matched `A2` events already show acceptable localization (median IoU `~0.71`), meaning remaining false positives/misses are genuine model-level class boundary ambiguities rather than boundary jitter.
- **Next Highest-Leverage Experiment (Stage 3, requires explicit user approval):** Retrain `SomniMapNet` on GPU with class-weighted / Focal + Dice CAP loss (increasing `cap_weight` from `0.3` to `1.0`) and N1-balanced staging weights, keeping `test_locked` untouched until final evaluation.
