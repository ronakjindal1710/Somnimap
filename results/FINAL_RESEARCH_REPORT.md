# SomniMap — Final Research Report

**Date:** 2026-11-26  
**Hardware:** NVIDIA RTX A5000 (24 GB VRAM), PyTorch 2.5.1+cu121  
**Dataset:** PhysioNet CAP Sleep Database (107/108 EDF recordings preprocessed; `nfle12.edf` absent from raw folder; subject-independent 70/15/15 stratified split: Train=75, Val=17 [16 available], Test_locked=16)

---

## 1. Executive Summary

We completed the full SomniMap experimental pipeline (Phases 0–8) for joint 5-class sleep staging (Wake, N1, N2, N3, REM) and Cyclic Alternating Pattern (CAP A1/A2/A3) event detection.

- **Phase 2 → Phase 3 (Temporal Context):** Adding 20-epoch temporal context improved validation Macro-F1 from **0.6787** (`CNN1D`) to **0.7747** (`CNN_BiLSTM`), and `TemporalUNet` completed on GPU with **0.7585** Macro-F1 ($\kappa = 0.7757$).
- **Phase 4 (`SomniMapNet` E3–E8 Ablations on Validation):**
  - `bilstm` (0.7746) outperformed `tcn` (0.7704) and `transformer` (0.7422) bottlenecks.
  - 20-epoch windows (0.7746) outperformed 10-epoch (0.7523) and 40-epoch (0.7674) windows.
  - Multi-channel PSG with masked SE channel attention reached **0.7824** Macro-F1 (N1 F1 = 0.4558), outperforming single-EEG (0.7746, +5.0 pp N1 F1) and no-channel-attention (0.7476, **+3.48 pp**).
  - Weighted CE (0.7824) outperformed Focal Loss (0.7663).
  - Joint dual-head `SomniMapNet` (`E8_dualhead`, stage + CAP) achieved **0.7770** Macro-F1 ($\kappa = 0.7857$) and the highest validation **N1 F1 (0.4690)**.
- **Phase 5 (CAP Event Detection, IoU $\ge 0.5$):** On validation, `E8_dualhead` achieved **CAP Rate Pearson $r = 0.8835$** and overall event F1 = 0.2984 (A1 = 0.3276, A3 = 0.4203, Mean IoU = 0.7650, Onset error = 0.92s).
- **Phase 6 (XAI & Calibration):** Temperature scaling ($T = 1.1757$ fitted on `val`) reduced validation ECE from **0.0287 → 0.0179**. 1D Grad-CAM and Captum Integrated Gradients highlighted physiologically meaningful EEG/EOG/EMG motifs and channel importance shares.
- **Phase 8 (Single-Shot Locked Test Evaluation, 16 Unseen Subjects):**
  - **Macro-F1:** **0.7382** (95% CI [0.6494, 0.8015])
  - **Balanced Accuracy:** **0.7513** (95% CI [0.6638, 0.8182])
  - **Cohen's $\kappa$:** **0.7130** (95% CI [0.6046, 0.7954])
  - **N1 F1:** **0.4930** (95% CI [0.3213, 0.5946])
  - **Calibrated ECE:** **0.0542 → 0.0335** ($T = 1.1757$)
  - **CAP Event F1 (IoU $\ge 0.5$):** **0.3780** (A1 = 0.4479, A3 = 0.4600, Mean IoU = 0.7758, Onset error = 0.80s, CAP Rate $r = 0.7366$).

---

## 2. Phase 3 — Temporal Context Models (Validation, `w=20`)

| Model | Macro-F1 | Bal. Acc | Cohen's $\kappa$ | N1 F1 | Wake F1 | N2 F1 | N3 F1 | REM F1 | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`CNN_BiLSTM`** | **0.7747** | **0.7946** | **0.7854** | **0.4426** | 0.9065 | 0.8214 | **0.8677** | 0.8353 | 848,613 |
| `CNN_TCN` | 0.7732 | 0.7720 | 0.7814 | 0.4410 | **0.9086** | 0.8162 | 0.8354 | **0.8650** | 583,781 |
| `TemporalUNet` | 0.7585 | 0.7603 | 0.7757 | 0.3921 | 0.8897 | **0.8334** | 0.8404 | 0.8369 | 1,187,789 |
| `CNN_Transformer` | 0.7571 | 0.7718 | 0.7678 | 0.3984 | 0.9017 | 0.8228 | 0.8364 | 0.8261 | 461,669 |

---

## 3. Phase 4 — `SomniMapNet` Ablations E3–E8 (Validation)

| Run Name | Ablation | Mode | Window | Loss | Chan Attn | CAP Head | Macro-F1 | $\kappa$ | N1 F1 | Params |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|
| `E3_bilstm` | Bottleneck (BiLSTM) | single_eeg | 20 | weighted_ce | True | False | 0.7746 | **0.7987** | 0.4058 | 1,808,197 |
| `E3_tcn` | Bottleneck (TCN) | single_eeg | 20 | weighted_ce | True | False | 0.7704 | 0.7820 | 0.4375 | 1,695,685 |
| `E3_transformer` | Bottleneck (Transformer) | single_eeg | 20 | weighted_ce | True | False | 0.7422 | 0.7646 | 0.3427 | 1,969,093 |
| `E3_w10` | Window = 10 epochs | single_eeg | 10 | weighted_ce | True | False | 0.7523 | 0.7653 | 0.3932 | 1,808,197 |
| `E3_w40` | Window = 40 epochs | single_eeg | 40 | weighted_ce | True | False | 0.7674 | 0.7883 | 0.3936 | 1,808,197 |
| **`E5_multichan`** | Multi-Channel PSG | multi_channel | 20 | weighted_ce | True | False | **0.7824** | 0.7935 | 0.4558 | 1,815,283 |
| `E6_focal` | Focal Loss ($\gamma=2$) | multi_channel | 20 | focal | True | False | 0.7663 | 0.7766 | 0.4218 | 1,815,283 |
| `E7_no_chan_attn` | No Channel Attention | multi_channel | 20 | weighted_ce | False | False | 0.7476 | 0.7552 | 0.3991 | 1,808,197 |
| **`E8_dualhead`** | Dual-Head (Stage + CAP) | multi_channel | 20 | weighted_ce | True | True | 0.7770 | 0.7857 | **0.4690** | 1,838,615 |

---

## 4. Phase 8 — Locked Test-Set Results (`test_locked`, 16 Subjects, Single-Shot)

Evaluated **once** using `checkpoints/phase4_SomniMapNet_bilstm_w20_E8_dualhead_best.pt` with temperature $T = 1.1757$:

- **Overall Sleep Staging (15,623 epochs):**
  - Accuracy: **0.7893** | Weighted-F1: **0.7920**
  - Macro-F1: **0.7382** (95% subject-bootstrap CI: **[0.6494, 0.8015]**)
  - Balanced Accuracy: **0.7513** (95% CI: **[0.6638, 0.8182]**)
  - Cohen's $\kappa$: **0.7130** (95% CI: **[0.6046, 0.7954]**)
  - ECE (uncalibrated → calibrated): **0.0542 → 0.0335** | Brier: **0.3133 → 0.3100**

- **Per-Class Performance:**

| Stage | Precision | Sensitivity | Specificity | F1 | Support |
|---|---:|---:|---:|---:|---:|
| Wake | 0.7120 | 0.7787 | 0.9466 | 0.7439 | 2,264 |
| N1 | 0.4353 | 0.5682 | 0.9649 | 0.4930 | 711 |
| N2 | 0.8206 | 0.8141 | 0.8793 | 0.8173 | 6,314 |
| N3 | 0.8648 | 0.7660 | 0.9637 | 0.8124 | 3,633 |
| REM | 0.8188 | 0.8297 | 0.9616 | 0.8242 | 2,701 |

- **CAP Event Detection (IoU $\ge 0.5$, NREM frames):**
  - Overall: Precision = **0.3340**, Recall = **0.4354**, F1 = **0.3780**, Mean IoU = **0.7758**, Onset err = **0.80s**, Offset err = **1.12s**
  - A1: F1 = **0.4479** (P = 0.4087, R = 0.4955, Mean IoU = 0.7573)
  - A2: F1 = **0.0694** (P = 0.0661, R = 0.0730, Mean IoU = 0.6860)
  - A3: F1 = **0.4600** (P = 0.3739, R = 0.5979, Mean IoU = 0.8092)
  - Recording-level CAP Rate Pearson $r$: **0.7366**
- **Outlier Note:** Subject `nfle25` (535 epochs) had low test accuracy (0.1776, $\kappa = -0.0046$), widening the subject-level 95% bootstrap CI; most other test subjects achieved 0.76–0.91 accuracy (`plm2`: 0.9074, `n7`: 0.8959, `narco5`: 0.8934, `nfle21`: 0.8818).
