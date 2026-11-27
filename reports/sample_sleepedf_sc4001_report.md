# SomniMap Research Analysis Report — `SC4001E0-PSG.edf`

**Generated:** 2026-11-27 09:17:11 UTC  
**Research Disclaimer:** *SomniMap is an exploratory sleep-signal research tool, not a validated clinical diagnostic device. Predictive outputs must not be interpreted as medical diagnoses.*

---

## 1. Recording & Channel Provenance
- **File:** `SC4001E0-PSG.edf` | **Duration:** 79500.0 s (2650 × 30-s epochs)
- **Header Sampling Rate:** 100.0 Hz | **Loaded Subset Rate:** 100.0 Hz
- **Model Checkpoint:** `phase4_SomniMapNet_bilstm_w20_E8_dualhead_best.pt` (`bilstm`, `in_channels=6`, temperature $T=1.1757$, stride=5)
- **Canonical Channels Mapped:** `['EOG', 'EMG_CHIN', 'EEG_C4_A1']`
- **Optional Multimodal Channels:** `{'RESP_FLOW': 'Resp oro-nasal'}`
- **Compatibility / Channel Warnings:**
  - Primary CAP EEG (C4-A1/C3-A2) absent; mapped 'EEG Fpz-Cz' as primary EEG.
  - Missing canonical channels (masked to 0, not fabricated): ['EEG_F4_C4', 'EEG_C4_P4', 'EEG_FP2_F4']

> [!WARNING]
> **HIGH DOMAIN-SHIFT / UNCALIBRATED MONTAGE WARNING: 4/5 canonical EEG channels missing/masked to zero (['EEG_C3_A2', 'EEG_F4_C4', 'EEG_C4_P4', 'EEG_FP2_F4']); fallback EEG montage mapping used (recording montage differs from CAP Sleep Database); out-of-distribution predicted REM share (64.1% > 50%); out-of-distribution predicted N2 share (7.2% < 10%). High model confidence reflects in-distribution softmax calibration (T=1.1757) and must NOT be interpreted as in-domain physiological reliability under missing/mismatched montages.**

## 2. Predictive Outputs — Sleep-Stage Architecture
| Stage | Epochs | Percentage (%) |
|---|---:|---:|
| Wake | 453 | 17.09% |
| N1 | 16 | 0.60% |
| N2 | 190 | 7.17% |
| N3 | 293 | 11.06% |
| REM | 1698 | 64.08% |

- **Mean Calibrated Confidence:** 0.898 | **Mean Normalized Entropy:** 0.170
- **Low-Confidence Epochs (< 0.55):** 110 / 2650 (4.2%)

## 3. Predictive Outputs — CAP Event Analysis (Predicted NREM)
- **Predicted NREM CAP Rate:** 2.69% (macro CAP rate Pearson $r = 0.9319$ on `val`)
- **Detected Events (>= 2.5 s, smoothed, merge gap <= 1.5 s):** Total = 77 (`A1=71`, `A2=3`, `A3=3`)
- *Note:* Macro NREM CAP rate correlates strongly with ground truth on in-domain `val` (`r = 0.9319`), but per-event precision (`0.4558` overall) and `A2` subtype sensitivity (`F1 ~0.27–0.29` on `val`) remain exploratory; do not treat individual `A2` counts or exact event boundaries as clinical truth.

## 4. Exploratory Signal-Derived Physiology (Brain–Heart–Respiration)
*Exploratory signal-derived physiological measurements only; not a clinical diagnosis.*

- **Cardiac (ECG):** Unavailable (ECG channel not present in EDF)
- **Respiratory Rate:** Mean = **19.04 breaths/min**
- **SpO2:** Unavailable (SpO2 channel not present in EDF)