# SomniMap Research Analysis Report — `n10.edf`

**Generated:** 2026-11-27 09:15:55 UTC  
**Research Disclaimer:** *SomniMap is an exploratory sleep-signal research tool, not a validated clinical diagnostic device. Predictive outputs must not be interpreted as medical diagnoses.*

---

## 1. Recording & Channel Provenance
- **File:** `n10.edf` | **Duration:** 29400.0 s (980 × 30-s epochs)
- **Header Sampling Rate:** 512.0 Hz | **Loaded Subset Rate:** 512.0 Hz
- **Model Checkpoint:** `phase4_SomniMapNet_bilstm_w20_E8_dualhead_best.pt` (`bilstm`, `in_channels=6`, temperature $T=1.1757$, stride=5)
- **Canonical Channels Mapped:** `['EEG_C4_A1', 'EEG_F4_C4', 'EEG_C4_P4', 'EEG_FP2_F4', 'EOG', 'EMG_CHIN']`
- **Optional Multimodal Channels:** `{'ECG': 'ECG1-ECG2'}`

## 2. Predictive Outputs — Sleep-Stage Architecture
| Stage | Epochs | Percentage (%) |
|---|---:|---:|
| Wake | 135 | 13.78% |
| N1 | 13 | 1.33% |
| N2 | 336 | 34.29% |
| N3 | 327 | 33.37% |
| REM | 169 | 17.24% |

- **Mean Calibrated Confidence:** 0.826 | **Mean Normalized Entropy:** 0.269
- **Low-Confidence Epochs (< 0.55):** 97 / 980 (9.9%)

## 3. Predictive Outputs — CAP Event Analysis (Predicted NREM)
- **Predicted NREM CAP Rate:** 11.13% (macro CAP rate Pearson $r = 0.9319$ on `val`)
- **Detected Events (>= 2.5 s, smoothed, merge gap <= 1.5 s):** Total = 324 (`A1=268`, `A2=25`, `A3=31`)
- *Note:* Macro NREM CAP rate correlates strongly with ground truth on in-domain `val` (`r = 0.9319`), but per-event precision (`0.4558` overall) and `A2` subtype sensitivity (`F1 ~0.27–0.29` on `val`) remain exploratory; do not treat individual `A2` counts or exact event boundaries as clinical truth.

## 4. Exploratory Signal-Derived Physiology (Brain–Heart–Respiration)
*Exploratory signal-derived physiological measurements only; not a clinical diagnosis.*

- **Cardiac (ECG):** Mean HR = **76.8 bpm** | SDNN = **75.27 ms** | RMSSD = **52.62 ms** | Beats = 37634
  - Mean HR change during predicted CAP events vs pre-event baseline: **+0.22 bpm**
- **Respiratory:** Unavailable (Respiratory channel not present in EDF)
- **SpO2:** Unavailable (SpO2 channel not present in EDF)