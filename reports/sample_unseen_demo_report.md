# SomniMap Research Analysis Report — `SC4001E0-PSG.edf`

**Generated:** 2026-11-27 06:47:02 UTC  
**Research Disclaimer:** *SomniMap is an exploratory sleep-signal research tool, not a validated clinical diagnostic device. Predictive outputs must not be interpreted as medical diagnoses.*

---

## 1. Recording & Channel Provenance
- **File:** `SC4001E0-PSG.edf` | **Duration:** 79500.0 s (2650 × 30-s epochs)
- **Header Sampling Rate:** 100.0 Hz | **Loaded Subset Rate:** 100.0 Hz
- **Model Checkpoint:** `phase4_SomniMapNet_bilstm_w20_E8_dualhead_best.pt` (`bilstm`, `in_channels=6`, temperature $T=1.1757$, stride=10)
- **Canonical Channels Mapped:** `['EOG', 'EMG_CHIN', 'EEG_C4_A1']`
- **Optional Multimodal Channels:** `{'RESP_FLOW': 'Resp oro-nasal'}`
- **Compatibility / Channel Warnings:**
  - Primary CAP EEG (C4-A1/C3-A2) absent; mapped 'EEG Fpz-Cz' as primary EEG.
  - Missing canonical channels (masked to 0, not fabricated): ['EEG_F4_C4', 'EEG_C4_P4', 'EEG_FP2_F4']

## 2. Predictive Outputs — Sleep-Stage Architecture
| Stage | Epochs | Percentage (%) |
|---|---:|---:|
| Wake | 453 | 17.09% |
| N1 | 21 | 0.79% |
| N2 | 184 | 6.94% |
| N3 | 297 | 11.21% |
| REM | 1695 | 63.96% |

- **Mean Calibrated Confidence:** 0.903 | **Mean Normalized Entropy:** 0.161
- **Low-Confidence Epochs (< 0.55):** 120 / 2650 (4.5%)

## 3. Predictive Outputs — CAP Event Analysis (Predicted NREM)
- **Predicted NREM CAP Rate:** 2.68%
- **Detected Events (>= 2.0 s, smoothed):** Total = 90 (`A1=86`, `A2=2`, `A3=2`)
- *Note:* CAP A2 detection has lower sensitivity in automated scoring; event counts should be treated as exploratory.

## 4. Exploratory Signal-Derived Physiology (Brain–Heart–Respiration)
*Exploratory signal-derived physiological measurements only; not a clinical diagnosis.*

- **Cardiac (ECG):** Unavailable (ECG channel not present in EDF)
- **Respiratory Rate:** Mean = **19.04 breaths/min**
- **SpO2:** Unavailable (SpO2 channel not present in EDF)