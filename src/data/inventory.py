"""EDF dataset inventory script for the PhysioNet CAP Sleep Database.

Reads EDF headers record by record (no fixed channel index or sample rate
assumptions), merges demographics from gender-age.xlsx, checks annotation
file availability, and writes data/manifest.csv.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mne
import openpyxl
import pandas as pd
import yaml


def load_demographics(xlsx_path: Path) -> dict[str, dict[str, Any]]:
    """Load sex and age metadata from gender-age.xlsx, normalizing known ID quirks."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active
    demo: dict[str, dict[str, Any]] = {}
    for row in ws.iter_rows(values_only=True):
        if not row or row[0] is None:
            continue
        raw_id = str(row[0]).strip().upper()
        # Fix known PhysioNet typo: SBD1/SBD2 -> SDB1/SDB2
        if raw_id.startswith("SBD"):
            raw_id = "SDB" + raw_id[3:]
        sex = str(row[1]).strip().upper() if len(row) > 1 and row[1] is not None else None
        age = row[2] if len(row) > 2 else None
        demo[raw_id] = {"sex": sex, "age": age}
    return demo


def classify_channel(label: str) -> str:
    """Heuristically classify a PSG channel label into modality."""
    up = label.upper().strip()
    if any(k in up for k in ["SAO2", "SPO2", "PLETH", "OXI", "RESP", "AIRFLOW", "THOR", "ABDO", "FLOW", "MIC", "SNORE", "POS", "STAT", "TORACE", "ADDOME", "FLUSSO"]):
        return "RESP_OTHER"
    if "ECG" in up or "EKG" in up:
        return "ECG"
    if "EOG" in up or "LOC" in up or "ROC" in up:
        return "EOG"
    if "EMG" in up or "CHIN" in up or "TIB" in up or "DX" in up or "SX" in up or "MENT" in up:
        return "EMG"
    if any(k in up for k in ["EEG", "FP", "C4", "C3", "F4", "F3", "P4", "P3", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "CZ", "FZ", "PZ", "F2", "C2", "A1", "A2"]):
        return "EEG"
    return "OTHER"


def inspect_edf_header(edf_path: Path) -> dict[str, Any]:
    """Read EDF header metadata via MNE-Python without loading signal arrays."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    info = raw.info
    labels = [ch.strip() for ch in raw.ch_names]
    sfreq = float(info["sfreq"])
    n_channels = len(labels)
    duration_sec = float(raw.n_times / sfreq) if sfreq > 0 else 0.0
    meas_date = info.get("meas_date")
    start_dt = meas_date.isoformat() if meas_date is not None else ""

    # MNE resamples heterogeneous-rate EDF channels to the max rate on read_raw_edf;
    # record both MNE's common sfreq and per-channel original sfreqs if available.
    orig_sfreqs = {}
    if hasattr(raw, "_raw_extras") and raw._raw_extras:
        extra = raw._raw_extras[0]
        if "n_samps" in extra and "record_length" in extra:
            rec_len_val = extra["record_length"]
            rec_len = float(rec_len_val[0]) if hasattr(rec_len_val, "__len__") else float(rec_len_val)
            if rec_len > 0:
                for lbl, ns in zip(labels, extra["n_samps"]):
                    orig_sfreqs[lbl] = float(ns) / rec_len

    eeg_channels = []
    eog_channels = []
    emg_channels = []
    channel_sfreqs = {}
    channel_units = {}

    for i, lbl in enumerate(labels):
        ch_sf = orig_sfreqs.get(lbl, sfreq)
        channel_sfreqs[lbl] = ch_sf
        unit_obj = info["chs"][i].get("unit", "")
        channel_units[lbl] = str(unit_obj)
        mod = classify_channel(lbl)
        if mod == "EEG":
            eeg_channels.append(lbl)
        elif mod == "EOG":
            eog_channels.append(lbl)
        elif mod == "EMG":
            emg_channels.append(lbl)

    raw.close()

    return {
        "n_channels": n_channels,
        "duration_sec": round(duration_sec, 2),
        "n_epochs_30s": int(duration_sec // 30),
        "start_datetime": start_dt,
        "eeg_channels": "|".join(eeg_channels),
        "n_eeg": len(eeg_channels),
        "eog_channels": "|".join(eog_channels),
        "n_eog": len(eog_channels),
        "emg_channels": "|".join(emg_channels),
        "n_emg": len(emg_channels),
        "unique_eeg_sfreqs": "|".join(sorted({str(int(round(channel_sfreqs[c]))) for c in eeg_channels})),
        "all_labels": json.dumps(labels),
        "channel_sfreqs_json": json.dumps(channel_sfreqs),
        "channel_units_json": json.dumps(channel_units),
    }


def build_inventory(config_path: Path) -> pd.DataFrame:
    """Scan all recordings listed in RECORDS and write manifest.csv."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dataset_dir"])
    manifest_out = Path(cfg["paths"]["manifest_csv"])
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    records_file = raw_dir / "RECORDS"
    edf_names = [line.strip() for line in records_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    demo = load_demographics(raw_dir / "gender-age.xlsx")

    rows = []
    for edf_name in edf_names:
        record_id = edf_name.replace(".edf", "")
        group = re.sub(r"\d+$", "", record_id)
        edf_path = raw_dir / edf_name
        txt_path = raw_dir / f"{record_id}.txt"
        st_path = raw_dir / f"{edf_name}.st"

        d = demo.get(record_id.upper(), {"sex": None, "age": None})
        row: dict[str, Any] = {
            "record_id": record_id,
            "group": group,
            "is_healthy": group == "n",
            "sex": d["sex"],
            "age": d["age"],
            "edf_exists": edf_path.exists(),
            "edf_size_mb": round(edf_path.stat().st_size / (1024 * 1024), 2) if edf_path.exists() else 0.0,
            "txt_exists": txt_path.exists(),
            "st_exists": st_path.exists(),
            "header_ok": False,
            "header_error": "",
        }

        if edf_path.exists():
            try:
                hdr = inspect_edf_header(edf_path)
                row.update(hdr)
                row["header_ok"] = True
            except Exception as exc:
                row["header_error"] = str(exc)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(manifest_out, index=False)
    return df


if __name__ == "__main__":
    cfg_path = Path("C:/Users/prave/.gemini/antigravity/scratch/SomniMap/experiments/configs/base_config.yaml")
    df = build_inventory(cfg_path)
    print(f"Inventory complete: {len(df)} records, header_ok={df['header_ok'].sum()}/{len(df)}")
    print(f"Healthy: {df['is_healthy'].sum()}, Pathological: {(~df['is_healthy']).sum()}")
    print("Group counts:\n", df["group"].value_counts().to_string())
    print("\nMissing demographics:", df["age"].isna().sum())
    print("Unique EEG sampling rates:", df["unique_eeg_sfreqs"].value_counts().head(10).to_string())
