"""Annotation parser and coverage audit for CAP Sleep Database (.txt exports).

Parses RemLogic event export (.txt) files across all 108 recordings, maps R&K
sleep stages to SomniMap's 5-class scheme (keeping MT/UNSCORED masked), audits
CAP A1/A2/A3 event coverage, and writes data/annotation_coverage.csv.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


STAGE_MAP: dict[str, int] = {
    "SLEEP-S0": 0,   # Wake
    "SLEEP-S1": 1,   # N1
    "SLEEP-S2": 2,   # N2
    "SLEEP-S3": 3,   # N3
    "SLEEP-S4": 3,   # N3 (merged S3+S4)
    "SLEEP-REM": 4,  # REM
    "SLEEP-MT": -1,  # Movement time -> masked
    "SLEEP-UNSCORED": -1,  # Unscored -> masked
}

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM", -1: "Masked"}


def parse_txt_annotations(txt_path: Path) -> dict[str, Any]:
    """Parse a single RemLogic .txt annotation file."""
    lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    table_start = None
    col_map: dict[str, int] = {}
    for idx, line in enumerate(lines):
        if line.strip().startswith("Sleep Stage") and "Event" in line:
            header_parts = [h.strip() for h in line.split("\t")]
            for c_i, h_name in enumerate(header_parts):
                hl = h_name.lower()
                if hl.startswith("sleep stage"):
                    col_map["stage"] = c_i
                elif hl.startswith("time"):
                    col_map["time"] = c_i
                elif hl == "event":
                    col_map["event"] = c_i
                elif hl.startswith("duration"):
                    col_map["duration"] = c_i
                elif hl == "location":
                    col_map["location"] = c_i
            table_start = idx + 1
            break

    if table_start is None or "event" not in col_map:
        return {
            "parse_ok": False,
            "error": "Table header not found",
            "sleep_events": [],
            "cap_events": [],
            "other_events": [],
        }

    sleep_events = []
    cap_events = []
    other_events = Counter()

    min_cols = max(col_map["event"], col_map.get("duration", 0)) + 1
    for line in lines[table_start:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < min_cols:
            parts = re.split(r"\s{2,}|\t", line.strip())
        if len(parts) < min_cols:
            continue

        stage_col = parts[col_map["stage"]].strip() if "stage" in col_map else ""
        time_str = parts[col_map["time"]].strip() if "time" in col_map else ""
        event = parts[col_map["event"]].strip().upper()
        try:
            dur = float(parts[col_map["duration"]].strip()) if "duration" in col_map else 0.0
        except ValueError:
            dur = 0.0
        loc_idx = col_map.get("location")
        location = parts[loc_idx].strip() if (loc_idx is not None and loc_idx < len(parts)) else ""

        if event.startswith("SLEEP-"):
            mapped = STAGE_MAP.get(event, -1)
            sleep_events.append({
                "time_str": time_str,
                "raw_event": event,
                "stage_id": mapped,
                "duration_s": dur,
                "location": location,
            })
        elif event.startswith("MCAP-"):
            cap_events.append({
                "time_str": time_str,
                "raw_event": event,
                "subtype": event.replace("MCAP-", ""),
                "concurrent_stage": stage_col,
                "duration_s": dur,
                "location": location,
            })
        else:
            other_events[event] += 1

    return {
        "parse_ok": True,
        "error": "",
        "sleep_events": sleep_events,
        "cap_events": cap_events,
        "other_events": dict(other_events),
    }


def audit_annotations(config_path: Path) -> pd.DataFrame:
    """Audit annotations across all records in manifest.csv."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dataset_dir"])
    manifest_df = pd.read_csv(cfg["paths"]["manifest_csv"])
    cov_out = Path(cfg["paths"]["annotation_coverage_csv"])
    cov_out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, mrow in manifest_df.iterrows():
        rec_id = mrow["record_id"]
        txt_path = raw_dir / f"{rec_id}.txt"
        if not txt_path.exists():
            rows.append({"record_id": rec_id, "group": mrow["group"], "parse_ok": False, "error": "Missing .txt"})
            continue

        parsed = parse_txt_annotations(txt_path)
        stage_counts = Counter(ev["stage_id"] for ev in parsed["sleep_events"])
        cap_counts = Counter(ev["subtype"] for ev in parsed["cap_events"])
        cap_total_dur = sum(ev["duration_s"] for ev in parsed["cap_events"])

        n_valid_stage = sum(v for k, v in stage_counts.items() if k >= 0)
        n_masked_stage = stage_counts.get(-1, 0)
        n_cap = sum(cap_counts.values())

        rows.append({
            "record_id": rec_id,
            "group": mrow["group"],
            "is_healthy": mrow["is_healthy"],
            "parse_ok": parsed["parse_ok"],
            "error": parsed["error"],
            "n_sleep_events": len(parsed["sleep_events"]),
            "n_valid_epochs": n_valid_stage,
            "n_masked_epochs": n_masked_stage,
            "n_Wake": stage_counts.get(0, 0),
            "n_N1": stage_counts.get(1, 0),
            "n_N2": stage_counts.get(2, 0),
            "n_N3": stage_counts.get(3, 0),
            "n_REM": stage_counts.get(4, 0),
            "n_cap_total": n_cap,
            "n_cap_A1": cap_counts.get("A1", 0),
            "n_cap_A2": cap_counts.get("A2", 0),
            "n_cap_A3": cap_counts.get("A3", 0),
            "cap_total_dur_s": round(cap_total_dur, 1),
            "has_cap_annotations": n_cap > 0,
            "other_events_json": json.dumps(parsed["other_events"]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(cov_out, index=False)
    return df


if __name__ == "__main__":
    cfg_path = Path("C:/Users/prave/.gemini/antigravity/scratch/SomniMap/experiments/configs/base_config.yaml")
    df = audit_annotations(cfg_path)
    print(f"Annotation audit complete: {df['parse_ok'].sum()}/{len(df)} parsed successfully")
    print(f"Total valid 5-class epochs: {df['n_valid_epochs'].sum():,} | Masked: {df['n_masked_epochs'].sum():,}")
    for col, name in [("n_Wake", "Wake"), ("n_N1", "N1"), ("n_N2", "N2"), ("n_N3", "N3"), ("n_REM", "REM")]:
        tot = df[col].sum()
        pct = 100.0 * tot / max(1, df["n_valid_epochs"].sum())
        print(f"  {name:5s}: {tot:7,d} ({pct:5.2f}%)")
    print(f"CAP recordings with >=1 A-phase event: {df['has_cap_annotations'].sum()}/{len(df)}")
    print(f"Total CAP events: {df['n_cap_total'].sum():,} (A1={df['n_cap_A1'].sum():,}, A2={df['n_cap_A2'].sum():,}, A3={df['n_cap_A3'].sum():,})")
