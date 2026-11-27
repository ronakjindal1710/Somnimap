"""Stage 6 — Unseen External Dataset Demo on PhysioNet Sleep-EDF Expanded (3 subjects).

Evaluates raw-EDF SomniMapNet inference on 3 genuinely unseen external recordings
(zero overlap with CAP Sleep). Reports domain-shift caveats and keeps results
strictly separate from Phase 8 CAP `test_locked`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mne
import numpy as np

from src.config import load_config
from src.inference.predictor import predict_from_edf
from src.inference.report_export import build_analysis_report_markdown
from src.physiology.multimodal import analyze_multimodal_physiology
from src.training.metrics import compute_sleep_metrics

SLEEP_EDF_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}


def extract_sleep_edf_gt(hyp_path: Path, n_epochs: int) -> np.ndarray:
    """Convert Sleep-EDF Hypnogram EDF+ annotations to 30-s 5-class stage IDs (-1 if unscored)."""
    ann = mne.read_annotations(str(hyp_path))
    y = np.full(n_epochs, -1, dtype=np.int64)
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        if desc not in SLEEP_EDF_MAP:
            continue
        st_id = SLEEP_EDF_MAP[desc]
        e0 = max(0, int(round(onset / 30.0)))
        e1 = min(n_epochs, int(round((onset + dur) / 30.0)))
        if e1 > e0:
            y[e0:e1] = st_id
    return y


def main() -> None:
    cfg = load_config()
    proj_root = Path(cfg["paths"]["project_root"])
    ext_dir = proj_root / "data" / "external_demo"
    ext_dir.mkdir(parents=True, exist_ok=True)

    # Fetch 3 unseen Sleep-EDF Expanded subjects (recording 1 each)
    files = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=[0, 1, 2], recording=[1], path=str(ext_dir), on_missing="ignore"
    )

    manifest_entries = []
    eval_rows = []
    all_yt, all_yp = [], []
    sample_report_written = False

    for idx, pair in enumerate(files):
        psg_path = Path(pair[0])
        hyp_path = Path(pair[1])
        print(f"[{idx+1}/{len(files)}] Running SomniMap raw-EDF inference on {psg_path.name}...", flush=True)

        res = predict_from_edf(psg_path, stride_epochs=10, cfg=cfg)
        pred = res["predictions"]
        phys = analyze_multimodal_physiology(
            res["extras_raw"], res["loaded_sfreq"], pred["stage_preds"], pred.get("cap_events")
        )

        y_gt = extract_sleep_edf_gt(hyp_path, pred["n_epochs"])
        m = compute_sleep_metrics(y_gt, pred["stage_preds"], probs=pred["stage_probs"])
        mask = y_gt >= 0
        all_yt.append(y_gt[mask])
        all_yp.append(pred["stage_preds"][mask])

        manifest_entries.append({
            "subject_idx": idx,
            "psg_file": psg_path.name,
            "hypnogram_file": hyp_path.name,
            "duration_sec": res["edf_info"]["duration_sec"],
            "header_sfreq_hz": res["edf_info"]["header_sfreq_hz"],
            "canonical_mapping": res["edf_info"]["canonical_mapping"],
            "warnings": res["edf_info"]["warnings"],
        })
        eval_rows.append({
            "psg_file": psg_path.name,
            "n_scored_epochs": int(mask.sum()),
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "balanced_acc": round(m["balanced_accuracy"], 4),
            "cohen_kappa": round(m["cohen_kappa"], 4),
            "N1_f1": round(m["per_class"]["N1"]["f1"], 4),
            "cap_rate_nrem": pred.get("cap_rate_nrem"),
        })

        if not sample_report_written:
            rep_md = build_analysis_report_markdown(res, phys)
            rep_path = proj_root / "reports" / "sample_unseen_demo_report.md"
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            rep_path.write_text(rep_md, encoding="utf-8")
            sample_report_written = True

    agg = compute_sleep_metrics(np.concatenate(all_yt), np.concatenate(all_yp))
    summary = {
        "dataset": "PhysioNet Sleep-EDF Expanded (External Unseen Demo)",
        "domain_shift_notes": [
            "Sleep-EDF uses Fpz-Cz / Pz-Oz EEG (100 Hz) and horizontal EOG + submental EMG, whereas SomniMapNet was trained on CAP Sleep C4-A1/F4-C4/C4-P4/Fp2-F4 + ROC-LOC + EMG1-EMG2 (3/6 canonical channels absent and masked to 0).",
            "Sleep-EDF does not contain CAP A1/A2/A3 ground-truth labels; CAP outputs are exploratory only.",
        ],
        "aggregate_sleep_staging": {
            "n_scored_epochs": agg["n_samples"],
            "accuracy": round(agg["accuracy"], 4),
            "balanced_acc": round(agg["balanced_accuracy"], 4),
            "macro_f1": round(agg["macro_f1"], 4),
            "cohen_kappa": round(agg["cohen_kappa"], 4),
            "N1_f1": round(agg["per_class"]["N1"]["f1"], 4),
        },
        "per_recording": eval_rows,
    }

    (proj_root / "data" / "splits" / "external_demo_manifest.json").write_text(
        json.dumps({"recordings": manifest_entries}, indent=2), encoding="utf-8"
    )
    (proj_root / "results" / "external_sleep_edf_demo.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n=== External Sleep-EDF Demo Summary ===")
    print(json.dumps(summary["aggregate_sleep_staging"], indent=2))


if __name__ == "__main__":
    main()
