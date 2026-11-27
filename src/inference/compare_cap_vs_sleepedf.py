"""Compare full-channel held-out CAP (val) vs partial-channel external Sleep-EDF inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.config import get_repo_root, load_config
from src.data.annotation_parser import parse_txt_annotations
from src.data.preprocessing import align_annotations_to_epochs
from src.inference.predictor import predict_from_edf
from src.inference.report_export import (
    assess_domain_shift,
    build_analysis_report_markdown,
    build_analysis_summary_json,
)
from src.physiology.multimodal import analyze_multimodal_physiology


def _run_one(
    edf_path: Path,
    txt_path: Path | None,
    cfg: dict[str, Any],
    stride_epochs: int = 5,
) -> dict[str, Any]:
    res = predict_from_edf(edf_path, stride_epochs=stride_epochs, cfg=cfg)
    pred = res["predictions"]
    phys = analyze_multimodal_physiology(
        res["extras_raw"], res["loaded_sfreq"], pred["stage_preds"], pred.get("cap_events")
    )
    ds = assess_domain_shift(res["edf_info"], pred)

    gt_metrics: dict[str, Any] = {"available": False}
    if txt_path is not None and txt_path.exists():
        parsed = parse_txt_annotations(txt_path)
        if parsed.get("parse_ok") and res["edf_info"].get("start_datetime"):
            gt_st, _ = align_annotations_to_epochs(
                parsed, res["edf_info"]["start_datetime"], pred["n_epochs"]
            )
            valid = gt_st >= 0
            if valid.any():
                y_true = gt_st[valid]
                y_pred = pred["stage_preds"][valid]
                gt_metrics = {
                    "available": True,
                    "n_scored_epochs": int(valid.sum()),
                    "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                    "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
                }

    md = build_analysis_report_markdown(res, phys)
    js = json.loads(build_analysis_summary_json(res, phys))
    return {
        "summary_json": js,
        "domain_shift": ds,
        "gt_stage_comparison": gt_metrics,
        "markdown_report": md,
    }


def main() -> None:
    cfg = load_config()
    root = get_repo_root()
    raw_dir = Path(cfg["paths"]["raw_dataset_dir"])
    ext_dir = root / "data" / "external_demo" / "physionet-sleep-data"

    cap_edf = raw_dir / "n10.edf"
    cap_txt = raw_dir / "n10.txt"
    ext_edf = ext_dir / "SC4001E0-PSG.edf"

    cap_out = _run_one(cap_edf, cap_txt, cfg, stride_epochs=5)
    ext_out = _run_one(ext_edf, None, cfg, stride_epochs=5)

    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "sample_cap_val_n10_report.md").write_text(
        cap_out["markdown_report"], encoding="utf-8"
    )
    (root / "reports" / "sample_sleepedf_sc4001_report.md").write_text(
        ext_out["markdown_report"], encoding="utf-8"
    )

    comparison = {
        "note": "All 108 CAP Sleep Database recordings are partitioned into train(75)/val(17)/test_locked(16). n10 is a non-training val recording; test_locked is untouched.",
        "heldout_cap_val_demo_candidates": ["n10", "ins5", "rbd17", "sdb3"],
        "full_channel_cap_val_n10": {
            "summary": cap_out["summary_json"],
            "gt_stage_comparison": cap_out["gt_stage_comparison"],
        },
        "partial_channel_sleep_edf_SC4001E0": {
            "summary": ext_out["summary_json"],
            "gt_stage_comparison": ext_out["gt_stage_comparison"],
        },
    }
    out_json = root / "results" / "heldout_cap_vs_sleepedf_comparison.json"
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print("Saved comparison to:", out_json)
    print("n10 (full-channel CAP val):", json.dumps({
        "stages": cap_out["summary_json"]["stage_percentages"],
        "mean_conf": cap_out["summary_json"]["mean_confidence"],
        "domain_shift": cap_out["domain_shift"]["high_domain_shift"],
        "gt_stage": cap_out["gt_stage_comparison"],
        "cap_rate_nrem": cap_out["summary_json"]["cap_rate_nrem"],
        "cap_counts": cap_out["summary_json"]["cap_event_counts"],
        "ecg_available": cap_out["summary_json"]["physiology"]["ecg"].get("available"),
    }, indent=2))
    print("SC4001E0 (partial-channel Sleep-EDF):", json.dumps({
        "stages": ext_out["summary_json"]["stage_percentages"],
        "mean_conf": ext_out["summary_json"]["mean_confidence"],
        "domain_shift": ext_out["domain_shift"]["high_domain_shift"],
        "domain_shift_reasons": ext_out["domain_shift"]["reasons"],
        "cap_rate_nrem": ext_out["summary_json"]["cap_rate_nrem"],
        "cap_counts": ext_out["summary_json"]["cap_event_counts"],
        "ecg_available": ext_out["summary_json"]["physiology"]["ecg"].get("available"),
    }, indent=2))


if __name__ == "__main__":
    main()
