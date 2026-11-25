"""Subject-wise, group-stratified 70/15/15 split manifest generator.

Ensures zero subject/recording leakage across train, validation, and test
partitions, preserves healthy vs. pathological representation and diagnostic
group balance, and logs sleep-stage + CAP coverage per partition.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def allocate_group_counts(n: int, ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> tuple[int, int, int]:
    """Deterministically allocate n subjects of a group into (train, val, test)."""
    if n == 0:
        return (0, 0, 0)
    if n == 1:
        return (1, 0, 0)
    if n == 2:
        return (1, 1, 0)

    n_val = max(1, round(n * ratios[1]))
    n_test = max(1, round(n * ratios[2]))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        if n_val > n_test:
            n_val = max(0, n - n_train - n_test)
        else:
            n_test = max(0, n - n_train - n_val)
    return (n_train, n_val, n_test)


def build_split_manifest(config_path: Path) -> dict[str, Any]:
    """Create and save locked 70/15/15 subject-wise group-stratified split."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["splits"]["seed"])
    rng = random.Random(seed)

    manifest_df = pd.read_csv(cfg["paths"]["manifest_csv"])
    cov_df = pd.read_csv(cfg["paths"]["annotation_coverage_csv"])
    merged = manifest_df.merge(
        cov_df.drop(columns=["group", "is_healthy"], errors="ignore"),
        on="record_id",
        how="inner",
    )

    eligible = merged[(merged["header_ok"]) & (merged["parse_ok"]) & (merged["n_valid_epochs"] > 0)].copy()
    excluded = merged[~merged["record_id"].isin(eligible["record_id"])].copy()

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for grp, grp_df in sorted(eligible.groupby("group"), key=lambda x: x[0]):
        recs = sorted(grp_df["record_id"].tolist())
        rng.shuffle(recs)
        n_tr, n_va, n_te = allocate_group_counts(len(recs))
        train_ids.extend(recs[:n_tr])
        val_ids.extend(recs[n_tr : n_tr + n_va])
        test_ids.extend(recs[n_tr + n_va : n_tr + n_va + n_te])

    # Sanity assertions: zero overlap, complete coverage
    s_tr, s_va, s_te = set(train_ids), set(val_ids), set(test_ids)
    assert len(s_tr & s_va) == 0, "Overlap between train and val!"
    assert len(s_tr & s_te) == 0, "Overlap between train and test!"
    assert len(s_va & s_te) == 0, "Overlap between val and test!"
    assert len(s_tr | s_va | s_te) == len(eligible), "Missing eligible records!"

    def summarize_partition(ids: list[str]) -> dict[str, Any]:
        sub = eligible[eligible["record_id"].isin(ids)]
        stage_cols = ["n_Wake", "n_N1", "n_N2", "n_N3", "n_REM"]
        tot_valid = int(sub["n_valid_epochs"].sum())
        stage_dist = {
            c.replace("n_", ""): {
                "count": int(sub[c].sum()),
                "pct": round(100.0 * float(sub[c].sum()) / max(1, tot_valid), 2),
            }
            for c in stage_cols
        }
        return {
            "n_subjects": len(sub),
            "n_healthy": int(sub["is_healthy"].sum()),
            "n_pathological": int((~sub["is_healthy"]).sum()),
            "groups": sub["group"].value_counts().to_dict(),
            "n_valid_epochs": tot_valid,
            "n_cap_records": int(sub["has_cap_annotations"].sum()),
            "n_cap_events": int(sub["n_cap_total"].sum()),
            "stage_distribution": stage_dist,
            "record_ids": sorted(ids),
        }

    split_obj = {
        "version": "1.0-locked",
        "seed": seed,
        "strategy": "subject-wise, diagnostic-group-stratified 70/15/15",
        "total_records": len(merged),
        "eligible_records": len(eligible),
        "excluded_records": excluded[["record_id", "header_error", "error"]].to_dict(orient="records"),
        "partitions": {
            "train": summarize_partition(train_ids),
            "val": summarize_partition(val_ids),
            "test_locked": summarize_partition(test_ids),
        },
    }

    out_path = Path(cfg["paths"]["split_manifest_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split_obj, indent=2), encoding="utf-8")
    return split_obj


if __name__ == "__main__":
    cfg_path = Path("C:/Users/prave/.gemini/antigravity/scratch/SomniMap/experiments/configs/base_config.yaml")
    res = build_split_manifest(cfg_path)
    print(f"Split manifest locked: {res['eligible_records']}/{res['total_records']} eligible records")
    for name, p in res["partitions"].items():
        pct_subj = 100.0 * p["n_subjects"] / max(1, res["eligible_records"])
        print(
            f"  {name:12s}: {p['n_subjects']:3d} subjects ({pct_subj:4.1f}%) | "
            f"Healthy={p['n_healthy']:2d}, Path={p['n_pathological']:2d} | "
            f"Epochs={p['n_valid_epochs']:6,d} | CAP_recs={p['n_cap_records']:2d}"
        )
