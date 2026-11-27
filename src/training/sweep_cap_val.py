"""Focused validation-only CAP post-processing sweep (checks A2 & FP ceiling).

Runs SomniMapNet E8_dualhead once on `val` (stride=5) and sweeps:
- smooth_frames in {5, 7}
- bg_thresh in {0.35, 0.40, 0.45}
- a2_mult in {1.5, 2.0, 2.5}
- min_dur_s in {2.0, 2.5}
- merge_gap_s in {1.0, 1.5}
Never touches `test_locked`. Saves results to `results/stage2_cap_sweep_val.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.models.somnimap_net import SomniMapNet
from src.training.benchmark_postproc_val import eval_cap_configs, predict_recording_overlapping


def main() -> None:
    cfg = load_config()
    proc_dir = Path(cfg["paths"]["processed_dir"])
    proj_root = Path(cfg["paths"]["project_root"])
    splits = json.loads(Path(cfg["paths"]["split_manifest_json"]).read_text(encoding="utf-8"))
    val_ids = splits["partitions"]["val"]["record_ids"]

    ckpt_path = proj_root / "checkpoints" / "phase4_SomniMapNet_bilstm_w20_E8_dualhead_best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SomniMapNet(in_channels=6, context_type="bilstm", use_cap_head=True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Caching stride=5 val predictions once...", flush=True)
    rec_s5 = {
        rid: predict_recording_overlapping(model, proc_dir, rid, device, stride_epochs=5, temperature=1.1757)
        for rid in val_ids
    }
    stage_preds = {r: np.argmax(d["stage_probs"], axis=-1) for r, d in rec_s5.items()}

    rows = []
    for sm in (5, 7):
        for bg in (0.35, 0.40, 0.45):
            for a2m in (1.5, 2.0, 2.5):
                for md in (2.0, 2.5):
                    for mg in (1.0, 1.5):
                        res = eval_cap_configs(rec_s5, stage_preds, False, sm, bg, a2m, md, mg)
                        ov = res["overall"]
                        a2 = res["per_subtype"]["A2"]
                        rows.append({
                            "smooth": sm,
                            "bg_thresh": bg,
                            "a2_mult": a2m,
                            "min_dur_s": md,
                            "merge_gap_s": mg,
                            "precision": ov["precision"],
                            "recall": ov["recall"],
                            "f1": ov["f1"],
                            "fp": ov["fp"],
                            "a1_f1": res["per_subtype"]["A1"]["f1"],
                            "a2_f1": a2["f1"],
                            "a2_prec": a2["precision"],
                            "a2_rec": a2["recall"],
                            "a3_f1": res["per_subtype"]["A3"]["f1"],
                            "cap_rate_r": res["cap_rate_pearson_r"],
                        })

    rows.sort(key=lambda r: (r["f1"], r["a2_f1"]), reverse=True)
    best_overall = rows[0]
    best_a2 = max(rows, key=lambda r: (r["a2_f1"], r["f1"]))

    out = {
        "n_configs_tested": len(rows),
        "best_overall_f1": best_overall,
        "best_a2_f1": best_a2,
        "top_10": rows[:10],
    }
    out_p = proj_root / "results" / "stage2_cap_sweep_val.json"
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Best Overall:", best_overall)
    print("Best A2:", best_a2)


if __name__ == "__main__":
    main()
