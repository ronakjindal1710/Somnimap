"""Phase 8 — Single-Shot Locked Test-Set Evaluation (16 Unseen Subjects) with 95% CIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import SomniMapSequenceDataset
from src.models.somnimap_net import SomniMapNet
from src.training.eval_cap import evaluate_cap_on_records, plot_cap_examples
from src.training.metrics import STAGE_NAMES, compute_brier, compute_ece, compute_sleep_metrics
from src.training.mlflow_utils import log_run_to_mlflow, save_confusion_matrix_plot


def subject_bootstrap_ci(
    per_subject_yt: list[np.ndarray],
    per_subject_yp: list[np.ndarray],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Cluster bootstrap over subjects for 95% CIs of Macro-F1, Bal. Acc, Cohen's kappa, N1 F1."""
    rng = np.random.default_rng(seed)
    n_subj = len(per_subject_yt)
    f1s, bals, kaps, n1s = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_subj, size=n_subj)
        yt = np.concatenate([per_subject_yt[i] for i in idx])
        yp = np.concatenate([per_subject_yp[i] for i in idx])
        m = compute_sleep_metrics(yt, yp)
        f1s.append(m["macro_f1"])
        bals.append(m["balanced_accuracy"])
        kaps.append(m["cohen_kappa"])
        n1s.append(m["per_class"]["N1"]["f1"])
    return {
        "macro_f1_ci95": [round(float(np.percentile(f1s, 2.5)), 4), round(float(np.percentile(f1s, 97.5)), 4)],
        "balanced_acc_ci95": [round(float(np.percentile(bals, 2.5)), 4), round(float(np.percentile(bals, 97.5)), 4)],
        "cohen_kappa_ci95": [round(float(np.percentile(kaps, 2.5)), 4), round(float(np.percentile(kaps, 97.5)), 4)],
        "N1_f1_ci95": [round(float(np.percentile(n1s, 2.5)), 4), round(float(np.percentile(n1s, 97.5)), 4)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 Locked Test-Set Evaluation (Run Exactly Once)")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--context_type", type=str, default="bilstm")
    parser.add_argument("--multi_channel", action="store_true", default=False)
    parser.add_argument("--use_cap_head", action="store_true", default=False)
    parser.add_argument("--no_chan_attn", action="store_true", default=False)
    parser.add_argument("--window_epochs", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    from src.config import load_config
    cfg = load_config()

    proc_dir = Path(cfg["paths"]["processed_dir"])
    proj_root = Path(cfg["paths"]["project_root"])
    split_data = json.loads(Path(cfg["paths"]["split_manifest_json"]).read_text(encoding="utf-8"))
    test_ids = split_data["partitions"]["test_locked"]["record_ids"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "multi_channel" if args.multi_channel else "single_eeg"
    in_ch = 6 if args.multi_channel else 1

    model = SomniMapNet(
        in_channels=in_ch, context_type=args.context_type, use_cap_head=args.use_cap_head
    ).to(device)
    if args.no_chan_attn and model.chan_attn is not None:
        model.chan_attn = None
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    per_subj_yt, per_subj_yp, all_logits, all_yt = [], [], [], []
    per_subject_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for rec_id in test_ids:
            ds = SomniMapSequenceDataset(
                proc_dir, [rec_id], window_epochs=args.window_epochs, stride_epochs=args.window_epochs, mode=mode
            )
            loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
            s_lg, s_yt = [], []
            for batch in loader:
                x = batch["x"].to(device)
                cm = batch["ch_mask"].to(device)
                lg = model(x, ch_mask=cm)["stage_logits"].cpu().numpy().reshape(-1, 5)
                s_lg.append(lg)
                s_yt.append(batch["y_stage"].numpy().reshape(-1))
            lg_np = np.vstack(s_lg)
            yt_np = np.concatenate(s_yt)
            yp_np = np.argmax(lg_np, axis=-1)
            per_subj_yt.append(yt_np)
            per_subj_yp.append(yp_np)
            all_logits.append(lg_np)
            all_yt.append(yt_np)

            sm = compute_sleep_metrics(yt_np, yp_np)
            per_subject_rows.append({
                "record_id": rec_id,
                "n_samples": sm["n_samples"],
                "macro_f1": round(sm["macro_f1"], 4),
                "cohen_kappa": round(sm["cohen_kappa"], 4),
                "accuracy": round(sm["accuracy"], 4),
            })

    logits_all = np.vstack(all_logits)
    yt_all = np.concatenate(all_yt)
    yp_all = np.argmax(logits_all, axis=-1)

    probs_uncal = torch.softmax(torch.tensor(logits_all), dim=-1).numpy()
    probs_cal = torch.softmax(torch.tensor(logits_all) / args.temperature, dim=-1).numpy()

    metrics = compute_sleep_metrics(yt_all, yp_all, probs=probs_uncal)
    mask = yt_all >= 0
    metrics["ece_calibrated"] = round(compute_ece(probs_cal[mask], yt_all[mask]), 4)
    metrics["brier_calibrated"] = round(compute_brier(probs_cal[mask], yt_all[mask]), 4)
    metrics["temperature"] = args.temperature
    metrics["ci95"] = subject_bootstrap_ci(per_subj_yt, per_subj_yp, n_boot=1000, seed=42)
    metrics["per_subject"] = per_subject_rows

    cm_path = proj_root / "results" / "confusion_matrices" / "phase8_test_locked_cm.png"
    save_confusion_matrix_plot(metrics["confusion_matrix"], "SomniMapNet Test (Locked 16 Subjects)", cm_path)

    if args.use_cap_head:
        cap_summary, cap_rec_df, cap_examples = evaluate_cap_on_records(
            model, proc_dir, test_ids, device, window_epochs=args.window_epochs, mode=mode, iou_thresh=0.5
        )
        metrics["cap_event_eval"] = cap_summary
        cap_rec_df.to_csv(proj_root / "results" / "phase5_cap_records_test_locked.csv", index=False)
        plot_cap_examples(cap_examples, proj_root / "results" / "phase5_cap_examples_test_locked.png")

    out_path = proj_root / "results" / "phase8_test_locked_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log_run_to_mlflow(
        tracking_uri=cfg["paths"]["mlflow_tracking_uri"],
        experiment_name="SomniMap_Phase8_TestLocked",
        run_name=Path(args.ckpt).stem + "_test_locked",
        params={
            "ckpt": args.ckpt,
            "context_type": args.context_type,
            "multi_channel": args.multi_channel,
            "use_cap_head": args.use_cap_head,
            "temperature": args.temperature,
        },
        metrics=metrics,
        cm_path=cm_path,
    )

    print("\n=== Phase 8 Locked Test-Set Results (16 Subjects) ===")
    print(f"Macro-F1:     {metrics['macro_f1']:.4f}  95% CI {metrics['ci95']['macro_f1_ci95']}")
    print(f"Bal. Acc:     {metrics['balanced_accuracy']:.4f}  95% CI {metrics['ci95']['balanced_acc_ci95']}")
    print(f"Cohen Kappa:  {metrics['cohen_kappa']:.4f}  95% CI {metrics['ci95']['cohen_kappa_ci95']}")
    print(f"N1 F1:        {metrics['per_class']['N1']['f1']:.4f}  95% CI {metrics['ci95']['N1_f1_ci95']}")
    print(f"ECE (uncal -> cal): {metrics['ece']:.4f} -> {metrics['ece_calibrated']:.4f}")
    for st in STAGE_NAMES:
        pc = metrics["per_class"][st]
        print(f"  {st:4s} | F1={pc['f1']:.4f} | Sens={pc['sensitivity']:.4f} | Spec={pc['specificity']:.4f}")


if __name__ == "__main__":
    main()
