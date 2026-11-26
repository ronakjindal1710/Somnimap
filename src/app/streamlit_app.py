"""Phase 7 — SomniMap Interactive Clinical & Research Dashboard (8 Screens)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

st.set_page_config(page_title="SomniMap — Sleep & CAP Analytics", layout="wide", page_icon="🌙")

CFG_PATH = REPO_ROOT / "experiments" / "configs" / "base_config.yaml"
with open(CFG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

PROJ_ROOT = Path(CFG["paths"]["project_root"])
PROC_DIR = Path(CFG["paths"]["processed_dir"])
RESULTS_DIR = PROJ_ROOT / "results"
SPLIT_JSON = Path(CFG["paths"]["split_manifest_json"])

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
CH_NAMES = ["EEG_C4_A1", "EEG_F4_C4", "EEG_C4_P4", "EEG_FP2_F4", "EOG", "EMG_CHIN", "EEG_C3_A2"]


@st.cache_data
def load_split_manifest() -> dict:
    return json.loads(SPLIT_JSON.read_text(encoding="utf-8"))


@st.cache_data
def load_csv_if_exists(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_json_if_exists(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


st.sidebar.title("🌙 SomniMap")
st.sidebar.caption("Multi-Scale Sleep Staging & CAP Event Detection")

SCREENS = [
    "1. Cohort & Split Overview",
    "2. Model Benchmarks & Ablations",
    "3. PSG Signal & Hypnogram Explorer",
    "4. Live SomniMapNet Inference",
    "5. CAP Event Detection (A1/A2/A3)",
    "6. Explainability (Grad-CAM & IG)",
    "7. Calibration & Reliability",
    "8. Final Locked Test Report",
]
screen = st.sidebar.radio("Navigation", SCREENS)

split_data = load_split_manifest()


# ---------------------------------------------------------------------------
# Screen 1: Cohort & Split Overview
# ---------------------------------------------------------------------------
if screen == SCREENS[0]:
    st.title("1. Cohort & Subject-Independent Split Overview")
    parts = split_data["partitions"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Subjects", split_data.get("total_records", 108))
    c2.metric("Train Subjects", parts["train"]["n_subjects"])
    c3.metric("Validation Subjects", parts["val"]["n_subjects"])
    c4.metric("Test (Locked) Subjects", parts["test_locked"]["n_subjects"])

    rows = []
    for sp, info in parts.items():
        for grp, cnt in info.get("groups", {}).items():
            rows.append({"Split": sp, "Diagnostic Group": grp.upper(), "Subjects": cnt})
    grp_df = pd.DataFrame(rows)
    fig = px.bar(grp_df, x="Diagnostic Group", y="Subjects", color="Split", barmode="group",
                 title="Diagnostic Group Stratification Across Locked Splits (70/15/15)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Preprocessing Summary")
    prep_df = load_csv_if_exists(str(PROC_DIR / "preprocessing_summary.csv"))
    if not prep_df.empty:
        st.dataframe(prep_df, use_container_width=True, height=320)
    else:
        st.info("Preprocessing summary not found.")


# ---------------------------------------------------------------------------
# Screen 2: Model Benchmarks & Ablations
# ---------------------------------------------------------------------------
elif screen == SCREENS[1]:
    st.title("2. Model Benchmarks & Phase 4 Ablations (Validation Split)")
    p2 = load_csv_if_exists(str(RESULTS_DIR / "phase2_baselines.csv"))
    p3 = load_csv_if_exists(str(RESULTS_DIR / "phase3_temporal.csv"))
    p4 = load_csv_if_exists(str(RESULTS_DIR / "phase4_somnimap.csv"))

    tab2, tab3, tab4 = st.tabs(["Phase 2: Epoch Baselines", "Phase 3: Temporal Context", "Phase 4: SomniMapNet E3–E8"])
    with tab2:
        if not p2.empty:
            st.dataframe(p2, use_container_width=True)
            fig = px.bar(p2, x="model", y=["macro_f1", "cohen_kappa", "N1_f1"], barmode="group",
                         title="Phase 2 Epoch-Only Baselines")
            st.plotly_chart(fig, use_container_width=True)
    with tab3:
        if not p3.empty:
            st.dataframe(p3, use_container_width=True)
            fig = px.bar(p3, x="model", y=["macro_f1", "cohen_kappa", "N1_f1"], barmode="group",
                         title="Phase 3 Temporal Context Models (20-epoch windows)")
            st.plotly_chart(fig, use_container_width=True)
    with tab4:
        if not p4.empty:
            st.dataframe(p4, use_container_width=True)
            fig = px.bar(p4, x="run_name", y=["macro_f1", "cohen_kappa", "N1_f1"], barmode="group",
                         title="Phase 4 SomniMapNet Ablations (E3–E8)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Phase 4 results not yet generated.")

    cm_files = sorted((RESULTS_DIR / "confusion_matrices").glob("*.png")) if (RESULTS_DIR / "confusion_matrices").exists() else []
    if cm_files:
        sel_cm = st.selectbox("Inspect Validation Confusion Matrix", [f.name for f in cm_files])
        st.image(str(RESULTS_DIR / "confusion_matrices" / sel_cm), width=520)


# ---------------------------------------------------------------------------
# Screen 3: PSG Signal & Hypnogram Explorer
# ---------------------------------------------------------------------------
elif screen == SCREENS[2]:
    st.title("3. PSG Signal & Hypnogram Explorer")
    avail_ids = sorted([d.name for d in PROC_DIR.iterdir() if d.is_dir() and (d / "metadata.json").exists()])
    val_ids = set(split_data["partitions"]["val"]["record_ids"])
    default_pool = [r for r in avail_ids if r in val_ids] or avail_ids
    rec_id = st.selectbox("Select Recording (Validation Split)", default_pool)

    rec_dir = PROC_DIR / rec_id
    meta = json.loads((rec_dir / "metadata.json").read_text(encoding="utf-8"))
    y_stage = np.load(rec_dir / "labels_stage.npy")
    y_cap = np.load(rec_dir / "labels_cap.npy")
    epochs = np.load(rec_dir / "epochs.npy", mmap_mode="r")

    st.caption(f"Epochs: {meta['n_epochs']} | Valid Stage Epochs: {meta['n_valid_stage_epochs']} | Channel Mask: {meta['channel_mask']}")
    valid_ep = np.where(y_stage >= 0, y_stage, np.nan)
    fig_hyp = go.Figure()
    fig_hyp.add_trace(go.Scatter(y=valid_ep, mode="lines", line_shape="hv", name="Sleep Stage"))
    fig_hyp.update_yaxes(tickvals=[0, 1, 2, 3, 4], ticktext=STAGE_NAMES, autorange="reversed")
    fig_hyp.update_layout(title=f"Ground-Truth Hypnogram — {rec_id}", xaxis_title="Epoch (30s)", height=280)
    st.plotly_chart(fig_hyp, use_container_width=True)

    ep_idx = st.slider("Inspect 30-s Epoch", 0, len(y_stage) - 1, min(100, len(y_stage) - 1))
    st.write(f"**Stage:** {STAGE_NAMES[y_stage[ep_idx]] if y_stage[ep_idx] >= 0 else 'UNSCORED/MT'} | **CAP positive 10Hz frames:** {int((y_cap[ep_idx] > 0).sum())}/300")
    t = np.arange(3000) / 100.0
    fig_sig = go.Figure()
    for c_i, cname in enumerate(CH_NAMES):
        if meta["channel_mask"][c_i] == 1:
            fig_sig.add_trace(go.Scatter(x=t, y=epochs[ep_idx, :, c_i] + c_i * 4, name=cname, line=dict(width=1)))
    fig_sig.update_layout(title=f"Normalized PSG Waveforms — Epoch {ep_idx}", xaxis_title="Time (s)", height=420)
    st.plotly_chart(fig_sig, use_container_width=True)


# ---------------------------------------------------------------------------
# Screen 4: Live SomniMapNet Inference
# ---------------------------------------------------------------------------
elif screen == SCREENS[3]:
    st.title("4. Live SomniMapNet Inference")
    ckpt_files = sorted((PROJ_ROOT / "checkpoints").glob("phase4_*.pt")) if (PROJ_ROOT / "checkpoints").exists() else []
    if not ckpt_files:
        st.warning("No Phase 4 checkpoints found in checkpoints/.")
    else:
        sel_ckpt = st.selectbox("Select Checkpoint", [c.name for c in ckpt_files])
        val_ids = [r for r in split_data["partitions"]["val"]["record_ids"] if (PROC_DIR / r / "metadata.json").exists()]
        rec_id = st.selectbox("Validation Recording", val_ids)
        if st.button("Run Inference"):
            import torch
            from src.data.dataset import SomniMapSequenceDataset
            from src.models.somnimap_net import SomniMapNet

            ckpt_path = PROJ_ROOT / "checkpoints" / sel_ckpt
            is_multi = "multichan" in sel_ckpt or "dualhead" in sel_ckpt or "focal" in sel_ckpt or "no_chan_attn" in sel_ckpt
            has_cap = "dualhead" in sel_ckpt
            ctx = "bilstm"
            for c_opt in ("tcn", "transformer", "bilstm"):
                if c_opt in sel_ckpt:
                    ctx = c_opt
                    break
            mode = "multi_channel" if is_multi else "single_eeg"
            in_ch = 6 if is_multi else 1

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SomniMapNet(in_channels=in_ch, context_type=ctx, use_cap_head=has_cap).to(device)
            if "no_chan_attn" in sel_ckpt:
                model.chan_attn = None
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state"])
            model.eval()

            ds = SomniMapSequenceDataset(PROC_DIR, [rec_id], window_epochs=20, stride_epochs=20, mode=mode)
            preds, gts = [], []
            with torch.no_grad():
                for i in range(len(ds)):
                    b = ds[i]
                    x = b["x"].unsqueeze(0).to(device)
                    cm = b["ch_mask"].unsqueeze(0).to(device)
                    lg = model(x, ch_mask=cm)["stage_logits"][0].argmax(dim=-1).cpu().numpy()
                    preds.append(lg)
                    gts.append(b["y_stage"].numpy())
            y_pr = np.concatenate(preds)
            y_gt = np.concatenate(gts)
            mask = y_gt >= 0
            acc = float((y_pr[mask] == y_gt[mask]).mean())
            st.metric("Recording Epoch Accuracy", f"{acc*100:.2f}%")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=np.where(mask, y_gt, np.nan), name="Ground Truth", line_shape="hv"))
            fig.add_trace(go.Scatter(y=y_pr, name="Predicted", line_shape="hv", line=dict(dash="dot")))
            fig.update_yaxes(tickvals=[0, 1, 2, 3, 4], ticktext=STAGE_NAMES, autorange="reversed")
            fig.update_layout(title=f"Hypnogram Comparison — {rec_id}", height=340)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Screen 5: CAP Event Detection
# ---------------------------------------------------------------------------
elif screen == SCREENS[4]:
    st.title("5. CAP Event Detection Evaluation (A1 / A2 / A3)")
    cap_eval = load_json_if_exists(str(RESULTS_DIR / "phase5_cap_eval_val.json"))
    if not cap_eval:
        st.info("Phase 5 CAP evaluation results not yet generated.")
    else:
        ov = cap_eval["overall"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Event F1 (IoU≥0.5)", ov["f1"])
        c2.metric("Precision", ov["precision"])
        c3.metric("Recall", ov["recall"])
        c4.metric("CAP Rate Pearson r", cap_eval.get("cap_rate_pearson_r", 0.0))

        sub_rows = [{"Subtype": k, **v} for k, v in cap_eval["per_subtype"].items()]
        st.subheader("Per-Subtype CAP Event Metrics (IoU ≥ 0.5)")
        st.dataframe(pd.DataFrame(sub_rows), use_container_width=True)

        ex_img = RESULTS_DIR / "phase5_cap_examples_val.png"
        if ex_img.exists():
            st.image(str(ex_img), caption="CAP Event Detection Timeline Examples")


# ---------------------------------------------------------------------------
# Screen 6: Explainability (Grad-CAM & IG)
# ---------------------------------------------------------------------------
elif screen == SCREENS[5]:
    st.title("6. Explainability (1D Grad-CAM & Integrated Gradients)")
    xai_dir = RESULTS_DIR / "xai"
    xai_json = load_json_if_exists(str(xai_dir / "xai_calibration_summary.json"))
    imgs = sorted(xai_dir.glob("*.png")) if xai_dir.exists() else []
    case_imgs = [p for p in imgs if p.name != "reliability_diagram.png"]
    if case_imgs:
        for p in case_imgs:
            st.image(str(p), caption=p.stem)
    else:
        st.info("Phase 6 XAI plots not yet generated.")
    if "xai_cases" in xai_json:
        st.json(xai_json["xai_cases"])


# ---------------------------------------------------------------------------
# Screen 7: Calibration & Reliability
# ---------------------------------------------------------------------------
elif screen == SCREENS[6]:
    st.title("7. Model Calibration & Temperature Scaling")
    xai_dir = RESULTS_DIR / "xai"
    xai_json = load_json_if_exists(str(xai_dir / "xai_calibration_summary.json"))
    cal = xai_json.get("calibration", {})
    if cal:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fitted Temperature T", cal["temperature"])
        c2.metric("ECE (Before → After)", f"{cal['ece_before']} → {cal['ece_after']}")
        c3.metric("Brier (Before → After)", f"{cal['brier_before']} → {cal['brier_after']}")
        c4.metric("NLL (Before → After)", f"{cal['nll_before']} → {cal['nll_after']}")
    rel_img = xai_dir / "reliability_diagram.png"
    if rel_img.exists():
        st.image(str(rel_img), width=560)
    else:
        st.info("Reliability diagram not yet generated.")


# ---------------------------------------------------------------------------
# Screen 8: Final Locked Test Report
# ---------------------------------------------------------------------------
elif screen == SCREENS[7]:
    st.title("8. Final Locked Test-Set Evaluation (16 Unseen Subjects)")
    test_json = load_json_if_exists(str(RESULTS_DIR / "phase8_test_locked_metrics.json"))
    if not test_json:
        st.warning("Locked test-set evaluation (Phase 8) has not been executed yet. The test_locked split remains untouched.")
    else:
        st.json(test_json)
        test_cm = RESULTS_DIR / "confusion_matrices" / "phase8_test_locked_cm.png"
        if test_cm.exists():
            st.image(str(test_cm), width=540)
