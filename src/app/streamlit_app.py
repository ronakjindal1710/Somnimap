"""NIDRA Interactive Clinical & Research Dashboard (9 Screens)."""

from __future__ import annotations

import html
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

st.set_page_config(
    page_title="NIDRA — Neural Inference for Deep, REM & Arousal Architecture",
    layout="wide",
    page_icon="🌙",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.1rem;
        padding-bottom: 2.5rem;
    }
    [data-testid="stMetric"] {
        background-color: #181d27;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    [data-testid="stMetricValue"] {
        color: #f1f5f9;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        overflow: hidden;
    }
    h1, h2, h3 {
        letter-spacing: -0.01em;
    }
    .nidra-insights-card {
        background-color: #181d27;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #38bdf8;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 1.25rem;
        margin-bottom: 0.75rem;
    }
    .nidra-insights-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 10px;
        letter-spacing: 0.01em;
    }
    .nidra-insight-row {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 7px;
        font-size: 0.9rem;
        line-height: 1.48;
        color: #cbd5e1;
    }
    .nidra-insight-row:last-child {
        margin-bottom: 0;
    }
    .nidra-icon-ok { color: #34d399; font-weight: 700; }
    .nidra-icon-warn { color: #fbbf24; font-weight: 700; }
    .nidra-icon-info { color: #38bdf8; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_plotly(fig: go.Figure, title: str | None = None, height: int | None = None) -> go.Figure:
    """Apply cohesive dark editorial styling across all dashboard charts."""
    layout_kw: dict = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(24,29,39,0.55)",
        "font": {"color": "#f1f5f9", "size": 12},
        "margin": {"l": 48, "r": 24, "t": 56, "b": 44},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "bgcolor": "rgba(0,0,0,0)",
        },
        "xaxis": {"gridcolor": "rgba(255,255,255,0.07)", "zerolinecolor": "rgba(255,255,255,0.1)"},
        "yaxis": {"gridcolor": "rgba(255,255,255,0.07)", "zerolinecolor": "rgba(255,255,255,0.1)"},
    }
    if title is not None:
        layout_kw["title"] = {"text": title, "font": {"size": 15, "color": "#f1f5f9"}}
    if height is not None:
        layout_kw["height"] = height
    fig.update_layout(**layout_kw)
    return fig


def render_key_insights(title: str, insights: list[tuple[str, str]]) -> None:
    """Render a cohesive dark editorial Key Insights card."""
    icon_cls_map = {
        "✓": "nidra-icon-ok",
        "⚠": "nidra-icon-warn",
        "ℹ": "nidra-icon-info",
    }
    rows_html = []
    for icon, text in insights:
        cls = icon_cls_map.get(icon, "nidra-icon-info")
        rows_html.append(
            f'<div class="nidra-insight-row"><span class="{cls}">{html.escape(icon)}</span>'
            f'<span>{html.escape(text)}</span></div>'
        )
    st.markdown(
        f'<div class="nidra-insights-card">'
        f'<div class="nidra-insights-title">{html.escape(title)}</div>'
        f'{"".join(rows_html)}'
        f"</div>",
        unsafe_allow_html=True,
    )

from src.config import load_config

CFG = load_config()

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


st.sidebar.title("🌙 NIDRA")
st.sidebar.caption("Neural Inference for Deep, REM & Arousal Architecture")

SCREENS = [
    "1. Cohort & Split Overview",
    "2. Model Benchmarks & Ablations",
    "3. PSG Signal & Hypnogram Explorer",
    "4. Live NIDRA Inference",
    "5. CAP Event Detection (A1/A2/A3)",
    "6. Explainability (Grad-CAM & IG)",
    "7. Calibration & Reliability",
    "8. Final Locked Test Report",
    "9. Raw EDF Upload, Multimodal & Report",
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
    fig = px.bar(grp_df, x="Diagnostic Group", y="Subjects", color="Split", barmode="group")
    style_plotly(fig, title="Diagnostic Group Stratification Across Locked Splits (70/15/15)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Preprocessing Summary")
    prep_df = load_csv_if_exists(str(PROC_DIR / "preprocessing_summary.csv"))
    if not prep_df.empty:
        st.dataframe(prep_df, use_container_width=True, height=320)
    else:
        st.info("Preprocessing summary not found.")

    n_healthy = sum(p.get("n_healthy", 0) for p in parts.values())
    n_path = sum(p.get("n_pathological", 0) for p in parts.values())
    total_valid_ep = sum(p.get("n_valid_epochs", 0) for p in parts.values())
    total_cap_ev = sum(p.get("n_cap_events", 0) for p in parts.values())
    grp_totals: dict[str, int] = {}
    for p in parts.values():
        for g, cnt in p.get("groups", {}).items():
            grp_totals[g.upper()] = grp_totals.get(g.upper(), 0) + cnt
    top_grps = ", ".join(f"{k} ({v})" for k, v in sorted(grp_totals.items(), key=lambda x: -x[1])[:4])
    s1_insights = [
        ("✓", f"Subject-independent 70/15/15 split covers {split_data.get('total_records', 108)} recordings ({n_healthy} Normal healthy vs. {n_path} pathological across 7 disorders; largest cohorts: {top_grps})."),
        ("ℹ", f"Annotated corpus spans {total_valid_ep:,} valid 30-s sleep epochs and {total_cap_ev:,} scored CAP Phase-A events."),
    ]
    if not prep_df.empty and "oob_stage" in prep_df.columns:
        oob_s = int(prep_df["oob_stage"].sum())
        oob_c = int(prep_df["oob_cap"].sum()) if "oob_cap" in prep_df.columns else 0
        low_ch = int((prep_df["n_channels_present"] < 6).sum()) if "n_channels_present" in prep_df.columns else 0
        s1_insights.append(
            ("✓" if (oob_s == 0 and oob_c == 0) else "⚠",
             f"Preprocessing alignment integrity: {oob_s} out-of-bounds stage events, {oob_c} out-of-bounds CAP events; {low_ch} recordings have < 6 canonical channels present (masked, never fabricated).")
        )
    render_key_insights("Key Insights — Cohort & Split Integrity", s1_insights)


# ---------------------------------------------------------------------------
# Screen 2: Model Benchmarks & Ablations
# ---------------------------------------------------------------------------
elif screen == SCREENS[1]:
    st.title("2. Model Benchmarks & Phase 4 Ablations (Validation Split)")
    p2 = load_csv_if_exists(str(RESULTS_DIR / "phase2_baselines.csv"))
    p3 = load_csv_if_exists(str(RESULTS_DIR / "phase3_temporal.csv"))
    p4 = load_csv_if_exists(str(RESULTS_DIR / "phase4_somnimap.csv"))

    tab2, tab3, tab4 = st.tabs(["Phase 2: Epoch Baselines", "Phase 3: Temporal Context", "Phase 4: NIDRA (SomniMapNet E3–E8)"])
    with tab2:
        if not p2.empty:
            st.dataframe(p2, use_container_width=True)
            fig = px.bar(p2, x="model", y=["macro_f1", "cohen_kappa", "N1_f1"], barmode="group")
            style_plotly(fig, title="Phase 2 Epoch-Only Baselines")
            st.plotly_chart(fig, use_container_width=True)
    with tab3:
        if not p3.empty:
            st.dataframe(p3, use_container_width=True)
            fig = px.bar(p3, x="model", y=["macro_f1", "cohen_kappa", "N1_f1"], barmode="group")
            style_plotly(fig, title="Phase 3 Temporal Context Models (20-epoch windows)")
            st.plotly_chart(fig, use_container_width=True)
    with tab4:
        if not p4.empty:
            st.dataframe(p4, use_container_width=True)
            fig = px.bar(p4, x="run_name", y=["macro_f1", "cohen_kappa", "N1_f1"], barmode="group")
            style_plotly(fig, title="Phase 4 NIDRA Ablations (E3–E8)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Phase 4 results not yet generated.")

    cm_files = sorted((RESULTS_DIR / "confusion_matrices").glob("*.png")) if (RESULTS_DIR / "confusion_matrices").exists() else []
    if cm_files:
        sel_cm = st.selectbox("Inspect Validation Confusion Matrix", [f.name for f in cm_files])
        st.image(str(RESULTS_DIR / "confusion_matrices" / sel_cm), width=520)

    s2_insights = [
        ("ℹ", "All models consume standardized 100 Hz signals (0.3–35 Hz EEG/EOG, 10–45 Hz chin EMG, 50 Hz notch) segmented into 30-s sleep epochs and 10 Hz CAP frames."),
    ]
    if not p2.empty and not p3.empty:
        b2 = p2.sort_values("macro_f1", ascending=False).iloc[0]
        b3 = p3.sort_values("macro_f1", ascending=False).iloc[0]
        s2_insights.append(
            ("✓", f"Adding 20-epoch temporal context lifts validation Macro-F1 from {b2['macro_f1']:.3f} ({b2['model']}) to {b3['macro_f1']:.3f} ({b3['model']}).")
        )
    if not p4.empty:
        b4 = p4.sort_values("macro_f1", ascending=False).iloc[0]
        e8 = p4[p4["run_name"].astype(str).str.contains("E8", na=False)]
        if not e8.empty:
            r8 = e8.iloc[0]
            s2_insights.append(
                ("✓", f"Dual-head multi-channel NIDRA ({r8['run_name']}) achieves Macro-F1 = {r8['macro_f1']:.3f} and N1 F1 = {r8['N1_f1']:.3f}, linking macro sleep stages with 10 Hz micro-arousals.")
            )
        else:
            s2_insights.append(
                ("✓", f"Top Phase 4 NIDRA ablation ({b4['run_name']}) reaches Macro-F1 = {b4['macro_f1']:.3f} and N1 F1 = {b4['N1_f1']:.3f}.")
            )
    render_key_insights("Key Insights — Architecture & Ablation Gains", s2_insights)


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
    fig_hyp.update_layout(xaxis_title="Epoch (30s)")
    style_plotly(fig_hyp, title=f"Ground-Truth Hypnogram — {rec_id}", height=300)
    st.plotly_chart(fig_hyp, use_container_width=True)

    ep_idx = st.slider("Inspect 30-s Epoch", 0, len(y_stage) - 1, min(100, len(y_stage) - 1))
    st.write(f"**Stage:** {STAGE_NAMES[y_stage[ep_idx]] if y_stage[ep_idx] >= 0 else 'UNSCORED/MT'} | **CAP positive 10Hz frames:** {int((y_cap[ep_idx] > 0).sum())}/300")
    t = np.arange(3000) / 100.0
    fig_sig = go.Figure()
    for c_i, cname in enumerate(CH_NAMES):
        if meta["channel_mask"][c_i] == 1:
            fig_sig.add_trace(go.Scatter(x=t, y=epochs[ep_idx, :, c_i] + c_i * 4, name=cname, line=dict(width=1)))
    fig_sig.update_layout(xaxis_title="Time (s)")
    style_plotly(fig_sig, title=f"Normalized PSG Waveforms — Epoch {ep_idx}", height=430)
    st.plotly_chart(fig_sig, use_container_width=True)

    active_stage = STAGE_NAMES[y_stage[ep_idx]] if y_stage[ep_idx] >= 0 else "UNSCORED/MT"
    cap_pos = int((y_cap[ep_idx] > 0).sum())
    cap_sec = cap_pos * 0.1
    missing_chs = [CH_NAMES[i] for i, m in enumerate(meta["channel_mask"]) if m == 0]
    s3_insights = [
        ("ℹ", f"Recording {rec_id} Epoch {ep_idx}: Active stage is {active_stage} with {cap_pos}/300 positive 10 Hz CAP frames ({cap_sec:.1f}s of Phase-A micro-arousal activity)."),
        ("✓" if (active_stage in ("Wake", "REM") and cap_pos == 0) or (active_stage in ("N1", "N2", "N3")) else "ℹ",
         "Clinically, CAP is defined exclusively during NREM sleep (N1/N2/N3), so Wake and REM epochs remain near 0/300 frames while N2/N3 capture transient slow-wave (A1) and fast-EEG (A2/A3) arousal bursts."),
    ]
    if missing_chs:
        s3_insights.append(("⚠", f"Missing/masked channels in {rec_id}: {', '.join(missing_chs)} (zeroed with mask=0, never fabricated)."))
    render_key_insights("Key Insights — Epoch Physiology & CAP Context", s3_insights)


# ---------------------------------------------------------------------------
# Screen 4: Live NIDRA Inference
# ---------------------------------------------------------------------------
elif screen == SCREENS[3]:
    st.title("4. Live NIDRA Inference")
    ckpt_files = sorted((PROJ_ROOT / "checkpoints").glob("phase4_*.pt")) if (PROJ_ROOT / "checkpoints").exists() else []
    if not ckpt_files:
        st.warning("No Phase 4 checkpoints found in checkpoints/.")
    else:
        ckpt_names = [c.name for c in ckpt_files]
        default_ckpt = "phase4_SomniMapNet_bilstm_w20_E8_dualhead_best.pt"
        default_idx = ckpt_names.index(default_ckpt) if default_ckpt in ckpt_names else 0
        sel_ckpt = st.selectbox("Select Checkpoint", ckpt_names, index=default_idx)
        val_ids = [r for r in split_data["partitions"]["val"]["record_ids"] if (PROC_DIR / r / "metadata.json").exists()]
        rec_id = st.selectbox("Validation Recording", val_ids)
        stride_s4 = st.select_slider(
            "Window Stride (epochs)",
            options=[1, 2, 5, 10, 20],
            value=5,
            key="s4_stride",
            help="Stride=5 matches the calibrated post-processing sweep defaults; smaller strides increase overlap/runtime.",
        )
        if st.button("Run Inference"):
            from src.inference.predictor import SomniMapPredictor

            rec_dir = PROC_DIR / rec_id
            meta = json.loads((rec_dir / "metadata.json").read_text(encoding="utf-8"))
            epochs_full = np.load(rec_dir / "epochs.npy")
            y_gt = np.load(rec_dir / "labels_stage.npy")
            ch_mask = np.array(meta["channel_mask"], dtype=np.float32)

            model_epochs = epochs_full[:, :, :6].copy()
            model_ch_mask = ch_mask[:6].copy()
            if model_ch_mask[0] == 0 and ch_mask[6] == 1:
                model_epochs[:, :, 0] = epochs_full[:, :, 6]
                model_ch_mask[0] = 1

            predictor = SomniMapPredictor(ckpt_name=sel_ckpt, cfg=CFG)
            pred = predictor.predict_arrays(model_epochs, model_ch_mask, stride_epochs=stride_s4)
            y_pr = pred["stage_preds"]
            mask = y_gt >= 0
            acc = float((y_pr[mask] == y_gt[mask]).mean())
            c1, c2, c3 = st.columns(3)
            c1.metric("Recording Epoch Accuracy", f"{acc*100:.2f}%")
            c2.metric("Mean Confidence", f"{float(np.mean(pred['stage_confidence'])):.3f}")
            c3.metric("Low-Conf Epochs (<0.55)", f"{len(pred['low_confidence_epochs'])}/{len(y_pr)}")
            st.caption(f"Provenance: `{pred['provenance']['checkpoint']}` | T={pred['provenance']['temperature']} | stride={stride_s4}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=np.where(mask, y_gt, np.nan), name="Ground Truth (Reference)", line_shape="hv"))
            fig.add_trace(go.Scatter(y=y_pr, name="Predicted (NIDRA)", line_shape="hv", line=dict(dash="dot")))
            fig.update_yaxes(tickvals=[0, 1, 2, 3, 4], ticktext=STAGE_NAMES, autorange="reversed")
            style_plotly(fig, title=f"Hypnogram Comparison — {rec_id}", height=360)
            st.plotly_chart(fig, use_container_width=True)

            mean_c = float(np.mean(pred["stage_confidence"]))
            n_low = len(pred["low_confidence_epochs"])
            low_pct = 100.0 * n_low / max(1, len(y_pr))
            s4_insights = [
                ("✓" if acc >= 0.80 else "ℹ",
                 f"On {rec_id}, NIDRA achieves {acc*100:.2f}% epoch accuracy ({'strong hypnogram alignment >= 80%' if acc >= 0.80 else 'moderate alignment — inspect transitional boundaries'}) with mean calibrated confidence {mean_c:.3f}."),
                ("ℹ", f"Low-confidence epochs (< 0.55): {n_low}/{len(y_pr)} ({low_pct:.1f}%), typically concentrated around Wake/N1/N2 and REM sleep-stage transitions."),
            ]
            if len(ch_mask) > 5 and ch_mask[5] == 0:
                s4_insights.append(("⚠", f"EMG_CHIN is missing (channel_mask[5]=0) in {rec_id}, which can introduce N1/N2 vs. REM onset ambiguity."))
            render_key_insights(f"Key Insights — Live Inference on {rec_id}", s4_insights)


# ---------------------------------------------------------------------------
# Screen 5: CAP Event Detection
# ---------------------------------------------------------------------------
elif screen == SCREENS[4]:
    st.title("5. CAP Event Detection Evaluation (A1 / A2 / A3)")
    cap_base = load_json_if_exists(str(RESULTS_DIR / "phase5_cap_eval_val.json"))
    stage2 = load_json_if_exists(str(RESULTS_DIR / "stage2_postproc_val.json"))
    sweep2 = load_json_if_exists(str(RESULTS_DIR / "stage2_cap_sweep_val.json"))

    if sweep2 or (stage2 and "cap" in stage2):
        st.subheader("Calibrated Stage 2 Post-Processing (Validation Split)")
        pipelines: dict[str, dict[str, Any]] = {}
        if sweep2 and "best_overall_f1" in sweep2:
            b = sweep2["best_overall_f1"]
            pipelines["sweep_best_overall"] = {
                "label": "Frozen Sweep Best Overall F1 (smooth=7, bg=0.35, a2=2.0, dur=2.5s, gap=1.5s)",
                "overall": {"f1": b["f1"], "precision": b["precision"], "recall": b["recall"], "fp": b["fp"]},
                "cap_rate_pearson_r": b["cap_rate_r"],
                "per_subtype": {
                    "A1": {"f1": b["a1_f1"]},
                    "A2": {"precision": b["a2_prec"], "recall": b["a2_rec"], "f1": b["a2_f1"]},
                    "A3": {"f1": b["a3_f1"]},
                },
                "caption": f"Frozen val sweep winner (`stage2_cap_sweep_val.json`) | FP={b['fp']} | A1 F1={b['a1_f1']} | A2 F1={b['a2_f1']} | A3 F1={b['a3_f1']}",
            }
        if sweep2 and "best_a2_f1" in sweep2:
            b2 = sweep2["best_a2_f1"]
            pipelines["sweep_best_a2"] = {
                "label": "Frozen Sweep Best A2 F1 (smooth=7, bg=0.35, a2=2.5, dur=2.5s, gap=1.5s)",
                "overall": {"f1": b2["f1"], "precision": b2["precision"], "recall": b2["recall"], "fp": b2["fp"]},
                "cap_rate_pearson_r": b2["cap_rate_r"],
                "per_subtype": {
                    "A1": {"f1": b2["a1_f1"]},
                    "A2": {"precision": b2["a2_prec"], "recall": b2["a2_rec"], "f1": b2["a2_f1"]},
                    "A3": {"f1": b2["a3_f1"]},
                },
                "caption": f"Frozen val sweep A2-focused (`stage2_cap_sweep_val.json`) | FP={b2['fp']} | A1 F1={b2['a1_f1']} | A2 F1={b2['a2_f1']} | A3 F1={b2['a3_f1']}",
            }
        if stage2 and "cap" in stage2:
            mode_labels = {
                "s5_smooth5_bg0.35_a2x1.5_dur2.0_gtNREM": "Stage 2 Benchmark (Best Overall F1, GT-NREM)",
                "s5_smooth7_bg0.50_a2x2.0_dur2.0_gtNREM": "Stage 2 Benchmark (Best A2 F1, GT-NREM)",
                "s5_smooth7_bg0.50_a2x2.0_dur2.0_predNREM": "Stage 2 Benchmark (Annotation-Free Pred-NREM)",
                "baseline_s20_gtNREM": "Uncalibrated Baseline (stride=20, min_dur=0.5s)",
            }
            for k, lbl in mode_labels.items():
                if k in stage2["cap"]:
                    d = stage2["cap"][k]
                    ov_d = d["overall"]
                    pipelines[k] = {
                        "label": lbl,
                        "overall": ov_d,
                        "cap_rate_pearson_r": d.get("cap_rate_pearson_r", 0.0),
                        "per_subtype": d["per_subtype"],
                        "caption": f"TP={ov_d['tp']} | FP={ov_d['fp']} | FN={ov_d['fn']} | Mean IoU={ov_d['mean_iou']} | Onset err={ov_d['onset_err_s']}s",
                    }

        avail_keys = list(pipelines.keys())
        sel_mode = st.selectbox("Select Evaluation Pipeline", avail_keys, format_func=lambda k: pipelines[k]["label"])
        sel_data = pipelines[sel_mode]
        ov = sel_data["overall"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Event F1 (IoU≥0.5)", ov["f1"])
        c2.metric("Precision", ov["precision"])
        c3.metric("Recall", ov["recall"])
        c4.metric("CAP Rate Pearson r", sel_data["cap_rate_pearson_r"])
        st.caption(sel_data["caption"])

        sub_rows = [{"Subtype": k, **v} for k, v in sel_data["per_subtype"].items()]
        st.subheader("Per-Subtype CAP Event Metrics (IoU ≥ 0.5)")
        st.dataframe(pd.DataFrame(sub_rows), use_container_width=True)

        st.subheader("Before vs. After Post-Processing Comparison (Validation)")
        comp_rows = []
        for d in pipelines.values():
            comp_rows.append({
                "Pipeline": d["label"],
                "Precision": d["overall"]["precision"],
                "Recall": d["overall"]["recall"],
                "Overall F1": d["overall"]["f1"],
                "FP": d["overall"]["fp"],
                "A1 F1": d["per_subtype"]["A1"]["f1"],
                "A2 F1": d["per_subtype"]["A2"]["f1"],
                "A3 F1": d["per_subtype"]["A3"]["f1"],
                "CAP Rate r": d["cap_rate_pearson_r"],
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
    elif cap_base:
        st.caption("Showing Uncalibrated Phase 5 Baseline (`phase5_cap_eval_val.json`).")
        ov = cap_base["overall"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Event F1 (IoU≥0.5)", ov["f1"])
        c2.metric("Precision", ov["precision"])
        c3.metric("Recall", ov["recall"])
        c4.metric("CAP Rate Pearson r", cap_base.get("cap_rate_pearson_r", 0.0))
        sub_rows = [{"Subtype": k, **v} for k, v in cap_base["per_subtype"].items()]
        st.dataframe(pd.DataFrame(sub_rows), use_container_width=True)
    else:
        st.info("Phase 5 CAP evaluation results not yet generated.")

    ex_img = RESULTS_DIR / "phase5_cap_examples_val.png"
    if ex_img.exists():
        st.image(str(ex_img), caption="CAP Event Detection Timeline Examples (Phase 5 Baseline)")

    if sweep2 or (stage2 and "cap" in stage2):
        sub_p = sel_data["per_subtype"]
        a1_f1 = sub_p["A1"]["f1"]
        a2_f1 = sub_p["A2"]["f1"]
        a3_f1 = sub_p["A3"]["f1"]
        s5_insights = [
            ("✓", f"Selected pipeline achieves Overall Event F1 = {ov['f1']:.4f} (Precision = {ov['precision']:.4f}, Recall = {ov['recall']:.4f} at IoU >= 0.5) and macro CAP Rate Pearson r = {sel_data['cap_rate_pearson_r']:.4f}."),
            ("ℹ", f"Subtype balance: A3 ({a3_f1:.4f}) and A1 slow-wave bursts ({a1_f1:.4f}) lead performance, while intermediate A2 events remain harder ({a2_f1:.4f}) due to morphological overlap between A1 and A3."),
            ("✓", "Strong macro CAP-rate correlation (r > 0.90) makes NIDRA reliable for overnight NREM sleep-fragmentation profiling, even where individual A2 event boundaries are exploratory."),
        ]
        render_key_insights("Key Insights — CAP Arousal Detection & Fragmentation", s5_insights)


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
        cases = xai_json["xai_cases"]
        s6_insights = []
        for c in cases[:2]:
            imp = c.get("channel_importance", [])
            if imp:
                top_i = int(np.argmax(imp))
                top_ch = CH_NAMES[top_i] if top_i < len(CH_NAMES) else f"ch{top_i}"
                s6_insights.append(
                    ("ℹ", f"Case `{c['tag']}` (True={c['true']}, Pred={c['pred']}): Primary attribution came from {top_ch} ({imp[top_i]*100:.1f}% share).")
                )
        s6_insights.append(
            ("✓", "Physiological drivers: slow delta power (0.5–4 Hz) on frontal/central EEG leads anchors N3 deep sleep, spindle/K-complex bursts mark N2, and high-frequency desynchronization + EOG/EMG cues separate Wake/REM/A3 arousals.")
        )
        render_key_insights("Key Insights — Spectral & Channel Attribution", s6_insights)


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

    if cal:
        s7_insights = [
            ("✓", f"Temperature scaling (T = {cal['temperature']}) reduces validation ECE from {cal['ece_before']} to {cal['ece_after']} and improves NLL ({cal['nll_before']} -> {cal['nll_after']}), aligning predicted confidence with true accuracy."),
        ]
        test_ref = load_json_if_exists(str(RESULTS_DIR / "phase8_test_locked_metrics.json"))
        cm = test_ref.get("confusion_matrix")
        if cm and len(cm) == 5:
            cm_arr = np.array(cm, dtype=np.int64)
            np.fill_diagonal(cm_arr, 0)
            ti, pi = np.unravel_index(int(np.argmax(cm_arr)), cm_arr.shape)
            s7_insights.append(
                ("⚠", f"Top off-diagonal confusion pair in evaluation: True {STAGE_NAMES[ti]} -> Predicted {STAGE_NAMES[pi]} ({int(cm_arr[ti, pi])} epochs), reflecting overlapping transitional EEG/EOG features.")
            )
        s7_insights.append(
            ("ℹ", "Brief transitional epochs (especially N1 light sleep) naturally carry higher normalized entropy and lower calibrated confidence, enabling uncertainty-aware human review.")
        )
        render_key_insights("Key Insights — Calibration & Error Patterns", s7_insights)


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

        import re as _re
        subj_list = test_json.get("per_subject", [])
        grp_scores: dict[str, list[float]] = {}
        for s in subj_list:
            m = _re.match(r"([a-zA-Z]+)", s["record_id"])
            g = m.group(1).upper() if m else "OTHER"
            grp_scores.setdefault(g, []).append(float(s["macro_f1"]))
        grp_means = {g: float(np.mean(v)) for g, v in grp_scores.items() if v}
        s8_insights = [
            ("✓", f"Locked 16-subject test performance: Macro-F1 = {test_json['macro_f1']:.4f} (95% CI [{test_json['ci95']['macro_f1'][0]:.4f}, {test_json['ci95']['macro_f1'][1]:.4f}]), Accuracy = {test_json['accuracy']*100:.2f}%, Cohen's kappa = {test_json['cohen_kappa']:.3f}."),
        ]
        if grp_means:
            best_g = max(grp_means.items(), key=lambda x: x[1])
            low_g = min(grp_means.items(), key=lambda x: x[1])
            s8_insights.append(
                ("ℹ", f"Cross-condition stability: highest mean Macro-F1 in {best_g[0]} ({best_g[1]:.3f}) vs. lowest in {low_g[0]} ({low_g[1]:.3f}) across {len(subj_list)} unseen test subjects.")
            )
        if subj_list:
            worst_s = min(subj_list, key=lambda x: x["macro_f1"])
            best_s = max(subj_list, key=lambda x: x["macro_f1"])
            s8_insights.append(
                ("⚠", f"Subject spread ranges from {best_s['record_id']} (Macro-F1={best_s['macro_f1']:.3f}) to {worst_s['record_id']} (Macro-F1={worst_s['macro_f1']:.3f}), driven by individual sleep fragmentation and channel/artifact noise.")
            )
        render_key_insights("Key Insights — Locked Test & Cross-Disorder Generalization", s8_insights)


# ---------------------------------------------------------------------------
# Screen 9: Raw EDF Upload, Multimodal Physiology & Report Export
# ---------------------------------------------------------------------------
elif screen == SCREENS[8]:
    import shutil
    import tempfile
    from src.inference.edf_loader import inspect_edf
    from src.inference.predictor import predict_from_edf
    from src.inference.report_export import (
        assess_domain_shift,
        build_analysis_report_markdown,
        build_analysis_summary_json,
        evaluate_against_reference_txt,
    )
    from src.physiology.multimodal import analyze_multimodal_physiology

    st.title("9. Raw EDF Upload, Multimodal Physiology & Report Export")
    st.info(
        "**Research Disclaimer:** NIDRA (*Neural Inference for Deep, REM & Arousal Architecture*) is an exploratory sleep-signal research platform, not a clinical diagnostic system. "
        "Model predictions and signal-derived physiological metrics are shown separately."
    )

    uploaded_edf = st.file_uploader("Upload an overnight PSG EDF file (.edf) [Required]", type=["edf"])
    uploaded_txt = st.file_uploader("Optional: Upload RemLogic .txt reference annotations (Ground-Truth Overlay Only)", type=["txt"])
    stride_opt = st.select_slider(
        "Inference Window Stride (epochs)",
        options=[1, 2, 5, 10, 20],
        value=5,
        key="s9_stride",
        help="Stride=5 matches the calibrated post-processing sweep defaults; smaller strides increase overlap/runtime.",
    )

    if uploaded_edf is not None:
        tmp_path: Path | None = None
        tmp_txt_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                uploaded_edf.seek(0)
                shutil.copyfileobj(uploaded_edf, tmp, length=4 * 1024 * 1024)
                tmp_path = Path(tmp.name)
            if uploaded_txt is not None:
                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as ttxt:
                    uploaded_txt.seek(0)
                    shutil.copyfileobj(uploaded_txt, ttxt, length=1024 * 1024)
                    tmp_txt_path = Path(ttxt.name)

            info = inspect_edf(tmp_path)
            st.subheader("Channel Inventory & Compatibility Check")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Duration (hrs)", round(info["duration_sec"] / 3600.0, 2))
            c2.metric("30-s Epochs", info["n_epochs_30s"])
            c3.metric("Header Rate (Hz)", info["header_sfreq_hz"])
            c4.metric("Compatible", "Yes" if info["compatible"] else "No")

            for w in info["warnings"]:
                st.warning(w)
            with st.expander("Channel Mapping & Multimodal Extras"):
                st.json({"canonical_mapping": info["canonical_mapping"], "multimodal_extras": info["multimodal_extras"]})

            if info["compatible"] and st.button("Run NIDRA + Multimodal Analysis"):
                with st.spinner("Preprocessing EDF and running NIDRA inference..."):
                    res = predict_from_edf(tmp_path, stride_epochs=stride_opt, cfg=CFG)
                    pred = res["predictions"]
                    phys = analyze_multimodal_physiology(
                        res["extras_raw"], res["loaded_sfreq"], pred["stage_preds"], pred.get("cap_events")
                    )

                gt_eval = None
                if tmp_txt_path is not None:
                    try:
                        gt_eval = evaluate_against_reference_txt(
                            tmp_txt_path, info.get("start_datetime"), pred
                        )
                        if gt_eval is None:
                            st.info("Optional .txt reference could not be aligned (missing EDF start_datetime or unsupported .txt format).")
                    except Exception as exc:
                        st.info(f"Optional .txt reference could not be aligned ({exc}).")

                st.subheader("Predictive Outputs — Hypnogram & Uncertainty")
                domain_shift = assess_domain_shift(info, pred)
                if domain_shift["high_domain_shift"]:
                    st.error(domain_shift["summary"])

                u1, u2, u3 = st.columns(3)
                u1.metric("Mean Calibrated Confidence", f"{float(np.mean(pred['stage_confidence'])):.3f}")
                u2.metric("Mean Normalized Entropy", f"{float(np.mean(pred['stage_entropy'])):.3f}")
                n_low = len(pred["low_confidence_epochs"])
                u3.metric("Low-Conf Epochs (<0.55)", f"{n_low}/{pred['n_epochs']} ({100.0*n_low/max(1,pred['n_epochs']):.1f}%)")

                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(y=pred["stage_preds"], name="Predicted (NIDRA)", line_shape="hv"))
                if gt_eval is not None:
                    gt_st = gt_eval["gt_stage_array"]
                    fig_h.add_trace(go.Scatter(
                        y=np.where(gt_st >= 0, gt_st, np.nan),
                        name="Ground Truth Reference (.txt)",
                        line_shape="hv",
                        line=dict(dash="dot"),
                    ))
                fig_h.update_yaxes(tickvals=[0, 1, 2, 3, 4], ticktext=STAGE_NAMES, autorange="reversed")
                fig_h.update_layout(xaxis_title="Epoch (30s)")
                style_plotly(fig_h, title="Sleep-Stage Hypnogram (Predictions vs Optional Reference)", height=330)
                st.plotly_chart(fig_h, use_container_width=True)

                fig_u = go.Figure()
                fig_u.add_trace(go.Scatter(y=pred["stage_confidence"], name="Calibrated Confidence", line=dict(width=1.2)))
                fig_u.add_trace(go.Scatter(y=pred["stage_entropy"], name="Normalized Entropy", line=dict(width=1.2, dash="dot")))
                fig_u.update_layout(xaxis_title="Epoch (30s)")
                style_plotly(fig_u, title="Per-Epoch Predictive Uncertainty", height=280)
                st.plotly_chart(fig_u, use_container_width=True)

                if gt_eval is not None:
                    st.subheader("Ground-Truth Evaluation (vs. Uploaded Reference .txt)")
                    st.caption("Computed post-hoc against uploaded reference annotations (zero model leakage)")
                    g1, g2, g3, g4 = st.columns(4)
                    g1.metric("Overall Accuracy", f"{gt_eval['stage_accuracy'] * 100:.2f}%")
                    g2.metric("Macro F1", f"{gt_eval['stage_macro_f1']:.4f}")
                    g3.metric("Macro Precision", f"{gt_eval['stage_macro_precision']:.4f}")
                    g4.metric("Macro Recall", f"{gt_eval['stage_macro_recall']:.4f}")
                    st.dataframe(pd.DataFrame(gt_eval["per_stage"]), use_container_width=True)

                    if gt_eval.get("cap_frame_metrics") or gt_eval.get("cap_event_metrics"):
                        cf = gt_eval.get("cap_frame_metrics")
                        ce = gt_eval.get("cap_event_metrics")
                        st.markdown("**CAP Ground-Truth Metrics (Valid NREM)**")
                        if cf and ce:
                            m1, m2, m3, m4, m5, m6 = st.columns(6)
                            m1.metric("Frame Prec", f"{cf['precision']:.4f}")
                            m2.metric("Frame Rec", f"{cf['recall']:.4f}")
                            m3.metric("Frame F1", f"{cf['f1']:.4f}")
                            m4.metric("Event Prec (IoU≥0.5)", f"{ce['precision']:.4f}")
                            m5.metric("Event Rec (IoU≥0.5)", f"{ce['recall']:.4f}")
                            m6.metric("Event F1 (IoU≥0.5)", f"{ce['f1']:.4f}")
                        elif cf:
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Frame Precision", f"{cf['precision']:.4f}")
                            m2.metric("Frame Recall", f"{cf['recall']:.4f}")
                            m3.metric("Frame F1", f"{cf['f1']:.4f}")

                if "cap_events" in pred:
                    st.subheader("Predictive Outputs — CAP Events (Predicted NREM)")
                    st.caption(
                        "**Model Caveat:** Post-processed CAP detection uses frozen `val` sweep defaults (`smooth=7`, `min_dur_s=2.5 s`, `merge_gap_s=1.5 s`, `a2_mult=2.0`). "
                        "Macro NREM CAP rate correlates strongly with ground truth on in-domain `val` (`r = 0.9319`), but per-event precision (`0.4558`) and `A2` subtype F1 (`~0.27–0.29` on `val`) remain exploratory — treat `A2` counts and exact boundaries with caution."
                    )
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("NREM CAP Rate", f"{pred['cap_rate_nrem']*100:.1f}%")
                    cc2.metric("A1 Events", pred["cap_event_counts"]["A1"])
                    cc3.metric("A2 Events (Exploratory)", pred["cap_event_counts"]["A2"])
                    cc4.metric("A3 Events", pred["cap_event_counts"]["A3"])
                    if pred["cap_events"]:
                        with st.expander(f"Inspect Detected CAP Events ({len(pred['cap_events'])} total)"):
                            ev_df = pd.DataFrame(pred["cap_events"])[["subtype_name", "start_s", "end_s", "duration_s", "epoch_idx"]]
                            st.dataframe(ev_df, use_container_width=True, height=240)

                st.subheader("Exploratory Signal-Derived Physiology (Brain–Heart–Respiration)")
                st.caption(phys["disclaimer"])
                p1, p2, p3 = st.columns(3)
                ecg_d = phys["ecg"]
                resp_d = phys["respiratory"]
                spo2_d = phys["spo2"]
                p1.metric("ECG Mean HR (bpm)", ecg_d["mean_hr_bpm"] if ecg_d.get("available") else "Unavailable")
                p2.metric("Resp Rate (br/min)", resp_d["mean_br_bpm"] if resp_d.get("available") else "Unavailable")
                p3.metric("SpO2 Mean (%)", spo2_d["mean_pct"] if spo2_d.get("available") else "Unavailable")

                if not ecg_d.get("available"):
                    st.caption(f"ECG: {ecg_d.get('reason', 'Not available')}")
                else:
                    st.write("**Stage-Stratified Cardiac Metrics (ECG):**")
                    present_stages = [st_name for st_name in STAGE_NAMES if pred["stage_counts"].get(st_name, 0) > 0]
                    by_st = ecg_d.get("by_stage", {})
                    ecg_rows = [
                        {"Stage": st_name, **by_st[st_name]}
                        if st_name in by_st
                        else {"Stage": st_name, "mean_hr_bpm": np.nan, "sdnn_ms": np.nan, "rmssd_ms": np.nan}
                        for st_name in present_stages
                    ]
                    if ecg_rows:
                        st.dataframe(pd.DataFrame(ecg_rows), use_container_width=True)
                    if ecg_d.get("mean_cap_hr_delta_bpm") is not None:
                        st.caption(f"Mean HR shift during predicted CAP vs pre-event baseline: **{ecg_d['mean_cap_hr_delta_bpm']:+.2f} bpm**")

                if not resp_d.get("available"):
                    st.caption(f"Respiratory: {resp_d.get('reason', 'Not available')}")
                elif resp_d.get("by_stage"):
                    st.write("**Stage-Stratified Respiratory Rate (breaths/min):**")
                    st.dataframe(pd.DataFrame([{"Stage": k, **v} for k, v in resp_d["by_stage"].items()]), use_container_width=True)

                if not spo2_d.get("available"):
                    st.caption(f"SpO2: {spo2_d.get('reason', 'Not available')}")

                md_rep = build_analysis_report_markdown(res, phys, recording_name=uploaded_edf.name, gt_eval=gt_eval)
                js_rep = build_analysis_summary_json(res, phys, recording_name=uploaded_edf.name, gt_eval=gt_eval)
                d1, d2 = st.columns(2)
                d1.download_button("Download Markdown Report (.md)", md_rep, file_name=f"nidra_{uploaded_edf.name}_report.md")
                d2.download_button("Download JSON Summary (.json)", js_rep, file_name=f"nidra_{uploaded_edf.name}_summary.json")

                sp = pred["stage_percentages"]
                dur_hrs = info["duration_sec"] / 3600.0
                s9_insights = [
                    ("ℹ", f"Recording `{uploaded_edf.name}` ({dur_hrs:.2f} hrs, {pred['n_epochs']} epochs): Wake {sp['Wake']:.1f}%, N1 {sp['N1']:.1f}%, N2 {sp['N2']:.1f}%, N3 {sp['N3']:.1f}%, REM {sp['REM']:.1f}%."),
                ]
                if "cap_rate_nrem" in pred:
                    cap_pct = pred["cap_rate_nrem"] * 100.0
                    stability = "elevated arousal fragmentation" if cap_pct >= 35.0 else "moderate/low NREM arousal fragmentation"
                    s9_insights.append(
                        ("ℹ", f"Predicted NREM CAP arousal rate is {cap_pct:.1f}% ({len(pred['cap_events'])} events; {stability}).")
                    )
                vitals_parts = []
                if ecg_d.get("available"):
                    vitals_parts.append(f"ECG HR {ecg_d['mean_hr_bpm']} bpm (SDNN {ecg_d['overall_sdnn_ms']} ms)")
                if resp_d.get("available"):
                    vitals_parts.append(f"Resp {resp_d['mean_br_bpm']} br/min")
                if spo2_d.get("available"):
                    vitals_parts.append(f"SpO2 mean {spo2_d['mean_pct']}% (min {spo2_d['min_pct']}%)")
                if vitals_parts:
                    s9_insights.append(("✓", "Multimodal vitals extracted: " + " | ".join(vitals_parts) + "."))
                else:
                    s9_insights.append(("ℹ", "No ECG, Respiratory, or SpO2 channels detected — multimodal vitals marked unavailable without fabrication."))

                if domain_shift["high_domain_shift"]:
                    s9_insights.append(("⚠", "High domain-shift / montage mismatch detected — treat stage shares and calibrated confidence as exploratory."))
                else:
                    s9_insights.append(("✓", "Channel coverage and stage distribution are in-domain for NIDRA."))

                if gt_eval is not None:
                    gt_msg = f"Ground-truth comparison (.txt): Stage Accuracy = {gt_eval['stage_accuracy']*100:.2f}%, Macro F1 = {gt_eval['stage_macro_f1']:.4f}"
                    if gt_eval.get("cap_event_metrics"):
                        gt_msg += f", CAP Event F1 (IoU>=0.5) = {gt_eval['cap_event_metrics']['f1']:.4f}"
                    s9_insights.append(("✓", gt_msg + "."))

                render_key_insights(f"Key Insights — {uploaded_edf.name}", s9_insights)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            if tmp_txt_path is not None:
                tmp_txt_path.unlink(missing_ok=True)
