"""Standalone NIDRA (SomniMapNet) predictor for raw EDF recordings and preprocessed arrays."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from src.config import get_repo_root, load_config
from src.models.somnimap_net import SomniMapNet
from src.training.eval_cap import CAP_NAMES, extract_events_from_frames
from src.training.metrics import STAGE_NAMES
from src.inference.preprocess_edf import preprocess_raw_edf


def load_checkpoint_registry() -> dict[str, Any]:
    reg_path = get_repo_root() / "checkpoints" / "registry.json"
    if reg_path.exists():
        return json.loads(reg_path.read_text(encoding="utf-8"))
    return {"default_inference_checkpoint": "phase4_SomniMapNet_bilstm_w20_E8_dualhead_best.pt", "checkpoints": {}}


def decode_cap_probabilities(
    cap_probs: np.ndarray,
    stage_preds: np.ndarray,
    smooth_frames: int = 7,
    bg_thresh: float = 0.35,
    a2_mult: float = 2.0,
    min_dur_s: float = 2.5,
    merge_gap_s: float = 1.5,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Decode [N_ep, 300, 4] CAP probabilities into 10 Hz labels and event list (masked to predicted NREM)."""
    n_ep = cap_probs.shape[0]
    p = cap_probs.reshape(-1, 4).copy()
    if smooth_frames > 1:
        p = uniform_filter1d(p, size=smooth_frames, axis=0)

    a_probs = p[..., 1:].copy()
    a_probs[..., 1] *= a2_mult
    best_sub = np.argmax(a_probs, axis=-1) + 1
    is_cap = (1.0 - p[..., 0]) >= bg_thresh
    raw_frames = np.where(is_cap, best_sub, 0)

    nrem_ep = np.isin(stage_preds, [1, 2, 3])
    nrem_frames = np.repeat(nrem_ep, 300)
    masked_frames = np.where(nrem_frames, raw_frames, 0)

    events = extract_events_from_frames(masked_frames, sfreq_hz=10.0, min_dur_s=min_dur_s, merge_gap_s=merge_gap_s)
    for ev in events:
        ev["subtype_name"] = CAP_NAMES.get(ev["subtype"], str(ev["subtype"]))
        ev["duration_s"] = round(ev["end_s"] - ev["start_s"], 2)
        ev["epoch_idx"] = int(ev["start_s"] // 30.0)
    return masked_frames.reshape(n_ep, 300), events


class SomniMapPredictor:
    """Reusable SomniMapNet inference wrapper with provenance and calibration."""

    def __init__(
        self,
        ckpt_name: str | None = None,
        device: str | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.registry = load_checkpoint_registry()
        self.ckpt_name = ckpt_name or self.registry.get(
            "default_inference_checkpoint", "phase4_SomniMapNet_bilstm_w20_E8_dualhead_best.pt"
        )
        proj_root = Path(self.cfg["paths"]["project_root"])
        self.ckpt_path = proj_root / "checkpoints" / self.ckpt_name
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        state = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        model_state = state["model_state"]
        ckpt_cfg = state.get("config") or {}
        meta = self.registry.get("checkpoints", {}).get(self.ckpt_name, {})

        has_chan_attn = any(k.startswith("chan_attn.") for k in model_state)
        has_cap_head = any(k.startswith("cap_head.") for k in model_state)

        inferred_in_ch = 6 if (has_chan_attn or any(tag in self.ckpt_name for tag in ("_E5_", "_E6_", "_E7_", "_E8_", "multichan", "dualhead"))) else 1
        self.in_channels = int(ckpt_cfg.get("in_channels", meta.get("in_channels", inferred_in_ch)))

        if "context_type" in ckpt_cfg:
            inferred_ctx = str(ckpt_cfg["context_type"])
        elif "context_type" in meta:
            inferred_ctx = str(meta["context_type"])
        elif "_tcn_" in self.ckpt_name or any("tcn" in k.lower() for k in model_state):
            inferred_ctx = "tcn"
        elif "_transformer_" in self.ckpt_name or any("transformer" in k.lower() or "pos_enc" in k.lower() for k in model_state):
            inferred_ctx = "transformer"
        else:
            inferred_ctx = "bilstm"
        self.context_type = inferred_ctx

        self.use_cap_head = bool(ckpt_cfg.get("use_cap_head", meta.get("use_cap_head", has_cap_head)))
        self.chan_attn = bool(ckpt_cfg.get("use_chan_attn", meta.get("chan_attn", has_chan_attn)))

        w_match = re.search(r"_w(\d+)_", self.ckpt_name)
        inferred_win = int(w_match.group(1)) if w_match else 20
        self.window_epochs = int(ckpt_cfg.get("window_epochs", meta.get("window_epochs", inferred_win)))
        self.temperature = float(ckpt_cfg.get("temperature", meta.get("temperature", 1.1757)))

        self.model = SomniMapNet(
            in_channels=self.in_channels,
            context_type=self.context_type,
            use_cap_head=self.use_cap_head,
        ).to(self.device)
        if not self.chan_attn and self.model.chan_attn is not None:
            self.model.chan_attn = None

        self.model.load_state_dict(model_state)
        self.model.eval()

    @torch.no_grad()
    def predict_arrays(
        self,
        model_epochs: np.ndarray,
        model_ch_mask: np.ndarray,
        stride_epochs: int = 5,
        batch_size: int = 8,
    ) -> dict[str, Any]:
        """Run overlapping-window inference on [N_ep, 3000, C] normalized arrays."""
        n_ep = int(model_epochs.shape[0])
        if n_ep < self.window_epochs:
            raise ValueError(f"Need at least {self.window_epochs} epochs, got {n_ep}")

        # Prepare [N_ep, C, 3000]
        if self.in_channels == 1:
            x_full = np.transpose(model_epochs[:, :, :1], (0, 2, 1)).astype(np.float32)
            cm_vec = np.ones(1, dtype=np.float32)
        else:
            x_full = np.transpose(model_epochs[:, :, :6], (0, 2, 1)).astype(np.float32)
            cm_vec = model_ch_mask[:6].astype(np.float32)

        starts = list(range(0, n_ep - self.window_epochs + 1, stride_epochs))
        if starts[-1] != n_ep - self.window_epochs:
            starts.append(n_ep - self.window_epochs)

        stage_prob_sum = np.zeros((n_ep, 5), dtype=np.float64)
        stage_count = np.zeros(n_ep, dtype=np.float64)
        cap_prob_sum = np.zeros((n_ep, 300, 4), dtype=np.float64) if self.use_cap_head else None
        cap_count = np.zeros(n_ep, dtype=np.float64)

        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i : i + batch_size]
            xb = np.stack([x_full[s : s + self.window_epochs] for s in batch_starts], axis=0)
            cmb = np.repeat(cm_vec[None, :], len(batch_starts), axis=0)
            out = self.model(
                torch.from_numpy(xb).to(self.device),
                ch_mask=torch.from_numpy(cmb).to(self.device),
            )
            st_p = torch.softmax(out["stage_logits"] / self.temperature, dim=-1).cpu().numpy()
            cp_p = torch.softmax(out["cap_logits"], dim=-1).cpu().numpy() if (self.use_cap_head and "cap_logits" in out) else None

            for j, s in enumerate(batch_starts):
                e = s + self.window_epochs
                stage_prob_sum[s:e] += st_p[j]
                stage_count[s:e] += 1.0
                if cap_prob_sum is not None and cp_p is not None:
                    cap_prob_sum[s:e] += cp_p[j]
                    cap_count[s:e] += 1.0

        stage_probs = stage_prob_sum / np.maximum(stage_count[:, None], 1.0)
        stage_preds = np.argmax(stage_probs, axis=-1).astype(np.int64)
        stage_conf = np.max(stage_probs, axis=-1)
        # Normalized Shannon entropy in [0, 1]
        entropy = -np.sum(stage_probs * np.log(np.clip(stage_probs, 1e-8, 1.0)), axis=-1) / np.log(5.0)

        stage_counts = {STAGE_NAMES[k]: int((stage_preds == k).sum()) for k in range(5)}
        stage_pcts = {k: round(100.0 * v / max(1, n_ep), 2) for k, v in stage_counts.items()}

        res: dict[str, Any] = {
            "n_epochs": n_ep,
            "stage_preds": stage_preds,
            "stage_probs": stage_probs,
            "stage_confidence": stage_conf,
            "stage_entropy": entropy,
            "low_confidence_epochs": np.where(stage_conf < 0.55)[0].tolist(),
            "stage_counts": stage_counts,
            "stage_percentages": stage_pcts,
            "provenance": {
                "checkpoint": self.ckpt_name,
                "context_type": self.context_type,
                "in_channels": self.in_channels,
                "temperature": self.temperature,
                "stride_epochs": stride_epochs,
                "device": str(self.device),
            },
        }

        if cap_prob_sum is not None:
            cap_probs = cap_prob_sum / np.maximum(cap_count[:, None, None], 1.0)
            cap_frames, cap_events = decode_cap_probabilities(cap_probs, stage_preds)
            nrem_frames = int(np.isin(stage_preds, [1, 2, 3]).sum() * 300)
            cap_pos_frames = int((cap_frames > 0).sum())
            res["cap_probs"] = cap_probs
            res["cap_frames"] = cap_frames
            res["cap_events"] = cap_events
            res["cap_rate_nrem"] = round(cap_pos_frames / max(1, nrem_frames), 4)
            res["cap_event_counts"] = {
                name: sum(1 for ev in cap_events if ev["subtype_name"] == name) for name in ("A1", "A2", "A3")
            }
        return res


def predict_from_edf(
    edf_path: str | Path,
    ckpt_name: str | None = None,
    stride_epochs: int = 5,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """End-to-end raw-EDF preprocessing + SomniMapNet inference."""
    cfg = cfg or load_config()
    pre = preprocess_raw_edf(edf_path, cfg)
    predictor = SomniMapPredictor(ckpt_name=ckpt_name, cfg=cfg)
    pred = predictor.predict_arrays(pre["model_epochs"], pre["model_ch_mask"], stride_epochs=stride_epochs)
    return {
        "edf_info": pre["info"],
        "epochs": pre["epochs"],
        "qc_flags": pre["qc_flags"],
        "channel_mask": pre["channel_mask"],
        "loaded_sfreq": pre["loaded_sfreq"],
        "extras_raw": pre["extras_raw"],
        "predictions": pred,
    }


NidraPredictor = SomniMapPredictor
