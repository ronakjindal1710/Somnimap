"""PyTorch Dataset classes for SomniMap (epoch-level and sequence-window).

Strictly enforces subject-level partition boundaries:
- Windows are constructed within each recording independently; never across
  recording or split boundaries.
- Supports Stage 1 (single primary EEG: C4-A1 with C3-A2 fallback) and Stage 2
  (multi-channel 6/7-channel montage with channel-presence masks).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


class SomniMapEpochDataset(Dataset):
    """Epoch-level dataset for Phase 2 baselines (1D CNN, ResNet1D, InceptionTime)."""

    def __init__(
        self,
        processed_dir: Path,
        record_ids: list[str],
        mode: Literal["single_eeg", "multi_channel"] = "single_eeg",
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.mode = mode
        self.records: list[dict] = []
        self.index: list[tuple[int, int]] = []  # (rec_idx, epoch_idx)

        for rec_id in record_ids:
            rec_dir = self.processed_dir / rec_id
            if not (rec_dir / "metadata.json").exists():
                continue
            meta = json.loads((rec_dir / "metadata.json").read_text(encoding="utf-8"))
            y_stage = np.load(rec_dir / "labels_stage.npy")
            valid_idx = np.where(y_stage >= 0)[0]
            if len(valid_idx) == 0:
                continue

            rec_idx = len(self.records)
            self.records.append({
                "record_id": rec_id,
                "epochs_path": rec_dir / "epochs.npy",
                "y_stage": y_stage,
                "ch_mask": np.array(meta["channel_mask"], dtype=np.float32),
                "primary_eeg_idx": 0 if meta["channel_mask"][0] == 1 else 6,
            })
            for ep_i in valid_idx:
                self.index.append((rec_idx, int(ep_i)))

        self._mmap_cache: dict[int, np.ndarray] = {}

    def _get_epochs(self, rec_idx: int) -> np.ndarray:
        if rec_idx not in self._mmap_cache:
            self._mmap_cache[rec_idx] = np.load(self.records[rec_idx]["epochs_path"], mmap_mode="r")
        return self._mmap_cache[rec_idx]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec_idx, ep_i = self.index[idx]
        rec = self.records[rec_idx]
        ep = self._get_epochs(rec_idx)[ep_i]  # [3000, 7]

        if self.mode == "single_eeg":
            x = np.array(ep[:, rec["primary_eeg_idx"]], dtype=np.float32)[None, :]  # [1, 3000]
            ch_mask = np.ones(1, dtype=np.float32)
        else:
            x = np.array(ep[:, :6].T, dtype=np.float32)  # [6, 3000]
            ch_mask = rec["ch_mask"][:6]

        return {
            "x": torch.from_numpy(x),
            "ch_mask": torch.from_numpy(ch_mask),
            "y_stage": torch.tensor(int(rec["y_stage"][ep_i]), dtype=torch.long),
        }


class SomniMapSequenceDataset(Dataset):
    """Context-window sequence dataset for Phase 3–5 temporal & multi-task models."""

    def __init__(
        self,
        processed_dir: Path,
        record_ids: list[str],
        window_epochs: int = 20,
        stride_epochs: int = 10,
        mode: Literal["single_eeg", "multi_channel"] = "single_eeg",
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.window_epochs = window_epochs
        self.stride_epochs = stride_epochs
        self.mode = mode
        self.records: list[dict] = []
        self.windows: list[tuple[int, int]] = []  # (rec_idx, start_epoch)

        for rec_id in record_ids:
            rec_dir = self.processed_dir / rec_id
            if not (rec_dir / "metadata.json").exists():
                continue
            meta = json.loads((rec_dir / "metadata.json").read_text(encoding="utf-8"))
            y_stage = np.load(rec_dir / "labels_stage.npy")
            y_cap = np.load(rec_dir / "labels_cap.npy")
            n_ep = len(y_stage)
            if n_ep < window_epochs:
                continue

            rec_idx = len(self.records)
            self.records.append({
                "record_id": rec_id,
                "epochs_path": rec_dir / "epochs.npy",
                "y_stage": y_stage,
                "y_cap": y_cap,
                "ch_mask": np.array(meta["channel_mask"], dtype=np.float32),
                "primary_eeg_idx": 0 if meta["channel_mask"][0] == 1 else 6,
            })

            starts = list(range(0, n_ep - window_epochs + 1, stride_epochs))
            if starts[-1] != n_ep - window_epochs:
                starts.append(n_ep - window_epochs)
            for s in starts:
                if (y_stage[s : s + window_epochs] >= 0).any():
                    self.windows.append((rec_idx, s))

        self._mmap_cache: dict[int, np.ndarray] = {}

    def _get_epochs(self, rec_idx: int) -> np.ndarray:
        if rec_idx not in self._mmap_cache:
            self._mmap_cache[rec_idx] = np.load(self.records[rec_idx]["epochs_path"], mmap_mode="r")
        return self._mmap_cache[rec_idx]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec_idx, s = self.windows[idx]
        e = s + self.window_epochs
        rec = self.records[rec_idx]
        win = self._get_epochs(rec_idx)[s:e]  # [T_ep, 3000, 7]

        if self.mode == "single_eeg":
            x = np.array(win[:, :, rec["primary_eeg_idx"]], dtype=np.float32)[:, None, :]  # [T_ep, 1, 3000]
            ch_mask = np.ones(1, dtype=np.float32)
        else:
            x = np.transpose(np.array(win[:, :, :6], dtype=np.float32), (0, 2, 1))  # [T_ep, 6, 3000]
            ch_mask = rec["ch_mask"][:6]

        return {
            "x": torch.from_numpy(x),
            "ch_mask": torch.from_numpy(ch_mask),
            "y_stage": torch.from_numpy(rec["y_stage"][s:e].astype(np.int64)),
            "y_cap": torch.from_numpy(rec["y_cap"][s:e].astype(np.int64)),
        }
