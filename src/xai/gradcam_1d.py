"""1D Grad-CAM for SomniMapNet (sleep-stage and CAP event attributions)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class GradCAM1D:
    """Computes 1D Grad-CAM over a target Conv1d/ResConvBlock1D layer (e.g., model.dec2)."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_act),
            target_layer.register_full_backward_hook(self._save_grad),
        ]

    def _save_act(self, _module, _inp, out) -> None:
        self.activations = out

    def _save_grad(self, _module, _grad_in, grad_out) -> None:
        self.gradients = grad_out[0]

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def explain_stage(
        self,
        x: torch.Tensor,
        ch_mask: torch.Tensor | None,
        epoch_idx: int,
        target_class: int,
        out_len: int = 3000,
    ) -> torch.Tensor:
        """Return normalized 1D Grad-CAM [out_len] for a specific epoch & sleep stage."""
        self.model.zero_grad()
        out = self.model(x, ch_mask=ch_mask)
        score = out["stage_logits"][0, epoch_idx, target_class]
        score.backward()

        act = self.activations[0]  # [D, T_ep * 300]
        grad = self.gradients[0]   # [D, T_ep * 300]
        weights = grad.mean(dim=-1, keepdim=True)  # [D, 1]
        cam = torch.relu((weights * act).sum(dim=0))  # [T_ep * 300]

        # Slice the target epoch (300 steps at 10 Hz) and upsample to out_len (100 Hz = 3000)
        s = epoch_idx * 300
        e = s + 300
        cam_ep = cam[s:e].view(1, 1, -1)
        cam_up = F.interpolate(cam_ep, size=out_len, mode="linear", align_corners=False).squeeze()
        cam_up = cam_up - cam_up.min()
        if cam_up.max() > 1e-8:
            cam_up = cam_up / cam_up.max()
        return cam_up.detach().cpu()

    def explain_cap(
        self,
        x: torch.Tensor,
        ch_mask: torch.Tensor | None,
        epoch_idx: int,
        cap_class: int,
        out_len: int = 3000,
    ) -> torch.Tensor:
        """Return normalized 1D Grad-CAM [out_len] for CAP subtype (1=A1, 2=A2, 3=A3) in an epoch."""
        self.model.zero_grad()
        out = self.model(x, ch_mask=ch_mask)
        if "cap_logits" not in out:
            raise ValueError("Model does not have cap_head enabled.")
        score = out["cap_logits"][0, epoch_idx, :, cap_class].sum()
        score.backward()

        act = self.activations[0]
        grad = self.gradients[0]
        weights = grad.mean(dim=-1, keepdim=True)
        cam = torch.relu((weights * act).sum(dim=0))

        s = epoch_idx * 300
        e = s + 300
        cam_ep = cam[s:e].view(1, 1, -1)
        cam_up = F.interpolate(cam_ep, size=out_len, mode="linear", align_corners=False).squeeze()
        cam_up = cam_up - cam_up.min()
        if cam_up.max() > 1e-8:
            cam_up = cam_up / cam_up.max()
        return cam_up.detach().cpu()
