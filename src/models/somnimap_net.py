"""Flagship SomniMap-Net architecture (Blueprint Sec. 5 & 10).

Components:
1. Multi-Resolution Stem: parallel 1D convolutions (short/medium/long motifs)
   + residual refinement blocks.
2. Channel Attention: SE-style gating over available PSG channels, strictly
   masking missing channels to 0 using `ch_mask`.
3. U-Net Temporal Encoder-Decoder: downsamples to bottleneck, applies a
   configurable temporal context module (`bilstm`, `tcn`, `transformer`, or `none`),
   and upsamples with skip connections.
4. Dual Multi-Task Heads:
   - Sleep-stage head: per-epoch 5-class logits [B, T_ep, 5]
   - CAP event head: 10 Hz frame-level 4-class logits [B, T_ep, 300, 4] (0=none, A1/A2/A3)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResStem1D(nn.Module):
    """Parallel 1D convolutions capturing 0.15s, 0.35s, and 0.75s waveform motifs."""

    def __init__(self, in_ch: int = 1, branch_ch: int = 16) -> None:
        super().__init__()
        self.b1 = nn.Conv1d(in_ch, branch_ch, kernel_size=15, stride=2, padding=7, bias=False)
        self.b2 = nn.Conv1d(in_ch, branch_ch, kernel_size=35, stride=2, padding=17, bias=False)
        self.b3 = nn.Conv1d(in_ch, branch_ch, kernel_size=75, stride=2, padding=37, bias=False)
        out_ch = branch_ch * 3
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.act(self.bn(out))


class ChannelAttentionSE(nn.Module):
    """Masked Squeeze-and-Excitation attention across PSG channels."""

    def __init__(self, n_channels: int, feat_dim: int) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.fc = nn.Sequential(
            nn.Linear(n_channels * feat_dim, max(16, n_channels * 4)),
            nn.GELU(),
            nn.Linear(max(16, n_channels * 4), n_channels),
        )

    def forward(self, ch_feats: torch.Tensor, ch_mask: torch.Tensor) -> torch.Tensor:
        # ch_feats: [B*T, C, D, L], ch_mask: [B*T, C]
        bt, c, d, l = ch_feats.shape
        masked = ch_feats * ch_mask.view(bt, c, 1, 1)
        pooled = masked.mean(dim=-1).reshape(bt, c * d)
        scores = self.fc(pooled).masked_fill(ch_mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1).view(bt, c, 1, 1)
        return (masked * attn).sum(dim=1)  # [B*T, D, L]


class ResConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 7) -> None:
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + self.skip(x))


class SomniMapNet(nn.Module):
    """Multi-scale residual temporal U-Net with channel attention and multi-task heads."""

    def __init__(
        self,
        in_channels: int = 1,
        base_ch: int = 48,
        context_type: str = "bilstm",
        n_stage_classes: int = 5,
        n_cap_classes: int = 4,
        use_cap_head: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.context_type = context_type
        self.use_cap_head = use_cap_head

        # Per-channel multi-resolution stem (3000 -> 1500)
        self.stem = MultiResStem1D(in_ch=1, branch_ch=base_ch // 3)  # out = base_ch (48)
        if in_channels > 1:
            self.chan_attn = ChannelAttentionSE(n_channels=in_channels, feat_dim=base_ch)
        else:
            self.chan_attn = None

        # U-Net Encoder (applied across concatenated sequence [B, base_ch, T_ep * 1500])
        self.enc1 = ResConvBlock1D(base_ch, base_ch, k=9)           # L/1 (1500/ep)
        self.enc2 = ResConvBlock1D(base_ch, base_ch * 2, k=7)       # L/5 (300/ep -> 10 Hz!)
        self.enc3 = ResConvBlock1D(base_ch * 2, base_ch * 3, k=7)   # L/25 (60/ep)
        self.enc4 = ResConvBlock1D(base_ch * 3, base_ch * 4, k=5)   # L/125 (12/ep)

        bn_dim = base_ch * 4  # 192
        # Temporal Context Bottleneck at epoch-level (pooled from enc4)
        if context_type == "bilstm":
            self.ctx = nn.LSTM(bn_dim, bn_dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        elif context_type == "tcn":
            self.ctx = nn.Sequential(
                nn.Conv1d(bn_dim, bn_dim, 3, padding=1, dilation=1, bias=False), nn.BatchNorm1d(bn_dim), nn.GELU(),
                nn.Conv1d(bn_dim, bn_dim, 3, padding=2, dilation=2, bias=False), nn.BatchNorm1d(bn_dim), nn.GELU(),
                nn.Conv1d(bn_dim, bn_dim, 3, padding=4, dilation=4, bias=False), nn.BatchNorm1d(bn_dim), nn.GELU(),
            )
        elif context_type == "transformer":
            self.pos_emb = nn.Parameter(torch.randn(1, 64, bn_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=bn_dim, nhead=4, dim_feedforward=bn_dim * 2, dropout=dropout, activation="gelu", batch_first=True
            )
            self.ctx = nn.TransformerEncoder(layer, num_layers=2)
        elif context_type == "none":
            self.ctx = None
        else:
            raise ValueError(f"Unknown context_type: {context_type}")

        # U-Net Decoder back to 10 Hz (300 steps/epoch)
        self.up3 = nn.Conv1d(bn_dim, base_ch * 3, 1)
        self.dec3 = ResConvBlock1D(base_ch * 6, base_ch * 3, k=5)
        self.up2 = nn.Conv1d(base_ch * 3, base_ch * 2, 1)
        self.dec2 = ResConvBlock1D(base_ch * 4, base_ch * 2, k=7)  # 10 Hz feature map (300 steps/ep)

        dec_dim = base_ch * 2  # 96
        self.stage_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dec_dim + bn_dim, n_stage_classes),
        )
        if use_cap_head:
            self.cap_head = nn.Sequential(
                nn.Conv1d(dec_dim, dec_dim // 2, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(dec_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(dec_dim // 2, n_cap_classes, kernel_size=1),
            )
        else:
            self.cap_head = None

    def forward(self, x: torch.Tensor, ch_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # x: [B, T_ep, C, 3000], ch_mask: [B, C]
        b, t_ep, c, s = x.shape
        x_flat = x.reshape(b * t_ep * c, 1, s)
        stem_out = self.stem(x_flat)  # [B*T*C, D, 1500]
        _, d, l_stem = stem_out.shape

        if self.chan_attn is not None:
            if ch_mask is None:
                ch_mask = torch.ones((b, c), device=x.device, dtype=x.dtype)
            ch_m = ch_mask.unsqueeze(1).expand(b, t_ep, c).reshape(b * t_ep, c)
            fused = self.chan_attn(stem_out.view(b * t_ep, c, d, l_stem), ch_m)
        elif c > 1:
            if ch_mask is None:
                ch_mask = torch.ones((b, c), device=x.device, dtype=x.dtype)
            ch_m = ch_mask.unsqueeze(1).expand(b, t_ep, c).reshape(b * t_ep, c)
            masked = stem_out.view(b * t_ep, c, d, l_stem) * ch_m.view(b * t_ep, c, 1, 1)
            fused = masked.sum(dim=1) / ch_m.sum(dim=1).clamp(min=1.0).view(b * t_ep, 1, 1)
        else:
            fused = stem_out.view(b * t_ep, d, l_stem)

        # Concatenate across epochs: [B, D, T_ep * 1500]
        seq_feat = fused.view(b, t_ep, d, l_stem).permute(0, 2, 1, 3).reshape(b, d, t_ep * l_stem)

        e1 = self.enc1(seq_feat)                   # T_ep * 1500
        e2 = self.enc2(F.max_pool1d(e1, 5))        # T_ep * 300 (10 Hz!)
        e3 = self.enc3(F.max_pool1d(e2, 5))        # T_ep * 60
        e4 = self.enc4(F.max_pool1d(e3, 5))        # T_ep * 12

        # Epoch-level context bottleneck
        ep_bn = F.avg_pool1d(e4, kernel_size=12, stride=12).transpose(1, 2)  # [B, T_ep, bn_dim]
        if self.context_type == "bilstm":
            ep_ctx, _ = self.ctx(ep_bn)
        elif self.context_type == "tcn":
            ep_ctx = self.ctx(ep_bn.transpose(1, 2)).transpose(1, 2)
        elif self.context_type == "transformer":
            ep_ctx = self.ctx(ep_bn + self.pos_emb[:, :t_ep, :])
        else:
            ep_ctx = ep_bn

        # Inject context back into U-Net bottleneck
        bn_seq = e4 + F.interpolate(ep_ctx.transpose(1, 2), size=e4.shape[-1], mode="nearest")

        d3 = self.up3(F.interpolate(bn_seq, size=e3.shape[-1], mode="linear", align_corners=False))
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(F.interpolate(d3, size=e2.shape[-1], mode="linear", align_corners=False))
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # [B, dec_dim, T_ep * 300]

        # Sleep-stage head: fuse decoder epoch-pooled features + context bottleneck
        ep_dec = F.avg_pool1d(d2, kernel_size=300, stride=300).transpose(1, 2)  # [B, T_ep, dec_dim]
        stage_logits = self.stage_head(torch.cat([ep_dec, ep_ctx], dim=-1))     # [B, T_ep, 5]

        out = {"stage_logits": stage_logits}
        if self.cap_head is not None:
            cap_logits = self.cap_head(d2)  # [B, 4, T_ep * 300]
            out["cap_logits"] = cap_logits.transpose(1, 2).reshape(b, t_ep, 300, -1)
        return out
