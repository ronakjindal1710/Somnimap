"""Phase 3 temporal/sequence models for 20-epoch sleep-staging windows.

All models take [B, T_ep, 1, 3000] (single-EEG) or [B, T_ep, C, 3000]
and output per-epoch 5-class logits [B, T_ep, 5].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpochEncoder1D(nn.Module):
    """Shared lightweight 1D CNN epoch feature extractor -> [B*T, D]."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=25, stride=2, padding=12, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(64, embed_dim, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, 3000]
        b, t, c, s = x.shape
        h = self.net(x.view(b * t, c, s)).squeeze(-1)
        return self.drop(h).view(b, t, -1)


class CNNBiLSTM(nn.Module):
    """EpochEncoder1D + 2-layer BiLSTM context."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 128, hidden: int = 128, n_classes: int = 5) -> None:
        super().__init__()
        self.encoder = EpochEncoder1D(in_channels=in_channels, embed_dim=embed_dim)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        seq, _ = self.bilstm(h)
        return self.head(seq)


class TCNBlock1D(nn.Module):
    """Non-causal dilated 1D residual block across epochs."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.2) -> None:
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class CNNTCN(nn.Module):
    """EpochEncoder1D + non-causal dilated TCN across epochs."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 128, n_classes: int = 5) -> None:
        super().__init__()
        self.encoder = EpochEncoder1D(in_channels=in_channels, embed_dim=embed_dim)
        self.tcn = nn.Sequential(
            TCNBlock1D(embed_dim, dilation=1),
            TCNBlock1D(embed_dim, dilation=2),
            TCNBlock1D(embed_dim, dilation=4),
            TCNBlock1D(embed_dim, dilation=8),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).transpose(1, 2)  # [B, D, T]
        h = self.tcn(h).transpose(1, 2)      # [B, T, D]
        return self.head(h)


class CNNTransformer(nn.Module):
    """EpochEncoder1D + 2-layer TransformerEncoder across epochs."""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128,
        max_epochs_win: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        self.encoder = EpochEncoder1D(in_channels=in_channels, embed_dim=embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_epochs_win, embed_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 2,
            dropout=0.25,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)  # [B, T, D]
        t = h.shape[1]
        h = self.transformer(h + self.pos_emb[:, :t, :])
        return self.head(h)


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 7) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=k // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalUNet(nn.Module):
    """U-Sleep-inspired 1D U-Net over concatenated epoch sequence [B, T_ep, C, 3000]."""

    def __init__(self, in_channels: int = 1, base_ch: int = 24, n_classes: int = 5) -> None:
        super().__init__()
        # Strided stem (stride=5 -> 20 Hz, 600 steps/epoch) for fast CPU sequence U-Net
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=15, stride=5, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
        )
        # Encoder
        self.enc1 = ConvBlock1D(base_ch, base_ch, k=9)          # 600/ep
        self.enc2 = ConvBlock1D(base_ch, base_ch * 2, k=7)      # 150/ep
        self.enc3 = ConvBlock1D(base_ch * 2, base_ch * 4, k=7)  # 30/ep
        self.enc4 = ConvBlock1D(base_ch * 4, base_ch * 6, k=5)  # 6/ep
        self.bottleneck = ConvBlock1D(base_ch * 6, base_ch * 8, k=5)  # 2/ep

        # Decoder
        self.up4 = nn.Conv1d(base_ch * 8, base_ch * 6, kernel_size=1)
        self.dec4 = ConvBlock1D(base_ch * 12, base_ch * 6, k=5)

        self.up3 = nn.Conv1d(base_ch * 6, base_ch * 4, kernel_size=1)
        self.dec3 = ConvBlock1D(base_ch * 8, base_ch * 4, k=5)

        self.up2 = nn.Conv1d(base_ch * 4, base_ch * 2, kernel_size=1)
        self.dec2 = ConvBlock1D(base_ch * 4, base_ch * 2, k=7)

        self.up1 = nn.Conv1d(base_ch * 2, base_ch, kernel_size=1)
        self.dec1 = ConvBlock1D(base_ch * 2, base_ch, k=7)

        self.classifier = nn.Conv1d(base_ch, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_ep, C, 3000] -> [B, C, T_ep * 3000]
        b, t_ep, c, s = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(b, c, t_ep * s)

        e0 = self.stem(x_flat)                 # T_ep * 600
        e1 = self.enc1(e0)                     # T_ep * 600
        e2 = self.enc2(F.max_pool1d(e1, 4))    # T_ep * 150
        e3 = self.enc3(F.max_pool1d(e2, 5))    # T_ep * 30
        e4 = self.enc4(F.max_pool1d(e3, 5))    # T_ep * 6
        bn = self.bottleneck(F.max_pool1d(e4, 3))  # T_ep * 2

        d4 = self.up4(F.interpolate(bn, size=e4.shape[-1], mode="linear", align_corners=False))
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(F.interpolate(d4, size=e3.shape[-1], mode="linear", align_corners=False))
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(F.interpolate(d3, size=e2.shape[-1], mode="linear", align_corners=False))
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(F.interpolate(d2, size=e1.shape[-1], mode="linear", align_corners=False))
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Segment-level pooling back to T_ep epochs: [B, n_classes, T_ep] -> [B, T_ep, n_classes]
        step_per_ep = d1.shape[-1] // t_ep
        out = F.avg_pool1d(self.classifier(d1), kernel_size=step_per_ep, stride=step_per_ep)
        return out.transpose(1, 2)


def build_temporal_model(name: str, in_channels: int = 1, n_classes: int = 5) -> nn.Module:
    if name == "CNN_BiLSTM":
        return CNNBiLSTM(in_channels=in_channels, n_classes=n_classes)
    if name == "CNN_TCN":
        return CNNTCN(in_channels=in_channels, n_classes=n_classes)
    if name == "CNN_Transformer":
        return CNNTransformer(in_channels=in_channels, n_classes=n_classes)
    if name == "TemporalUNet":
        return TemporalUNet(in_channels=in_channels, n_classes=n_classes)
    raise ValueError(f"Unknown temporal model: {name}")
