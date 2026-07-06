#!/usr/bin/env python3
"""
sr_model.py
===========
A compact 3D residual-channel-attention network (RCAN-style) for super-resolving
soft, low-contrast polymeric-scaffold micro-CT, plus an edge-aware loss.

Design choice: *pre-upsampling residual learning*. The network receives the LR volume
already interpolated onto the HR grid and predicts the RESIDUAL to HR. This (a) sidesteps
the awkward non-integer scale factor (2.521) entirely, and (b) focuses the model on exactly
the high-frequency edge/fringe structure that interpolation misses — the error we saw
concentrated on the boundaries in the |interp - HR| map.

Standard PyTorch ops only (Conv3d etc.) — all AOT-compiled in the cu128 wheel, so no
custom-kernel JIT (the Blackwell gotcha) is involved.
"""
from __future__ import annotations
import torch
import torch.nn as nn


# --------------------------------------------------------------------- building blocks
class ChannelAttention(nn.Module):
    """Squeeze-and-excite style 3D channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        r = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, r, 1), nn.ReLU(inplace=True),
            nn.Conv3d(r, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class RCAB(nn.Module):
    """Residual channel-attention block."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            ChannelAttention(channels, reduction),
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualGroup(nn.Module):
    def __init__(self, channels: int, n_blocks: int, reduction: int = 8):
        super().__init__()
        self.blocks = nn.Sequential(*[RCAB(channels, reduction) for _ in range(n_blocks)])
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv(self.blocks(x))


class SR3DNet(nn.Module):
    """3D RCAN-lite operating at HR grid; predicts the residual image."""
    def __init__(self, channels: int = 64, n_groups: int = 4, n_blocks: int = 4,
                 reduction: int = 8):
        super().__init__()
        self.head = nn.Conv3d(1, channels, 3, padding=1)
        self.groups = nn.Sequential(*[ResidualGroup(channels, n_blocks, reduction)
                                      for _ in range(n_groups)])
        self.body_conv = nn.Conv3d(channels, channels, 3, padding=1)
        self.tail = nn.Conv3d(channels, 1, 3, padding=1)

    def forward(self, x):                 # x: interpolated-LR on HR grid [B,1,D,H,W]
        f = self.head(x)
        b = self.groups(f)
        b = self.body_conv(b) + f         # long skip (residual-in-residual)
        residual = self.tail(b)
        return x + residual               # global residual: predict edges on top of interp

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------- losses
def charbonnier(pred, target, eps: float = 1e-3):
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def gradient_l1(pred, target):
    """Edge-aware term: L1 between finite-difference gradients along D, H, W."""
    loss = 0.0
    for ax in (2, 3, 4):
        gp = pred.diff(dim=ax)
        gt = target.diff(dim=ax)
        loss = loss + (gp - gt).abs().mean()
    return loss / 3.0


class SRLoss(nn.Module):
    """Charbonnier (fidelity) + edge-aware gradient L1 (boundary fidelity)."""
    def __init__(self, l1_w: float = 1.0, edge_w: float = 0.1):
        super().__init__()
        self.l1_w, self.edge_w = l1_w, edge_w

    def forward(self, pred, target):
        l1 = charbonnier(pred, target)
        edge = gradient_l1(pred, target)
        total = self.l1_w * l1 + self.edge_w * edge
        return total, {"l1": float(l1.detach()), "edge": float(edge.detach())}


def psnr(pred, target, data_range: float = 1.0):
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-12)
    return float(20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10(mse))


if __name__ == "__main__":
    # tiny self-check
    net = SR3DNet(channels=8, n_groups=1, n_blocks=1)
    x = torch.randn(1, 1, 16, 16, 16)
    y = net(x)
    print("params:", net.param_count(), "in:", tuple(x.shape), "out:", tuple(y.shape))
