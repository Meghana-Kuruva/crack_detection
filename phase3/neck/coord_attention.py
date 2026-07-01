"""
Phase 3 — Coordinate Attention
================================
Implements "Coordinate Attention for Efficient Mobile Network Design"
(Hou et al., CVPR 2021) adapted for thin-crack oriented bounding box detection.

WHY coordinate attention over vanilla spatial attention (CBAM, SE):
  - Vanilla SE: only channel attention, no spatial sensitivity — misses elongated cracks
  - Vanilla spatial attention: isotropic (pool along both H and W simultaneously)
    → loses directional information about crack orientation
  - Coordinate Attention: factorizes into separate 1-D encodings along H and W
    → captures "crack runs horizontally across this row" or "crack runs vertically"
    → preserves precise positional information unlike global average pooling
    → cheap: two 1-D pools (O(H) + O(W)) instead of an O(HW) self-attention

Architecture:
    x  → pool_H (B, C, H, 1)  ↗
         pool_W (B, C, 1, W)  → concat along H dim → shared_conv(1×1) → split
                                 → h_attn (B, C, H, 1), w_attn (B, C, 1, W)
    out = x * sigmoid(h_attn) * sigmoid(w_attn)

Usage:
    ca = CoordinateAttention(in_ch=128, reduction=32)
    out = ca(x)   # same shape as x, recalibrated by directional attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention module.

    Args:
        in_ch     : number of input channels
        reduction : channel reduction ratio for the intermediate projection
                    (default 32 means mid = max(in_ch // reduction, 8))
    """

    def __init__(self, in_ch: int, reduction: int = 32):
        super().__init__()
        mid = max(in_ch // reduction, 8)

        # Shared 1×1 conv that projects the concatenated H and W encodings
        # We use a factorised version: BN + ReLU + 1×1
        self.pool_h  = nn.AdaptiveAvgPool2d((None, 1))  # → (B, C, H, 1)
        self.pool_w  = nn.AdaptiveAvgPool2d((1, None))  # → (B, C, 1, W)

        # Project concatenated 1-D features (along the H axis after transposition)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # Two separate 1×1 convs to generate H-direction and W-direction gates
        self.gate_h = nn.Conv2d(mid, in_ch, 1, bias=False)
        self.gate_w = nn.Conv2d(mid, in_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── 1-D pooling along each axis ───────────────────────────────────────
        # pool_h: average all W positions → (B, C, H, 1)
        h_enc = self.pool_h(x)               # (B, C, H, 1)
        # pool_w: average all H positions → (B, C, 1, W) → transpose to (B, C, W, 1)
        w_enc = self.pool_w(x).transpose(-2, -1)  # (B, C, W, 1)

        # Concatenate along the "spatial" dimension (H + W combined along dim 2)
        # so the shared conv sees a single (B, C, H+W, 1) input
        hw = torch.cat([h_enc, w_enc], dim=2)   # (B, C, H+W, 1)

        # ── Shared projection ─────────────────────────────────────────────────
        hw = self.shared_conv(hw)               # (B, mid, H+W, 1)

        # ── Split and gate ────────────────────────────────────────────────────
        h_feat = hw[:, :, :H, :]               # (B, mid, H, 1)
        w_feat = hw[:, :, H:, :].transpose(-2, -1)  # (B, mid, 1, W)

        # Generate attention gates
        h_gate = torch.sigmoid(self.gate_h(h_feat))   # (B, C, H, 1)
        w_gate = torch.sigmoid(self.gate_w(w_feat))   # (B, C, 1, W)

        # Apply: gates broadcast over W and H dimensions respectively
        out = x * h_gate * w_gate               # (B, C, H, W)
        return out
