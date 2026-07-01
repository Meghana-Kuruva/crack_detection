"""
Phase 3 — Strip Pooling
========================
Implements horizontal and vertical strip pooling for elongated crack detection.

WHY strip pooling:
  Standard max/avg pooling is isotropic — it aggregates context in a square region.
  Railway cracks are elongated: a longitudinal crack can span >80% of the sleeper
  length (hundreds of pixels at stride 8), but is only 1-3 px wide.

  Strip pooling captures context along the FULL extent of a row (horizontal strip)
  or column (vertical strip):
    - Horizontal strip pool: avg pool with kernel (1, W) → each output pixel sees
      the entire row, making it easy to detect "continuous line across this row"
    - Vertical strip pool: avg pool with kernel (H, 1) → analogous for columns

  After the pooled context is upsampled back and added to the original feature,
  the model can suppress "point-like" ballast gap FPs (they don't activate horizontal
  strips consistently) while amplifying "line-like" cracks.

Reference: Hou et al. "Strip Pooling: Rethinking Spatial Pooling for Scene Parsing",
           CVPR 2020.

Usage:
    sp = StripPooling(in_ch=128)
    out = sp(x)     # same shape as x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StripPooling(nn.Module):
    """
    Strip pooling: horizontal + vertical 1-D global average pooling.

    Architecture:
        x  →  H-strip pool (B, C, H, 1) → project → upsample back to (B, C, H, W)
           +  V-strip pool (B, C, 1, W) → project → upsample back to (B, C, H, W)
           +  identity x
        → fused (B, C, H, W)

    The 1×1 projections before/after pooling reduce channels during the aggregate
    step (cheaper) and restore them on the way back.

    Args:
        in_ch     : input channels (= output channels)
        reduction : how much to reduce channels in the strip branch (default 4×)
    """

    def __init__(self, in_ch: int, reduction: int = 4):
        super().__init__()
        mid = max(in_ch // reduction, 8)

        # H-strip branch: pool along W → (B, C, H, 1)
        self.h_pool  = nn.AdaptiveAvgPool2d((None, 1))
        self.h_proj  = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.h_expand = nn.Conv2d(mid, in_ch, 1, bias=False)

        # V-strip branch: pool along H → (B, C, 1, W)
        self.v_pool  = nn.AdaptiveAvgPool2d((1, None))
        self.v_proj  = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.v_expand = nn.Conv2d(mid, in_ch, 1, bias=False)

        # Final fuse: combine h + v + identity
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
        )

        self.bn_id = nn.BatchNorm2d(in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── Horizontal strip: pools all W positions ───────────────────────────
        h = self.h_pool(x)             # (B, C, H, 1)
        h = self.h_proj(h)             # (B, mid, H, 1)
        h = self.h_expand(h)           # (B, C, H, 1)
        h = h.expand(-1, -1, H, W)    # broadcast to (B, C, H, W)

        # ── Vertical strip: pools all H positions ─────────────────────────────
        v = self.v_pool(x)             # (B, C, 1, W)
        v = self.v_proj(v)             # (B, mid, 1, W)
        v = self.v_expand(v)           # (B, C, 1, W)
        v = v.expand(-1, -1, H, W)    # broadcast to (B, C, H, W)

        # ── Fuse: additive combination then normalise ─────────────────────────
        out = self.fuse(h + v) + self.bn_id(x)
        return out


class StripAttention(nn.Module):
    """
    Strip attention: like strip pooling but with learned attention weights
    instead of simple average.  More expensive but more expressive.

    Useful when you want the model to SELECT which part of a strip is most
    informative (e.g., the rail-seat region) rather than averaging.
    """

    def __init__(self, in_ch: int, reduction: int = 4):
        super().__init__()
        mid = max(in_ch // reduction, 8)

        # H-strip attention weights (B, 1, H, 1) gating each row
        self.h_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),   # (B, C, H, 1)
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=False),  # (B, 1, H, 1)
            nn.Sigmoid(),
        )

        # V-strip attention weights (B, 1, 1, W) gating each column
        self.v_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),   # (B, C, 1, W)
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=False),  # (B, 1, 1, W)
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h_gate: (B, 1, H, 1) — which rows matter (horizontal crack lines)
        h_gate = self.h_attn(x)   # (B, 1, H, 1)
        # v_gate: (B, 1, 1, W) — which columns matter (vertical crack lines)
        v_gate = self.v_attn(x)   # (B, 1, 1, W)
        # Outer product attention: pixel (i,j) is attended if row i AND column j matter
        return x * (h_gate * v_gate)   # broadcasts to (B, C, H, W)
