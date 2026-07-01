"""
Phase 3 — Asymptotic Feature Pyramid Network (AFPN)
=====================================================
Implements AFPN spanning P2–P5, with Coordinate Attention and Strip Pooling
inserted after fusion to exploit crack elongation geometry.

Why AFPN over standard FPN for cracks:
  Standard FPN fuses P5→P4→P3→P2 top-down (big semantic gap between adjacent levels).
  BiFPN does top-down + bottom-up but ALL pairs at once — large semantic gap for P2↔P5.

  AFPN uses "adjacent-scale asymptotic fusion": always merge only scales that differ
  by one stride level, progressively building from the finest scale upward:
    1.  T4 = merge(P4, up(P5))          — P4+P5 share similar semantics
    2.  T3 = merge(P3, up(T4))          — T4 has P5 context, now blended with P3
    3.  T2 = merge(P2, up(T3))          — T3 has P4+P5 context, now blended with P2

  Then a bottom-up pass re-injects fine detail:
    1.  O2 = T2 (output as-is)
    2.  O3 = merge(T3, down(O2))
    3.  O4 = merge(T4, down(O3))
    4.  O5 = merge(P5, down(O4))

  This avoids the semantic gap problem: fine (P2) and coarse (P5) never directly
  interact; the semantic gap is bridged incrementally.

  Reference: Yang et al., "AFPN: Asymptotic Feature Pyramid Network for Object
             Detection", CVPR 2023.

Usage:
    afpn = AFPN(in_channels=[64, 128, 256, 512])   # P2..P5 channel widths
    o2, o3, o4, o5 = afpn(p2, p3, p4, p5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .coord_attention import CoordinateAttention
from .strip_pooling   import StripPooling


# ══════════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

def _conv_bn_relu(in_ch: int, out_ch: int, k: int = 1, s: int = 1) -> nn.Sequential:
    p = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BiFuseBlock(nn.Module):
    """
    Fuse two feature maps of the same spatial size and channel count.
    Uses a learned scalar weight for each input (fast normalised attention
    from BiFPN) rather than simple addition.

    α₁ · F₁ + α₂ · F₂ where αᵢ = softmax(wᵢ) (fast normalised)
    """

    def __init__(self, ch: int):
        super().__init__()
        # Two 1-D learnable weights (one per input)
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        # Post-fusion conv to mix channels
        self.conv = _conv_bn_relu(ch, ch, k=3)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        # Fast normalised weights (non-negative, sum to 1 approximately)
        w = F.relu(self.w)
        total = w.sum() + 1e-8
        out = (w[0] * f1 + w[1] * f2) / total
        return self.conv(out)


class UpsampleFuse(nn.Module):
    """Upsample the lower-resolution tensor then fuse with the higher-resolution one."""

    def __init__(self, lo_ch: int, hi_ch: int, out_ch: int):
        super().__init__()
        # Align channels before fusion
        self.adapt_lo = _conv_bn_relu(lo_ch, hi_ch)
        self.fuse     = BiFuseBlock(hi_ch)
        # Optionally reduce to out_ch if different
        self.out_proj = _conv_bn_relu(hi_ch, out_ch) if hi_ch != out_ch else nn.Identity()

    def forward(self, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        # lo: lower-res (coarser scale), hi: higher-res (finer scale)
        lo_up = F.interpolate(self.adapt_lo(lo), size=hi.shape[-2:],
                              mode='bilinear', align_corners=False)
        return self.out_proj(self.fuse(lo_up, hi))


class DownsampleFuse(nn.Module):
    """Downsample the higher-resolution tensor then fuse with the lower-resolution one."""

    def __init__(self, hi_ch: int, lo_ch: int, out_ch: int):
        super().__init__()
        self.adapt_hi = _conv_bn_relu(hi_ch, lo_ch)
        self.down     = nn.MaxPool2d(2, stride=2)
        self.fuse     = BiFuseBlock(lo_ch)
        self.out_proj = _conv_bn_relu(lo_ch, out_ch) if lo_ch != out_ch else nn.Identity()

    def forward(self, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        hi_down = self.down(self.adapt_hi(hi))
        # Align spatial size precisely (pooling may be off by 1 pixel)
        if hi_down.shape[-2:] != lo.shape[-2:]:
            hi_down = F.interpolate(hi_down, size=lo.shape[-2:],
                                    mode='bilinear', align_corners=False)
        return self.out_proj(self.fuse(hi_down, lo))


# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION ENRICHMENT (CA + Strip pooling applied after fusion)
# ══════════════════════════════════════════════════════════════════════════════

class LevelEnrich(nn.Module):
    """
    Per-feature-map enrichment after AFPN fusion:
      Coordinate Attention (directional sensitivity) +
      Strip Pooling (elongated structure emphasis).

    Applied at each output level so every feature map gains directional
    crack-aware attention before going into the detection head.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.ca = CoordinateAttention(ch)
        self.sp = StripPooling(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CA and SP are complementary: CA gates by direction, SP aggregates context
        return self.sp(self.ca(x))


# ══════════════════════════════════════════════════════════════════════════════
# FULL AFPN
# ══════════════════════════════════════════════════════════════════════════════

class AFPN(nn.Module):
    """
    Asymptotic FPN spanning P2–P5 with CA + Strip Pooling at each output level.

    Args:
        in_channels : list of [p2_ch, p3_ch, p4_ch, p5_ch] from the backbone
        out_ch      : uniform output channel count for all levels
                      (None = keep per-level channels from in_channels)

    Returns:
        (o2, o3, o4, o5): enriched feature maps for [P2, P3, P4, P5]
    """

    def __init__(
        self,
        in_channels: List[int],   # [p2_ch, p3_ch, p4_ch, p5_ch]
        out_ch:      int = None,
    ):
        super().__init__()
        c2, c3, c4, c5 = in_channels
        if out_ch is None:
            o2, o3, o4, o5 = c2, c3, c4, c5
        else:
            o2 = o3 = o4 = o5 = out_ch

        # ── Top-down pass (asymptotic: only adjacent-scale merges) ───────────
        # T4 = merge(P4, up(P5))
        self.td_54 = UpsampleFuse(lo_ch=c5, hi_ch=c4, out_ch=c4)
        # T3 = merge(P3, up(T4))
        self.td_43 = UpsampleFuse(lo_ch=c4, hi_ch=c3, out_ch=c3)
        # T2 = merge(P2, up(T3))
        self.td_32 = UpsampleFuse(lo_ch=c3, hi_ch=c2, out_ch=c2)

        # ── Bottom-up pass (inject fine-scale detail back up) ────────────────
        # O3 = merge(T3, down(T2))
        self.bu_23 = DownsampleFuse(hi_ch=c2, lo_ch=c3, out_ch=o3)
        # O4 = merge(T4, down(O3))
        self.bu_34 = DownsampleFuse(hi_ch=o3, lo_ch=c4, out_ch=o4)
        # O5 = merge(P5, down(O4))
        self.bu_45 = DownsampleFuse(hi_ch=o4, lo_ch=c5, out_ch=o5)

        # Project P2 output to o2 channels
        self.p2_proj = _conv_bn_relu(c2, o2) if c2 != o2 else nn.Identity()

        # ── Attention enrichment at each output level ─────────────────────────
        self.enrich2 = LevelEnrich(o2)
        self.enrich3 = LevelEnrich(o3)
        self.enrich4 = LevelEnrich(o4)
        self.enrich5 = LevelEnrich(o5)

        # Store output channel sizes for downstream modules
        self.out_channels = [o2, o3, o4, o5]

    def forward(
        self,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ── Top-down pass ────────────────────────────────────────────────────
        t4 = self.td_54(p5, p4)    # P4 enriched with P5 context
        t3 = self.td_43(t4, p3)    # P3 enriched with P4+P5 context
        t2 = self.td_32(t3, p2)    # P2 enriched with P3+P4+P5 context

        # ── Bottom-up pass ───────────────────────────────────────────────────
        o2 = self.p2_proj(t2)
        o3 = self.bu_23(t2, t3)    # T3 + fine detail from T2
        o4 = self.bu_34(o3, t4)    # T4 + detail from O3
        o5 = self.bu_45(o4, p5)    # P5 + detail from O4

        # ── Directional attention enrichment ─────────────────────────────────
        o2 = self.enrich2(o2)
        o3 = self.enrich3(o3)
        o4 = self.enrich4(o4)
        o5 = self.enrich5(o5)

        return o2, o3, o4, o5
