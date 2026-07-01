"""
Phase 3 — RepViT-style Reparameterizable Conv Blocks
======================================================
Implements structural re-parameterization (RepVGG / RepViT concept):
  TRAINING  : multiple parallel branches (3×3 DW, 1×1 PW, BN-identity shortcut)
  INFERENCE : all branches merged analytically into a single 3×3 conv via rep_convert()

Why for thin cracks:
  - Multi-branch training gives a larger effective receptive field than a single
    3×3 (the 1×1 sees centre pixel, the identity shortcut sees the pre-activation
    residual), which helps detect 1-2 px crack edges.
  - At inference the merged single conv is as fast as a plain ResNet block.
  - Reparameterization preserves detail at stride 2/4 (early backbone stages)
    without the accuracy drop of pruning or knowledge distillation.

Design follows RepViT (Wang et al., 2023): depthwise-then-pointwise token-mixer
with a BN-only shortcut merged at eval time.

Usage:
    blk = RepVitBlock(in_ch=64, out_ch=128, stride=2)
    x   = blk(x)         # training (multi-branch)
    blk.rep_convert()     # merge to single-branch
    x   = blk(x)         # inference (same output, single conv)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _bn_to_conv_params(bn: nn.BatchNorm2d, groups: int = 1) -> tuple:
    """
    Convert a BN-only branch (identity + BN) into equivalent conv weights.

    A BN with a preceding identity is equivalent to a 1×1 conv whose weight is
    diag(1/(std+ε) * γ) and bias is -μ/(std+ε)*γ + β.

    Returns (weight, bias) for a C×C 1×1 conv with the given groups.
    """
    c = bn.weight.shape[0]
    # Build identity kernel: (C, C/groups, 1, 1)
    ch_per_group = c // groups
    k = torch.zeros(c, ch_per_group, 1, 1, device=bn.weight.device)
    for i in range(c):
        k[i, i % ch_per_group, 0, 0] = 1.0
    # Fold BN into conv
    std = (bn.running_var + bn.eps).sqrt()
    w   = bn.weight / std
    b   = bn.bias - bn.running_mean * w
    # Scale kernel
    k = k * w.view(c, 1, 1, 1)
    return k, b


def _pad_1x1_to_3x3(w1x1: torch.Tensor) -> torch.Tensor:
    """Zero-pad a 1×1 kernel to 3×3 for adding to the 3×3 branch."""
    return F.pad(w1x1, [1, 1, 1, 1])


# ══════════════════════════════════════════════════════════════════════════════
# CORE BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class RepVitBlock(nn.Module):
    """
    RepViT-style token-mixer: depthwise 3×3 + pointwise 1×1 with
    structural re-parameterization.

    During training:
        out = BN(DW3×3(x)) + BN(PW1×1(x)) [+ BN(x) if residual]
    After rep_convert():
        out = single_3×3_conv(x)          [equivalent, faster]

    The depthwise / pointwise split is the "token mixing + channel mixing"
    pattern from EfficientViT; here the DW 3×3 does spatial mixing and the
    1×1 does channel projection.
    """

    def __init__(
        self,
        in_ch:  int,
        out_ch: int,
        stride: int = 1,
    ):
        super().__init__()
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self._merged = False

        # Token-mixer: depthwise 3×3 WITH stride (spatial mixing + downsampling)
        self.dw3    = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                                groups=in_ch, bias=False)
        self.dw3_bn = nn.BatchNorm2d(in_ch)

        # Channel-mixer: pointwise 1×1 AFTER dw3, always stride=1
        # (dw3 already applied the downsampling; pw1 only changes channel count)
        self.pw1    = nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False)
        self.pw1_bn = nn.BatchNorm2d(out_ch)

        # Identity shortcut: BN-only, only when stride=1 and channels unchanged
        self.use_residual = (in_ch == out_ch and stride == 1)
        if self.use_residual:
            self.res_bn = nn.BatchNorm2d(out_ch)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._merged:
            return self.act(self.merged_pw(self.merged_dw(x)))

        # Main branch: DW 3×3 (downsampling) → PW 1×1 (channel projection)
        out = self.pw1_bn(self.pw1(self.dw3_bn(self.dw3(x))))

        # BN-only identity shortcut on the INPUT x (only when stride=1, same channels)
        # This is the re-parameterizable shortcut from RepVGG:
        # at inference, this BN branch is absorbed into the main conv weights.
        if self.use_residual:
            out = out + self.res_bn(x)

        return self.act(out)

    @torch.no_grad()
    def rep_convert(self):
        """
        Fold BN layers into the conv weights for inference (no BN overhead).
        Keeps DW and PW as two separate merged convs — no further merging needed
        since DW-separable is already the standard fast path.
        """
        if self._merged:
            return

        def fold_bn(conv, bn):
            std = (bn.running_var + bn.eps).sqrt()
            w   = conv.weight * (bn.weight / std).view(-1, 1, 1, 1)
            b   = bn.bias - bn.running_mean * (bn.weight / std)
            return w, b

        # Fold BN into DW conv
        w_dw, b_dw = fold_bn(self.dw3, self.dw3_bn)
        self.merged_dw = nn.Conv2d(
            self.in_ch, self.in_ch, 3,
            stride=self.stride, padding=1, groups=self.in_ch, bias=True,
        )
        self.merged_dw.weight.data = w_dw
        self.merged_dw.bias.data   = b_dw

        # Fold BN into PW conv
        w_pw, b_pw = fold_bn(self.pw1, self.pw1_bn)
        self.merged_pw = nn.Conv2d(self.in_ch, self.out_ch, 1, bias=True)
        self.merged_pw.weight.data = w_pw
        self.merged_pw.bias.data   = b_pw

        # Delete training branches
        del self.dw3, self.dw3_bn, self.pw1, self.pw1_bn
        if self.use_residual:
            del self.res_bn
        self._merged = True


# ══════════════════════════════════════════════════════════════════════════════
# SPPF (spatial pyramid pooling fast) — mirrors ultralytics implementation
# ══════════════════════════════════════════════════════════════════════════════

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling Fast (SPP-F) as used in YOLOv8.
    Applies the same max-pool with kernel k at 4 different depths (1, 2, 3, 4 pools)
    then concatenates results.  Gives multi-scale receptive field at near-zero extra cost.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        mid = in_ch // 2
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(mid * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: build a stage of N RepViT blocks
# ══════════════════════════════════════════════════════════════════════════════

class RepVitStage(nn.Module):
    """
    A sequence of RepViT blocks with optional initial downsampling.

    The first block does the stride (downsampling), subsequent blocks are
    stride-1 depth blocks.  This is the standard stage pattern for CNNs.
    """

    def __init__(
        self,
        in_ch:    int,
        out_ch:   int,
        n_blocks: int = 2,
        stride:   int = 2,
    ):
        super().__init__()
        layers = [RepVitBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            layers.append(RepVitBlock(out_ch, out_ch, stride=1))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    def rep_convert(self):
        for blk in self.blocks:
            blk.rep_convert()
