"""
Phase 3 — DyHead: Dynamic Head for Unified Object Detection
=============================================================
Implements DyHead (Dai et al., CVPR 2021) adapted for thin-crack OBB detection.

DyHead unifies three types of attention in one block:
  1. SCALE-AWARE attention  — learns to weight each feature level (P2/P3/P4/P5)
     differently: thin cracks should weight P2/P3 more, while scene context
     should weight P4/P5 more.
  2. SPATIAL-AWARE attention — deformable attention that concentrates on crack
     pixels instead of spreading uniformly.  The sampling offsets naturally
     follow thin elongated structures.
  3. TASK-AWARE attention — separate attention for the classification branch
     (is this a crack?) vs regression branch (where/how big).

Design doc (Section 4):
  "DyHead: scale/spatial/task attention.  DyHead's spatial-aware attention
   concentrates on thin structures, scale-aware attention handles cracks of
   varying width, and task-aware attention lets OBB and classification specialise."

  "backbone → AFPN(P2–P5) → FiLM(context) → CoordAttn/StripPool → DyHead → heads"

This implementation keeps the DyHead after AFPN (already has CA+SP) so we skip
redundant attention and focus on scale-aware and task-aware components.

Usage:
    dyhead = DyHead(ch=256, num_blocks=2)
    feats  = [p2, p3, p4, p5]   # all at 256 channels after AFPN
    out    = dyhead(feats)       # same list structure, scale/task-refined
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ══════════════════════════════════════════════════════════════════════════════
# SCALE-AWARE ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

class ScaleAwareAttn(nn.Module):
    """
    Learns per-level weights so the model can emphasise fine (P2/P3) or
    coarse (P4/P5) scales according to crack size.

    Implementation: SE-style (Squeeze-Excitation) across the scale dimension.
    For L levels, this is a small MLP from (B, C*L) → (B, L) → sigmoid weights.

    After scaling, all levels are summed into one fused feature, then split back.
    """

    def __init__(self, ch: int, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels
        # Squeeze: global avg pool each level → (B, C, 1, 1) → flatten → (B, C*L)
        # Excite:  MLP → (B, L) per-level weights
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite  = nn.Sequential(
            nn.Linear(ch * num_levels, ch // 4),
            nn.ReLU(inplace=True),
            nn.Linear(ch // 4, num_levels),
            nn.Sigmoid(),
        )
        self.ch = ch

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        feats: L tensors of shape (B, C, H_l, W_l) — possibly different spatial sizes
        """
        # All tensors must have the same C; spatial can differ
        B = feats[0].shape[0]

        # Squeeze each level: (B, C, H, W) → (B, C) via global avg pool
        squeezed = []
        for f in feats:
            s = self.squeeze(f).view(B, self.ch)
            squeezed.append(s)
        squeezed = torch.cat(squeezed, dim=1)   # (B, C*L)

        # Excite: MLP → L per-level weights
        weights = self.excite(squeezed)           # (B, L)

        # Apply: scale each level
        out = []
        for i, f in enumerate(feats):
            w = weights[:, i].view(B, 1, 1, 1)   # (B, 1, 1, 1)
            out.append(f * w)

        return out


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL-AWARE ATTENTION (lightweight deformable alternative)
# ══════════════════════════════════════════════════════════════════════════════

class SpatialAwareAttn(nn.Module):
    """
    Spatial attention using locally-predicted offsets to focus on crack pixels.

    Full DCNv2 (deformable conv v2) requires a CUDA extension; we implement a
    pure-PyTorch approximation using predicted attention maps:
      - Predict an attention weight map (B, 1, H, W) via 3×3 conv
      - Multiply feature map to suppress off-crack regions

    This captures the INTENT of deformable conv (concentrate on thin structures)
    without requiring non-standard CUDA ops — critical for Colab compatibility.

    For publication-quality results, swap this for torchvision.ops.deform_conv2d
    if available on your hardware.
    """

    def __init__(self, ch: int, kernel_size: int = 3):
        super().__init__()
        # Spatial attention gate: predicts (B, 1, H, W) attention weights
        self.attn_gate = nn.Sequential(
            nn.Conv2d(ch, ch // 4, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.attn_gate(x)    # (B, 1, H, W)
        return x * gate


# ══════════════════════════════════════════════════════════════════════════════
# TASK-AWARE ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

class TaskAwareAttn(nn.Module):
    """
    Generates task-specific feature maps for classification and regression.

    The original DyHead uses hyperfunction-based task attention; here we use
    a simpler but effective version: SE-style attention with two separate
    excitation pathways (one for cls, one for reg), gated by the input.

    This splits the shared feature into two parallel task branches:
      - cls_feat: emphasises discriminative features (crack vs ballast)
      - reg_feat: emphasises geometric features (orientation, length)

    The combined output is the mean — downstream heads can access either branch
    by hooking into cls_feat / reg_feat if needed.
    """

    def __init__(self, ch: int):
        super().__init__()
        mid = max(ch // 4, 8)

        # Shared squeeze
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Two separate excitation branches
        self.cls_exc = nn.Sequential(
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )
        self.reg_exc = nn.Sequential(
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = self.squeeze(x).view(B, C)     # (B, C)

        cls_gate = self.cls_exc(s).view(B, C, 1, 1)
        reg_gate = self.reg_exc(s).view(B, C, 1, 1)

        cls_feat = x * cls_gate
        reg_feat = x * reg_gate

        # Return combined (the downstream OBBHead will handle cls/reg separately)
        return (cls_feat + reg_feat) / 2


# ══════════════════════════════════════════════════════════════════════════════
# ONE DYHEAD BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class DyHeadBlock(nn.Module):
    """
    One DyHead block: scale-aware → spatial-aware → task-aware, applied to all levels.
    """

    def __init__(self, ch: int, num_levels: int = 4):
        super().__init__()
        self.scale_attn   = ScaleAwareAttn(ch, num_levels=num_levels)
        self.spatial_attn = SpatialAwareAttn(ch)
        self.task_attn    = TaskAwareAttn(ch)
        # Post-block norm
        self.norm = nn.ModuleList([nn.GroupNorm(1, ch) for _ in range(num_levels)])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # 1. Scale-aware: re-weight across levels
        feats = self.scale_attn(feats)

        # 2. Spatial-aware: focus on thin structures within each level
        feats = [self.spatial_attn(f) for f in feats]

        # 3. Task-aware: produce task-specialised representations
        feats = [self.task_attn(f) for f in feats]

        # Post-block norm
        feats = [self.norm[i](feats[i]) for i in range(len(feats))]

        return feats


# ══════════════════════════════════════════════════════════════════════════════
# STACKED DYHEAD
# ══════════════════════════════════════════════════════════════════════════════

class DyHead(nn.Module):
    """
    Stack of DyHead blocks applied to [P2, P3, P4, P5] feature maps.

    All levels must have the same channel count ch (ensured by AFPN's out_ch parameter).

    Args:
        ch         : channel count (same for all levels)
        num_blocks : how many DyHead blocks to stack (default 2)
        num_levels : number of feature levels (default 4 for P2–P5)
    """

    def __init__(self, ch: int, num_blocks: int = 2, num_levels: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            DyHeadBlock(ch, num_levels=num_levels)
            for _ in range(num_blocks)
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        for blk in self.blocks:
            feats = blk(feats)
        return feats
