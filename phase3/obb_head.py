"""
Phase 3 — Enhanced OBB Detection Head
=======================================
Implements Sections 2.3, 2.4, 2.5 from the design doc:
  2.3 — P2 detection level (stride 4) for 1-3 px thin crack recall
  2.4 — Asymmetric strip convolutions in head: parallel 1×k + k×1 DW branches
  2.5 — Texture-contrast side feature: Laplacian + learned Gabor-like filters

The OBB head produces per-anchor predictions:
  - Classification scores: (B, nc, H, W)
  - Box regression via DFL: (B, 4*reg_max, H, W)
  - Angle prediction: (B, 1, H, W)

We build one EnhancedOBBHead per detection level (P2, P3, P4, P5) and stack them.

KEY ENHANCEMENTS OVER STANDARD YOLOV8 OBB HEAD:
  1. StripConvStem: replaces one 3×3 with parallel 1×k + k×1 branches.
     This makes the head directionally sensitive — a 1×9 kernel sees a 9px horizontal
     strip; k×1 sees a 9px vertical strip.  Crack-like responses align with one
     of these and activate strongly; round ballast-gap responses do not.

  2. TextureContrast: a tiny side branch computing local texture statistics
     (Laplacian for high-freq energy + learned Gabor-like DW filters) whose
     output is concatenated into the CLASSIFICATION branch.
     - On concrete: Laplacian energy is low + Gabor activations are uniform
     - On cracks: Laplacian energy is HIGH (edge = crack walls)
     - On ballast: Laplacian energy is HIGH but RANDOM (not elongated)
     This gives the classifier an explicit "is this concrete vs ballast?" cue
     that standard box features lack.

  3. P2 head: a dedicated head for stride-4 features capturing 1-3 px thin cracks
     that P3 (stride 8) has already pooled away.

Usage:
    head = EnhancedOBBHead(in_ch=64, nc=3, reg_max=16, strip_k=9)
    cls_pred, box_pred, ang_pred = head(p2_feat)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


# ══════════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

def _conv_bn_act(in_ch, out_ch, k=3, s=1, p=None, act=True):
    if p is None:
        p = k // 2
    layers = [
        nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


# ══════════════════════════════════════════════════════════════════════════════
# STRIP CONV STEM (Section 2.4)
# ══════════════════════════════════════════════════════════════════════════════

class StripConvStem(nn.Module):
    """
    Replaces one standard 3×3 conv in the head stem with parallel asymmetric branches:
      Branch H: 1×k depthwise → 1×1 pointwise   (captures horizontal crack lines)
      Branch V: k×1 depthwise → 1×1 pointwise   (captures vertical crack lines)
      Branch sq: 3×3 depthwise → 1×1 pointwise  (baseline isotropic response)

    All branches have the same output channel count and are summed.
    This adds only (2k + 9) × in_ch multiply-accumulate ops vs 9 × in_ch for
    a single 3×3 — essentially free, while tripling directional sensitivity.

    Args:
        ch     : number of input AND output channels
        k      : strip kernel length (7 or 9 recommended)
    """

    def __init__(self, ch: int, k: int = 9):
        super().__init__()
        assert k % 2 == 1, "k must be odd"
        mid = ch

        # Horizontal strip: 1×k depthwise (captures H crack lines)
        self.h_dw   = nn.Conv2d(ch, mid, (1, k), padding=(0, k // 2), groups=ch, bias=False)
        self.h_bn   = nn.BatchNorm2d(mid)
        self.h_pw   = nn.Conv2d(mid, ch, 1, bias=False)

        # Vertical strip: k×1 depthwise (captures V crack lines)
        self.v_dw   = nn.Conv2d(ch, mid, (k, 1), padding=(k // 2, 0), groups=ch, bias=False)
        self.v_bn   = nn.BatchNorm2d(mid)
        self.v_pw   = nn.Conv2d(mid, ch, 1, bias=False)

        # Isotropic baseline: 3×3 depthwise (catches diagonal/mixed responses)
        self.sq_dw  = nn.Conv2d(ch, mid, 3, padding=1, groups=ch, bias=False)
        self.sq_bn  = nn.BatchNorm2d(mid)
        self.sq_pw  = nn.Conv2d(mid, ch, 1, bias=False)

        # Fused BN + act after sum
        self.fuse_bn  = nn.BatchNorm2d(ch)
        self.act      = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h  = self.h_pw(self.h_bn(self.h_dw(x)))
        v  = self.v_pw(self.v_bn(self.v_dw(x)))
        sq = self.sq_pw(self.sq_bn(self.sq_dw(x)))
        return self.act(self.fuse_bn(h + v + sq))


# ══════════════════════════════════════════════════════════════════════════════
# TEXTURE CONTRAST SIDE BRANCH (Section 2.5)
# ══════════════════════════════════════════════════════════════════════════════

class TextureContrastBranch(nn.Module):
    """
    Computes local texture features for the classification head.

    Two channels:
      Channel 0: Laplacian energy — measures local second-derivative magnitude.
                 High on crack edges, moderate on ballast, low on smooth concrete.
      Channel 1–7: Learned Gabor-like DW filters (bank of oriented wavelets).
                   Initialised to actual Gabor kernels at 4 orientations (0, 45, 90, 135°).
                   These respond to oriented texture — cracks light up one orientation
                   strongly; isotropic ballast lights up all orientations uniformly.

    The output (B, out_ch, H, W) is concatenated into the classification head's
    stem before the final 1×1 conv.

    Args:
        in_ch    : neck feature channels (for the Laplacian channel)
        out_ch   : texture feature channels to concatenate into cls head
        k_gabor  : kernel size for the learned Gabor-like filters (must be odd)
    """

    def __init__(self, in_ch: int, out_ch: int = 8, k_gabor: int = 7):
        super().__init__()
        n_gabor  = out_ch - 1    # one slot taken by Laplacian

        # Fixed Laplacian kernel (not learned — it IS the high-freq energy operator)
        lap_k = torch.tensor([[0., -1., 0.],
                               [-1., 4., -1.],
                               [0., -1., 0.]])
        # Apply same kernel to all input channels (depthwise, groups=in_ch)
        self.register_buffer(
            'lap_kernel',
            lap_k.view(1, 1, 3, 3).expand(in_ch, 1, 3, 3).clone()
        )
        self.in_ch = in_ch

        # Learned Gabor-like bank: n_gabor depthwise filters
        self.gabor_bank = nn.Conv2d(
            in_ch, n_gabor, k_gabor,
            padding=k_gabor // 2, groups=1, bias=False
        )
        # Initialise to actual Gabor wavelets (4 orientations × ceil(n_gabor/4) per orient)
        self._init_gabor(k_gabor, n_gabor)

        # 1×1 project to out_ch (absorbs the lap channel + gabor channels)
        self.project = nn.Sequential(
            nn.Conv2d(1 + n_gabor, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.lap_bn = nn.BatchNorm2d(1)

    def _init_gabor(self, k: int, n: int):
        """Initialise conv weights as Gabor wavelets at evenly spaced orientations."""
        with torch.no_grad():
            w = self.gabor_bank.weight.data   # (n, in_ch, k, k)
            orientations = [i * math.pi / max(n, 1) for i in range(n)]
            for i, theta in enumerate(orientations):
                g = self._gabor_kernel(k, theta)              # (k, k)
                # Broadcast the same 2-D Gabor kernel to all in_ch channels
                w[i] = g.unsqueeze(0).expand(self.in_ch, k, k).clone()
            # Global normalise so initial magnitude is ~1
            self.gabor_bank.weight.data = w / (w.abs().max() + 1e-8)

    @staticmethod
    def _gabor_kernel(k: int, theta: float, sigma: float = 1.5, lam: float = 3.0) -> torch.Tensor:
        """Generate a single Gabor kernel."""
        half = k // 2
        y, x = torch.meshgrid(
            torch.linspace(-half, half, k),
            torch.linspace(-half, half, k),
            indexing='ij',
        )
        x_rot = x * math.cos(theta) + y * math.sin(theta)
        y_rot = -x * math.sin(theta) + y * math.cos(theta)
        g = torch.exp(-0.5 * (x_rot ** 2 + y_rot ** 2) / sigma ** 2) * \
            torch.cos(2 * math.pi * x_rot / lam)
        return g - g.mean()   # zero-mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_ch, H, W) — neck feature map

        Returns (B, out_ch, H, W) texture features.
        The Laplacian is computed on the mean across channels (intensity proxy).
        """
        # Laplacian energy: (B, in_ch, H, W) → depthwise → (B, in_ch, H, W) → mean → (B, 1, H, W)
        lap = F.conv2d(x, self.lap_kernel, padding=1, groups=self.in_ch)
        lap = lap.pow(2).mean(dim=1, keepdim=True).sqrt()   # (B, 1, H, W) — local edge energy
        lap = self.lap_bn(lap)

        # Gabor-like bank
        gabor = self.gabor_bank(x)    # (B, n_gabor, H, W)

        # Concatenate and project
        tex = torch.cat([lap, gabor], dim=1)   # (B, 1+n_gabor, H, W)
        return self.project(tex)


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED OBB HEAD (one per detection level)
# ══════════════════════════════════════════════════════════════════════════════

class EnhancedOBBHead(nn.Module):
    """
    OBB detection head for one feature level with:
      - StripConvStem for directional sensitivity
      - TextureContrastBranch feeding extra context into the cls branch

    Outputs match the ultralytics OBB head format:
      cls_pred : (B, nc,         H, W)
      box_pred : (B, 4*reg_max,  H, W)  — DFL regression
      ang_pred : (B, 1,          H, W)  — angle in [-π/4, π/4]

    The head is intentionally lightweight (2 stems) so P2 (large spatial) is fast.
    """

    def __init__(
        self,
        in_ch:     int,
        nc:        int   = 3,     # number of object classes
        reg_max:   int   = 16,    # DFL regression depth (matches YOLOv8)
        mid_ch:    int   = None,  # intermediate channels (default = max(in_ch, 64))
        strip_k:   int   = 9,     # asymmetric strip kernel length
        tex_out:   int   = 8,     # texture branch output channels
    ):
        super().__init__()
        mid = mid_ch or max(in_ch, 64)

        # ── Classification branch ─────────────────────────────────────────────
        self.cls_stem = nn.Sequential(
            _conv_bn_act(in_ch, mid, 3),
            StripConvStem(mid, k=strip_k),
        )
        # Texture contrast branch feeds extra channels into cls head
        self.tex = TextureContrastBranch(in_ch, out_ch=tex_out)
        # Final cls conv: mid + tex_out channels → nc
        self.cls_pred = nn.Conv2d(mid + tex_out, nc, 1, bias=True)

        # ── Box regression branch (DFL) ───────────────────────────────────────
        self.box_stem = nn.Sequential(
            _conv_bn_act(in_ch, mid, 3),
            _conv_bn_act(mid, mid, 3),
        )
        self.box_pred = nn.Conv2d(mid, 4 * reg_max, 1, bias=True)

        # ── Angle prediction branch ───────────────────────────────────────────
        # Shared with box stem to save params; separate final conv for angle
        self.ang_pred = nn.Conv2d(mid, 1, 1, bias=True)

        # Init classification bias to prevent overconfident early predictions
        # (standard focal-loss initialisation: prior probability ~0.01)
        nn.init.constant_(self.cls_pred.bias, -math.log((1 - 0.01) / 0.01))
        nn.init.zeros_(self.ang_pred.bias)
        nn.init.zeros_(self.box_pred.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Texture features from input (before stem processing)
        tex = self.tex(x)                          # (B, tex_out, H, W)

        # Classification branch with strip conv + texture concat
        cls_feat = self.cls_stem(x)                # (B, mid, H, W)
        cls_with_tex = torch.cat([cls_feat, tex], dim=1)  # (B, mid+tex_out, H, W)
        cls_pred = self.cls_pred(cls_with_tex)     # (B, nc, H, W)

        # Box regression branch
        box_feat = self.box_stem(x)                # (B, mid, H, W)
        box_pred = self.box_pred(box_feat)         # (B, 4*reg_max, H, W)

        # Angle prediction (reuses box_feat)
        ang_pred = torch.sigmoid(self.ang_pred(box_feat)) * math.pi - math.pi / 2
        # Maps sigmoid output (0,1) to angle range (-π/2, π/2), then the model
        # learns to concentrate predictions in the valid OBB range [-π/4, π/4].

        return cls_pred, box_pred, ang_pred


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-LEVEL OBB HEAD (P2 + P3 + P4 + P5)
# ══════════════════════════════════════════════════════════════════════════════

class MultiLevelOBBHead(nn.Module):
    """
    Stacks four EnhancedOBBHeads for P2/P3/P4/P5.

    Note on P2: the P2 head is identical in structure to P3/P4/P5 but operates
    on the larger feature map (4× more pixels at stride 4 vs stride 8).  This
    is the design doc's "add a P2 detection level" change — nothing architecturally
    different, just one more head for the extra resolution.

    All heads share the same nc / reg_max / strip_k for consistency.

    Args:
        in_channels : [p2_ch, p3_ch, p4_ch, p5_ch] from AFPN/DyHead
        nc          : number of object classes
        reg_max     : DFL depth
        strip_k     : asymmetric strip kernel length
    """

    def __init__(
        self,
        in_channels: List[int],
        nc:          int  = 3,
        reg_max:     int  = 16,
        strip_k:     int  = 9,
        tex_out:     int  = 8,
    ):
        super().__init__()
        self.heads = nn.ModuleList([
            EnhancedOBBHead(ic, nc=nc, reg_max=reg_max, strip_k=strip_k, tex_out=tex_out)
            for ic in in_channels
        ])
        self.strides = [4, 8, 16, 32]   # P2, P3, P4, P5
        self.nc      = nc
        self.reg_max = reg_max

    def forward(
        self,
        feats: List[torch.Tensor],   # [p2, p3, p4, p5] from DyHead
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Returns a list of (cls_pred, box_pred, ang_pred) tuples, one per level.
        Each cls_pred: (B, nc, H_l, W_l)
        Each box_pred: (B, 4*reg_max, H_l, W_l)
        Each ang_pred: (B, 1, H_l, W_l)
        """
        return [head(f) for head, f in zip(self.heads, feats)]
