"""
Phase 3 — Hybrid Backbone: RepViT stem + EfficientViT attention stages
=======================================================================
Implements the design doc's Section 3 recommendation:
  "a hybrid backbone — convolutional early stages (RepViT at stride 2/4)
   followed by lightweight linear-attention later stages (EfficientViT)."

Architecture:
    Input  (3,  H,  W)
      ↓ stem (stride 2)
    S1     (32, H/2, W/2)       — conv-only, 3×3 details
      ↓ stage1 (stride 2)
    P2     (64, H/4, W/4)       ← P2 feature output (stride 4) for thin cracks
      ↓ stage2 (stride 2)
    P3     (128, H/8, H/8)
      ↓ EfficientViT blocks
    P3'    (128, H/8, W/8)      ← P3 feature output with local+global context
      ↓ stage3 (stride 2)
    P4     (256, H/16, W/16)
      ↓ EfficientViT blocks
    P4'    (256, H/16, W/16)    ← P4 feature output
      ↓ stage4 (stride 2)
    P5     (512, H/32, W/32)
      ↓ EfficientViT blocks + SPPF
    P5'    (512, H/32, W/32)    ← P5 feature output (global context)

Returns (p2, p3, p4, p5) — same interface as YOLOv8 backbone for neck compatibility.

Design rationale:
  - Strides 2/4 (stem → S1 → P2) are pure RepViT conv to preserve 1-2 px crack edges
    before spatial resolution is halved a third time.
  - Strides 8+ use EfficientViT attention to aggregate crack trajectory context
    over the larger effective receptive field, helping distinguish ballast gaps
    (random, high spatial frequency) from cracks (directional, elongated).
  - SPPF at P5 provides multi-scale pooling for scene-level context (sleeper vs
    background), matching the YOLOv8 P5 design exactly.

Default channel widths are chosen to match yolov8s (P3=128, P4=256, P5=512)
so Phase 2 auxiliary heads can be reused without reinitialisation.
"""

import torch
import torch.nn as nn

from .rep_vit       import RepVitBlock, RepVitStage, SPPF
from .efficient_vit import EfficientViTStage


class HybridBackbone(nn.Module):
    """
    Hybrid backbone returning (P2, P3, P4, P5) feature pyramids.

    Channel widths:
        p2_ch  = 64     (stride 4  — detail, for P2 detection level)
        p3_ch  = 128    (stride 8  — matches yolov8s neck input)
        p4_ch  = 256    (stride 16)
        p5_ch  = 512    (stride 32)

    Args:
        in_ch       : input image channels (3 for RGB)
        base_ch     : stem output channels (default 32)
        width_mult  : global channel multiplier; set <1 for a lighter model
        evit_depth  : EfficientViT blocks per stage (default 2 per stage)
        evit_heads  : number of attention heads
        evit_head_dim : dimension per head
    """

    def __init__(
        self,
        in_ch:         int   = 3,
        base_ch:       int   = 32,    # stem channels
        width_mult:    float = 1.0,   # 1.0 = yolov8s width, 0.5 = nano
        evit_depth:    int   = 2,     # EfficientViT blocks per stage
        evit_heads:    int   = 4,
        evit_head_dim: int   = 32,
    ):
        super().__init__()

        def ch(c): return max(int(c * width_mult), 8)   # at least 8 channels

        # ── Stem: stride 2, no downsampling at stride 4 yet ─────────────────
        # A 3-conv stem (instead of one 4×4 stride-4 patch embed) is recommended
        # for crack detection: it processes each pixel 3 times before pooling,
        # so a 1-2 px crack contributes to more feature neurons.
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch(base_ch), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch(base_ch)),
            nn.GELU(),
            # Extra depth conv to enrich the stride-2 representation
            nn.Conv2d(ch(base_ch), ch(base_ch), 3, padding=1, groups=ch(base_ch), bias=False),
            nn.BatchNorm2d(ch(base_ch)),
            nn.GELU(),
            nn.Conv2d(ch(base_ch), ch(base_ch), 1, bias=False),
            nn.BatchNorm2d(ch(base_ch)),
            nn.GELU(),
        )
        # Output: (B, base_ch, H/2, W/2)

        # ── Stage 1 → P2 (stride 4) ──────────────────────────────────────────
        # Pure conv: we are still at high resolution and want to preserve edges.
        self.stage1 = RepVitStage(ch(base_ch), ch(64), n_blocks=2, stride=2)
        self.p2_ch  = ch(64)
        # Output: (B, 64, H/4, W/4)  ← P2

        # ── Stage 2 → P3 (stride 8) ──────────────────────────────────────────
        self.stage2 = RepVitStage(ch(64), ch(128), n_blocks=2, stride=2)
        # EfficientViT adds long-range context at this stride (N = HW/64, manageable)
        self.evit2  = EfficientViTStage(ch(128), depth=evit_depth,
                                        num_heads=evit_heads, head_dim=evit_head_dim)
        self.p3_ch  = ch(128)
        # Output: (B, 128, H/8, W/8)  ← P3

        # ── Stage 3 → P4 (stride 16) ─────────────────────────────────────────
        self.stage3 = RepVitStage(ch(128), ch(256), n_blocks=3, stride=2)
        self.evit3  = EfficientViTStage(ch(256), depth=evit_depth,
                                        num_heads=evit_heads, head_dim=evit_head_dim)
        self.p4_ch  = ch(256)
        # Output: (B, 256, H/16, W/16)  ← P4

        # ── Stage 4 → P5 (stride 32) ─────────────────────────────────────────
        self.stage4 = RepVitStage(ch(256), ch(512), n_blocks=3, stride=2)
        self.evit4  = EfficientViTStage(ch(512), depth=evit_depth,
                                        num_heads=evit_heads, head_dim=evit_head_dim)
        # SPPF mirrors the YOLOv8 P5 design: multi-scale max-pool before neck
        self.sppf   = SPPF(ch(512), ch(512), k=5)
        self.p5_ch  = ch(512)
        # Output: (B, 512, H/32, W/32)  ← P5

        # Weight initialisation
        self._init_weights()

    def forward(self, x: torch.Tensor):
        """
        Returns:
            p2 : (B, 64,  H/4,  W/4)   — thin-crack detail
            p3 : (B, 128, H/8,  W/8)   — mid-scale features
            p4 : (B, 256, H/16, W/16)  — semantic features
            p5 : (B, 512, H/32, W/32)  — global context
        """
        s1 = self.stem(x)             # (B, 32, H/2, W/2)
        p2 = self.stage1(s1)          # (B, 64, H/4, W/4)
        p3 = self.evit2(self.stage2(p2))   # (B, 128, H/8, W/8)
        p4 = self.evit3(self.stage3(p3))   # (B, 256, H/16, W/16)
        p5 = self.sppf(self.evit4(self.stage4(p4)))  # (B, 512, H/32, W/32)
        return p2, p3, p4, p5

    def rep_convert(self):
        """Merge all RepVit branches for deployment."""
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            stage.rep_convert()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
