"""
Phase 2 — Head 1: Crack Segmentation Head
==========================================
Section 7 of the design doc.

WHY crack segmentation helps with FPs:
  A ballast gap, when segmented pixel-wise, falls into the NON-SLEEPER region.
  Crack segmentation forces the model to localise structures at sub-pixel
  precision — a ballast gap that the detector wants to fire on will produce a
  segmentation output that extends beyond the sleeper boundary, which is then
  penalised by L_gate.  This provides an implicit negative signal that pure
  detection never sees.

Architecture:
  Input: P3 neck features (stride 8, highest-resolution in the neck —
         enough to resolve 1–3 px thin cracks).
  Output: (B, 1, H/8, W/8) logits → upsample to input resolution for loss.

  Lightweight U-Net decoder:
    P3 → [3×3 conv, BN, ReLU] × 2 → bilinear 2× upsample → [conv] → 2×
    → [conv] → 2× → [1×1 output]

  Channel counts kept small (64→32→16) because:
    (a) P3 already has rich features from the YOLO neck
    (b) We want fast training — the neck features do the heavy lifting
    (c) Fewer params = less overfitting on the small Roboflow dataset

Projection head for contrastive loss (Section 6.3):
  Attached to the bottleneck of this head, producing a D=128 embedding
  for the SupervisedContrastiveLoss in multitask_loss.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBnRelu(nn.Module):
    """Standard Conv-BN-ReLU block."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CrackSegHead(nn.Module):
    """
    Lightweight crack segmentation decoder.

    Takes the P3 neck feature map (stride-8, hooked from layer 15 of YOLOv8)
    and produces a per-pixel crack logit map at stride 8.

    Loss: Focal-Tversky (FocalTverskyLoss in losses/focal_tversky.py).

    The head also exposes a contrastive_proj() method that returns a
    D=128 L2-normalised embedding per image for the contrastive loss.
    """

    def __init__(
        self,
        in_channels:   int = 128,    # P3 channels: 128 for yolov8s, 64 for yolov8n
        mid_channels:  int = 64,
        proj_dim:      int = 128,    # dimension of contrastive projection
    ):
        super().__init__()

        # ── Decoder ──────────────────────────────────────────────────────────
        # Two Conv-BN-ReLU at P3 resolution, then upsample ×8 back to input.
        # We choose 3 bilinear×2 upsamples instead of ConvTranspose2d
        # because bilinear + conv is smoother for thin structures.
        self.conv1 = _ConvBnRelu(in_channels, mid_channels)
        self.conv2 = _ConvBnRelu(mid_channels, mid_channels // 2)

        # Upsampling path (×8 total: three ×2 steps)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            _ConvBnRelu(mid_channels // 2, mid_channels // 2),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            _ConvBnRelu(mid_channels // 2, mid_channels // 4),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            _ConvBnRelu(mid_channels // 4, mid_channels // 4),
        )

        # Final 1×1 → single output channel (crack vs background)
        self.head = nn.Conv2d(mid_channels // 4, 1, kernel_size=1)

        # ── Contrastive projection head ───────────────────────────────────────
        # Global average pool of mid-level features → small MLP → L2-norm.
        # Used by the supervised contrastive loss to carve the
        # crack/ballast-gap decision boundary in embedding space.
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mid_channels, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

        self._bottleneck = None     # stored during forward for projection access

    def forward(self, p3_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p3_features : (B, in_channels, H/8, W/8) from YOLOv8 neck layer 15

        Returns:
            logits : (B, 1, H, W) at the original input resolution
        """
        x = self.conv1(p3_features)
        x = self.conv2(x)
        self._bottleneck = x       # save for contrastive projection

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.head(x)        # (B, 1, H, W)

    def contrastive_projection(self) -> torch.Tensor:
        """
        Return L2-normalised embeddings from the last forward pass.
        Call this AFTER forward() within the same step.
        Returns (B, proj_dim).
        """
        assert self._bottleneck is not None, "Call forward() before contrastive_projection()"
        emb = self.proj(self._bottleneck)
        return F.normalize(emb, dim=-1)
