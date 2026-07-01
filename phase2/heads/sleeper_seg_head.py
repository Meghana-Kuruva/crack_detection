"""
Phase 2 — Head 2: Sleeper Segmentation Head
============================================
Section 7 of the design doc.

WHY add a sleeper segmentation head to the detection model:
  In Phase 1 the sleeper mask came from a separate lightweight network run on
  the downsampled full frame BEFORE tiling.  That is correct and cheap at
  inference, but it means the segmentation and detection features are trained
  INDEPENDENTLY — the detector never learns to use "is this tile on a sleeper?"
  as an internal cue.

  Adding a sleeper seg head to the shared neck means:
    1. The shared features are jointly optimised to support both detection
       AND sleeper localisation — they become implicitly context-aware.
    2. The L_gate term (Section 6.4) can back-propagate through BOTH heads
       simultaneously, directly training the detector to not fire off-sleeper.
    3. At inference we can optionally replace the Phase 1 classical segmenter
       with this learned head (attached to the tile, not the full frame) —
       or run both and ensemble.

Architecture:
  Input: P5 neck features (stride 32, most semantic — best for knowing
         "what region am I in?" context at tile level).
  Output: (B, 1, H/32, W/32) logits → upsample to input resolution for loss.

  Kept VERY lightweight because P5 is already semantically rich.
  A single ConvBnRelu + 3×bilinear-upsample is sufficient.

Loss: Dice + BCE (DiceBCELoss in losses/focal_tversky.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SleeperSegHead(nn.Module):
    """
    Lightweight sleeper-region segmentation decoder.

    Attached to the P5 neck feature map (stride 32) of the YOLOv8 backbone.
    Produces a per-pixel sleeper probability at the input resolution.

    Loss: DiceBCELoss (Dice + Binary Cross-Entropy).

    The predicted sleeper probability is also used:
      - In L_gate: region_gate_loss(crack_prob, sleeper_prob)
      - At inference: replaces or augments the Phase 1 classical segmenter
        when tiles are being processed (provides tile-level context cue)
    """

    def __init__(
        self,
        in_channels:  int = 512,    # P5 channels: 512 for yolov8s, 256 for yolov8n
        mid_channels: int = 64,
    ):
        super().__init__()

        # Compress P5 channels to something cheap
        self.compress = _ConvBnRelu(in_channels, mid_channels)

        # 5 bilinear×2 upsamples: stride 32 → stride 1
        # 2^5 = 32 so we need 5 upsamples total
        # We keep the channel count fixed for simplicity
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                _ConvBnRelu(mid_channels, mid_channels),
            )
            for _ in range(5)
        ])

        self.head = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, p5_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p5_features : (B, in_channels, H/32, W/32) from YOLOv8 neck layer 21

        Returns:
            logits : (B, 1, H, W) — apply sigmoid for probability
        """
        x = self.compress(p5_features)
        for up in self.up_blocks:
            x = up(x)
        return self.head(x)    # (B, 1, H, W)

    def predict_prob(
        self,
        p5_features: torch.Tensor,
        target_size: tuple = None,
    ) -> torch.Tensor:
        """
        Inference helper: returns (B, H, W) sleeper probability in [0, 1].

        Args:
            p5_features : P5 neck features
            target_size : (H, W) to resize to (useful when tile size varies)
        """
        logits = self.forward(p5_features)
        prob   = torch.sigmoid(logits.squeeze(1))

        if target_size is not None and prob.shape[-2:] != target_size:
            prob = F.interpolate(
                prob.unsqueeze(1),
                size=target_size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)

        return prob
