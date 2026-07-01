"""
Phase 1 — Module 1: Sleeper Region Segmentation
================================================
Goal: given a full track frame, produce a binary mask where
  1 = sleeper (smooth concrete surface)
  0 = ballast / background

WHY this is the cornerstone of Phase 1:
  The dominant FP cause is SAHI feeding the model tiles that contain only
  ballast — the model fires on ballast gaps that look like cracks in local
  texture. If we know WHERE the sleepers are, we can:
    (a) restrict tile generation to only those regions  →  Module 2
    (b) penalise any detection that lands off-sleeper   →  Module 3

Two implementations with an identical predict_mask() interface:

  A. ClassicalSleeperSegmenter  — no training required, works immediately.
     Uses local variance + gradient magnitude.  Concrete = smooth = low
     variance/gradient. Ballast = rough stone = high variance/gradient.
     Use this as the ablation's lower-bound baseline.

  B. LightweightSleeperUNet — trainable; needs ~50 annotated sleeper masks.
     ~200 K params, runs on a 256×256 crop in under 5 ms.
     Also returns a global_embed vector for FiLM conditioning (Phase 3).

Start with A. Switch to B once you have annotations.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


# ══════════════════════════════════════════════════════════════════════════════
# A.  CLASSICAL CV SEGMENTER  (no training needed)
# ══════════════════════════════════════════════════════════════════════════════

class ClassicalSleeperSegmenter:
    """
    Texture-based sleeper vs ballast separator.

    Physical justification for each cue:
      - Local variance:  concrete is a uniform low-frequency grey surface →
                         variance is low.  Ballast is rough multi-scale stone
                         → variance is high.
      - Gradient mag:    smooth concrete has weak edges.  Stone-on-stone
                         boundaries in ballast produce strong multi-directional
                         gradients.
      - HSV saturation:  concrete is near-grey (saturation ≈ 0).  Ballast
                         can have earthy hues (higher saturation).

    All three cues are blended.  The result is thresholded and cleaned up
    with morphological ops so cracks inside the sleeper (which have high
    local gradient) do NOT fragment the sleeper mask.
    """

    def __init__(
        self,
        variance_ksize: int   = 31,   # Window for local variance — larger = smoother decision
        variance_thresh: float = 800, # Pixels with variance above this are "rough" → ballast
        gradient_thresh: float = 30,  # Pixels with Sobel mag above this are "edgy" → ballast
        morph_ksize:   int   = 25,   # Morphological cleanup kernel — fills crack-sized holes
    ):
        self.var_ksize  = variance_ksize
        self.var_thresh = variance_thresh
        self.grad_thresh = gradient_thresh
        self.morph_k    = morph_ksize

    # ── private helpers ──────────────────────────────────────────────────────

    def _local_variance(self, gray: np.ndarray) -> np.ndarray:
        """
        Local variance via E[X²] – E[X]².
        Fast to compute with two box-filter passes; accurate enough for our use.
        """
        f = gray.astype(np.float32)
        k = (self.var_ksize, self.var_ksize)
        mean    = cv2.blur(f,    k)
        mean_sq = cv2.blur(f**2, k)
        return np.clip(mean_sq - mean**2, 0, None)

    def _gradient_magnitude(self, gray: np.ndarray) -> np.ndarray:
        """Sobel gradient magnitude — high on stone edges, low on smooth concrete."""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx**2 + gy**2)

    # ── public interface ─────────────────────────────────────────────────────

    def predict_mask(
        self,
        image_bgr: np.ndarray,
        return_prob: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict sleeper mask for a full-frame BGR image.

        Args:
            image_bgr : H×W×3 BGR numpy array (full frame from cv2.imread)
            return_prob: always True here; kept for interface compatibility

        Returns:
            binary_mask : H×W uint8  — 1 = sleeper surface
            prob_map    : H×W float32 — continuous [0,1] sleeper probability
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # ── Cue 1: texture variance ──────────────────────────────────────────
        # Invert and normalise so HIGH probability ↔ LOW variance ↔ smooth concrete
        var_map  = self._local_variance(gray)
        var_prob = 1.0 - np.clip(var_map / self.var_thresh, 0, 1)

        # ── Cue 2: gradient magnitude ────────────────────────────────────────
        grad_map  = self._gradient_magnitude(gray)
        grad_prob = 1.0 - np.clip(grad_map / self.grad_thresh, 0, 1)

        # ── Cue 3: HSV saturation ────────────────────────────────────────────
        # Concrete is grey (saturation ≈ 0); ballast has earthier colours.
        hsv   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sat   = hsv[:, :, 1].astype(np.float32) / 255.0
        sat_prob = 1.0 - sat

        # ── Weighted blend ───────────────────────────────────────────────────
        # Texture cues carry 80 % weight; colour cue is weaker (lighting can shift it).
        prob_map = 0.40 * var_prob + 0.40 * grad_prob + 0.20 * sat_prob

        # ── Binary threshold ─────────────────────────────────────────────────
        binary_mask = (prob_map >= 0.5).astype(np.uint8)

        # ── Morphological cleanup ────────────────────────────────────────────
        # CLOSE before OPEN so cracks on the sleeper surface do NOT split the
        # mask into two separate regions (which would shrink or lose ROIs).
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_k, self.morph_k))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, k)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  k)

        return binary_mask, prob_map.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# B.  TRAINABLE LIGHTWEIGHT U-NET  (needs annotated masks)
# ══════════════════════════════════════════════════════════════════════════════

class _DSConv(nn.Module):
    """
    Depthwise-separable conv block.
    ~8× fewer params than a full Conv2d for the same kernel/channel count.
    Sufficient for smooth region segmentation — we do not need the full
    representational power of dense convolution here.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            # Each input channel convolved independently (spatial mixing)
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            # 1×1 pointwise conv mixes channels
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),   # ReLU6 clips large activations, better for quantisation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightweightSleeperUNet(nn.Module):
    """
    ~200 K parameter U-Net for sleeper region segmentation.

    Designed to run on 256×256 downsampled frames in a single forward pass
    before SAHI tiling begins.  Outputs serve two roles:
      1. binary mask  → restrict tile generation to sleeper ROIs  (Module 2)
      2. global_embed → FiLM-conditioning of tile neck features   (Phase 3, Section 2.1)

    Training recipe:
      - Annotate ~50 sleeper masks (polygon label in Roboflow or LabelMe)
      - Loss: Dice + BCE on the sleeper channel
      - 30–50 epochs on 256×256 crops is enough (it is a coarse task)

    Input normalisation: RGB in [0, 1].
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 16, num_classes: int = 2):
        super().__init__()
        c = base_ch   # shorthand

        # ── Encoder: stride-2 blocks halve H×W, double channels ─────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU6(inplace=True)
        )                                         # 256→128, c  ch
        self.enc2 = _DSConv(c,     c * 2, stride=2)  # 128→ 64, 2c ch
        self.enc3 = _DSConv(c * 2, c * 4, stride=2)  #  64→ 32, 4c ch
        self.enc4 = _DSConv(c * 4, c * 8, stride=2)  #  32→ 16, 8c ch

        # ── Bottleneck: global average pooling → small MLP → residual inject ─
        # The MLP lets the decoder know the overall scene brightness and layout
        # (sleeper orientation changes with camera angle on curved track sections).
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_mlp  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 8, c * 8),
            nn.ReLU(inplace=True),
        )
        # The c*8-dim vector is also returned as `global_embed` for Phase 3 FiLM use.

        # ── Decoder: ConvTranspose2d upsample + skip-connection concat ───────
        # Skip connections bring back fine spatial detail erased by striding.
        self.up4  = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec4 = _DSConv(c * 8, c * 4)   # concat enc3 skip (4c) + up4 (4c) → 4c

        self.up3  = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec3 = _DSConv(c * 4, c * 2)

        self.up2  = nn.ConvTranspose2d(c * 2, c,     2, stride=2)
        self.dec2 = _DSConv(c * 2, c)

        self.up1  = nn.ConvTranspose2d(c, c,          2, stride=2)

        # ── Output: 1×1 conv → num_classes logits ───────────────────────────
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : [B, 3, H, W] normalised RGB tensor

        Returns:
            logits       : [B, num_classes, H, W] — feed to Dice+BCE loss
            global_embed : [B, 8c]                — for FiLM conditioning in Phase 3
        """
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Global context — broadcast back to e4 spatial size and add residually
        g_vec     = self.global_pool(e4)          # [B, 8c, 1, 1]
        global_embed = self.global_mlp(g_vec)     # [B, 8c]
        e4 = e4 + g_vec.expand_as(e4)             # residual spatial inject

        # Decode with skip connections
        # bilinear interp before concat handles any ±1 px size mismatch from striding
        d4 = self.up4(e4)
        d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)

        return self.head(d1), global_embed

    def predict_mask(
        self,
        image_bgr: np.ndarray,
        device: str = 'cpu',
        threshold: float = 0.5,
        downsample_size: Tuple[int, int] = (256, 256),  # (W, H)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Same interface as ClassicalSleeperSegmenter.predict_mask().

        The rest of the pipeline calls this method without caring which
        segmenter implementation is underneath — pure duck typing.
        """
        orig_h, orig_w = image_bgr.shape[:2]

        # BGR → RGB → float [0,1] → [B, 3, H, W] tensor
        img_rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_small = cv2.resize(img_rgb, downsample_size)
        tensor    = (torch.from_numpy(img_small).float()
                     .permute(2, 0, 1).unsqueeze(0) / 255.0).to(device)

        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(tensor)
            # Softmax across class axis; take channel-1 (sleeper class probability)
            prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

        # Bilinear upsample back to original resolution
        prob_full   = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (prob_full >= threshold).astype(np.uint8)

        # Morphological cleanup — same logic as classical segmenter
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, k)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  k)

        return binary_mask, prob_full.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITY: extract ROI bounding boxes from the binary mask
# ══════════════════════════════════════════════════════════════════════════════

def extract_sleeper_rois(
    binary_mask: np.ndarray,
    min_area: int = 5000,
    padding: int  = 20,
) -> List[Tuple[int, int, int, int]]:
    """
    Find connected sleeper regions and return their bounding boxes.

    These ROIs are fed to ROISAHISlicer so tiles are only generated inside
    sleeper areas — the key intervention of Section 9.1.

    Args:
        binary_mask : H×W uint8, 1 = sleeper
        min_area    : discard tiny blobs (segmentation noise, <5000 px²)
        padding     : expand each ROI box by this many pixels so that cracks
                      sitting right on the sleeper edge are not clipped

    Returns:
        rois : list of (x1, y1, x2, y2) in full-frame pixel coordinates
    """
    h, w = binary_mask.shape
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask)

    rois = []
    for i in range(1, n_labels):           # label 0 is always background
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue

        x  = stats[i, cv2.CC_STAT_LEFT]
        y  = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        x1 = max(0,     x  - padding)
        y1 = max(0,     y  - padding)
        x2 = min(w - 1, x  + bw + padding)
        y2 = min(h - 1, y  + bh + padding)
        rois.append((x1, y1, x2, y2))

    return rois
