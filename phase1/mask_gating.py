"""
Phase 1 — Module 3: Mask Gating
================================
After YOLO runs on each tile and detections are re-projected to full-frame
coordinates, this module filters/penalises detections that land outside the
sleeper mask.

Two modes (Section 8):
  hard — binary: drop any detection whose sleeper-footprint overlap < threshold.
           Implements the "post-hoc mask filtering" baseline (Section 8.1).
           Near-zero compute, directly removes boundary/ballast FPs.
           Use as the LOWER BOUND in the ablation table.

  soft — multiply conf by p(sleeper|box)^alpha (Section 8.2).
           Preserves recall on cracks physically at the sleeper edge where
           the mask is uncertain.  A detection half-inside ballast gets its
           confidence heavily penalised but is not hard-dropped.
           Default production mode.

The key computation is compute_box_sleeper_overlap():
  Rasterise the OBB polygon → sample the continuous probability map inside.
  Mean probability = p(sleeper | box).  Using the probability map (not the
  binary mask) avoids hard cliff effects at the boundary.
"""

import cv2
import numpy as np
from typing import Tuple


# ══════════════════════════════════════════════════════════════════════════════
# CORE HELPER: p(sleeper | OBB)
# ══════════════════════════════════════════════════════════════════════════════

def compute_box_sleeper_overlap(
    obb_corners:      np.ndarray,    # (4, 2) float, full-frame pixel coords
    sleeper_prob_map: np.ndarray,    # H×W float32, per-pixel sleeper probability
) -> float:
    """
    Mean sleeper probability inside the OBB footprint = p(sleeper | box).

    Returns:
      1.0  →  box is entirely on the sleeper surface
      0.0  →  box is entirely in ballast / outside image

    Why we use the PROBABILITY MAP rather than the binary mask:
      At the sleeper-ballast boundary the probability ramps from 1→0 over
      several pixels.  Sampling the binary mask would give a hard 0 or 1 for
      every pixel at the boundary, making the overlap score jump discontinuously
      as a box shifts by one pixel.  The probability map gives a smooth gradient
      that the soft-gating exponent can exploit.

    Why we RASTERISE the polygon rather than sample the centre:
      Cracks near the sleeper edge have their centre on the sleeper but part of
      their box over ballast (or vice-versa).  A centre-only sample would
      wrongly classify them.  Rasterising captures the actual footprint.
    """
    h, w = sleeper_prob_map.shape

    # Rasterise the OBB polygon into a binary mask the same size as the full frame
    poly = obb_corners.astype(np.int32)
    obb_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(obb_mask, [poly], 1)

    # Sample the probability map only at pixels that are inside the OBB
    inside_pixels = sleeper_prob_map[obb_mask == 1]

    if len(inside_pixels) == 0:
        # Box is entirely out of frame bounds
        return 0.0

    return float(inside_pixels.mean())


# ══════════════════════════════════════════════════════════════════════════════
# MASK GATING CLASS
# ══════════════════════════════════════════════════════════════════════════════

class MaskGating:
    """
    Gate crack detections by their sleeper-mask overlap score.

    Expected effect (Table 14 of design doc):
      "Largest FP reduction — directly removes boundary/ballast FPs."
      "Could cut off-sleeper FPs by a large fraction."
      Confidence: High — attacks root cause directly.
      Compute cost: Low (the segmentation already ran on the full frame).
    """

    def __init__(
        self,
        mode:           str   = 'soft', # 'hard' or 'soft'
        hard_threshold: float = 0.30,   # Hard mode: drop if p(sleeper|box) < this
        soft_alpha:     float = 2.0,    # Soft mode: conf' = conf × overlap^alpha
                                        # alpha=2 → a box with 50% overlap gets conf×0.25
        min_confidence: float = 0.05,   # After soft gating, discard if conf' < this floor
    ):
        assert mode in ('hard', 'soft'), "mode must be 'hard' or 'soft'"
        self.mode           = mode
        self.hard_threshold = hard_threshold
        self.soft_alpha     = soft_alpha
        self.min_confidence = min_confidence

    def gate(
        self,
        boxes:            np.ndarray,  # (N, 4, 2) OBB corners, full-frame coords
        confidences:      np.ndarray,  # (N,)
        class_ids:        np.ndarray,  # (N,)
        sleeper_prob_map: np.ndarray,  # H×W float32 from segmenter
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply mask gating to a batch of detections.

        Returns:
            gated_boxes    : (M, 4, 2) — kept/suppressed detections
            gated_confs    : (M,)      — original conf in hard mode; penalised in soft mode
            gated_classes  : (M,)
            overlap_scores : (M,)      — p(sleeper|box) for each kept detection
                                         useful for downstream metrics + NMM re-scoring
        """
        if len(boxes) == 0:
            empty = np.empty(0, dtype=np.float32)
            return (np.empty((0, 4, 2), dtype=np.float32),
                    empty, empty.astype(np.int32), empty)

        # ── Compute p(sleeper | box) for every detection ─────────────────────
        overlap_scores = np.array([
            compute_box_sleeper_overlap(boxes[i], sleeper_prob_map)
            for i in range(len(boxes))
        ], dtype=np.float32)

        if self.mode == 'hard':
            # Hard binary gate: drop if the box footprint is mostly off-sleeper.
            # Simple and interpretable; good for the ablation's lower bound.
            keep = overlap_scores >= self.hard_threshold

        else:  # soft
            # Penalise proportionally.
            # alpha = 2 example behaviour:
            #   overlap 1.0 → conf unchanged          (box fully on sleeper)
            #   overlap 0.7 → conf × 0.49             (mostly on sleeper, slight penalty)
            #   overlap 0.5 → conf × 0.25             (ambiguous, heavy penalty)
            #   overlap 0.2 → conf × 0.04             (mostly in ballast, near-zero)
            penalised = confidences * (overlap_scores ** self.soft_alpha)

            # Hard floor: if even after penalisation the confidence is above the
            # floor it still counts; otherwise discard (avoids cluttered output)
            keep         = penalised >= self.min_confidence
            confidences  = penalised   # return the penalised confidences downstream

        return (
            boxes[keep],
            confidences[keep],
            class_ids[keep],
            overlap_scores[keep],
        )
