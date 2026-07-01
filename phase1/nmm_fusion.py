"""
Phase 1 — Module 4: Mask-Aware NMM Fusion
==========================================
Replaces plain NMS when combining detections from overlapping tiles.

Why plain NMS is wrong here (Section 1d — "overlap voting amplifies stable FPs"):
  A ballast gap that appears consistently across N overlapping tiles gets its
  confidence reinforced by overlap voting.  NMS keeps the highest-confidence
  instance — which is still high.  The gap survives.

NMM — Non-Max Merging:
  Instead of keeping one box and suppressing all overlapping ones, NMM
  MERGES overlapping boxes into a single box whose corners are the
  confidence-weighted average of the cluster.  This is better for thin
  crack detection near tile edges where the highest-confidence box is not
  necessarily the one with the fullest view of the crack.

Mask-aware re-scoring (Section 9.6):
  Before merging, each detection is re-scored by:
    rescored_conf = original_conf × p(sleeper|box)^alpha × boundary_weight

  where boundary_weight penalises detections near the sleeper-ballast edge.
  This demotes stable ballast detections before the merge step, so they
  cannot survive the confidence floor even if they appear in many tiles.

  The re-scored confidence is used ONLY for clustering and ordering.
  The final merged confidence that is reported is the MAX of the original
  (un-rescored) confidences in the cluster — so thresholds in downstream
  evaluation still have their usual meaning.
"""

import cv2
import numpy as np
from typing import Tuple

try:
    from .mask_gating import compute_box_sleeper_overlap   # package import
except ImportError:
    from mask_gating import compute_box_sleeper_overlap    # sys.path import


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: rotated IoU between two OBBs
# ══════════════════════════════════════════════════════════════════════════════

def rotated_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two oriented bounding boxes specified as 4 corner points.

    Method: cv2.minAreaRect → cv2.rotatedRectangleIntersection.
    Accurate for near-rectangular boxes (all crack OBBs are), fast in C++.

    Args:
        box1, box2 : (4, 2) corner arrays in pixel coordinates

    Returns:
        iou : float in [0, 1]
    """
    rect1 = cv2.minAreaRect(box1.astype(np.float32))
    rect2 = cv2.minAreaRect(box2.astype(np.float32))

    ret, pts = cv2.rotatedRectangleIntersection(rect1, rect2)

    if ret == cv2.INTERSECT_NONE or pts is None:
        return 0.0

    intersect_area = cv2.contourArea(pts)

    area1 = rect1[1][0] * rect1[1][1]
    area2 = rect2[1][0] * rect2[1][1]
    union = area1 + area2 - intersect_area

    return float(intersect_area / union) if union > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: distance-from-boundary weight
# ══════════════════════════════════════════════════════════════════════════════

def boundary_distance_weight(
    box_center:     np.ndarray,   # (2,) [cx, cy] full-frame pixel coords
    sleeper_mask:   np.ndarray,   # H×W uint8 binary sleeper mask
    sigma:          float = 20.0, # Controls how fast the weight rises with distance
) -> float:
    """
    Returns HIGH weight if the detection centre is deep inside a sleeper region,
    LOW weight if it is near the sleeper-ballast boundary.

    WHY: true cracks are distributed across the sleeper surface; FPs cluster
    at the boundary (where the mask is uncertain).  This weight demotes
    edge-of-sleeper detections without completely zeroing them — we do not
    want to miss cracks that extend to the very edge of a sleeper.

    Implementation: distance transform of the binary mask.
      dist_transform[row,col] = Euclidean distance (pixels) from (row,col) to
      the nearest non-sleeper (boundary/ballast) pixel.
      Boundary pixels → distance 0 → weight 0.5 (sigmoid at 0).
      Deep interior → large distance → weight → 1.0.

    Weight formula: sigmoid(distance / sigma)
      At dist = 0  : weight = 0.50
      At dist = σ  : weight = 0.73
      At dist = 3σ : weight = 0.95
    """
    dist_map = cv2.distanceTransform(sleeper_mask, cv2.DIST_L2, cv2.DIST_MASK_5)

    h, w = dist_map.shape
    cx = int(np.clip(box_center[0], 0, w - 1))
    cy = int(np.clip(box_center[1], 0, h - 1))

    dist   = float(dist_map[cy, cx])
    weight = 1.0 / (1.0 + np.exp(-dist / sigma))
    return weight


# ══════════════════════════════════════════════════════════════════════════════
# MASK-AWARE NMM
# ══════════════════════════════════════════════════════════════════════════════

class MaskAwareNMM:
    """
    Mask-aware Non-Max Merging for cross-tile detection fusion.

    Algorithm:
      1. Re-score each detection:
           rescored = conf × p(sleeper|box)^alpha × boundary_weight
      2. Sort by rescored confidence, highest first.
      3. Greedy cluster: for each unassigned detection, collect all
         overlapping detections (IoU > iou_threshold) into its cluster.
      4. Merge each cluster:
           corners  = weighted-average of cluster corners (weights = rescored conf)
           conf     = max of original confs in cluster
           class_id = majority vote in cluster
      5. Drop any merged box whose MAX rescored confidence < confidence_floor.
    """

    def __init__(
        self,
        iou_threshold:    float = 0.30,  # Boxes with IoU > this are the same detection
        confidence_floor: float = 0.10,  # Drop merged box if its best rescored conf < this
        sleeper_alpha:    float = 2.0,   # Exponent for p(sleeper|box) re-weighting
        boundary_sigma:   float = 20.0,  # Sigma for distance-from-boundary weight (pixels)
        class_agnostic:   bool  = False, # If True, merge detections across classes
    ):
        self.iou_thr    = iou_threshold
        self.conf_floor = confidence_floor
        self.alpha      = sleeper_alpha
        self.sigma      = boundary_sigma
        self.cls_agnostic = class_agnostic

    def fuse(
        self,
        boxes:          np.ndarray,   # (N, 4, 2) OBB corners, full-frame coords
        confidences:    np.ndarray,   # (N,) original model confidences
        class_ids:      np.ndarray,   # (N,) class indices
        sleeper_prob:   np.ndarray,   # H×W float32 sleeper probability map
        sleeper_mask:   np.ndarray,   # H×W uint8 binary sleeper mask
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fuse all tile detections into a deduplicated, mask-re-scored set.

        Returns:
            merged_boxes   : (M, 4, 2)
            merged_confs   : (M,)       — original scale confidences
            merged_classes : (M,)
        """
        if len(boxes) == 0:
            return (np.empty((0, 4, 2), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.int32))

        # ── Step 1: compute re-scored confidence for every detection ──────────
        centers   = boxes.mean(axis=1)   # (N, 2) [cx, cy]
        rescored  = np.zeros(len(boxes), dtype=np.float32)

        for i in range(len(boxes)):
            overlap     = compute_box_sleeper_overlap(boxes[i], sleeper_prob)
            bound_w     = boundary_distance_weight(centers[i], sleeper_mask, self.sigma)
            rescored[i] = confidences[i] * (overlap ** self.alpha) * bound_w

        # ── Step 2: sort by rescored confidence ──────────────────────────────
        order      = np.argsort(-rescored)
        boxes      = boxes[order]
        confidences = confidences[order]
        class_ids  = class_ids[order]
        rescored   = rescored[order]

        # ── Steps 3–5: greedy cluster → merge → floor ────────────────────────
        used = np.zeros(len(boxes), dtype=bool)
        out_boxes, out_confs, out_cls = [], [], []

        for i in range(len(boxes)):
            if used[i]:
                continue

            # Seed a new cluster with detection i
            cluster = [i]
            used[i] = True

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if (not self.cls_agnostic) and (class_ids[j] != class_ids[i]):
                    continue
                if rotated_iou(boxes[i], boxes[j]) >= self.iou_thr:
                    cluster.append(j)
                    used[j] = True

            c_idx   = np.array(cluster)
            c_boxes = boxes[c_idx]          # (K, 4, 2)
            c_rsc   = rescored[c_idx]       # (K,) rescored weights
            c_conf  = confidences[c_idx]    # (K,) original confs
            c_cls   = class_ids[c_idx]      # (K,)

            # Drop cluster if even its best rescored confidence is below the floor.
            # This is where stable ballast-gap clusters get removed: they have
            # high original confidence but low rescored (because p(sleeper|box) ≈ 0).
            if c_rsc.max() < self.conf_floor:
                continue

            # Merge: weighted-average corners using rescored confidence as weights
            weights    = c_rsc / c_rsc.sum()                             # (K,) normalised
            merged_box = np.einsum('k,kij->ij', weights, c_boxes)        # (4, 2) weighted mean

            out_boxes.append(merged_box)
            out_confs.append(float(c_conf.max()))                        # report original conf
            out_cls.append(int(np.bincount(c_cls.astype(np.int32)).argmax()))

        if not out_boxes:
            return (np.empty((0, 4, 2), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.int32))

        return (
            np.array(out_boxes,  dtype=np.float32),
            np.array(out_confs,  dtype=np.float32),
            np.array(out_cls,    dtype=np.int32),
        )
