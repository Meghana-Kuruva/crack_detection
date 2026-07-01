"""
Phase 1 — Module 2: ROI-Restricted SAHI Tiler
==============================================
Standard SAHI slices the ENTIRE image into uniform overlapping tiles.

The problem (Section 1a + 1d):
  Tiles in the ballast region contain no cracks but confuse the model into
  producing FPs because locally a ballast gap and a crack look identical.
  When the same ballast gap appears in several overlapping tiles, SAHI's
  merge step reinforces its confidence — exactly the wrong behaviour.

This module replaces uniform SAHI with ROI-restricted tiling:
  1. Accept sleeper ROIs from the segmentation module  (Module 1)
  2. Generate tiles ONLY within those ROIs with configurable overlap
  3. Optionally pad each tile with border context (Section 9.2) so the model
     sees a bit of the surrounding region — restoring some of the context
     SAHI strips, without the cost of the full FiLM global branch (Phase 3)
  4. Store per-tile metadata so detections can be re-projected to full-frame
     coordinates after inference

Design note on tile size (Section 9.3):
  Smaller tiles → more thin-crack recall but worse context, more FPs.
  Larger tiles → better context, weaker thin-crack recall.
  Default 640 px is a reasonable middle ground; sweep {384,512,640,768}
  and pick the precision/recall knee.  Do NOT solve recall by shrinking tiles
  — instead use P2 detection level (Phase 3) + context branch.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SlicedTile:
    """
    One tile cut from the full frame.

    Carries all metadata needed to re-project detections back to full-frame
    coordinates and to know which pixels count as the "valid detection zone."
    """
    image:      np.ndarray  # H×W×3 BGR crop
    x_offset:   int         # full-frame column of this tile's top-left pixel
    y_offset:   int         # full-frame row   of this tile's top-left pixel
    # Valid zone — the central region inside the context-pad margin.
    # Detections whose centre falls outside this zone are discarded; they
    # belong to an adjacent tile's valid zone and will be caught there.
    valid_x1:   int         # relative to tile image top-left
    valid_y1:   int
    valid_x2:   int
    valid_y2:   int
    roi_idx:    int         # which sleeper ROI this tile came from
    tile_w:     int         # actual pixel width  of the tile crop
    tile_h:     int         # actual pixel height of the tile crop


class ROISAHISlicer:
    """
    Generates overlapping tiles restricted to sleeper ROIs.

    Typical usage (see pipeline.py for the full integration):

        slicer = ROISAHISlicer(tile_size=640, overlap_ratio=0.2, context_pad=32)
        tiles  = slicer.slice(full_frame_bgr, sleeper_rois)
        for tile in tiles:
            results = yolo.predict(source=tile.image, ...)
            full_corners, full_confs, full_cls = slicer.reproject_detections(
                tile, corners, confs, cls
            )
    """

    def __init__(
        self,
        tile_size:              int   = 640,  # Target tile side-length (pixels)
        overlap_ratio:          float = 0.2,  # Fraction of tile overlapping adjacent tiles
        context_pad:            int   = 32,   # Extra pixels added around each tile for context
        min_tile_area_fraction: float = 0.10, # Skip residual tiles smaller than 10% of tile_size²
    ):
        self.tile_size  = tile_size
        self.overlap    = overlap_ratio
        self.ctx_pad    = context_pad
        self.min_frac   = min_tile_area_fraction

        # Stride < tile_size so adjacent tiles share overlap pixels.
        # E.g. tile=640, overlap=0.2 → stride=512.
        # Larger overlap → better crack-at-edge recall but more duplicate
        # detections.  Pair higher overlap with mask-aware fusion (Module 4).
        self.stride = int(tile_size * (1.0 - overlap_ratio))

    # ── public interface ─────────────────────────────────────────────────────

    def slice(
        self,
        image_bgr:    np.ndarray,
        sleeper_rois: List[Tuple[int, int, int, int]],
    ) -> List[SlicedTile]:
        """
        Generate all tiles from the given sleeper ROIs.

        Args:
            image_bgr    : H×W×3 full frame
            sleeper_rois : list of (x1,y1,x2,y2) from extract_sleeper_rois()

        Returns:
            tiles : list of SlicedTile, ready for YOLO inference
        """
        img_h, img_w = image_bgr.shape[:2]
        tiles: List[SlicedTile] = []

        for roi_idx, (rx1, ry1, rx2, ry2) in enumerate(sleeper_rois):
            roi_w = rx2 - rx1
            roi_h = ry2 - ry1

            # If the entire ROI fits in one tile, emit it as a single tile
            # rather than generating lots of tiny overlapping crops.
            if roi_w <= self.tile_size and roi_h <= self.tile_size:
                t = self._make_tile(image_bgr, img_w, img_h,
                                    rx1, ry1, rx2, ry2, roi_idx)
                if t is not None:
                    tiles.append(t)
                continue

            # Slide tile windows across the ROI
            y = ry1
            while y < ry2:
                x = rx1
                while x < rx2:
                    x2 = min(x + self.tile_size, rx2)
                    y2 = min(y + self.tile_size, ry2)

                    # Skip tiles that are too small (boundary remnants)
                    if (x2 - x) * (y2 - y) < self.tile_size**2 * self.min_frac:
                        x += self.stride
                        continue

                    t = self._make_tile(image_bgr, img_w, img_h,
                                        x, y, x2, y2, roi_idx)
                    if t is not None:
                        tiles.append(t)

                    x += self.stride
                y += self.stride

        return tiles

    def reproject_detections(
        self,
        tile:           SlicedTile,
        boxes_corners:  np.ndarray,   # (N, 4, 2) OBB corners in tile-pixel coords
        confidences:    np.ndarray,   # (N,)
        class_ids:      np.ndarray,   # (N,)
        filter_valid:   bool = True,  # drop detections outside the valid zone
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Shift OBB corner coordinates from tile-local → full-frame.

        Also discards detections whose centre falls in the context-pad margin
        (filter_valid=True).  Those detections will appear in the adjacent
        tile's valid zone and be processed there — avoids double-counting
        the same crack from two overlapping tiles.

        Args:
            tile          : the SlicedTile these detections came from
            boxes_corners : (N, 4, 2) float, tile-local pixel coords
            confidences   : (N,)
            class_ids     : (N,)
            filter_valid  : True in production; False to inspect all raw dets

        Returns:
            full_frame_boxes : (M, 4, 2)
            confidences      : (M,)
            class_ids        : (M,)
        """
        if len(boxes_corners) == 0:
            return (np.empty((0, 4, 2), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.int32))

        # Shift all corner x,y by the tile's position in the full frame
        full = boxes_corners.astype(np.float32).copy()
        full[:, :, 0] += tile.x_offset   # x columns
        full[:, :, 1] += tile.y_offset   # y rows

        if not filter_valid:
            return full, confidences, class_ids

        # Centre of each OBB = mean of its 4 corners
        centers = full.mean(axis=1)   # (N, 2) — [cx, cy]

        # Valid zone in full-frame coordinates
        vx1 = tile.x_offset + tile.valid_x1
        vy1 = tile.y_offset + tile.valid_y1
        vx2 = tile.x_offset + tile.valid_x2
        vy2 = tile.y_offset + tile.valid_y2

        inside = (
            (centers[:, 0] >= vx1) & (centers[:, 0] < vx2) &
            (centers[:, 1] >= vy1) & (centers[:, 1] < vy2)
        )

        return full[inside], confidences[inside], class_ids[inside]

    # ── private ──────────────────────────────────────────────────────────────

    def _make_tile(
        self,
        image_bgr: np.ndarray,
        img_w: int,
        img_h: int,
        x1: int, y1: int,
        x2: int, y2: int,
        roi_idx: int,
    ) -> Optional[SlicedTile]:
        """
        Crop a single context-padded tile.

        Context padding (Section 9.2):
          The physical crop is enlarged by ctx_pad pixels on each side.
          The "valid detection zone" inside the padded crop is the original
          (x1,y1)→(x2,y2) rectangle.

          WHY: each tile now has a few dozen pixels of surrounding context,
          so the network sees enough of the scene to judge "am I near a
          sleeper edge?" without the full global branch.  Cheap ablation
          variant of the FiLM conditioning in Phase 3.
        """
        pad = self.ctx_pad

        # Padded crop boundaries, clamped to image edges
        px1 = max(0,       x1 - pad)
        py1 = max(0,       y1 - pad)
        px2 = min(img_w,   x2 + pad)
        py2 = min(img_h,   y2 + pad)

        crop = image_bgr[py1:py2, px1:px2].copy()
        if crop.size == 0:
            return None

        # Valid zone — relative to the crop's top-left corner
        return SlicedTile(
            image     = crop,
            x_offset  = px1,
            y_offset  = py1,
            valid_x1  = x1 - px1,
            valid_y1  = y1 - py1,
            valid_x2  = x2 - px1,
            valid_y2  = y2 - py1,
            roi_idx   = roi_idx,
            tile_w    = crop.shape[1],
            tile_h    = crop.shape[0],
        )
