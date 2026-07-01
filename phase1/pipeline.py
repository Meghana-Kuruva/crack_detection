"""
Phase 1 — Complete Inference + Training Pipeline
================================================
Integrates all 5 modules in the correct order:

  INFERENCE (per frame):
    1. SleeperSegmentation   → binary mask + probability map
    2. ROISAHISlicer         → tiles restricted to sleeper ROIs
    3. YOLOv8-OBB inference  → raw detections per tile
    4. MaskGating            → soft-penalise off-sleeper detections
    5. MaskAwareNMM          → merge + re-score cross-tile detections

  TRAINING LOOP (offline, after each training round):
    6. BootstrappedFPMiner   → mine hard negatives from held-out images
    7. run_mining_loop()     → iterate mine → merge → retrain (2–3 rounds)

Quick start:

    # ── Inference ────────────────────────────────────────────────────────────
    from pipeline import Phase1Pipeline
    pipeline = Phase1Pipeline(
        yolo_model_path = "path/to/best.pt",
        class_names     = ["cracks in midportion", "cracks under rail seat", "damage"],
    )
    import cv2
    img        = cv2.imread("test_frame.jpg")
    detections = pipeline.detect(img)
    annotated  = pipeline.visualize(img, detections)
    cv2.imwrite("result.jpg", annotated)

    # ── Hard-negative mining ─────────────────────────────────────────────────
    from hard_negative_mining import run_mining_loop
    from sleeper_segmentation import ClassicalSleeperSegmenter
    final_model = run_mining_loop(
        dataset_dir        = "/content/BTP-1",
        initial_model_path = "path/to/best.pt",
        held_out_images    = [...],          # paths to images not in training set
        segmenter          = ClassicalSleeperSegmenter(),
        output_base_dir    = "mining_output",
        n_rounds           = 3,
        device             = 0,              # GPU 0
    )
"""

import os
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from ultralytics import YOLO

from sleeper_segmentation import (
    ClassicalSleeperSegmenter,
    LightweightSleeperUNet,
    extract_sleeper_rois,
)
from roi_sahi_slicer  import ROISAHISlicer
from mask_gating      import MaskGating, compute_box_sleeper_overlap
from nmm_fusion       import MaskAwareNMM


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURE: one crack detection
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """A single crack detection with full provenance metadata."""
    corners:         np.ndarray  # (4, 2) OBB corners, full-frame pixel coords
    confidence:      float       # Original YOLO confidence score
    class_id:        int
    class_name:      str
    sleeper_overlap: float       # p(sleeper | box) — 0=ballast, 1=on sleeper


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class Phase1Pipeline:
    """
    Complete Phase 1 crack-detection pipeline.

    Module wiring:
      segmenter  →  slicer  →  yolo  →  gating  →  fusion  →  detections

    All hyperparameters have defaults that match the design doc's
    recommendations; override at construction time for ablation sweeps.
    """

    def __init__(
        self,
        yolo_model_path:         str,
        class_names:             List[str],

        # Segmenter
        use_trained_segmenter:   bool           = False,
        segmenter_model_path:    Optional[str]  = None,

        # Tiler  (Section 9.3 / 9.4)
        tile_size:               int   = 640,
        overlap_ratio:           float = 0.20,
        context_pad:             int   = 32,

        # Gating  (Section 8.2)
        gating_mode:             str   = 'soft',
        gating_alpha:            float = 2.0,

        # NMM fusion  (Section 9.6)
        nms_iou_threshold:       float = 0.30,
        confidence_floor:        float = 0.10,
        boundary_sigma:          float = 20.0,

        # YOLO inference
        yolo_conf:               float = 0.15,
        yolo_imgsz:              int   = 640,

        device:                  str   = 'cpu',
    ):
        self.class_names  = class_names
        self.device       = device
        self.yolo_conf    = yolo_conf
        self.yolo_imgsz   = yolo_imgsz

        # ── Load detector ────────────────────────────────────────────────────
        print(f"[Pipeline] Loading YOLOv8-OBB from {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)

        # ── Segmenter ────────────────────────────────────────────────────────
        if use_trained_segmenter:
            assert segmenter_model_path is not None, \
                "Provide segmenter_model_path when use_trained_segmenter=True"
            net = LightweightSleeperUNet()
            net.load_state_dict(torch.load(segmenter_model_path, map_location=device))
            net.to(device).eval()
            self.segmenter = net
            print("[Pipeline] Using trained U-Net segmenter")
        else:
            self.segmenter = ClassicalSleeperSegmenter()
            print("[Pipeline] Using classical CV segmenter (no training required)")

        # ── Tiler ────────────────────────────────────────────────────────────
        self.slicer = ROISAHISlicer(
            tile_size=tile_size,
            overlap_ratio=overlap_ratio,
            context_pad=context_pad,
        )

        # ── Gating ───────────────────────────────────────────────────────────
        self.gating = MaskGating(
            mode=gating_mode,
            soft_alpha=gating_alpha,
        )

        # ── NMM fusion ───────────────────────────────────────────────────────
        self.fusion = MaskAwareNMM(
            iou_threshold=nms_iou_threshold,
            confidence_floor=confidence_floor,
            sleeper_alpha=gating_alpha,
            boundary_sigma=boundary_sigma,
        )

        print("[Pipeline] Ready.\n")

    # ── inference ─────────────────────────────────────────────────────────────

    def detect(
        self,
        image_bgr:         np.ndarray,
        return_debug_info: bool = False,
    ):
        """
        Run the full Phase 1 pipeline on one BGR frame.

        Args:
            image_bgr        : H×W×3 BGR numpy array (from cv2.imread)
            return_debug_info: if True, also return a dict with intermediate stats

        Returns:
            detections       : List[Detection]
            (optional) debug : dict with timing and intermediate counts
        """
        t0 = time.time()

        # ── Step 1: Sleeper segmentation ─────────────────────────────────────
        # Cheap (~5ms on CPU at 256×256).  Produces:
        #   binary_mask → ROI extraction + hard gating + NMM distance transform
        #   prob_map    → soft gating + NMM re-scoring (continuous, not binary)
        binary_mask, prob_map = self.segmenter.predict_mask(image_bgr)
        sleeper_rois = extract_sleeper_rois(binary_mask)

        if not sleeper_rois:
            print("[Pipeline] Warning: no sleeper ROIs detected — check segmenter.")
            if return_debug_info:
                return [], {'error': 'no_rois', 'time_ms': (time.time()-t0)*1000}
            return []

        # ── Step 2: ROI-restricted tiling ────────────────────────────────────
        # Only generate tiles inside sleeper ROIs.
        # Ballast-only tiles never exist → the FPs that live there cannot occur.
        tiles = self.slicer.slice(image_bgr, sleeper_rois)

        # ── Step 3: YOLO inference per tile ──────────────────────────────────
        all_boxes, all_confs, all_cls = [], [], []

        for tile in tiles:
            results = self.yolo.predict(
                source=tile.image,
                conf=self.yolo_conf,
                imgsz=self.yolo_imgsz,
                verbose=False,
                device=self.device,
            )

            obb = results[0].obb
            if obb is None or len(obb) == 0:
                continue

            # Tile-local OBB data
            corners = obb.xyxyxyxy.cpu().numpy()             # (N, 4, 2)
            confs   = obb.conf.cpu().numpy()                  # (N,)
            cls     = obb.cls.cpu().numpy().astype(np.int32)  # (N,)

            # Re-project to full-frame coords and drop context-margin dets
            fc, fconf, fcls = self.slicer.reproject_detections(
                tile, corners, confs, cls, filter_valid=True
            )

            if len(fc) > 0:
                all_boxes.append(fc)
                all_confs.append(fconf)
                all_cls.append(fcls)

        if not all_boxes:
            if return_debug_info:
                return [], {
                    'n_tiles': len(tiles), 'n_raw': 0, 'n_gated': 0, 'n_final': 0,
                    'time_ms': (time.time()-t0)*1000,
                }
            return []

        all_boxes = np.concatenate(all_boxes, axis=0)   # (N, 4, 2)
        all_confs = np.concatenate(all_confs, axis=0)   # (N,)
        all_cls   = np.concatenate(all_cls,   axis=0)   # (N,)
        n_raw     = len(all_boxes)

        # ── Step 4: Soft mask gating ─────────────────────────────────────────
        # Multiply each detection's confidence by p(sleeper|box)^alpha.
        # Boxes deep in ballast → confidence ≈ 0 → dropped at the floor.
        # Boxes on the sleeper → confidence unchanged.
        gb, gc, gcls, gov = self.gating.gate(all_boxes, all_confs, all_cls, prob_map)
        n_gated = len(gb)

        if n_gated == 0:
            if return_debug_info:
                return [], {
                    'n_tiles': len(tiles), 'n_raw': n_raw,
                    'n_gated': 0, 'n_final': 0,
                    'time_ms': (time.time()-t0)*1000,
                }
            return []

        # ── Step 5: Mask-aware NMM fusion ────────────────────────────────────
        # Merge overlapping cross-tile detections.
        # Re-scores by sleeper overlap × distance-from-boundary before merging,
        # so stable ballast gaps cannot survive through overlap voting.
        mb, mc, mcls = self.fusion.fuse(gb, gc, gcls, prob_map, binary_mask)

        # ── Assemble Detection objects ────────────────────────────────────────
        detections: List[Detection] = []
        for i in range(len(mb)):
            cid  = int(mcls[i])
            name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            ov   = compute_box_sleeper_overlap(mb[i], prob_map)
            detections.append(Detection(
                corners         = mb[i],
                confidence      = float(mc[i]),
                class_id        = cid,
                class_name      = name,
                sleeper_overlap = ov,
            ))

        if return_debug_info:
            return detections, {
                'n_tiles':   len(tiles),
                'n_raw':     n_raw,
                'n_gated':   n_gated,
                'n_final':   len(detections),
                'time_ms':   (time.time() - t0) * 1000,
            }

        return detections

    # ── visualisation ─────────────────────────────────────────────────────────

    def visualize(
        self,
        image_bgr:  np.ndarray,
        detections: List[Detection],
        show_mask:  bool = True,
        show_rois:  bool = True,
    ) -> np.ndarray:
        """
        Annotate a frame with:
          - semi-transparent green overlay = segmenter's sleeper mask
          - blue rectangles = sleeper ROI bounding boxes
          - red OBBs = final crack detections (after gating + NMM)

        Returns annotated BGR image (safe to cv2.imwrite or imshow).
        """
        vis = image_bgr.copy()

        binary_mask, _ = self.segmenter.predict_mask(image_bgr)

        # Sleeper mask overlay (green, 15% opacity)
        if show_mask:
            overlay = np.zeros_like(vis)
            overlay[binary_mask == 1] = (0, 180, 0)
            vis = cv2.addWeighted(vis, 0.85, overlay, 0.15, 0)

        # Sleeper ROI boxes (blue)
        if show_rois:
            for (x1, y1, x2, y2) in extract_sleeper_rois(binary_mask):
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 2)

        # Crack detections (red OBBs)
        for det in detections:
            poly = det.corners.astype(np.int32)
            cv2.polylines(vis, [poly], isClosed=True, color=(0, 0, 255), thickness=2)

            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            label = f"{det.class_name[:12]} {det.confidence:.2f}"
            cv2.putText(vis, label, (cx - 40, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1)

        return vis

    # ── FP metrics (Section 13) ───────────────────────────────────────────────

    def compute_fp_metrics(
        self,
        detections:       List[Detection],
        off_sleeper_thr:  float = 0.30,   # Detections below this overlap are "off-sleeper FPs"
    ) -> Dict:
        """
        Compute the FP-focused metrics described in Section 13 of the design doc:
          - off_sleeper_fp_count : detections whose footprint is mostly off-sleeper
          - on_sleeper_count     : detections on the sleeper (likely true positives)
          - mean_sleeper_overlap : average p(sleeper|box) across all detections

        Call this on BOTH the baseline (plain YOLO+SAHI) and Phase 1 outputs
        and report the drop in off_sleeper_fp_count as the headline result.
        """
        if not detections:
            return {
                'total':                0,
                'off_sleeper_fp_count': 0,
                'on_sleeper_count':     0,
                'mean_sleeper_overlap': 0.0,
            }

        overlaps      = np.array([d.sleeper_overlap for d in detections])
        off_sleeper   = int((overlaps < off_sleeper_thr).sum())
        on_sleeper    = int((overlaps >= off_sleeper_thr).sum())

        return {
            'total':                len(detections),
            'off_sleeper_fp_count': off_sleeper,
            'on_sleeper_count':     on_sleeper,
            'mean_sleeper_overlap': float(overlaps.mean()),
        }


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: run baseline (plain YOLO + standard SAHI) for ablation row 0
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline(
    yolo_model_path: str,
    image_bgr:       np.ndarray,
    yolo_conf:       float = 0.15,
    yolo_imgsz:      int   = 1024,
    tile_size:       int   = 640,
    overlap_ratio:   float = 0.20,
    device:          str   = 'cpu',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard SAHI: tile the WHOLE image (no ROI restriction), run YOLO,
    apply plain NMS.  Use this as the ABLATION BASELINE (row 0).

    Returns raw (boxes, confs, class_ids) before any mask intervention
    so you can compute the off-sleeper FP rate for comparison.
    """
    model  = YOLO(yolo_model_path)
    img_h, img_w = image_bgr.shape[:2]

    # Uniform grid tiling of the entire image
    stride  = int(tile_size * (1.0 - overlap_ratio))
    all_boxes, all_confs, all_cls = [], [], []

    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            crop = image_bgr[y:y2, x:x2]
            if crop.size == 0:
                continue

            res = model.predict(
                source=crop, conf=yolo_conf, imgsz=yolo_imgsz,
                verbose=False, device=device
            )

            obb = res[0].obb
            if obb is None or len(obb) == 0:
                continue

            corners = obb.xyxyxyxy.cpu().numpy()
            confs   = obb.conf.cpu().numpy()
            cls     = obb.cls.cpu().numpy().astype(np.int32)

            # Shift to full-frame coords (no valid-zone filtering in baseline)
            corners[:, :, 0] += x
            corners[:, :, 1] += y
            all_boxes.append(corners)
            all_confs.append(confs)
            all_cls.append(cls)

    if not all_boxes:
        return (np.empty((0, 4, 2)), np.empty(0), np.empty(0, dtype=int))

    return (
        np.concatenate(all_boxes, axis=0),
        np.concatenate(all_confs, axis=0),
        np.concatenate(all_cls,   axis=0),
    )


# ══════════════════════════════════════════════════════════════════════════════
# DEMO: Colab-ready usage example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Minimal demo — paste this into a Colab cell after uploading the phase1/ folder.

    Expected directory layout:
        /content/
          phase1/               ← this package
          best.pt               ← your trained YOLOv8-OBB model
          test_frame.jpg        ← one test image from the Roboflow dataset
    """
    import sys
    sys.path.insert(0, '/content/phase1')

    CLASS_NAMES = [
        "cracks in midportion",
        "cracks under rail seat",
        "damage",
    ]

    # ── Inference demo ────────────────────────────────────────────────────────
    pipeline = Phase1Pipeline(
        yolo_model_path = "/content/best.pt",
        class_names     = CLASS_NAMES,
        tile_size       = 640,
        overlap_ratio   = 0.20,
        context_pad     = 32,
        gating_mode     = 'soft',
        gating_alpha    = 2.0,
        yolo_conf       = 0.15,
        yolo_imgsz      = 640,
        device          = '0',   # GPU; use 'cpu' if no GPU
    )

    img = cv2.imread("/content/test_frame.jpg")
    assert img is not None, "Test image not found"

    # ── Compare baseline vs Phase 1 ───────────────────────────────────────────
    print("Running BASELINE (plain SAHI, no masking)...")
    base_boxes, base_confs, base_cls = run_baseline(
        yolo_model_path = "/content/best.pt",
        image_bgr       = img,
        device          = '0',
    )

    print(f"  Baseline detections : {len(base_boxes)}")

    print("\nRunning Phase 1 pipeline...")
    dets, debug = pipeline.detect(img, return_debug_info=True)
    metrics     = pipeline.compute_fp_metrics(dets)

    print(f"  Phase 1 detections  : {metrics['total']}")
    print(f"  Off-sleeper FPs     : {metrics['off_sleeper_fp_count']}")
    print(f"  On-sleeper (true?)  : {metrics['on_sleeper_count']}")
    print(f"  Mean sleeper overlap: {metrics['mean_sleeper_overlap']:.3f}")
    print(f"  Timing              : {debug['time_ms']:.1f} ms")
    print(f"  Tiles generated     : {debug['n_tiles']}")
    print(f"  Raw YOLO dets       : {debug['n_raw']}")
    print(f"  After gating        : {debug['n_gated']}")

    # Save annotated result
    annotated = pipeline.visualize(img, dets, show_mask=True, show_rois=True)
    cv2.imwrite("/content/phase1_result.jpg", annotated)
    print("\nSaved → /content/phase1_result.jpg")

    # ── Hard-negative mining demo ─────────────────────────────────────────────
    # Uncomment and set paths to run the mining loop after initial training:
    #
    # from hard_negative_mining import run_mining_loop
    # from sleeper_segmentation import ClassicalSleeperSegmenter
    # import glob
    #
    # held_out = glob.glob("/content/BTP-1/test/images/*.jpg")
    #
    # final_model = run_mining_loop(
    #     dataset_dir        = "/content/BTP-1",
    #     initial_model_path = "/content/best.pt",
    #     held_out_images    = held_out,
    #     segmenter          = ClassicalSleeperSegmenter(),
    #     output_base_dir    = "/content/mining_output",
    #     n_rounds           = 3,
    #     device             = 0,
    # )
    # print("Final model after mining:", final_model)
