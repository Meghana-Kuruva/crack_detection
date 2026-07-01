"""
Phase 2 — Pseudo Label Generator
==================================
Generates training targets for the auxiliary heads from existing OBB labels
and the Phase 1 classical segmenter — without any new manual annotation.

WHY pseudo-labels are necessary:
  The Roboflow dataset only has OBB detection labels.  The three auxiliary heads
  need:
    1. Crack pixel masks       (crack segmentation head — Focal-Tversky)
    2. Sleeper region masks    (sleeper seg head — Dice+BCE)
    3. Crack-type class labels (crack-type head — CrossEntropy)

  We generate all three automatically:
    1. Rasterise GT OBB polygons → binary crack mask.
       Accuracy: approximate (OBBs are loose), but enough to drive Focal-Tversky
       to focus on thin elongated structures vs background.
    2. Run Phase 1 ClassicalSleeperSegmenter on each training image.
       Accuracy: the Phase 1 segmenter is the baseline; the learned head will
       exceed it by training end (that's the point of Phase 2).
    3. Map existing class names → crack-type indices using engineering knowledge:
         "cracks under rail seat"  → transverse (class 1)
         "cracks in midportion"    → longitudinal (class 0) — approximate
         "damage"                  → spalling    (class 3)

All pseudo-labels are used with appropriately reduced confidence — the model
learns from imperfect supervision and still improves features.

Usage:
    from pseudo_labels import PseudoLabelGenerator
    gen = PseudoLabelGenerator(class_names=['cracks in midportion', ...])
    # Inside the training dataloader collate:
    crack_mask, sleeper_mask, type_label = gen.generate(image_bgr, obb_label_path)
"""

import cv2
import numpy as np
import torch
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Phase 1 classical segmenter
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
from sleeper_segmentation import ClassicalSleeperSegmenter


# Mapping from Roboflow class name → crack-type index.
# Based on railway engineering knowledge:
#   - "cracks in midportion" = cracks at the centre of the sleeper span
#     → typically longitudinal (run along the length under bending stress)
#   - "cracks under rail seat" = cracks where rail bears on sleeper
#     → typically transverse (rail-seat cracks from vertical load concentration)
#   - "damage" = general visible degradation = spalling / surface damage
DEFAULT_CLASS_TO_TYPE = {
    'cracks in midportion':    0,   # longitudinal
    'cracks under rail seat':  1,   # transverse
    'damage':                  3,   # spalling
}


class PseudoLabelGenerator:
    """
    Generates crack mask, sleeper mask, and crack-type label for one training image.

    Thread-safe: all operations are on numpy arrays; the classical segmenter
    has no global state.  Safe to use in a DataLoader with num_workers > 0.
    """

    def __init__(
        self,
        class_names:        List[str],
        class_to_type:      Dict[str, int] = None,
        sleeper_seg_device: str = 'cpu',
        mask_dilation_px:   int = 3,    # dilate OBB mask to cover crack width
    ):
        self.class_names   = class_names
        self.cls_to_type   = class_to_type or DEFAULT_CLASS_TO_TYPE
        self.dilation_px   = mask_dilation_px

        # Classical sleeper segmenter from Phase 1
        self.sleeper_seg   = ClassicalSleeperSegmenter()

    def generate(
        self,
        image_bgr:      np.ndarray,          # H×W×3 training image (BGR, full tile)
        label_path:     str,                 # path to YOLO-OBB label .txt file
        image_size:     Tuple[int, int] = None,  # (H, W) to resize masks to; None = keep
    ) -> Dict[str, torch.Tensor]:
        """
        Generate all pseudo-label targets for one image.

        Returns:
            dict with keys:
                'crack_mask'   : (1, H, W) float32 binary crack mask
                'sleeper_mask' : (1, H, W) float32 binary sleeper mask
                'crack_type'   : (1,) int64 dominant crack-type label
                'has_crack'    : bool — False for background (empty label) images
        """
        H, W = image_bgr.shape[:2]

        # ── 1. Parse OBB label file ───────────────────────────────────────────
        boxes, class_ids = self._parse_obb_label(label_path, W, H)

        # ── 2. Crack segmentation pseudo-mask ────────────────────────────────
        crack_mask = self._make_crack_mask(boxes, H, W)

        # Dilate slightly so thin crack pixels are not missed by one-pixel errors
        if self.dilation_px > 0 and crack_mask.any():
            kernel = np.ones((self.dilation_px * 2 + 1,) * 2, np.uint8)
            crack_mask = cv2.dilate(crack_mask, kernel)

        # ── 3. Sleeper segmentation pseudo-mask ──────────────────────────────
        sleeper_binary, _ = self.sleeper_seg.predict_mask(image_bgr)

        # ── 4. Crack-type label ───────────────────────────────────────────────
        crack_type = self._dominant_crack_type(class_ids)

        # ── 5. Resize if needed ───────────────────────────────────────────────
        if image_size is not None and (H, W) != image_size:
            th, tw = image_size
            crack_mask     = cv2.resize(crack_mask,     (tw, th), interpolation=cv2.INTER_NEAREST)
            sleeper_binary = cv2.resize(sleeper_binary, (tw, th), interpolation=cv2.INTER_NEAREST)

        # Convert to tensors
        crack_t   = torch.from_numpy(crack_mask.astype(np.float32)).unsqueeze(0)   # (1,H,W)
        sleeper_t = torch.from_numpy(sleeper_binary.astype(np.float32)).unsqueeze(0)
        type_t    = torch.tensor([crack_type], dtype=torch.long)

        return {
            'crack_mask':   crack_t,
            'sleeper_mask': sleeper_t,
            'crack_type':   type_t,
            'has_crack':    len(boxes) > 0,
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _parse_obb_label(
        self,
        label_path: str,
        img_w: int,
        img_h: int,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Parse a YOLO-OBB label file.

        YOLO-OBB format:  class x1 y1 x2 y2 x3 y3 x4 y4  (all normalised)

        Returns:
            boxes     : list of (4, 2) pixel-coordinate corner arrays
            class_ids : list of int class indices
        """
        boxes, class_ids = [], []

        if not os.path.exists(label_path):
            return boxes, class_ids

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                cls    = int(parts[0])
                coords = list(map(float, parts[1:9]))   # 8 normalised values
                pts    = np.array(coords).reshape(4, 2)
                pts[:, 0] *= img_w
                pts[:, 1] *= img_h
                boxes.append(pts.astype(np.float32))
                class_ids.append(cls)

        return boxes, class_ids

    def _make_crack_mask(
        self,
        boxes:   List[np.ndarray],   # list of (4,2) corner arrays in pixel coords
        H:       int,
        W:       int,
    ) -> np.ndarray:
        """
        Rasterise OBB polygons into a binary crack mask.

        Each GT box is filled as a polygon.  The filled region approximates
        the crack footprint.  This is NOT a pixel-perfect crack mask, but
        it is good enough for Focal-Tversky to learn to predict thin structures.
        """
        mask = np.zeros((H, W), dtype=np.uint8)
        for pts in boxes:
            poly = pts.astype(np.int32)
            cv2.fillPoly(mask, [poly], 1)
        return mask

    def _dominant_crack_type(self, class_ids: List[int]) -> int:
        """
        Return the most common crack-type index in this image.
        Falls back to 0 (longitudinal) when no cracks are present.
        """
        if not class_ids:
            return 0

        types = []
        for cid in class_ids:
            if cid < len(self.class_names):
                name = self.class_names[cid]
                types.append(self.cls_to_type.get(name, 0))
            else:
                types.append(0)

        # Majority vote
        from collections import Counter
        return Counter(types).most_common(1)[0][0]


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset wrapper that adds pseudo-labels to each sample
# ══════════════════════════════════════════════════════════════════════════════

class OBBDatasetWithPseudoLabels(torch.utils.data.Dataset):
    """
    Wraps an image+label directory and augments each sample with pseudo-labels
    from PseudoLabelGenerator.

    Compatible with ultralytics data format (YOLO-OBB with image+label folders).

    Usage:
        ds = OBBDatasetWithPseudoLabels(
            images_dir   = '/content/BTP-1/train/images',
            labels_dir   = '/content/BTP-1/train/labels',
            class_names  = ['cracks in midportion', 'cracks under rail seat', 'damage'],
            img_size     = (640, 640),
        )
        loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)

    Each batch item has:
        'img'         : (3, H, W) float32 normalised tensor
        'crack_mask'  : (1, H, W) float32
        'sleeper_mask': (1, H, W) float32
        'crack_type'  : (1,) int64
        'label_path'  : str (for debugging)
    """

    def __init__(
        self,
        images_dir:  str,
        labels_dir:  str,
        class_names: List[str],
        img_size:    Tuple[int, int] = (640, 640),
        augment:     bool = True,
    ):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size   = img_size
        self.augment    = augment

        self.gen = PseudoLabelGenerator(class_names=class_names)

        # Collect all image paths
        self.img_paths = sorted(
            list(self.images_dir.glob('*.jpg')) +
            list(self.images_dir.glob('*.png'))
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path   = self.img_paths[idx]
        label_path = self.labels_dir / (img_path.stem + '.txt')

        # Read and resize image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            img_bgr = np.zeros((*self.img_size, 3), np.uint8)

        img_bgr = cv2.resize(img_bgr, self.img_size[::-1])  # cv2 takes (W,H)

        # Augmentation: multiple transforms for the tiny 119-image dataset
        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                img_bgr = cv2.flip(img_bgr, 1)
            # Vertical flip (cracks appear on both track orientations)
            if np.random.rand() > 0.8:
                img_bgr = cv2.flip(img_bgr, 0)
            # Random 90° rotation — cracks can run in any direction
            if np.random.rand() > 0.7:
                k = np.random.randint(1, 4)
                img_bgr = np.rot90(img_bgr, k).copy()
            # HSV jitter: simulate different lighting / rust / weathering
            if np.random.rand() > 0.5:
                hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 1] *= np.random.uniform(0.6, 1.4)   # saturation
                hsv[..., 2] *= np.random.uniform(0.6, 1.4)   # value
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # Gaussian blur: simulate out-of-focus or dusty camera
            if np.random.rand() > 0.8:
                k = np.random.choice([3, 5])
                img_bgr = cv2.GaussianBlur(img_bgr, (k, k), 0)

        # Generate pseudo-labels
        pseudo = self.gen.generate(img_bgr, str(label_path), image_size=self.img_size)

        # Convert image to float tensor
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_t   = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0

        return {
            'img':          img_t,
            'crack_mask':   pseudo['crack_mask'],
            'sleeper_mask': pseudo['sleeper_mask'],
            'crack_type':   pseudo['crack_type'],
            'label_path':   str(label_path),
        }
