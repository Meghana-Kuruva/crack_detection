"""
Phase 1 — Module 5: Bootstrapped Hard-Negative Mining
======================================================
This is the training-side fix for failure mode (c) from Section 1:

  "If ballast regions were never deliberately sampled as hard negatives,
   the classifier's 'crack' prototype is defined almost entirely against
   smooth concrete.  The decision boundary against ballast texture was
   never trained — so any ballast structure that resembles a crack lands
   on the wrong side of an UNDEFINED boundary."

The loop (Section 10.1):
  Round 0 : train on the original Roboflow OBB dataset  →  baseline model
  Round r (r = 1, 2, 3) :
    A. Run current model on HELD-OUT track images with a permissive threshold.
    B. Collect every high-confidence detection that falls OUTSIDE the sleeper
       mask.  The mask provides automatic ground-truth verification — no
       manual labelling is needed because we KNOW those regions are ballast.
    C. Save those crops as background images (empty YOLO label files).
    D. Merge hard negatives into training data and retrain.
  Terminate when off-sleeper FP rate converges (usually after 2–3 rounds).

Also implements:
  - Boundary tile oversampling (Section 10.2)
  - Synthetic hard negatives via copy-paste (Section 10.5)
"""

import os
import cv2
import numpy as np
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from ultralytics import YOLO
from mask_gating import compute_box_sleeper_overlap


# ══════════════════════════════════════════════════════════════════════════════
# MODULE A: Bootstrapped FP Miner
# ══════════════════════════════════════════════════════════════════════════════

class BootstrappedFPMiner:
    """
    Single-round hard-negative mining.

    Runs the current model on held-out images, collects off-sleeper FPs,
    and writes them as background training samples (empty label files).
    Designed for 2–3 iterations inside run_mining_loop().
    """

    def __init__(
        self,
        confidence_threshold: float = 0.30,  # Minimum conf to count as a "candidate FP"
                                              # (permissive: we want to see what the model
                                              #  is fairly sure about but is wrong)
        sleeper_overlap_max:  float = 0.20,  # Detections with p(sleeper|box) < this
                                              # are classified as off-sleeper FPs
        max_negatives_per_image: int = 10,   # Cap per image — avoids class imbalance
        crop_padding:         int   = 20,    # Extra pixels around each FP crop
    ):
        self.conf_thr      = confidence_threshold
        self.overlap_max   = sleeper_overlap_max
        self.max_per_image = max_negatives_per_image
        self.crop_pad      = crop_padding

    def mine_round(
        self,
        model_path:       str,
        held_out_images:  List[str],
        segmenter,                      # ClassicalSleeperSegmenter or LightweightSleeperUNet
        output_dir:       str,
        device:           str  = 'cpu',
        inference_imgsz:  int  = 1024,
    ) -> Dict:
        """
        Run one mining pass: detect FPs → verify via sleeper mask → save crops.

        Args:
            model_path      : path to current best .pt YOLOv8-OBB model
            held_out_images : file paths NOT used for training (MUST be separate)
            segmenter       : initialised sleeper segmenter
            output_dir      : where to write mined negative images + labels
            device          : 'cpu' or 'cuda' / 0 (int GPU index)
            inference_imgsz : image size for YOLO inference

        Returns:
            stats dict — log these to track convergence across rounds
        """
        neg_img_dir = os.path.join(output_dir, 'images')
        neg_lbl_dir = os.path.join(output_dir, 'labels')
        os.makedirs(neg_img_dir, exist_ok=True)
        os.makedirs(neg_lbl_dir, exist_ok=True)

        model = YOLO(model_path)

        stats = {
            'images_processed': 0,
            'total_detections': 0,
            'fp_candidates':    0,
            'negatives_saved':  0,
        }

        for img_path in held_out_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            stats['images_processed'] += 1
            img_h, img_w = img.shape[:2]
            stem = Path(img_path).stem

            # ── A: Segment sleeper region ─────────────────────────────────────
            binary_mask, prob_map = segmenter.predict_mask(img)

            # ── B: Run YOLO with a PERMISSIVE threshold ───────────────────────
            # We use the class confidence threshold here because we WANT to find
            # the FPs the model is confident about.  Raising the threshold would
            # miss the borderline ballast-gap cases we most need to hard-mine.
            results = model.predict(
                source=str(img_path),
                conf=self.conf_thr,
                imgsz=inference_imgsz,
                verbose=False,
                device=device,
            )

            obb = results[0].obb
            if obb is None or len(obb) == 0:
                continue

            corners_all = obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2) pixel coords
            confs_all   = obb.conf.cpu().numpy()       # (N,)
            stats['total_detections'] += len(corners_all)

            # ── C: Identify off-sleeper (FP) detections ───────────────────────
            fp_count = 0
            for i, (corners, conf) in enumerate(zip(corners_all, confs_all)):
                if fp_count >= self.max_per_image:
                    break

                overlap = compute_box_sleeper_overlap(corners, prob_map)

                # overlap < sleeper_overlap_max → box is predominantly in ballast
                # → the segmenter CONFIRMS this is not on a sleeper → it IS a FP
                if overlap >= self.overlap_max:
                    continue   # This detection is on a sleeper; leave it

                stats['fp_candidates'] += 1

                # ── D: Crop and save as background hard-negative ──────────────
                # The crop captures the ballast region the model misfired on.
                # An EMPTY label file tells YOLO "this image has no cracks"
                # — effectively a background negative with elevated sample weight
                # (more examples → effective weight up, no extra loss coding needed).

                xs = corners[:, 0]
                ys = corners[:, 1]
                x1 = int(max(0,       xs.min() - self.crop_pad))
                y1 = int(max(0,       ys.min() - self.crop_pad))
                x2 = int(min(img_w,   xs.max() + self.crop_pad))
                y2 = int(min(img_h,   ys.max() + self.crop_pad))

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                name = f"{stem}_hardneg_{i:04d}"
                cv2.imwrite(os.path.join(neg_img_dir, f"{name}.jpg"), crop)
                open(os.path.join(neg_lbl_dir, f"{name}.txt"), 'w').close()

                stats['negatives_saved'] += 1
                fp_count += 1

        print(f"[Mining] Round complete: {stats}")
        return stats


# ══════════════════════════════════════════════════════════════════════════════
# MODULE B: Boundary Tile Oversampler  (Section 10.2)
# ══════════════════════════════════════════════════════════════════════════════

class BoundaryTileOversampler:
    """
    Oversample training tiles that straddle the sleeper-ballast boundary.

    WHY: FPs cluster at the boundary; we should train on more boundary examples.
    Rather than changing the loss function, we simply generate N extra copies
    of each boundary tile — equivalent to a per-tile loss weight of N.

    How to use:
      1. Generate your training tiles (from ROISAHISlicer or standard SAHI).
      2. Call identify_boundary_tiles() to flag which tiles straddle the edge.
      3. Call duplicate_boundary_tiles() to copy them N times to output_dir.
      4. Include the output_dir in your YOLO training data.yaml.
    """

    def __init__(
        self,
        boundary_band_px:   int = 50,  # Pixels from mask edge counted as "boundary zone"
        oversample_factor:  int = 3,   # Emit this many copies of each boundary tile
    ):
        self.band           = boundary_band_px
        self.oversample     = oversample_factor

    def is_boundary_tile(
        self,
        tile_bbox:    Tuple[int, int, int, int],  # (x1,y1,x2,y2) in full-frame coords
        sleeper_mask: np.ndarray,                 # H×W uint8
    ) -> bool:
        """
        True if the tile overlaps the sleeper-ballast boundary band.

        Method: erode the mask by boundary_band_px → subtract from original →
        boundary pixels are those that vanished.  If any boundary pixels fall
        inside the tile's bounding box, it is a boundary tile.
        """
        kernel = np.ones((3, 3), np.uint8)
        eroded   = cv2.erode(sleeper_mask, kernel,
                             iterations=max(1, self.band // 3))
        boundary = sleeper_mask - eroded   # non-zero at the mask edge

        x1, y1, x2, y2 = tile_bbox
        # Clamp to mask dimensions
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(sleeper_mask.shape[1], x2)
        y2 = min(sleeper_mask.shape[0], y2)

        return bool(boundary[y1:y2, x1:x2].any())

    def duplicate_boundary_tiles(
        self,
        tile_image_paths: List[str],
        tile_label_paths: List[str],
        tile_bboxes:      List[Tuple],
        sleeper_masks:    List[np.ndarray],
        output_img_dir:   str,
        output_lbl_dir:   str,
    ) -> Tuple[List[str], List[str]]:
        """
        Copy boundary tiles oversample_factor times.

        Returns:
            (all_image_paths, all_label_paths) — original + copies, for data.yaml
        """
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_lbl_dir, exist_ok=True)

        out_imgs = list(tile_image_paths)
        out_lbls = list(tile_label_paths)

        for img_path, lbl_path, bbox, mask in zip(
            tile_image_paths, tile_label_paths, tile_bboxes, sleeper_masks
        ):
            if not self.is_boundary_tile(bbox, mask):
                continue

            stem = Path(img_path).stem
            img  = cv2.imread(str(img_path))
            with open(lbl_path) as f:
                lbl_content = f.read()

            for n in range(1, self.oversample):
                new_img_path = os.path.join(output_img_dir, f"{stem}_bdup{n}.jpg")
                new_lbl_path = os.path.join(output_lbl_dir, f"{stem}_bdup{n}.txt")
                cv2.imwrite(new_img_path, img)
                with open(new_lbl_path, 'w') as f:
                    f.write(lbl_content)
                out_imgs.append(new_img_path)
                out_lbls.append(new_lbl_path)

        print(f"[Boundary oversampling] "
              f"Original tiles: {len(tile_image_paths)}  "
              f"After oversampling: {len(out_imgs)}")
        return out_imgs, out_lbls


# ══════════════════════════════════════════════════════════════════════════════
# MODULE C: Synthetic Hard Negatives via Copy-Paste  (Section 10.5)
# ══════════════════════════════════════════════════════════════════════════════

class SyntheticNegativeGenerator:
    """
    Paste ballast texture patches onto sleeper regions and vice-versa.

    WHY: creates controlled hard-negative examples where the local texture
    says "ballast" but the location says "sleeper" (and vice-versa).  Forces
    the model to use BOTH cues, not just local texture — exactly the
    generalisation we need to resist ballast-gap FPs.

    Augmentation is soft (alpha blend, not hard paste) to avoid creating
    unrealistic training artefacts.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def paste_ballast_onto_sleeper(
        self,
        target_bgr:   np.ndarray,   # Training image (modified copy returned)
        ballast_patch: np.ndarray,  # Small BGR crop from a ballast region
        sleeper_mask: np.ndarray,   # Binary mask of target image's sleeper
        n_pastes:     int   = 2,
        alpha:        float = 0.70, # Blend weight for the patch (0=invisible, 1=fully replaced)
    ) -> np.ndarray:
        """
        Blend ballast texture into random sleeper positions.

        This creates images where the model sees ballast appearance INSIDE the
        known sleeper zone — it must learn to rely on location, not texture alone.
        """
        result        = target_bgr.copy()
        sleeper_pixels = np.argwhere(sleeper_mask == 1)   # (K, 2) [row, col]

        if len(sleeper_pixels) < 100:
            return result  # Not enough sleeper area to paste into

        ph, pw = ballast_patch.shape[:2]

        for _ in range(n_pastes):
            idx = self.rng.integers(len(sleeper_pixels))
            cy, cx = sleeper_pixels[idx]

            y1 = max(0, cy - ph // 2)
            x1 = max(0, cx - pw // 2)
            y2 = min(result.shape[0], y1 + ph)
            x2 = min(result.shape[1], x1 + pw)
            ph_actual = y2 - y1
            pw_actual = x2 - x1

            if ph_actual <= 0 or pw_actual <= 0:
                continue

            patch_crop = cv2.resize(ballast_patch, (pw_actual, ph_actual))
            # Alpha blend: (alpha) fraction of patch + (1-alpha) original
            result[y1:y2, x1:x2] = (
                alpha * patch_crop + (1 - alpha) * result[y1:y2, x1:x2]
            ).astype(np.uint8)

        return result

    def paste_crack_onto_clean_sleeper(
        self,
        target_bgr:   np.ndarray,   # Clean (crack-free) sleeper image
        crack_crop:   np.ndarray,   # Tight crop around a known real crack
        sleeper_mask: np.ndarray,   # Binary mask for placement restriction
        target_label_lines: List[str],  # Existing YOLO OBB label lines
        crack_label_line:   str,        # YOLO OBB label line for the crack crop
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Paste a real crack onto a clean sleeper surface to increase positive sample diversity.

        Returns:
            augmented_img   : the composite image
            updated_labels  : the original labels + the pasted crack's label (adjusted coords)

        Note: the label adjustment (re-normalising coords for the new position) is
        left as a TODO because it requires the original bounding-box corners — see
        the pipeline for how to integrate this with the Roboflow label format.
        """
        result = target_bgr.copy()
        sleeper_pixels = np.argwhere(sleeper_mask == 1)

        if len(sleeper_pixels) < 100:
            return result, target_label_lines

        ch, cw = crack_crop.shape[:2]
        idx    = self.rng.integers(len(sleeper_pixels))
        cy, cx = sleeper_pixels[idx]

        y1 = max(0, cy - ch // 2)
        x1 = max(0, cx - cw // 2)
        y2 = min(result.shape[0], y1 + ch)
        x2 = min(result.shape[1], x1 + cw)

        ch_a = y2 - y1
        cw_a = x2 - x1
        if ch_a > 0 and cw_a > 0:
            resized = cv2.resize(crack_crop, (cw_a, ch_a))
            # Paste at full opacity — the crack should be clearly visible
            result[y1:y2, x1:x2] = resized

        # Recompute OBB label coordinates for the new paste position.
        # YOLO-OBB format: "class x1 y1 x2 y2 x3 y3 x4 y4" (normalised to crop)
        # After paste at (x1, y1) with crop resized to (cw_a, ch_a):
        #   new_corner_x = x1 + norm_corner_x * cw_a  (absolute in target)
        #   new_norm_x   = new_corner_x / target_W
        updated_label = self._remap_obb_label(
            crack_label_line,
            src_hw      = (ch, cw),
            paste_xy    = (x1, y1),
            paste_hw    = (ch_a, cw_a),
            target_hw   = result.shape[:2],
        )
        updated_labels = target_label_lines + ([updated_label] if updated_label else [])

        return result, updated_labels

    @staticmethod
    def _remap_obb_label(
        label_line: str,
        src_hw:     tuple,   # (H, W) of the source crack crop
        paste_xy:   tuple,   # (x1, y1) top-left paste position in target image
        paste_hw:   tuple,   # (h, w) actual pasted region size (may differ from src due to clipping)
        target_hw:  tuple,   # (H, W) of the target image
    ):
        """
        Remap a YOLO-OBB label line from the source crop coordinate system to
        the target image coordinate system after a paste operation.

        YOLO-OBB format (normalised to the image the label belongs to):
            class_id  x1 y1  x2 y2  x3 y3  x4 y4

        After pasting the source crop at `paste_xy` with clipped size `paste_hw`:
          - Each normalised corner (nx, ny) in the source becomes:
              abs_x = paste_x1 + nx * paste_w          (absolute target pixels)
              abs_y = paste_y1 + ny * paste_h
          - Then renormalise to target image:
              new_nx = abs_x / target_W
              new_ny = abs_y / target_H
        """
        parts = label_line.strip().split()
        if len(parts) < 9:
            return None    # malformed — skip
        class_id = parts[0]
        try:
            raw = list(map(float, parts[1:9]))
        except ValueError:
            return None

        px1, py1 = paste_xy
        ph, pw   = paste_hw
        H, W     = target_hw

        new_coords = []
        for i in range(4):
            nx, ny = raw[i * 2], raw[i * 2 + 1]
            ax = max(0.0, min(W - 1.0, px1 + nx * pw))
            ay = max(0.0, min(H - 1.0, py1 + ny * ph))
            new_coords.extend([ax / W, ay / H])

        coord_str = ' '.join(f'{v:.6f}' for v in new_coords)
        return f"{class_id} {coord_str}"


# ══════════════════════════════════════════════════════════════════════════════
# FULL TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_mining_loop(
    dataset_dir:       str,
    initial_model_path: str,
    held_out_images:   List[str],
    segmenter,
    output_base_dir:   str,
    n_rounds:          int  = 3,
    device:            str  = 'cpu',
    train_epochs:      int  = 50,    # Shorter than initial train; hard negatives are the signal
    train_imgsz:       int  = 1024,
    train_batch:       int  = 8,
    train_lr:          float = 5e-4,  # Lower LR — fine-tuning, not learning from scratch
):
    """
    Full bootstrapped hard-negative mining loop.

    Runs n_rounds of: mine → merge into data → retrain.
    Returns the path to the final (best) model checkpoint.

    Args:
        dataset_dir        : original Roboflow YOLO OBB dataset directory
        initial_model_path : path to v1 model (trained on original data)
        held_out_images    : images NOT in the training set to mine FPs from
        segmenter          : initialised ClassicalSleeperSegmenter (or U-Net)
        output_base_dir    : directory to write mining artifacts and round checkpoints
        n_rounds           : number of mining iterations (2–3 recommended)
        device             : 'cuda' / 0 for GPU, 'cpu' for CPU-only
    """
    miner = BootstrappedFPMiner(
        confidence_threshold=0.30,
        sleeper_overlap_max=0.20,
        max_negatives_per_image=10,
    )

    current_model = initial_model_path

    for r in range(1, n_rounds + 1):
        print(f"\n{'═'*60}")
        print(f"  HARD-NEGATIVE MINING — ROUND {r}/{n_rounds}")
        print(f"{'═'*60}")

        # A. Mine FPs with the current model
        neg_dir = os.path.join(output_base_dir, f"round_{r}", "negatives")
        stats   = miner.mine_round(
            model_path=current_model,
            held_out_images=held_out_images,
            segmenter=segmenter,
            output_dir=neg_dir,
            device=device,
        )

        if stats['negatives_saved'] == 0:
            print("  No new hard negatives mined — FP rate has converged.  Stopping.")
            break

        # B. Build merged dataset: original training data + mined negatives
        merged_dir = os.path.join(output_base_dir, f"round_{r}", "dataset")
        _build_merged_dataset(
            original_dir=dataset_dir,
            negative_dir=neg_dir,
            output_dir=merged_dir,
            round_num=r,
        )

        # C. Fine-tune from the current best model
        model = YOLO(current_model)
        run_dir = os.path.join(output_base_dir, f"round_{r}", "training")
        model.train(
            data=os.path.join(merged_dir, "data.yaml"),
            epochs=train_epochs,
            imgsz=train_imgsz,
            batch=train_batch,
            lr0=train_lr,
            optimizer='AdamW',
            augment=True,
            mosaic=1.0,
            patience=15,
            device=0 if str(device) in ('cuda', '0', 0) else 'cpu',
            project=run_dir,
            name="mining_finetune",
            plots=False,
            save=True,
        )

        current_model = os.path.join(
            run_dir, "mining_finetune", "weights", "best.pt"
        )
        print(f"  Round {r} done → {current_model}")

    print(f"\nMining loop complete.  Final model: {current_model}")
    return current_model


def _build_merged_dataset(
    original_dir: str,
    negative_dir: str,
    output_dir:   str,
    round_num:    int,
):
    """
    Create a new dataset directory = original data + mined hard negatives.
    Writes a data.yaml pointing to the merged directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Copy the full original dataset structure
    for split in ('train', 'valid', 'test'):
        src = os.path.join(original_dir, split)
        dst = os.path.join(output_dir, split)
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # Append hard negatives into the training split
    for subdir in ('images', 'labels'):
        src = os.path.join(negative_dir, subdir)
        dst = os.path.join(output_dir, 'train', subdir)
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # Update data.yaml with absolute paths
    orig_yaml = os.path.join(original_dir, "data.yaml")
    with open(orig_yaml) as f:
        cfg = yaml.safe_load(f)

    cfg['path']  = output_dir
    cfg['train'] = 'train/images'
    cfg['val']   = 'valid/images'
    cfg['test']  = 'test/images'

    with open(os.path.join(output_dir, "data.yaml"), 'w') as f:
        yaml.dump(cfg, f)

    n_neg = len(list(Path(os.path.join(output_dir, 'train', 'images')).glob('*hardneg*')))
    print(f"  [Dataset merge] Round {round_num}: {n_neg} hard negatives added to training set")
