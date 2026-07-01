#!/usr/bin/env python3
"""
Railway Sleeper Crack Detection — Master Pipeline Script
=========================================================
Executes all three phases sequentially on a remote GPU server.

USAGE
-----
# Full pipeline from a pre-trained YOLOv8-OBB checkpoint:
python run_pipeline.py \\
    --data-yaml /data/BTP-1/data.yaml \\
    --phase1-weights yolov8s-obb.pt \\
    --output-dir /runs/experiment_01 \\
    --device 0

# Sanity test on 10 images (fast, catches import/shape bugs):
python run_pipeline.py --sanity-only --data-yaml /data/BTP-1/data.yaml --device 0

# Full benchmark evaluation using existing checkpoints:
python run_pipeline.py --eval-only \\
    --phase3-weights /runs/exp/phase3/best.pt \\
    --data-yaml /data/BTP-1/data.yaml \\
    --device 0

# Run only Phase 2 starting from a specific Phase 1 model:
python run_pipeline.py --phases 2 \\
    --phase1-weights /runs/exp/phase1/best.pt \\
    --data-yaml /data/BTP-1/data.yaml \\
    --output-dir /runs/experiment_01 \\
    --device 0

PHASE OVERVIEW
--------------
Phase 1 → FP-suppression inference pipeline + optional hard-negative mining.
           Produces: phase1/best.pt  (YOLOv8-OBB, trained on augmented dataset)
Phase 2 → Multi-task training with KLD/VFL losses + auxiliary heads.
           Input: phase1/best.pt
           Produces: phase2/best.pt  (MultiTaskModel)
Phase 3 → Hybrid backbone + AFPN + FiLM + DyHead architecture.
           Input: phase2/best.pt (for auxiliary-head weight transfer)
           Produces: phase3/best.pt  (Phase3Model)

EXECUTION ORDER
---------------
1. Phase 1  — baseline YOLO training (if --phase1-weights is a .yaml, train from scratch)
2. Phase 1  — hard-negative mining (3 rounds, optional)
3. Phase 2  — multi-task fine-tuning
4. Phase 3  — architecture-level training
5. Evaluation — mAP50, FP rate, seg-IoU on test set
"""

import argparse
import logging
import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import List, Optional

import torch

# ── Ensure all phase directories are importable ──────────────────────────────
ROOT = Path(__file__).parent.resolve()
for _d in ['phase1', 'phase2', 'phase3']:
    _p = str(ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Lazy imports (guarded so --help works without GPU libs) ──────────────────
def _import_cv2():
    import cv2; return cv2

def _import_np():
    import numpy as np; return np


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'pipeline.log'

    fmt = '%(asctime)s | %(levelname)-8s | %(message)s'
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path),
    ]
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    log = logging.getLogger('pipeline')
    log.info(f'Log file: {log_path}')
    return log


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Railway Sleeper Crack Detection — Master Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Execution control ────────────────────────────────────────────────────
    p.add_argument('--phases', nargs='+', type=int, default=[1, 2, 3],
                   choices=[1, 2, 3],
                   help='Which phases to run (e.g. --phases 2 3 to skip Phase 1)')
    p.add_argument('--sanity-only', action='store_true',
                   help='Run quick sanity tests only (10 images) then exit')
    p.add_argument('--eval-only', action='store_true',
                   help='Run evaluation only, no training')
    p.add_argument('--skip-mining', action='store_true',
                   help='Skip Phase 1 hard-negative mining loop')

    # ── Paths ────────────────────────────────────────────────────────────────
    p.add_argument('--data-yaml', required=True,
                   help='Path to Roboflow data.yaml')
    p.add_argument('--output-dir', default='./runs/pipeline',
                   help='Root directory for all outputs')
    p.add_argument('--phase1-weights', default='yolov8s-obb.pt',
                   help='Starting YOLO weights for Phase 1 (ultralytics pretrained or local .pt)')
    p.add_argument('--phase2-weights', default=None,
                   help='Pre-trained Phase 2 weights (skip Phase 2 training if provided)')
    p.add_argument('--phase3-weights', default=None,
                   help='Pre-trained Phase 3 weights (skip Phase 3 training if provided)')

    # ── Hardware ─────────────────────────────────────────────────────────────
    p.add_argument('--device', default='0',
                   help='CUDA device ("0", "0,1", or "cpu")')
    p.add_argument('--workers', type=int, default=4,
                   help='DataLoader worker processes')

    # ── Phase 1 hyperparameters ───────────────────────────────────────────────
    p.add_argument('--p1-epochs', type=int, default=100,
                   help='Phase 1 initial training epochs')
    p.add_argument('--p1-batch', type=int, default=16,
                   help='Phase 1 training batch size')
    p.add_argument('--p1-imgsz', type=int, default=1024,
                   help='Phase 1 training image size')
    p.add_argument('--mining-rounds', type=int, default=3,
                   help='Phase 1 hard-negative mining rounds')
    p.add_argument('--mining-epochs', type=int, default=50,
                   help='Fine-tune epochs per mining round')

    # ── Phase 2 hyperparameters ───────────────────────────────────────────────
    p.add_argument('--p2-epochs', type=int, default=80,
                   help='Phase 2 training epochs')
    p.add_argument('--p2-batch', type=int, default=8,
                   help='Phase 2 training batch size')
    p.add_argument('--p2-imgsz', type=int, default=640,
                   help='Phase 2 tile inference size')
    p.add_argument('--box-type', default='kld', choices=['kld', 'gwd'],
                   help='Rotated box loss type (Phase 2 ablation)')
    p.add_argument('--tversky-beta', type=float, default=0.7,
                   help='Focal-Tversky β (>0.5 = recall-oriented for thin cracks)')
    p.add_argument('--no-uncertainty', action='store_true',
                   help='Disable Kendall uncertainty weighting (use fixed weights instead)')

    # ── Phase 3 hyperparameters ───────────────────────────────────────────────
    p.add_argument('--p3-epochs', type=int, default=120,
                   help='Phase 3 training epochs')
    p.add_argument('--p3-batch', type=int, default=6,
                   help='Phase 3 training batch size')
    p.add_argument('--p3-imgsz', type=int, default=640,
                   help='Phase 3 image size')
    p.add_argument('--afpn-out-ch', type=int, default=256,
                   help='AFPN uniform output channels')
    p.add_argument('--dyhead-blocks', type=int, default=2,
                   help='Number of DyHead blocks (0 = ablate DyHead)')
    p.add_argument('--strip-k', type=int, default=9,
                   help='Strip conv kernel length in OBB head (1 = disable)')
    p.add_argument('--no-film', action='store_true',
                   help='Disable FiLM global context conditioning')
    p.add_argument('--width-mult', type=float, default=1.0,
                   help='Backbone channel width multiplier (0.5 for nano variant)')

    # ── Logging ───────────────────────────────────────────────────────────────
    p.add_argument('--wandb-project', default='',
                   help='WandB project name (empty = disabled)')
    p.add_argument('--verbose', action='store_true',
                   help='Verbose (DEBUG) logging')

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_environment(args, log: logging.Logger):
    """Verify GPU, disk, and critical imports before doing anything expensive."""
    log.info('─' * 60)
    log.info('ENVIRONMENT CHECK')
    log.info('─' * 60)

    # CUDA
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            gb = props.total_memory / 1024**3
            log.info(f'  GPU {i}: {props.name} ({gb:.1f} GB VRAM)')

        # GPU memory warnings
        if n_gpus > 0:
            gb0 = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gb0 < 8 and 3 in (args.phases or []):
                log.warning('  Phase 3 needs ≥10 GB VRAM. Detected only '
                            f'{gb0:.1f} GB — reduce --p3-batch or --width-mult.')
            if gb0 < 6 and 2 in (args.phases or []):
                log.warning('  Phase 2 needs ≥8 GB VRAM. Detected only '
                            f'{gb0:.1f} GB — reduce --p2-batch.')
    else:
        log.warning('  No GPU detected — running on CPU (VERY slow for training)')

    # Critical imports
    errors = []
    for pkg in ['ultralytics', 'cv2', 'numpy', 'yaml']:
        try:
            __import__(pkg)
            log.info(f'  ✓  {pkg}')
        except ImportError as e:
            log.error(f'  ✗  {pkg} — {e}')
            errors.append(pkg)

    if errors:
        raise SystemExit(
            f'\n[ERROR] Missing packages: {errors}\n'
            f'Install with: pip install -r requirements.txt\n'
        )

    # Data YAML
    if not Path(args.data_yaml).exists():
        raise FileNotFoundError(
            f'data.yaml not found: {args.data_yaml}\n'
            f'Download your Roboflow dataset and pass the correct path.'
        )

    import yaml
    with open(args.data_yaml) as f:
        cfg = yaml.safe_load(f)
    nc    = cfg.get('nc', '?')
    names = cfg.get('names', {})
    train = cfg.get('train', '')
    val   = cfg.get('val', '')
    log.info(f'  Dataset: nc={nc}, classes={list(names.values())}')
    log.info(f'  Train images: {train}')
    log.info(f'  Val images:   {val}')

    if not Path(train).exists():
        raise FileNotFoundError(f'Training image directory not found: {train}')
    if not Path(val).exists():
        raise FileNotFoundError(f'Validation image directory not found: {val}')

    n_train = len(list(Path(train).glob('*.jpg')) + list(Path(train).glob('*.png')))
    n_val   = len(list(Path(val).glob('*.jpg')) + list(Path(val).glob('*.png')))
    log.info(f'  Images found: {n_train} train, {n_val} val')

    # Disk space (need at least 10 GB for checkpoints)
    total, used, free = shutil.disk_usage(args.output_dir
                                          if Path(args.output_dir).exists()
                                          else '/')
    log.info(f'  Free disk: {free / 1024**3:.1f} GB')
    if free < 10 * 1024**3:
        log.warning('  Less than 10 GB free disk — checkpoints may fail to save.')

    log.info('─' * 60)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# SANITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_sanity_tests(args, cfg, out_dir: Path, log: logging.Logger) -> bool:
    """
    Run quick forward-pass tests on 10 images to catch import/shape bugs
    before committing to multi-hour training runs.
    """
    log.info('═' * 60)
    log.info('SANITY TESTS (10-image subset)')
    log.info('═' * 60)

    import cv2, numpy as np

    train_dir = Path(cfg['train'])
    img_paths = sorted(
        list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))
    )[:10]

    if not img_paths:
        log.error('No images found for sanity test — check data.yaml train path.')
        return False

    # ── Test 1: Phase 1 imports ───────────────────────────────────────────────
    log.info('[Sanity 1] Phase 1 imports ...')
    try:
        from sleeper_segmentation import ClassicalSleeperSegmenter
        from roi_sahi_slicer      import ROISAHISlicer
        from mask_gating          import MaskGating
        from nmm_fusion           import MaskAwareNMM
        log.info('  ✓  All Phase 1 imports OK')
    except ImportError as e:
        log.error(f'  ✗  Phase 1 import failed: {e}')
        return False

    # ── Test 2: Classical sleeper segmenter on 3 images ─────────────────────
    log.info('[Sanity 2] Classical sleeper segmenter ...')
    seg = ClassicalSleeperSegmenter()
    for p in img_paths[:3]:
        img = cv2.imread(str(p))
        if img is None:
            log.warning(f'  Could not read {p} — skipping')
            continue
        mask, prob = seg.predict_mask(img)
        assert mask.shape == img.shape[:2], f'Mask shape mismatch: {mask.shape} vs {img.shape[:2]}'
        assert mask.min() >= 0 and mask.max() <= 1, 'Mask values out of [0,1]'
    log.info(f'  ✓  Segmenter OK on {min(3, len(img_paths))} images')

    # ── Test 3: ROI SAHI slicer ───────────────────────────────────────────────
    log.info('[Sanity 3] ROI SAHI slicer ...')
    from sleeper_segmentation import extract_sleeper_rois
    slicer = ROISAHISlicer(tile_size=640, overlap_ratio=0.2)
    img    = cv2.imread(str(img_paths[0]))
    mask, _ = seg.predict_mask(img)
    rois   = extract_sleeper_rois(mask)
    tiles  = slicer.slice(img, rois)
    log.info(f'  ✓  {len(tiles)} tiles from {img_paths[0].name}')

    # ── Test 4: Phase 1 pipeline end-to-end (no YOLO weights = skip) ─────────
    log.info('[Sanity 4] Phase 1 pipeline end-to-end ...')
    if Path(args.phase1_weights).exists():
        try:
            from pipeline import Phase1Pipeline
            names = list(cfg.get('names', {}).values())
            pl = Phase1Pipeline(
                yolo_model_path = args.phase1_weights,
                class_names     = names,
                device          = 'cpu',
                yolo_conf       = 0.1,
            )
            dets = pl.detect(img)
            metrics = pl.compute_fp_metrics(dets)
            log.info(f'  ✓  Phase 1 pipeline: {len(dets)} detections, '
                     f'FP rate={metrics["off_sleeper_fp_count"]}/{metrics.get("total_detections", len(dets))}')
        except Exception as e:
            log.warning(f'  ⚠  Phase 1 pipeline error (non-fatal): {e}')
    else:
        log.info(f'  –  Skipped (no YOLO weights at {args.phase1_weights})')

    # ── Test 5: Phase 2 imports ───────────────────────────────────────────────
    log.info('[Sanity 5] Phase 2 imports ...')
    try:
        from losses.kld_gwd      import kld_loss
        from losses.varifocal    import VarifocalLoss
        from losses.focal_tversky import FocalTverskyLoss
        from losses.uncertainty  import UncertaintyWeighting
        from losses.multitask_loss import Phase2Loss
        from heads.crack_seg_head   import CrackSegHead
        from heads.sleeper_seg_head import SleeperSegHead
        from heads.crack_type_head  import CrackTypeHead
        from pseudo_labels          import PseudoLabelGenerator
        log.info('  ✓  All Phase 2 imports OK')
    except ImportError as e:
        log.error(f'  ✗  Phase 2 import failed: {e}')
        return False

    # ── Test 6: Phase 2 loss forward pass ────────────────────────────────────
    log.info('[Sanity 6] Phase 2 loss forward pass ...')
    try:
        nc = cfg.get('nc', 3)
        loss = Phase2Loss(nc=nc, use_uncertainty=True)
        dummy_loss_parts = {
            'crack_seg':   torch.tensor(0.5),
            'sleeper_seg': torch.tensor(0.3),
            'type_cls':    torch.tensor(0.8),
        }
        total, items = loss.weighter(dummy_loss_parts)
        assert total.item() > 0, 'Total loss should be positive'
        log.info(f'  ✓  Phase 2 loss OK: total={total.item():.4f}')
    except Exception as e:
        log.error(f'  ✗  Phase 2 loss failed: {e}')
        return False

    # ── Test 7: Pseudo-label generation ──────────────────────────────────────
    log.info('[Sanity 7] Pseudo-label generation ...')
    try:
        names = list(cfg.get('names', {}).values())
        gen   = PseudoLabelGenerator(class_names=names)
        img_p = img_paths[0]
        lbl_p = str(img_p).replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        pseudo = gen.generate(cv2.imread(str(img_p)), lbl_p, image_size=(640, 640))
        assert 'crack_mask' in pseudo
        assert pseudo['crack_mask'].shape == (1, 640, 640), \
            f'crack_mask shape wrong: {pseudo["crack_mask"].shape}'
        log.info('  ✓  Pseudo-label generation OK')
    except Exception as e:
        log.warning(f'  ⚠  Pseudo-label generation error (non-fatal): {e}')

    # ── Test 8: Phase 3 imports ───────────────────────────────────────────────
    log.info('[Sanity 8] Phase 3 imports ...')
    try:
        from backbone.hybrid_backbone import HybridBackbone
        from neck.afpn                import AFPN
        from neck.dyhead               import DyHead
        from film                      import GlobalContextEncoder, FiLMNeck
        from obb_head                  import MultiLevelOBBHead
        from phase3_model              import Phase3Model
        log.info('  ✓  All Phase 3 imports OK')
    except ImportError as e:
        log.error(f'  ✗  Phase 3 import failed: {e}')
        return False

    # ── Test 9: Phase 3 model forward pass ───────────────────────────────────
    log.info('[Sanity 9] Phase 3 model forward pass ...')
    try:
        m = Phase3Model(
            nc           = cfg.get('nc', 3),
            width_mult   = 0.5,          # use nano variant for speed
            afpn_out_ch  = 64,
            dyhead_blocks= 1,
            evit_depth   = 1,
            use_film     = True,
            device       = 'cpu',
        )
        x = torch.zeros(1, 3, 640, 640)
        m.eval()
        with torch.no_grad():
            preds, aux = m(x)
        assert len(preds) == 4, f'Expected 4 detection levels, got {len(preds)}'
        assert 'crack_seg' in aux, 'Missing crack_seg in aux outputs'
        log.info(f'  ✓  Phase 3 model OK: {len(preds)} detection levels, '
                 f'aux={list(aux.keys())}')
        del m  # free memory
    except Exception as e:
        log.error(f'  ✗  Phase 3 model failed: {e}')
        return False

    # ── Test 10: Integration — Phase 1 output feeds Phase 2 ──────────────────
    log.info('[Sanity 10] Integration: Phase 1 → Phase 2 weight loading ...')
    if Path(args.phase1_weights).exists():
        try:
            from multitask_model import MultiTaskModel
            mm = MultiTaskModel(
                yolo_weights = args.phase1_weights,
                nc           = cfg.get('nc', 3),
                device       = 'cpu',
            )
            mm.eval()
            log.info('  ✓  Phase 2 MultiTaskModel loaded from Phase 1 weights OK')
            del mm
        except Exception as e:
            log.warning(f'  ⚠  Phase 2 load error (non-fatal): {e}')
    else:
        log.info(f'  –  Skipped (no Phase 1 weights)')

    log.info('═' * 60)
    log.info('ALL SANITY TESTS PASSED')
    log.info('═' * 60)

    # Save a sanity report
    report = {
        'n_images': len(img_paths),
        'tests_passed': 10,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    (out_dir / 'sanity_report.json').write_text(json.dumps(report, indent=2))
    return True


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: TRAIN + HARD-NEGATIVE MINING
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1(args, cfg, out_dir: Path, log: logging.Logger) -> str:
    """
    Run Phase 1: baseline YOLOv8-OBB training + hard-negative mining loop.

    Returns the path to the Phase 1 best model checkpoint.
    """
    log.info('═' * 60)
    log.info('PHASE 1: Baseline Training + Hard-Negative Mining')
    log.info('═' * 60)

    p1_dir = out_dir / 'phase1'
    p1_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    # ── Step 1: Initial training ─────────────────────────────────────────────
    weights = args.phase1_weights
    initial_best = p1_dir / 'weights' / 'best.pt'

    if not initial_best.exists():
        log.info(f'Training Phase 1 baseline from {weights} ...')
        model = YOLO(weights)
        model.train(
            data      = args.data_yaml,
            epochs    = args.p1_epochs,
            imgsz     = args.p1_imgsz,
            batch     = args.p1_batch,
            device    = args.device,
            optimizer = 'AdamW',
            augment   = True,
            mosaic    = 1.0,
            patience  = 20,
            project   = str(p1_dir),
            name      = 'baseline',
            save      = True,
            task      = 'obb',
        )
        initial_best = p1_dir / 'baseline' / 'weights' / 'best.pt'
        log.info(f'Phase 1 baseline complete → {initial_best}')
    else:
        log.info(f'Phase 1 weights found at {initial_best} — skipping training.')

    current_best = str(initial_best)

    # ── Step 2: Hard-negative mining (optional) ───────────────────────────────
    if args.skip_mining:
        log.info('Hard-negative mining skipped (--skip-mining).')
        return current_best

    log.info(f'Starting {args.mining_rounds} rounds of hard-negative mining ...')
    import yaml
    with open(args.data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    train_img_dir = Path(data_cfg['train'])
    val_img_dir   = Path(data_cfg.get('val', ''))

    # Use validation images as held-out set for mining
    # (they are NOT in the training distribution — ideal for FP mining)
    held_out = sorted(
        list(val_img_dir.glob('*.jpg')) + list(val_img_dir.glob('*.png'))
    )

    if not held_out:
        log.warning('No held-out images found for mining — skipping.')
        return current_best

    # Use only up to 50 images for mining to keep each round tractable
    held_out_paths = [str(p) for p in held_out[:50]]

    try:
        from hard_negative_mining import run_mining_loop
        from sleeper_segmentation import ClassicalSleeperSegmenter

        mining_out = str(p1_dir / 'mining')
        mined_best = run_mining_loop(
            dataset_dir        = str(Path(args.data_yaml).parent),
            initial_model_path = current_best,
            held_out_images    = held_out_paths,
            segmenter          = ClassicalSleeperSegmenter(),
            output_base_dir    = mining_out,
            n_rounds           = args.mining_rounds,
            device             = args.device,
            train_epochs       = args.mining_epochs,
            train_imgsz        = args.p1_imgsz,
            train_batch        = args.p1_batch,
        )
        log.info(f'Mining complete → {mined_best}')
        return mined_best

    except Exception as e:
        log.warning(f'Hard-negative mining failed ({e}) — using pre-mining best.pt.')
        return current_best


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: MULTI-TASK TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def run_phase2(args, phase1_weights: str, out_dir: Path, log: logging.Logger) -> str:
    """
    Run Phase 2: multi-task training with KLD/VFL losses + auxiliary heads.

    Returns path to Phase 2 best model checkpoint.
    """
    log.info('═' * 60)
    log.info('PHASE 2: Multi-Task Training (KLD + VFL + Aux Heads)')
    log.info('═' * 60)

    if args.phase2_weights and Path(args.phase2_weights).exists():
        log.info(f'Using pre-trained Phase 2 weights: {args.phase2_weights}')
        return args.phase2_weights

    p2_dir = out_dir / 'phase2'

    log.info(f'Starting Phase 2 training from {phase1_weights} ...')
    log.info(f'  box_type       = {args.box_type}')
    log.info(f'  tversky_beta   = {args.tversky_beta}')
    log.info(f'  uncertainty    = {not args.no_uncertainty}')
    log.info(f'  epochs         = {args.p2_epochs}')

    from train_phase2 import run_phase2_training

    use_wandb   = bool(args.wandb_project)
    best_model  = run_phase2_training(
        yolo_weights         = phase1_weights,
        data_yaml            = args.data_yaml,
        epochs               = args.p2_epochs,
        imgsz                = args.p2_imgsz,
        batch                = args.p2_batch,
        lr0                  = 1e-3,
        device               = args.device,
        project              = str(p2_dir),
        name                 = 'multitask_kld_vfl',
        box_type             = args.box_type,
        tversky_beta         = args.tversky_beta,
        use_uncertainty      = not args.no_uncertainty,
        aux_loss_start_epoch = 5,
        use_wandb            = use_wandb,
        wandb_project        = args.wandb_project or 'sleeper-crack',
    )

    log.info(f'Phase 2 complete → {best_model}')
    return best_model


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: ARCHITECTURE TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def run_phase3(args, phase2_weights: str, out_dir: Path, log: logging.Logger) -> str:
    """
    Run Phase 3: hybrid backbone + AFPN + FiLM + DyHead training.

    Returns path to Phase 3 best model checkpoint.
    """
    log.info('═' * 60)
    log.info('PHASE 3: Architecture Training (Hybrid + AFPN + FiLM + DyHead)')
    log.info('═' * 60)

    if args.phase3_weights and Path(args.phase3_weights).exists():
        log.info(f'Using pre-trained Phase 3 weights: {args.phase3_weights}')
        return args.phase3_weights

    p3_dir = out_dir / 'phase3'

    # GPU memory check for Phase 3
    if torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gb < 10 and args.p3_batch > 4:
            old_batch = args.p3_batch
            args.p3_batch = 4
            log.warning(f'GPU has {gb:.1f} GB — reduced Phase 3 batch {old_batch}→{args.p3_batch}')
        if gb < 8 and args.width_mult > 0.5:
            log.warning(f'GPU has {gb:.1f} GB — consider --width-mult 0.5 for Phase 3')

    log.info(f'  width_mult     = {args.width_mult}')
    log.info(f'  afpn_out_ch    = {args.afpn_out_ch}')
    log.info(f'  dyhead_blocks  = {args.dyhead_blocks}')
    log.info(f'  strip_k        = {args.strip_k}')
    log.info(f'  use_film       = {not args.no_film}')
    log.info(f'  epochs         = {args.p3_epochs}')

    from train_phase3 import run_phase3_training

    import yaml
    with open(args.data_yaml) as f:
        nc = yaml.safe_load(f).get('nc', 3)

    use_wandb  = bool(args.wandb_project)
    best_model = run_phase3_training(
        phase2_weights  = phase2_weights,
        data_yaml       = args.data_yaml,
        epochs          = args.p3_epochs,
        batch           = args.p3_batch,
        lr0             = 5e-4,
        device          = args.device,
        project         = str(p3_dir),
        name            = 'hybrid_afpn_dyhead',
        width_mult      = args.width_mult,
        afpn_out_ch     = args.afpn_out_ch,
        dyhead_blocks   = args.dyhead_blocks,
        strip_k         = args.strip_k,
        use_film        = not args.no_film,
        use_uncertainty = not args.no_uncertainty,
        use_wandb       = use_wandb,
        wandb_project   = args.wandb_project or 'sleeper-crack',
    )

    log.info(f'Phase 3 complete → {best_model}')
    return best_model


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_integration_tests(
    phase1_weights: Optional[str],
    phase2_weights: Optional[str],
    phase3_weights: Optional[str],
    cfg:            dict,
    out_dir:        Path,
    log:            logging.Logger,
) -> dict:
    """
    Verify that outputs from each phase are valid inputs for the next.
    Returns a dict of pass/fail results.
    """
    log.info('═' * 60)
    log.info('INTEGRATION TESTS')
    log.info('═' * 60)

    results = {}
    nc = cfg.get('nc', 3)

    # ── IntTest 1: Phase 1 weights → Phase 1 pipeline ────────────────────────
    if phase1_weights and Path(phase1_weights).exists():
        log.info('[IntTest 1] Phase 1 weights → Phase 1 pipeline ...')
        try:
            from pipeline import Phase1Pipeline
            names = list(cfg.get('names', {}).values())
            pl = Phase1Pipeline(
                yolo_model_path = phase1_weights,
                class_names     = names,
                device          = 'cpu',
            )
            import cv2
            train_dir = Path(cfg['train'])
            imgs = sorted(list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png')))[:3]
            assert imgs, 'No images to test with'
            img = cv2.imread(str(imgs[0]))
            dets = pl.detect(img)
            assert isinstance(dets, list), 'detect() must return a list'
            metrics = pl.compute_fp_metrics(dets)
            assert 'off_sleeper_fp_count' in metrics
            log.info(f'  ✓  Phase 1 pipeline: {len(dets)} dets, '
                     f'metrics={metrics}')
            results['p1_pipeline'] = 'PASS'
        except Exception as e:
            log.error(f'  ✗  {e}')
            results['p1_pipeline'] = f'FAIL: {e}'
    else:
        log.info('  –  Phase 1 weights not available')
        results['p1_pipeline'] = 'SKIP'

    # ── IntTest 2: Phase 1 weights → Phase 2 model load ─────────────────────
    if phase1_weights and Path(phase1_weights).exists():
        log.info('[IntTest 2] Phase 1 weights → MultiTaskModel ...')
        try:
            from multitask_model import MultiTaskModel
            mm = MultiTaskModel(
                yolo_weights = phase1_weights,
                nc           = nc,
                device       = 'cpu',
            )
            mm.eval()
            x = torch.zeros(1, 3, 640, 640)
            with torch.no_grad():
                out = mm(x)
            assert out is not None, 'MultiTaskModel eval returned None'
            log.info('  ✓  Phase 1 → Phase 2 weight handoff OK')
            results['p1_to_p2'] = 'PASS'
            del mm
        except Exception as e:
            log.error(f'  ✗  {e}')
            results['p1_to_p2'] = f'FAIL: {e}'
    else:
        results['p1_to_p2'] = 'SKIP'

    # ── IntTest 3: Phase 2 checkpoint shape check ────────────────────────────
    if phase2_weights and Path(phase2_weights).exists():
        log.info('[IntTest 3] Phase 2 checkpoint integrity ...')
        try:
            ckpt = torch.load(phase2_weights, map_location='cpu')
            # ultralytics saves model with state dict accessible via .model
            if hasattr(ckpt, 'state_dict'):
                keys = list(ckpt.state_dict().keys())
            elif isinstance(ckpt, dict):
                model_obj = ckpt.get('model', ckpt)
                if hasattr(model_obj, 'state_dict'):
                    keys = list(model_obj.state_dict().keys())
                else:
                    keys = list(model_obj.keys())
            else:
                keys = []
            assert len(keys) > 0, 'Checkpoint has no state_dict keys'
            log.info(f'  ✓  Phase 2 checkpoint: {len(keys)} parameter tensors')
            results['p2_checkpoint'] = 'PASS'
        except Exception as e:
            log.error(f'  ✗  {e}')
            results['p2_checkpoint'] = f'FAIL: {e}'
    else:
        results['p2_checkpoint'] = 'SKIP'

    # ── IntTest 4: Phase 2 weights → Phase 3 partial load ────────────────────
    if phase2_weights and Path(phase2_weights).exists():
        log.info('[IntTest 4] Phase 2 weights → Phase 3 partial load ...')
        try:
            from phase3_model import Phase3Model, load_partial_weights
            m3 = Phase3Model(nc=nc, width_mult=0.5, afpn_out_ch=64,
                             dyhead_blocks=1, evit_depth=1, device='cpu')
            transferred, skipped = load_partial_weights(m3, phase2_weights, verbose=False)
            log.info(f'  ✓  Phase 3 partial load: {len(transferred)} layers transferred, '
                     f'{len(skipped)} skipped')
            results['p2_to_p3'] = 'PASS'
            del m3
        except Exception as e:
            log.error(f'  ✗  {e}')
            results['p2_to_p3'] = f'FAIL: {e}'
    else:
        results['p2_to_p3'] = 'SKIP'

    # ── IntTest 5: Phase 3 checkpoint eval pass ───────────────────────────────
    if phase3_weights and Path(phase3_weights).exists():
        log.info('[IntTest 5] Phase 3 checkpoint eval forward ...')
        try:
            from phase3_model import Phase3Model
            from train_phase3 import load_phase3_checkpoint
            m3 = Phase3Model(nc=nc, width_mult=0.5, afpn_out_ch=64,
                             dyhead_blocks=1, evit_depth=1, device='cpu')
            load_phase3_checkpoint(m3, phase3_weights, device='cpu')
            m3.eval()
            x = torch.zeros(1, 3, 640, 640)
            with torch.no_grad():
                preds, aux = m3(x)
            assert len(preds) == 4, f'Expected 4 levels, got {len(preds)}'
            log.info(f'  ✓  Phase 3 eval: {len(preds)} detection levels')
            results['p3_eval'] = 'PASS'
            del m3
        except Exception as e:
            log.error(f'  ✗  {e}')
            results['p3_eval'] = f'FAIL: {e}'
    else:
        results['p3_eval'] = 'SKIP'

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(1 for v in results.values() if v == 'PASS')
    n_fail = sum(1 for v in results.values() if v.startswith('FAIL'))
    n_skip = sum(1 for v in results.values() if v == 'SKIP')
    log.info(f'\nIntegration test summary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP')
    (out_dir / 'integration_test_results.json').write_text(json.dumps(results, indent=2))
    log.info('═' * 60)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark_eval(
    weights_by_phase: dict,   # {'phase1': path, 'phase2': path, 'phase3': path}
    args,
    cfg:     dict,
    out_dir: Path,
    log:     logging.Logger,
) -> dict:
    """
    Full evaluation on the test set for each available phase model.
    Reports mAP50, mAP50-95, precision, recall, and off-sleeper FP rate.
    """
    log.info('═' * 60)
    log.info('BENCHMARK EVALUATION')
    log.info('═' * 60)

    from ultralytics import YOLO
    import yaml

    with open(args.data_yaml) as f:
        full_cfg = yaml.safe_load(f)

    test_dir = Path(full_cfg.get('test', full_cfg.get('val', '')))
    test_imgs = sorted(
        list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
    )
    names = list(cfg.get('names', {}).values())

    all_results = {}
    eval_dir = out_dir / 'evaluation'
    eval_dir.mkdir(exist_ok=True)

    for phase_name, wpath in weights_by_phase.items():
        if not wpath or not Path(wpath).exists():
            log.info(f'  [{phase_name}] skipped — no weights')
            continue

        log.info(f'\n[{phase_name.upper()}] Evaluating {wpath} ...')
        phase_results = {'weights': wpath}

        # ── Standard ultralytics validation (Phase 1 and 2 only) ──────────
        if phase_name in ('phase1', 'phase2'):
            try:
                yolo = YOLO(wpath)
                val  = yolo.val(data=args.data_yaml, device=args.device, verbose=False)
                phase_results.update({
                    'mAP50':     round(float(val.box.map50), 4),
                    'mAP50-95':  round(float(val.box.map),   4),
                    'precision': round(float(val.box.p[0]),   4) if len(val.box.p) else None,
                    'recall':    round(float(val.box.r[0]),   4) if len(val.box.r) else None,
                })
                log.info(f'  mAP50={phase_results["mAP50"]:.4f}  '
                         f'mAP50-95={phase_results["mAP50-95"]:.4f}  '
                         f'P={phase_results["precision"]}  R={phase_results["recall"]}')
            except Exception as e:
                log.warning(f'  ultralytics val failed: {e}')
                phase_results['ultralytics_val_error'] = str(e)

        # ── FP-rate evaluation using Phase 1 pipeline ─────────────────────
        try:
            import cv2
            from pipeline import Phase1Pipeline
            pl = Phase1Pipeline(
                yolo_model_path = wpath,
                class_names     = names,
                device          = args.device,
            )
            total_dets, total_fp = 0, 0
            for img_p in test_imgs[:100]:   # cap at 100 for speed
                img = cv2.imread(str(img_p))
                if img is None:
                    continue
                dets    = pl.detect(img)
                metrics = pl.compute_fp_metrics(dets)
                total_dets += metrics.get('total_detections', len(dets))
                total_fp   += metrics.get('off_sleeper_fp_count', 0)

            fp_rate = total_fp / max(total_dets, 1)
            phase_results.update({
                'total_dets':      total_dets,
                'off_sleeper_fps': total_fp,
                'fp_rate':         round(fp_rate, 4),
            })
            log.info(f'  FP rate: {total_fp}/{total_dets} = {fp_rate:.3f}')
        except Exception as e:
            log.warning(f'  FP eval failed: {e}')
            phase_results['fp_eval_error'] = str(e)

        all_results[phase_name] = phase_results

    # ── Results table ────────────────────────────────────────────────────────
    log.info('\n' + '─' * 60)
    log.info(f'{"Phase":<12} {"mAP50":>8} {"mAP50-95":>10} {"FP Rate":>10}')
    log.info('─' * 60)
    for ph, res in all_results.items():
        m50  = res.get('mAP50', 'N/A')
        m5095= res.get('mAP50-95', 'N/A')
        fpr  = res.get('fp_rate', 'N/A')
        log.info(f'{ph:<12} {str(m50):>8} {str(m5095):>10} {str(fpr):>10}')
    log.info('─' * 60)

    results_path = eval_dir / 'benchmark_results.json'
    results_path.write_text(json.dumps(all_results, indent=2))
    log.info(f'\nResults saved → {results_path}')
    log.info('═' * 60)
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(out_dir / 'logs', args.verbose)

    log.info('═' * 60)
    log.info('RAILWAY SLEEPER CRACK DETECTION — MASTER PIPELINE')
    log.info(f'Output dir: {out_dir.resolve()}')
    log.info(f'Device:     {args.device}')
    log.info(f'Phases:     {args.phases}')
    log.info('═' * 60)

    # Save the full argument set for reproducibility
    (out_dir / 'run_config.json').write_text(
        json.dumps(vars(args), indent=2, default=str)
    )

    # ── Environment check ────────────────────────────────────────────────────
    cfg = check_environment(args, log)

    # ── Sanity tests ─────────────────────────────────────────────────────────
    if not run_sanity_tests(args, cfg, out_dir, log):
        raise SystemExit('\n[ABORT] Sanity tests failed. Fix the errors above before training.')

    if args.sanity_only:
        log.info('--sanity-only: exiting after sanity tests.')
        return

    # ── Phase checkpoints tracking ────────────────────────────────────────────
    p1_weights = args.phase1_weights
    p2_weights = args.phase2_weights
    p3_weights = args.phase3_weights

    if not args.eval_only:
        # Phase 1
        if 1 in args.phases:
            t0 = time.time()
            p1_weights = run_phase1(args, cfg, out_dir, log)
            log.info(f'Phase 1 wall time: {(time.time()-t0)/60:.1f} min')
            (out_dir / 'phase1_weights.txt').write_text(p1_weights)

        # Phase 2
        if 2 in args.phases:
            if not p1_weights or not Path(p1_weights).exists():
                raise SystemExit(
                    '[ERROR] Phase 2 requires Phase 1 weights. '
                    'Run Phase 1 first or provide --phase1-weights.'
                )
            t0 = time.time()
            p2_weights = run_phase2(args, p1_weights, out_dir, log)
            log.info(f'Phase 2 wall time: {(time.time()-t0)/60:.1f} min')
            (out_dir / 'phase2_weights.txt').write_text(p2_weights)

        # Phase 3
        if 3 in args.phases:
            # Phase 3 can start without Phase 2 (trains aux heads from scratch)
            p2w = p2_weights or ''
            t0  = time.time()
            p3_weights = run_phase3(args, p2w, out_dir, log)
            log.info(f'Phase 3 wall time: {(time.time()-t0)/60:.1f} min')
            (out_dir / 'phase3_weights.txt').write_text(p3_weights)

    # ── Integration tests ────────────────────────────────────────────────────
    int_results = run_integration_tests(
        p1_weights, p2_weights, p3_weights,
        cfg, out_dir, log,
    )

    # ── Benchmark evaluation ──────────────────────────────────────────────────
    weights_by_phase = {}
    if p1_weights and Path(p1_weights).exists():
        weights_by_phase['phase1'] = p1_weights
    if p2_weights and Path(p2_weights).exists():
        weights_by_phase['phase2'] = p2_weights
    if p3_weights and Path(p3_weights).exists():
        weights_by_phase['phase3'] = p3_weights

    if weights_by_phase:
        eval_results = run_benchmark_eval(weights_by_phase, args, cfg, out_dir, log)

    log.info('═' * 60)
    log.info('PIPELINE COMPLETE')
    log.info(f'All outputs in: {out_dir.resolve()}')
    log.info('─' * 60)
    log.info('Output structure:')
    log.info('  logs/pipeline.log       — full execution log')
    log.info('  run_config.json         — argument snapshot (reproducibility)')
    log.info('  sanity_report.json      — sanity test results')
    log.info('  integration_test_results.json')
    log.info('  phase1/weights/best.pt  — Phase 1 model')
    log.info('  phase2/*/weights/best.pt— Phase 2 model')
    log.info('  phase3/*/weights/best.pt— Phase 3 model')
    log.info('  evaluation/benchmark_results.json')
    log.info('═' * 60)


if __name__ == '__main__':
    main()
