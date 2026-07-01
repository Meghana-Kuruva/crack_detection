"""
Phase 2 — Training Script (Colab-ready)
=========================================
Run Phase 2 multi-task training starting from a Phase 1 best.pt checkpoint.

Copy this entire phase2/ folder to /content/phase2/ in Colab and run:

    !python /content/phase2/train_phase2.py

Or paste into a Colab cell:

    import sys; sys.path.insert(0, '/content/phase2')
    from train_phase2 import run_phase2_training
    run_phase2_training(
        yolo_weights = '/content/best.pt',
        data_yaml    = '/content/BTP-1/data.yaml',
        ...
    )

────────────────────────────────────────────────────────────────────────────
ABLATION GUIDE (Section 13 of design doc):
Run each row sequentially, changing one argument at a time:

  Row 6:  + KLD regression loss        box_type='kld'
  Row 7:  + VFL classification          (always on in Phase 2)
  Row 8:  + multi-task seg heads        aux_loss_start_epoch=0
  Row 9:  (Phase 1 SAHI already done)
  Row 10: (Phase 1 NMM already done)
  Row 11: (Phase 3)
  Row 12: (Phase 1 hard-neg mining)
  Row 13: + contrastive term            (enabled in multitask_loss.py)

Box-type ablation:  box_type='kld' vs box_type='gwd'
Uncertainty ablation: use_uncertainty=True vs False
Tversky β ablation: tversky_beta=0.7 (recall) vs 0.3 (precision)
────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
from pathlib import Path

# Make sure phase2 sub-modules and phase1 are importable
_PHASE2_DIR = Path(__file__).parent
sys.path.insert(0, str(_PHASE2_DIR))
sys.path.insert(0, str(_PHASE2_DIR.parent / 'phase1'))


def run_phase2_training(
    yolo_weights:          str   = 'yolov8s-obb.pt',
    data_yaml:             str   = 'data.yaml',
    epochs:                int   = 80,
    imgsz:                 int   = 640,
    batch:                 int   = 8,
    lr0:                   float = 1e-3,
    device                       = 0,
    project:               str   = 'phase2_runs',
    name:                  str   = 'multitask_kld_vfl',
    # Phase 2 specific
    box_type:              str   = 'kld',
    tversky_alpha:         float = 0.3,
    tversky_beta:          float = 0.7,
    gate_lambda:           float = 1.0,
    use_uncertainty:       bool  = True,
    aux_loss_start_epoch:  int   = 5,
    aux_head_lr_scale:     float = 2.0,
    wandb_project:         str   = 'Railway Sleeper Crack Detection',
    use_wandb:             bool  = False,
):
    """
    Main Phase 2 training function.

    Args:
        yolo_weights          : path to Phase 1 best.pt (or 'yolov8s-obb.pt' for scratch)
        data_yaml             : path to Roboflow data.yaml
        epochs                : total training epochs (80 recommended for Phase 2)
        imgsz                 : tile inference size (match Phase 1)
        batch                 : batch size (8 for T4 GPU with yolov8s)
        lr0                   : initial learning rate
        device                : 0 for GPU, 'cpu' for CPU
        project               : output project directory
        name                  : run name (for easy ablation tracking)
        box_type              : 'kld' (default) or 'gwd' for ablation
        tversky_alpha/beta    : control precision/recall trade-off in crack seg
        gate_lambda           : strength of L_gate region penalty
        use_uncertainty       : True = Kendall uncertainty, False = fixed weights
        aux_loss_start_epoch  : delay aux losses N epochs so detection stabilises
        aux_head_lr_scale     : aux heads trained at lr0 * this factor
        wandb_project         : WandB project name for logging
        use_wandb             : enable WandB logging
    """
    # ── Optional WandB setup ──────────────────────────────────────────────────
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project = wandb_project,
                name    = name,
                config  = {
                    'phase':                2,
                    'box_type':             box_type,
                    'tversky_alpha':        tversky_alpha,
                    'tversky_beta':         tversky_beta,
                    'gate_lambda':          gate_lambda,
                    'use_uncertainty':      use_uncertainty,
                    'aux_loss_start_epoch': aux_loss_start_epoch,
                    'epochs':               epochs,
                    'imgsz':                imgsz,
                    'batch':                batch,
                    'lr0':                  lr0,
                },
            )
        except ImportError:
            print("WandB not installed — skipping. pip install wandb to enable.")

    # ── Build trainer config dict ─────────────────────────────────────────────
    overrides = {
        # Standard ultralytics args
        'model':        yolo_weights,
        'data':         data_yaml,
        'epochs':       epochs,
        'imgsz':        imgsz,
        'batch':        batch,
        'lr0':          lr0,
        'optimizer':    'AdamW',
        'augment':      True,
        'mosaic':       1.0,
        'patience':     20,
        'device':       device,
        'project':      project,
        'name':         name,
        'save':         True,
        'plots':        False,
        'verbose':      True,
        # Phase 2 specific
        'box_type':             box_type,
        'tversky_alpha':        tversky_alpha,
        'tversky_beta':         tversky_beta,
        'gate_lambda':          gate_lambda,
        'use_uncertainty':      use_uncertainty,
        'aux_loss_start_epoch': aux_loss_start_epoch,
        'aux_head_lr_scale':    aux_head_lr_scale,
    }

    # ── Launch training ───────────────────────────────────────────────────────
    from custom_trainer import Phase2Trainer

    print(f"\n{'═'*65}")
    print(f"  PHASE 2 TRAINING")
    print(f"  Base model:   {yolo_weights}")
    print(f"  Box loss:     {box_type.upper()}")
    print(f"  Cls loss:     VFL (Varifocal)")
    print(f"  Seg loss:     Focal-Tversky (β={tversky_beta}) + Dice+BCE")
    print(f"  Weighting:    {'Uncertainty (Kendall)' if use_uncertainty else 'Fixed weights'}")
    print(f"  Aux start:    epoch {aux_loss_start_epoch}")
    print(f"  Epochs:       {epochs}")
    print(f"{'═'*65}\n")

    trainer = Phase2Trainer(overrides=overrides)
    trainer.train()

    # Return the best model path for Phase 3 or evaluation
    best = os.path.join(project, name, 'weights', 'best.pt')
    print(f"\n[Phase 2] Training complete. Best model → {best}")
    return best


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION SWEEP HELPER
# ══════════════════════════════════════════════════════════════════════════════

def run_phase2_ablation(
    yolo_weights: str,
    data_yaml:    str,
    base_project: str = 'phase2_ablation',
    device       = 0,
):
    """
    Sequentially run the ablation rows for Phase 2 (Section 13).

    Each row trains for the same number of epochs from the same base model
    so results are directly comparable in the ablation table.

    Rows:
        kld_only        — KLD box loss, plain BCE cls, no aux heads
        kld_vfl         — KLD box loss + VFL cls, no aux heads
        kld_vfl_seg     — + multi-task seg heads (Focal-Tversky + Dice+BCE)
        kld_vfl_seg_unc — + uncertainty weighting (full Phase 2)
        gwd_vfl_seg_unc — GWD instead of KLD (box-type ablation)
        kld_vfl_seg_fixed — fixed weights instead of uncertainty
    """
    ablation_configs = [
        dict(name='row06_kld_only',    box_type='kld',  aux_loss_start_epoch=9999),
        dict(name='row07_kld_vfl',     box_type='kld',  aux_loss_start_epoch=9999),
        dict(name='row08_kld_vfl_seg', box_type='kld',  aux_loss_start_epoch=0, use_uncertainty=False),
        dict(name='row08b_full_phase2',box_type='kld',  aux_loss_start_epoch=5, use_uncertainty=True),
        dict(name='row08c_gwd',        box_type='gwd',  aux_loss_start_epoch=5, use_uncertainty=True),
        dict(name='row08d_fixed_wt',   box_type='kld',  aux_loss_start_epoch=5, use_uncertainty=False),
    ]

    results = {}
    for cfg in ablation_configs:
        name = cfg.pop('name')
        print(f"\n[Ablation] Starting: {name}")
        best = run_phase2_training(
            yolo_weights = yolo_weights,
            data_yaml    = data_yaml,
            epochs       = 80,
            project      = base_project,
            name         = name,
            device       = device,
            **cfg,
        )
        results[name] = best

    print(f"\n[Ablation] All Phase 2 runs complete:")
    for run_name, model_path in results.items():
        print(f"  {run_name:35s} → {model_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPER: measure FP metrics before/after Phase 2
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_phase2(
    model_path:  str,
    data_yaml:   str,
    test_images: list,
    device       = 'cpu',
):
    """
    Evaluate a Phase 2 model using the Phase 1 pipeline's FP-focused metrics.

    Prints:
      - mAP50, mAP50-95, precision, recall (standard detection metrics)
      - off_sleeper_fp_count (the paper's key Phase 1+2 metric)

    Args:
        model_path  : path to Phase 2 best.pt
        data_yaml   : path to data.yaml for ultralytics validation
        test_images : list of image file paths for FP-metric measurement
        device      : inference device
    """
    from ultralytics import YOLO
    import yaml
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
    from pipeline import Phase1Pipeline

    # ── Standard ultralytics validation metrics ───────────────────────────────
    model_ul = YOLO(model_path)
    val      = model_ul.val(data=data_yaml, device=device, verbose=False)

    print(f"\n{'─'*55}")
    print(f"  Phase 2 Validation Metrics ({Path(model_path).name})")
    print(f"{'─'*55}")
    print(f"  mAP50      : {val.box.map50:.4f}")
    print(f"  mAP50-95   : {val.box.map:.4f}")
    print(f"  Precision  : {val.box.p[0]:.4f}")
    print(f"  Recall     : {val.box.r[0]:.4f}")

    # ── FP-focused metrics using Phase 1 pipeline ────────────────────────────
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    class_names = list(cfg.get('names', {0: 'crack'}).values())

    pipeline = Phase1Pipeline(
        yolo_model_path = model_path,
        class_names     = class_names,
        device          = str(device),
    )

    import cv2
    total_fp = 0
    total_det = 0

    for img_path in test_images[:50]:   # limit for speed during eval
        img  = cv2.imread(str(img_path))
        if img is None:
            continue
        dets    = pipeline.detect(img)
        metrics = pipeline.compute_fp_metrics(dets)
        total_fp  += metrics['off_sleeper_fp_count']
        total_det += metrics['total_detections']

    print(f"\n  FP-focused metrics (Phase 1 pipeline, {min(50, len(test_images))} images):")
    print(f"  Total detections     : {total_det}")
    print(f"  Off-sleeper FPs      : {total_fp}")
    print(f"  Off-sleeper FP rate  : {total_fp / max(total_det, 1):.3f}")
    print(f"{'─'*55}")

    return {
        'map50': val.box.map50,
        'map':   val.box.map,
        'off_sleeper_fp': total_fp,
        'total_det':      total_det,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """
    Default run for Colab.
    Edit the paths below before executing.
    """
    YOLO_WEIGHTS = '/content/best.pt'          # Phase 1 best model
    DATA_YAML    = '/content/BTP-1/data.yaml'  # Roboflow dataset yaml
    PROJECT_DIR  = '/content/drive/MyDrive/BTP/phase2'

    best_model = run_phase2_training(
        yolo_weights         = YOLO_WEIGHTS,
        data_yaml            = DATA_YAML,
        epochs               = 80,
        imgsz                = 640,
        batch                = 8,
        device               = 0,
        project              = PROJECT_DIR,
        name                 = 'kld_vfl_multitask',
        box_type             = 'kld',
        tversky_beta         = 0.7,
        use_uncertainty      = True,
        aux_loss_start_epoch = 5,
        use_wandb            = True,
        wandb_project        = 'Railway Sleeper Crack Detection',
    )
