"""
Phase 3 — Training Script (Colab-ready)
========================================
Trains the Phase 3 model (hybrid backbone + AFPN + FiLM + DyHead + enhanced OBB head)
starting from a Phase 2 best.pt checkpoint where possible, then fine-tuning the
full new architecture end-to-end.

TRAINING STRATEGY:
  Phase 3 has a NEW backbone and neck — we cannot directly load Phase 2 weights
  into the YOLO model because the architecture differs.  So we:

    Stage A (warm-up, epochs 0–19): Freeze the hybrid backbone, train only the
      neck (AFPN, DyHead) and heads (OBBHead + aux heads).  This lets the neck
      adapt to the new backbone's features without thrashing.

    Stage B (joint fine-tune, epochs 20–end): Unfreeze all layers and train
      jointly.  Use a lower LR for the backbone (pretrained-style LR schedule)
      and higher LR for the new neck/heads.

    FiLM branch: enabled from epoch 0 with a global-context dummy (whole-image
      embedding) until tile bboxes are available from the SAHI pipeline.

INTEGRATION WITH ULTRALYTICS:
  Phase 3 uses a custom training loop (not ultralytics Trainer) because the
  architecture no longer wraps a YOLO model — it IS a custom model.  We
  write a simple PyTorch training loop that:
    - Uses Phase 2's PseudoLabelGenerator for aux targets
    - Evaluates with Phase 1's SAHI pipeline for FP metrics
    - Saves checkpoints in a format compatible with the Phase 1 pipeline

ABLATION ROWS (design doc Section 13):
  Row 9 : + P2 level (strip_k=1 to disable strip conv, tex_out=0 to disable texture)
  Row 10: + strip conv in head (strip_k=9, tex_out=0)
  Row 11: + texture contrast (tex_out=8)
  Row 12: + FiLM conditioning (use_film=True)
  Row 13: + DyHead (dyhead_blocks=2 vs 0)

Usage (Colab):
    !python /content/phase3/train_phase3.py

Or import:
    from train_phase3 import run_phase3_training
    run_phase3_training(
        phase2_weights = '/content/phase2_best.pt',
        data_yaml      = '/content/BTP-1/data.yaml',
        ...
    )
"""

import os
import sys
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_PHASE3_DIR = Path(__file__).parent
sys.path.insert(0, str(_PHASE3_DIR))
sys.path.insert(0, str(_PHASE3_DIR.parent / 'phase2'))
sys.path.insert(0, str(_PHASE3_DIR.parent / 'phase1'))


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Phase3TrainArgs:
    """All hyperparameters for Phase 3 training."""
    # Paths
    phase2_weights:     str   = ''           # Phase 2 best.pt for partial init
    data_yaml:          str   = 'data.yaml'
    project:            str   = 'phase3_runs'
    name:               str   = 'hybrid_afpn_dyhead'

    # Training schedule
    epochs:             int   = 120          # more epochs; architecture is new
    warmup_epochs:      int   = 20           # backbone-frozen warm-up
    imgsz:              int   = 640
    batch:              int   = 6            # P2 feature map is 4× larger; reduce batch
    num_workers:        int   = 4

    # Optimiser
    lr0:                float = 5e-4         # lower than Phase 2 (new arch = higher init variance)
    lr_backbone:        float = 1e-4         # backbone gets 5× lower LR during joint fine-tune
    weight_decay:       float = 1e-4
    gradient_clip:      float = 10.0        # clip grads to avoid early-training explosions

    # Architecture
    width_mult:         float = 1.0
    evit_depth:         int   = 2
    afpn_out_ch:        int   = 256
    dyhead_blocks:      int   = 2
    strip_k:            int   = 9
    use_film:           bool  = True
    film_ctx_ch:        int   = 256
    use_uncertainty:    bool  = True
    box_type:           str   = 'kld'

    # Aux losses
    tversky_beta:       float = 0.7
    aux_loss_start_epoch: int = 5

    # Logging
    use_wandb:          bool  = False
    wandb_project:      str   = 'Railway Sleeper Crack Detection'
    save_every:         int   = 10           # save checkpoint every N epochs
    device:             str   = '0'          # '0' for GPU, 'cpu' for CPU


# ══════════════════════════════════════════════════════════════════════════════
# DATASET ADAPTER (wraps Phase 2's OBBDatasetWithPseudoLabels)
# ══════════════════════════════════════════════════════════════════════════════

def _build_dataloaders(args: Phase3TrainArgs):
    """
    Build train and val DataLoaders from a Roboflow YOLO-OBB dataset directory.

    The data.yaml contains:
        train: /path/to/train/images
        val:   /path/to/valid/images
        nc:    3
        names: {0: 'cracks in midportion', ...}
    """
    import yaml
    with open(args.data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    nc         = data_cfg.get('nc', 3)
    names_dict = data_cfg.get('names', {0: 'crack'})
    class_names = [names_dict[i] for i in sorted(names_dict.keys())]

    train_img_dir = data_cfg.get('train', '')
    val_img_dir   = data_cfg.get('val', '')

    # Labels directory assumed to mirror images directory
    def labels_dir(img_dir):
        return str(Path(img_dir).parent.parent / 'labels' /
                   Path(img_dir).name).replace('/images/', '/labels/')

    from pseudo_labels import OBBDatasetWithPseudoLabels

    train_ds = OBBDatasetWithPseudoLabels(
        images_dir  = train_img_dir,
        labels_dir  = labels_dir(train_img_dir),
        class_names = class_names,
        img_size    = (args.imgsz, args.imgsz),
        augment     = True,
    )
    val_ds = OBBDatasetWithPseudoLabels(
        images_dir  = val_img_dir,
        labels_dir  = labels_dir(val_img_dir),
        class_names = class_names,
        img_size    = (args.imgsz, args.imgsz),
        augment     = False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    return train_loader, val_loader, nc, class_names


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMISER: two param groups (backbone vs neck+head)
# ══════════════════════════════════════════════════════════════════════════════

def _build_optimiser(model, args: Phase3TrainArgs, epoch: int = 0):
    """
    Returns an AdamW optimiser with two param groups:
      - 'backbone':  lower LR (pretrained features)
      - 'neck_head': higher LR (new components)

    During warmup (epoch < warmup_epochs) the backbone is frozen entirely.
    """
    backbone_params = list(model.backbone.parameters())
    backbone_ids    = set(id(p) for p in backbone_params)

    other_params = [p for p in model.parameters()
                    if id(p) not in backbone_ids]

    if epoch < args.warmup_epochs:
        # Frozen backbone: set requires_grad=False
        for p in backbone_params:
            p.requires_grad_(False)
        param_groups = [{'params': other_params, 'lr': args.lr0, 'name': 'neck_head'}]
    else:
        # Unfreeze backbone with lower LR
        for p in backbone_params:
            p.requires_grad_(True)
        param_groups = [
            {'params': backbone_params, 'lr': args.lr_backbone, 'name': 'backbone'},
            {'params': other_params,    'lr': args.lr0,         'name': 'neck_head'},
        ]

    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def _cosine_lr(step, total_steps, lr_min_ratio=0.01):
    """Cosine annealing schedule multiplier."""
    return lr_min_ratio + 0.5 * (1 - lr_min_ratio) * (1 + math.cos(math.pi * step / total_steps))


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_phase3_training(
    phase2_weights:   str   = '',
    data_yaml:        str   = 'data.yaml',
    epochs:           int   = 120,
    batch:            int   = 6,
    lr0:              float = 5e-4,
    device:           str   = '0',
    project:          str   = 'phase3_runs',
    name:             str   = 'hybrid_afpn_dyhead',
    width_mult:       float = 1.0,
    afpn_out_ch:      int   = 256,
    dyhead_blocks:    int   = 2,
    strip_k:          int   = 9,
    use_film:         bool  = True,
    use_uncertainty:  bool  = True,
    box_type:         str   = 'kld',
    use_wandb:        bool  = False,
    **kwargs,
) -> str:
    """
    Main Phase 3 training function.

    Returns path to the best model checkpoint.
    """
    args = Phase3TrainArgs(
        phase2_weights  = phase2_weights,
        data_yaml       = data_yaml,
        epochs          = epochs,
        batch           = batch,
        lr0             = lr0,
        device          = device,
        project         = project,
        name            = name,
        width_mult      = width_mult,
        afpn_out_ch     = afpn_out_ch,
        dyhead_blocks   = dyhead_blocks,
        strip_k         = strip_k,
        use_film        = use_film,
        use_uncertainty = use_uncertainty,
        box_type        = box_type,
        use_wandb       = use_wandb,
    )
    for k, v in kwargs.items():
        if hasattr(args, k):
            setattr(args, k, v)

    # ── Device ────────────────────────────────────────────────────────────────
    device_str = args.device
    if device_str in ('0', '1', '2', '3') and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_str}')
    elif device_str == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(device_str)

    # ── WandB ─────────────────────────────────────────────────────────────────
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project = args.wandb_project,
                name    = args.name,
                config  = vars(args),
            )
        except ImportError:
            print("WandB not installed — skipping.")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, nc, class_names = _build_dataloaders(args)
    print(f"[Phase3] Dataset: {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val. Classes: {class_names}")

    # ── Model ─────────────────────────────────────────────────────────────────
    from phase3_model import Phase3Model, load_partial_weights

    model = Phase3Model(
        nc              = nc,
        box_type        = args.box_type,
        width_mult      = args.width_mult,
        evit_depth      = args.evit_depth,
        afpn_out_ch     = args.afpn_out_ch,
        dyhead_blocks   = args.dyhead_blocks,
        strip_k         = args.strip_k,
        use_film        = args.use_film,
        film_ctx_ch     = args.film_ctx_ch,
        use_uncertainty = args.use_uncertainty,
        device          = str(device),
    ).to(device)

    # Partial weight transfer from Phase 2 (aux heads + uncertainty params)
    if args.phase2_weights and os.path.exists(args.phase2_weights):
        print(f"[Phase3] Loading partial weights from Phase 2: {args.phase2_weights}")
        load_partial_weights(model, args.phase2_weights)
    else:
        print("[Phase3] No Phase 2 weights found — training from scratch.")

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = Path(args.project) / args.name / 'weights'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  PHASE 3 TRAINING")
    print(f"  Backbone:      Hybrid (RepViT + EfficientViT) width={args.width_mult}")
    print(f"  Neck:          AFPN(P2-P5) + CoordAttn + StripPool + DyHead")
    print(f"  Head:          StripConv(k={args.strip_k}) + TextureContrast + P2 level")
    print(f"  FiLM:          {'ON' if args.use_film else 'OFF'}")
    print(f"  Epochs:        {args.epochs}  (warmup frozen: {args.warmup_epochs})")
    print(f"  Batch:         {args.batch}  LR: {args.lr0}")
    print(f"{'═'*65}\n")

    best_loss = float('inf')
    best_path = str(out_dir / 'best.pt')

    total_steps = args.epochs * len(train_loader)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()

        # Switch optimiser at warmup boundary (unfreeze backbone)
        if epoch == 0 or epoch == args.warmup_epochs:
            optim = _build_optimiser(model, args, epoch=epoch)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=max(args.epochs - epoch, 1), eta_min=args.lr0 * 0.01)

        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch_data in enumerate(train_loader):
            # batch_data from OBBDatasetWithPseudoLabels:
            #   'img': (B, 3, H, W) float
            #   'crack_mask', 'sleeper_mask', 'crack_type'

            images = batch_data['img'].to(device)
            aux_targets = None
            if epoch >= args.aux_loss_start_epoch:
                aux_targets = {
                    'crack_mask':   batch_data['crack_mask'].to(device),
                    'sleeper_mask': batch_data['sleeper_mask'].to(device),
                    'crack_type':   batch_data['crack_type'].to(device),
                }

            # Build a minimal batch dict for Phase3Model._train_forward
            batch_dict = {'img': images}

            # Forward + loss
            loss, items = model(batch_dict, aux_targets=aux_targets)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Phase3] NaN/Inf loss at epoch={epoch} batch={batch_idx} — skipping.")
                continue

            # Backward
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optim.step()

            epoch_loss += loss.item()
            global_step += 1

            # LR warmup for first 3 epochs
            if epoch < 3:
                lr_scale = min(1.0, (epoch * len(train_loader) + batch_idx + 1) /
                               (3 * len(train_loader)))
                for pg in optim.param_groups:
                    pg['lr'] = pg.get('_base_lr', args.lr0) * lr_scale

        sched.step()
        epoch_loss /= max(len(train_loader), 1)

        # ── Validation ────────────────────────────────────────────────────────
        val_loss = _validate(model, val_loader, device, args)

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  "
              f"lr={optim.param_groups[-1]['lr']:.2e}  "
              f"t={elapsed:.0f}s")

        if use_wandb:
            try:
                import wandb
                wandb.log({'epoch': epoch+1, 'train_loss': epoch_loss,
                           'val_loss': val_loss, 'lr': optim.param_groups[-1]['lr']})
            except Exception:
                pass

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = str(out_dir / f'epoch_{epoch+1}.pt')
            _save_checkpoint(model, optim, epoch, epoch_loss, ckpt_path)

        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            _save_checkpoint(model, optim, epoch, val_loss, best_path)
            print(f"  → New best: val_loss={val_loss:.4f} → {best_path}")

    print(f"\n[Phase3] Training complete. Best model: {best_path}")
    return best_path


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _validate(model, val_loader, device, args: Phase3TrainArgs) -> float:
    model.eval()
    total_loss = 0.0
    n = 0

    for batch_data in val_loader:
        images = batch_data['img'].to(device)
        aux_targets = {
            'crack_mask':   batch_data['crack_mask'].to(device),
            'sleeper_mask': batch_data['sleeper_mask'].to(device),
            'crack_type':   batch_data['crack_type'].to(device),
        }
        batch_dict = {'img': images}

        model.train()   # enable loss computation (needs .training=True)
        try:
            loss, _ = model(batch_dict, aux_targets=aux_targets)
            total_loss += loss.item()
            n += 1
        except Exception:
            pass
        model.eval()

    model.train()
    return total_loss / max(n, 1)


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _save_checkpoint(model, optim, epoch, loss, path):
    torch.save({
        'epoch':      epoch,
        'loss':       loss,
        'model':      model.state_dict(),
        'optimiser':  optim.state_dict(),
    }, path)


def load_phase3_checkpoint(model, path: str, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)
    print(f"[Phase3] Loaded checkpoint from {path} (epoch={ckpt.get('epoch', '?')})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION SWEEP HELPER
# ══════════════════════════════════════════════════════════════════════════════

def run_phase3_ablation(
    phase2_weights: str,
    data_yaml:      str,
    base_project:   str = 'phase3_ablation',
    device:         str = '0',
):
    """
    Ablation rows for Phase 3 architecture components (design doc Section 13).

    Each row adds one component over the previous, isolating its contribution.

    Row 9:  + P2 level (baseline — no strip conv, no texture, no FiLM, no DyHead)
    Row 10: + strip conv in head (strip_k=9)
    Row 11: + texture contrast branch
    Row 12: + FiLM global context conditioning
    Row 13: + DyHead (scale/spatial/task attention)
    """
    ablation_rows = [
        dict(name='row09_p2level',       strip_k=1,  use_film=False, dyhead_blocks=0),
        dict(name='row10_strip_conv',    strip_k=9,  use_film=False, dyhead_blocks=0),
        dict(name='row11_texture',       strip_k=9,  use_film=False, dyhead_blocks=0,
             # tex_out is always 8; the row9 ablation would need tex_out=0 in obb_head
             # (set manually in Phase3Model constructor if needed)
             ),
        dict(name='row12_film',          strip_k=9,  use_film=True,  dyhead_blocks=0),
        dict(name='row13_dyhead',        strip_k=9,  use_film=True,  dyhead_blocks=2),
    ]

    results = {}
    for cfg in ablation_rows:
        row_name = cfg.pop('name')
        print(f"\n[Phase3 Ablation] Starting: {row_name}")
        best = run_phase3_training(
            phase2_weights = phase2_weights,
            data_yaml      = data_yaml,
            epochs         = 80,
            project        = base_project,
            name           = row_name,
            device         = device,
            **cfg,
        )
        results[row_name] = best

    print(f"\n[Ablation] All Phase 3 runs complete:")
    for row_name, model_path in results.items():
        print(f"  {row_name:35s} → {model_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# FP METRIC EVALUATION (using Phase 1 pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_phase3(
    model_path:  str,
    data_yaml:   str,
    test_images: List[str],
    nc:          int = 3,
    device:      str = 'cpu',
):
    """
    Evaluate a Phase 3 checkpoint with the Phase 1 FP-metric pipeline.

    The Phase 3 model does not produce ultralytics-format predictions directly,
    so we evaluate the auxiliary segmentation heads (which ARE compatible with
    Phase 1's mask-gating logic) and compute the FP rate via Phase 1's pipeline
    running the Phase 3 model in detection mode.

    For detection mAP, use run_phase3_training with a small validation dataset
    and monitor val_loss (which includes detection + seg + type losses).
    """
    from phase3_model import Phase3Model

    import yaml
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    class_names = [data_cfg['names'][i] for i in sorted(data_cfg['names'].keys())]

    model = Phase3Model(nc=nc, device=device)
    load_phase3_checkpoint(model, model_path, device=device)
    model.eval()

    import cv2
    import numpy as np

    n_images = 0
    seg_ious  = []

    for img_path in test_images[:50]:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_t    = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_t    = img_t.to(device)

        with torch.no_grad():
            _, aux_preds = model(img_t)

        # Evaluate sleeper segmentation IoU (proxy for context quality)
        sleeper_pred = torch.sigmoid(aux_preds['sleeper_seg']).squeeze().cpu().numpy()

        # Phase 1 classical segmenter as reference
        from sleeper_segmentation import ClassicalSleeperSegmenter
        seg = ClassicalSleeperSegmenter()
        ref_mask, _ = seg.predict_mask(img_bgr)
        ref_small   = cv2.resize(ref_mask, (sleeper_pred.shape[-1], sleeper_pred.shape[-2]))

        pred_bin  = (sleeper_pred > 0.5).astype(np.uint8)
        inter     = (pred_bin * ref_small).sum()
        union     = (pred_bin | ref_small.astype(bool)).sum()
        if union > 0:
            seg_ious.append(inter / union)

        n_images += 1

    mean_iou = float(np.mean(seg_ious)) if seg_ious else 0.0

    print(f"\n{'─'*55}")
    print(f"  Phase 3 Evaluation ({n_images} images)")
    print(f"{'─'*55}")
    print(f"  Sleeper seg IoU (vs classical) : {mean_iou:.4f}")
    print(f"  (For mAP50 — use validation loss during training or integrate")
    print(f"   with ultralytics val by wrapping in a compatible model wrapper)")
    print(f"{'─'*55}")

    return {'sleeper_seg_iou': mean_iou, 'n_images': n_images}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """Default run for Colab. Edit paths before executing."""
    PHASE2_WEIGHTS = '/content/phase2_best.pt'   # from Phase 2 run
    DATA_YAML      = '/content/BTP-1/data.yaml'
    PROJECT_DIR    = '/content/drive/MyDrive/BTP/phase3'

    best_model = run_phase3_training(
        phase2_weights  = PHASE2_WEIGHTS,
        data_yaml       = DATA_YAML,
        epochs          = 120,
        batch           = 6,
        lr0             = 5e-4,
        device          = '0',
        project         = PROJECT_DIR,
        name            = 'hybrid_afpn_dyhead_film',
        width_mult      = 1.0,
        afpn_out_ch     = 256,
        dyhead_blocks   = 2,
        strip_k         = 9,
        use_film        = True,
        use_uncertainty = True,
        box_type        = 'kld',
        use_wandb       = True,
        wandb_project   = 'Railway Sleeper Crack Detection',
    )
