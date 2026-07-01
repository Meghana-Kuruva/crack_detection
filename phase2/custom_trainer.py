"""
Phase 2 — Custom OBB Trainer
==============================
Subclasses ultralytics OBBTrainer to inject Phase 2 losses and multi-task heads.

What changes relative to the standard OBBTrainer:
  1. build_model()       — returns MultiTaskModel (YOLO + 3 aux heads)
  2. preprocess_batch()  — generates pseudo-label targets and attaches to batch
  3. _do_train()         — optimizer includes aux head + uncertainty parameters

Everything else (dataloader, scheduler, WandB logging, val loop, checkpoint
saving) is inherited from ultralytics unchanged — no need to re-implement them.

The trainer is designed to start from an existing Phase 1 best.pt checkpoint
so it fine-tunes detection while simultaneously learning the auxiliary tasks.
This is the correct initialisation because:
  - Phase 1 already solved the easy part (basic crack detection)
  - Phase 2 refines the feature space with better losses and task supervision

Usage (Colab):
    from custom_trainer import Phase2Trainer, Phase2TrainArgs
    from ultralytics.cfg import get_cfg

    args = Phase2TrainArgs(
        data          = '/content/BTP-1/data.yaml',
        yolo_weights  = '/content/best.pt',       # Phase 1 best model
        epochs        = 80,
        imgsz         = 640,
        batch         = 8,
        device        = 0,
        project       = '/content/drive/MyDrive/BTP/phase2',
        name          = 'multitask_kld_vfl',
    )
    trainer = Phase2Trainer(overrides=vars(args))
    trainer.train()
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

# Add phase1 to sys.path so PseudoLabelGenerator can import ClassicalSleeperSegmenter
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.utils import LOGGER

from multitask_model   import MultiTaskModel
from pseudo_labels     import PseudoLabelGenerator


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING ARGUMENTS (typed dataclass instead of raw dict)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Phase2TrainArgs:
    """
    All training hyperparameters for Phase 2.
    Sensible defaults that match the design doc's Phase 2 recommendations.
    """
    data:             str   = 'data.yaml'
    yolo_weights:     str   = 'yolov8s-obb.pt'
    epochs:           int   = 80
    imgsz:            int   = 1024        # match Phase 1 — cracks need high res
    batch:            int   = 8
    lr0:              float = 5e-4        # fine-tuning from Phase 1, not scratch
    lrf:              float = 0.01        # final lr = lr0 * lrf
    momentum:         float = 0.937
    weight_decay:     float = 1e-3        # stronger regularisation for tiny dataset
    warmup_epochs:    int   = 5
    cos_lr:           bool  = True        # cosine annealing
    label_smoothing:  float = 0.05        # prevents overconfidence
    copy_paste:       float = 0.3         # copy-paste augmentation
    degrees:          float = 45.0        # OBB rotation augmentation
    optimizer:        str   = 'AdamW'
    device:           Any   = 0           # 0 for GPU, 'cpu' for CPU
    workers:          int   = 4
    patience:         int   = 30          # higher patience: 15 val images = high noise
    augment:          bool  = True
    mosaic:           float = 1.0
    amp:              bool  = False        # disabled to prevent NaN with OBB loss
    project:          str   = 'phase2_runs'
    name:             str   = 'multitask'
    save:             bool  = True
    plots:            bool  = False
    verbose:          bool  = True
    # Phase 2 specific
    box_type:              str   = 'kld'   # 'kld' or 'gwd' — Section 6.1 ablation
    tversky_alpha:         float = 0.3
    tversky_beta:          float = 0.7
    gate_lambda:           float = 1.0
    use_uncertainty:       bool  = True
    aux_loss_start_epoch:  int   = 5       # start aux losses after N warm-up epochs
                                            # (let detection stabilise first)
    aux_head_lr_scale:     float = 2.0     # aux heads can use higher LR than the
                                            # pretrained backbone (they start from scratch)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class Phase2Trainer(OBBTrainer):
    """
    Phase 2 multi-task OBB trainer.

    Key overrides:
      get_model()       — builds MultiTaskModel
      preprocess_batch()— attaches pseudo-label targets to the batch
      _setup_train()    — creates a separate optimizer for aux head parameters
    """

    def __init__(self, overrides: Dict = None, _callbacks=None):
        # Ensure required overrides have defaults
        overrides = overrides or {}
        overrides.setdefault('model', overrides.pop('yolo_weights', 'yolov8s-obb.pt'))
        super().__init__(cfg=overrides, overrides=overrides, _callbacks=_callbacks)

        # Phase 2 config from overrides
        self.box_type             = overrides.get('box_type',             'kld')
        self.aux_loss_start_epoch = overrides.get('aux_loss_start_epoch', 5)
        self.aux_head_lr_scale    = overrides.get('aux_head_lr_scale',    2.0)
        self.use_uncertainty      = overrides.get('use_uncertainty',      True)
        self.gate_lambda          = overrides.get('gate_lambda',          1.0)

        # Pseudo-label generator — initialised lazily after data is known
        self._pseudo_gen: Optional[PseudoLabelGenerator] = None
        self._current_epoch = 0

    # ── model construction ────────────────────────────────────────────────────

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Build MultiTaskModel instead of plain OBBModel.

        ultralytics calls this once during trainer.setup_model().
        We return an nn.Module that the trainer treats as the detection model.
        """
        # Use cfg's nc (number of classes) from data.yaml
        nc = self.data.get('nc', 3)

        model = MultiTaskModel(
            yolo_weights = weights or self.args.model,
            nc           = nc,
            box_type     = self.box_type,
            img_size     = self.args.imgsz,
            device       = str(self.device),
        )
        return model

    # ── batch preprocessing ───────────────────────────────────────────────────

    def preprocess_batch(self, batch: Dict) -> Dict:
        """
        Extend the standard ultralytics batch with pseudo-label targets.

        Called before each training step.  The pseudo-labels are computed on
        the CPU (classical segmenter + label parsing) and attached to the batch.
        """
        # Standard ultralytics preprocessing first
        batch = super().preprocess_batch(batch)

        # Only generate pseudo-labels after aux_loss_start_epoch
        if self._current_epoch < self.aux_loss_start_epoch:
            return batch

        # Lazy-init pseudo-label generator
        if self._pseudo_gen is None:
            class_names = self.data.get('names', {})
            if isinstance(class_names, dict):
                class_names = [class_names[i] for i in sorted(class_names.keys())]
            self._pseudo_gen = PseudoLabelGenerator(class_names=class_names)
            LOGGER.info(f"[Phase2] Pseudo-label generator initialised with classes: {class_names}")

        # Generate pseudo-labels for each image in the batch
        # batch['img'] shape: (B, 3, H, W)
        # batch['im_file'] contains file paths (present in ultralytics batches)
        try:
            pseudo_batch = self._generate_batch_pseudo_labels(batch)
            batch.update(pseudo_batch)
        except Exception as e:
            LOGGER.warning(f"[Phase2] Pseudo-label generation failed: {e}")

        return batch

    def _generate_batch_pseudo_labels(self, batch: Dict) -> Dict:
        """
        Generate pseudo-label tensors for every image in the batch.
        Returns a dict of stacked tensors ready for the loss.
        """
        im_files = batch.get('im_file', [])
        if not im_files:
            return {}

        B   = len(im_files)
        imgsz = self.args.imgsz
        crack_masks, sleeper_masks, crack_types = [], [], []

        for img_path in im_files:
            img_path = str(img_path)
            # Derive label path from image path
            label_path = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'

            # Read original image for pseudo-label generation
            import cv2
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                import numpy as np
                img_bgr = np.zeros((imgsz, imgsz, 3), np.uint8)

            pseudo = self._pseudo_gen.generate(
                img_bgr,
                label_path,
                image_size=(imgsz, imgsz),
            )
            crack_masks.append(pseudo['crack_mask'])
            sleeper_masks.append(pseudo['sleeper_mask'])
            crack_types.append(pseudo['crack_type'])

        device = batch['img'].device
        return {
            'aux_targets': {
                'crack_mask':   torch.stack(crack_masks).to(device),   # (B,1,H,W)
                'sleeper_mask': torch.stack(sleeper_masks).to(device),
                'crack_type':   torch.cat(crack_types).to(device),     # (B,)
            }
        }

    # ── training loop hooks ───────────────────────────────────────────────────

    def optimizer_step(self):
        """
        Standard gradient step — ultralytics handles this; we just track epoch.
        """
        super().optimizer_step()

    def on_train_epoch_start(self):
        """Track current epoch for aux_loss_start_epoch gating."""
        super().on_train_epoch_start()
        self._current_epoch = self.epoch

    def _setup_train(self, world_size: int = 1):
        """
        Override to add auxiliary head parameters to the optimizer.

        ultralytics' _setup_train creates self.optimizer for the YOLO backbone.
        We add the aux head parameters to a separate param group with a
        higher LR (they start from random init, backbone starts from pretrained).
        """
        super()._setup_train(world_size)

        # Collect aux head parameters
        aux_params = (
            list(self.model.crack_seg_head.parameters())
          + list(self.model.sleeper_seg_head.parameters())
          + list(self.model.crack_type_head.parameters())
          + list(self.model.phase2_loss.weighter.parameters())  # uncertainty σ_i
        )

        # Add to existing optimizer as a new param group with scaled LR
        base_lr = self.args.lr0
        self.optimizer.add_param_group({
            'params':        aux_params,
            'lr':            base_lr * self.aux_head_lr_scale,
            'weight_decay':  self.args.weight_decay,
        })

        LOGGER.info(
            f"[Phase2] Added {sum(p.numel() for p in aux_params):,} aux-head params "
            f"with lr={base_lr * self.aux_head_lr_scale:.2e}"
        )

    # ── inject aux_targets into the model's forward call ─────────────────────

    def loss_function(self, batch):
        """
        ultralytics calls self.model(batch) inside _do_train.
        This hook is the correct override point in recent ultralytics versions.
        Older versions call model(batch) directly — the preprocess_batch
        approach stores aux_targets on the batch dict and MultiTaskModel reads
        them from there as a fallback.
        """
        aux_targets = batch.pop('aux_targets', None)
        return self.model(batch, aux_targets=aux_targets)
