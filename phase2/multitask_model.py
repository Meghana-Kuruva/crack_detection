"""
Phase 2 — Multi-Task YOLOv8-OBB Model
=======================================
Wraps a standard YOLOv8-OBB model and attaches three auxiliary heads
(crack segmentation, sleeper segmentation, crack-type classification)
via PyTorch forward hooks on the neck feature maps.

Architecture overview (Section 7):
    shared backbone + neck (YOLOv8-OBB layers 0–21)
          │
          ├── [hook layer 15, P3, stride 8]  → CrackSegHead
          ├── [hook layer 21, P5, stride 32] → SleeperSegHead
          ├── [hook layer 21, P5, stride 32] → CrackTypeHead
          └── [hook layer 15, P3]            → contrastive_projection()
          │
          └── [layer 22, OBBHead]            → detection predictions

Hook strategy:
  PyTorch forward hooks intercept the OUTPUT of a specific layer during the
  forward pass.  We hook layers 15 and 21 of the YOLO neck to capture P3 and
  P5 feature maps WITHOUT modifying the YOLO model's code.

  Hooks are registered on `model.model.model[layer_idx]` — the inner module
  list of the ultralytics DetectionModel.

  The hook callback stores the output tensor in a dictionary keyed by layer
  index.  After the YOLO forward pass, the auxiliary heads read from this dict.

Layer indices for common YOLOv8 variants:
  Both yolov8n-obb and yolov8s-obb have 23 layers (0–22) with the same
  structural indices — only the channel widths differ:
    Layer 15 = P3 neck output (C2f)  — channels: 64 (n), 128 (s), 192 (m)
    Layer 21 = P5 neck output (C2f)  — channels: 256 (n), 512 (s), 768 (m)

Channel count auto-detection:
  Rather than hardcoding per-variant channels, the model runs one dummy forward
  pass at construction time to measure the actual feature map channel counts,
  then builds the aux heads with the correct in_channels.

Usage:
    model = MultiTaskModel(
        yolo_weights = '/content/best.pt',
        model_variant = 's',     # 'n', 's', 'm'
        img_size = 640,
    )
    # Forward in train mode returns (total_loss, loss_items_dict) — see custom_trainer.py
    # Forward in eval mode returns standard ultralytics OBB predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from ultralytics import YOLO

from heads.crack_seg_head   import CrackSegHead
from heads.sleeper_seg_head import SleeperSegHead
from heads.crack_type_head  import CrackTypeHead
from losses.multitask_loss  import Phase2Loss


# Layer indices are the same across all YOLOv8 variants (only channels differ)
P3_LAYER_IDX = 15
P5_LAYER_IDX = 21


class MultiTaskModel(nn.Module):
    """
    YOLOv8-OBB detector + three auxiliary heads.

    Call in train mode with a batch dict to get (total_loss, items_dict).
    Call in eval mode with a tensor to get standard OBB predictions.

    The YOLO weights are loaded and the detection weights are fine-tuned jointly
    with the auxiliary heads, so the shared neck learns context-aware features.
    """

    def __init__(
        self,
        yolo_weights:    str,          # path to .pt file (Phase 1 best model)
        nc:              int   = 3,    # number of detection classes
        box_type:        str   = 'kld',
        img_size:        int   = 640,
        device:          str   = 'cpu',
    ):
        super().__init__()
        self.nc       = nc
        self.img_size = img_size
        self.device   = device

        # ── Load YOLOv8-OBB model ─────────────────────────────────────────────
        print(f"[MultiTaskModel] Loading base model from {yolo_weights}")
        yolo_wrapper   = YOLO(yolo_weights)
        self.yolo_core = yolo_wrapper.model    # inner nn.Module (DetectionModel)
        self.yolo_core.to(device)

        # ── Auto-detect P3 and P5 channel counts via a dummy forward ─────────
        p3_ch, p5_ch = self._probe_feature_channels(img_size)
        print(f"[MultiTaskModel] Detected channels — P3: {p3_ch}, P5: {p5_ch}")

        # ── Auxiliary heads ───────────────────────────────────────────────────
        self.crack_seg_head   = CrackSegHead(in_channels=p3_ch).to(device)
        self.sleeper_seg_head = SleeperSegHead(in_channels=p5_ch).to(device)
        self.crack_type_head  = CrackTypeHead(in_channels=p5_ch, n_types=4).to(device)

        # ── Phase 2 combined loss ─────────────────────────────────────────────
        self.phase2_loss = Phase2Loss(nc=nc, box_type=box_type).to(device)

        # ── Feature cache (filled by hooks) ──────────────────────────────────
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    # ── hook management ───────────────────────────────────────────────────────

    def _register_hooks(self):
        """
        Register forward hooks on P3 (layer 15) and P5 (layer 21) of the
        YOLOv8 neck.  Each hook copies the layer output into self._features.
        """
        layers = self.yolo_core.model   # nn.Sequential of all YOLO layers

        def make_hook(key: str):
            def hook(module, inp, output):
                # output can be a tuple for some layers; take the last tensor
                self._features[key] = output if isinstance(output, torch.Tensor) \
                                             else output[-1]
            return hook

        self._hooks.append(
            layers[P3_LAYER_IDX].register_forward_hook(make_hook('p3'))
        )
        self._hooks.append(
            layers[P5_LAYER_IDX].register_forward_hook(make_hook('p5'))
        )

    def remove_hooks(self):
        """Call this if you want to detach the aux heads (e.g., export ONNX)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _probe_feature_channels(self, img_size: int) -> Tuple[int, int]:
        """
        Run a single dummy forward pass to measure P3/P5 channel counts.
        Hooks must be registered first — called during __init__.
        """
        dummy_hooks = []

        def make_probe(key):
            def hook(module, inp, out):
                tensor = out if isinstance(out, torch.Tensor) else out[-1]
                self._features[key] = tensor
            return hook

        layers = self.yolo_core.model
        dummy_hooks.append(layers[P3_LAYER_IDX].register_forward_hook(make_probe('p3_probe')))
        dummy_hooks.append(layers[P5_LAYER_IDX].register_forward_hook(make_probe('p5_probe')))

        self.yolo_core.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size, device=self.device)
            try:
                self.yolo_core(dummy)
            except Exception:
                pass  # head might fail but hooks will have fired

        for h in dummy_hooks:
            h.remove()

        p3_ch = self._features.get('p3_probe', torch.zeros(1, 128)).shape[1]
        p5_ch = self._features.get('p5_probe', torch.zeros(1, 512)).shape[1]
        self._features.clear()
        return p3_ch, p5_ch

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x_or_batch,                             # tensor (eval) or batch dict (train)
        aux_targets: Optional[Dict] = None,     # pseudo-labels from PseudoLabelGenerator
    ):
        """
        Train mode : x_or_batch is an ultralytics batch dict.
                     Returns (total_loss, loss_items_dict).
        Eval mode  : x_or_batch is a (B,3,H,W) tensor.
                     Returns ultralytics OBB predictions (unchanged).
        """
        if not self.training:
            return self._eval_forward(x_or_batch)
        else:
            return self._train_forward(x_or_batch, aux_targets)

    def _eval_forward(self, images: torch.Tensor):
        """Standard detection-only forward (no aux heads at inference)."""
        self._features.clear()
        return self.yolo_core(images)

    def _train_forward(
        self,
        batch:       dict,
        aux_targets: Optional[Dict] = None,
    ):
        """
        Full multi-task forward during training.

        Steps:
          1. Run YOLO forward — hooks fire and populate self._features with P3/P5.
          2. Decode detection predictions from the YOLO head output.
          3. Run auxiliary heads on hooked features.
          4. Compute Phase 2 combined loss.
          5. Return (total_loss, items_dict) for the trainer's backward pass.
        """
        images = batch['img'].to(self.device)
        self._features.clear()

        # ── Step 1: YOLO forward (hooks populate P3 and P5) ─────────────────
        # We call the core model in train mode so it computes its own detection
        # loss via the built-in criterion.  We then ADD our auxiliary losses on top.
        yolo_loss, yolo_items = self.yolo_core(batch)

        p3 = self._features.get('p3')
        p5 = self._features.get('p5')

        # ── Step 2: Auxiliary predictions (if hooks fired correctly) ─────────
        aux_preds: Dict[str, torch.Tensor] = {}

        if p3 is not None:
            aux_preds['crack_seg'] = self.crack_seg_head(p3)

        if p5 is not None:
            aux_preds['sleeper_seg'] = self.sleeper_seg_head(p5)
            aux_preds['type_cls']    = self.crack_type_head(p5)

        # Contrastive embeddings from crack seg head bottleneck
        if p3 is not None and aux_targets is not None and 'contrastive_labels' in aux_targets:
            aux_preds['contrastive_emb'] = self.crack_seg_head.contrastive_projection()

        # ── Step 3: Auxiliary losses (only if we have both predictions and targets)
        aux_loss = torch.tensor(0.0, device=images.device)
        aux_items: Dict[str, float] = {}

        if aux_preds and aux_targets:
            # Resize pseudo-labels to match the output spatial size if needed
            aux_targets_resized = self._resize_targets(aux_targets, aux_preds, images.device)

            aux_loss_parts: Dict[str, torch.Tensor] = {}

            if 'crack_seg' in aux_preds and 'crack_mask' in aux_targets_resized:
                aux_loss_parts['crack_seg'] = self._seg_loss_crack(
                    aux_preds['crack_seg'], aux_targets_resized['crack_mask']
                )

            if 'sleeper_seg' in aux_preds and 'sleeper_mask' in aux_targets_resized:
                aux_loss_parts['sleeper_seg'] = self._seg_loss_sleeper(
                    aux_preds['sleeper_seg'], aux_targets_resized['sleeper_mask']
                )

            if 'type_cls' in aux_preds and 'crack_type' in aux_targets_resized:
                aux_loss_parts['type_cls'] = self._type_loss(
                    aux_preds['type_cls'], aux_targets_resized['crack_type']
                )

            if 'contrastive_emb' in aux_preds and 'contrastive_labels' in aux_targets:
                aux_loss_parts['contrastive'] = self._contrastive_loss(
                    aux_preds['contrastive_emb'], aux_targets['contrastive_labels']
                )

            # L_gate: crack activations outside sleeper should be zero
            if 'crack_seg' in aux_preds and 'sleeper_seg' in aux_preds:
                from losses.focal_tversky import region_gate_loss
                crack_prob   = torch.sigmoid(aux_preds['crack_seg'].squeeze(1))
                sleeper_prob = torch.sigmoid(aux_preds['sleeper_seg'].squeeze(1))
                if crack_prob.shape != sleeper_prob.shape:
                    sleeper_prob = F.interpolate(
                        sleeper_prob.unsqueeze(1),
                        size=crack_prob.shape[-2:],
                        mode='bilinear', align_corners=False
                    ).squeeze(1)
                aux_loss_parts['gate'] = region_gate_loss(crack_prob, sleeper_prob)

            # Sum auxiliary losses through uncertainty weighting
            if aux_loss_parts:
                aux_total, aux_items = self.phase2_loss.weighter(aux_loss_parts)
                aux_loss = aux_total

        # ── Step 4: Combine detection + auxiliary losses ─────────────────────
        # Uncertainty weights from Phase2Loss.weighter handle the balance.
        # The YOLO detection loss (yolo_loss) is already computed internally;
        # we add aux_loss on top.
        total_loss = yolo_loss + aux_loss

        # Merge items for logging
        items = {f'yolo_{k}': float(v) for k, v in yolo_items.items()} \
              | {f'aux_{k}':  v         for k, v in aux_items.items()}

        return total_loss, items

    # ── private loss helpers ──────────────────────────────────────────────────

    def _seg_loss_crack(self, pred, target):
        from losses.focal_tversky import FocalTverskyLoss
        if not hasattr(self, '_ftl'):
            self._ftl = FocalTverskyLoss().to(self.device)
        return self._ftl(pred, target.to(self.device))

    def _seg_loss_sleeper(self, pred, target):
        from losses.focal_tversky import DiceBCELoss
        if not hasattr(self, '_dbce'):
            self._dbce = DiceBCELoss().to(self.device)
        return self._dbce(pred, target.to(self.device))

    def _type_loss(self, pred, target):
        ce = nn.CrossEntropyLoss()
        return ce(pred, target.to(self.device).squeeze(1).long())

    def _contrastive_loss(self, emb, labels):
        from losses.focal_tversky import SupervisedContrastiveLoss
        if not hasattr(self, '_contrast'):
            self._contrast = SupervisedContrastiveLoss().to(self.device)
        return self._contrast(emb, labels.to(self.device))

    @staticmethod
    def _resize_targets(
        targets: Dict[str, torch.Tensor],
        preds:   Dict[str, torch.Tensor],
        device:  torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Resize mask pseudo-labels to match prediction spatial size if they differ.
        """
        out = {}
        for k, v in targets.items():
            v = v.to(device)
            if k in ('crack_mask', 'sleeper_mask') and k.replace('mask', 'seg') in preds:
                pred_key = k.replace('mask', 'seg')
                pred_size = preds[pred_key].shape[-2:]
                if v.shape[-2:] != pred_size:
                    v = F.interpolate(
                        v.float(),
                        size=pred_size,
                        mode='nearest',
                    )
            out[k] = v
        return out
