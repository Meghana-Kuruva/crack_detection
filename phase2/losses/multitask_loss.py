"""
Phase 2 — Loss 5: Combined Multi-Task Loss Orchestrator
========================================================
Wires all 7 loss terms into one callable that the custom trainer uses.

Loss terms and their sections:
  L_box        — KLD/GWD rotated-box regression      (Section 6.1)
  L_cls        — Varifocal classification loss        (Section 6.2)
  L_dfl        — Distribution focal loss              (Section 6.1, unchanged)
  L_crack_seg  — Focal-Tversky crack mask loss        (Section 6.5)
  L_sleeper_seg— Dice+BCE sleeper mask loss           (Section 6.5)
  L_type_cls   — CrossEntropy crack-type loss         (Section 7)
  L_contrastive— Supervised contrastive crack/ballast (Section 6.3)
  L_gate       — Region-conditioned off-sleeper penalty (Section 6.4)

Uncertainty weights (σ_i) are learnable — the weighter auto-balances them.

The class is designed to be a DROP-IN replacement for ultralytics' v8OBBLoss.
The trainer receives (total_loss, loss_items_dict) from a call to this class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .kld_gwd       import kld_loss, gwd_loss
from .varifocal     import VarifocalLoss, build_vfl_targets
from .focal_tversky import FocalTverskyLoss, DiceBCELoss, SupervisedContrastiveLoss, region_gate_loss
from .uncertainty   import UncertaintyWeighting


TASK_NAMES = ['box', 'cls', 'dfl', 'crack_seg', 'sleeper_seg', 'type_cls',
              'contrastive', 'gate']


class Phase2Loss(nn.Module):
    """
    Master Phase 2 loss.

    Called by the custom trainer as:
        total_loss, items = phase2_loss(preds, batch, aux_preds)

    where:
        preds      — standard ultralytics OBB head output (decoded boxes, scores)
        batch      — ultralytics batch dict (images, bboxes, cls, batch_idx, …)
        aux_preds  — dict from auxiliary heads:
                       'crack_seg'   : (B, 1, H, W) logits
                       'sleeper_seg' : (B, 1, H, W) logits
                       'type_cls'    : (B, n_crack_types) logits
                       'contrastive' : (N, D) L2-normed embeddings + labels dict

    Args:
        nc          : number of detection classes (from data.yaml)
        box_type    : 'kld' or 'gwd'  (Section 6.1 ablation)
        tversky_alpha, tversky_beta, tversky_gamma  (Section 6.5 ablation)
        contrastive_temperature  (Section 6.3)
        gate_lambda  : λ for L_gate  (Section 6.4)
        n_crack_types: 4 (longitudinal / transverse / map-cracking / spalling)
    """

    def __init__(
        self,
        nc:                      int   = 3,
        box_type:                str   = 'kld',
        tversky_alpha:           float = 0.3,
        tversky_beta:            float = 0.7,
        tversky_gamma:           float = 1.333,
        contrastive_temperature: float = 0.07,
        gate_lambda:             float = 1.0,
        n_crack_types:           int   = 4,
        use_uncertainty:         bool  = True,
    ):
        super().__init__()
        self.nc          = nc
        self.box_type    = box_type
        self.gate_lambda = gate_lambda

        # ── detection sub-losses ─────────────────────────────────────────────
        self.vfl      = VarifocalLoss(alpha=0.75, gamma=2.0)
        self.bce_dfl  = nn.BCEWithLogitsLoss(reduction='sum')  # DFL (unchanged)

        # ── auxiliary sub-losses ─────────────────────────────────────────────
        self.focal_tversky = FocalTverskyLoss(
            alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma
        )
        self.dice_bce  = DiceBCELoss()
        self.type_ce   = nn.CrossEntropyLoss()
        self.contrast  = SupervisedContrastiveLoss(
            temperature=contrastive_temperature, loss_type='ntxent'
        )

        # ── uncertainty combiner ─────────────────────────────────────────────
        if use_uncertainty:
            self.weighter = UncertaintyWeighting(task_names=TASK_NAMES)
        else:
            from .uncertainty import FixedWeightCombiner
            self.weighter = FixedWeightCombiner(weights={
                'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
                'crack_seg': 1.0, 'sleeper_seg': 0.5,
                'type_cls': 0.3, 'contrastive': 0.1, 'gate': 0.5,
            })

    # ── main entry point ──────────────────────────────────────────────────────

    def forward(
        self,
        # Standard ultralytics OBB detection output
        pred_bboxes:   torch.Tensor,    # (B, n_anchors, 5) decoded (cx,cy,w,h,θ)
        pred_scores:   torch.Tensor,    # (B, n_anchors, nc) class probs (sigmoid applied)
        target_bboxes: torch.Tensor,    # (n_pos, 5) GT boxes assigned to positive anchors
        target_labels: torch.Tensor,    # (n_pos,) GT class indices
        target_ious:   torch.Tensor,    # (n_pos,) predicted-vs-GT IoU for VFL targets
        fg_mask:       torch.Tensor,    # (B*n_anchors,) bool — which anchors are positive
        target_scores_sum: float,       # normalisation denominator
        batch_size:    int,

        # Auxiliary task predictions (optional — train progressively)
        aux: Optional[Dict[str, torch.Tensor]] = None,

        # Auxiliary task targets (optional — from pseudo_labels.py)
        aux_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple:
        """
        Compute all Phase 2 losses and return (total, items_dict).

        Returns:
            total_loss  : scalar (backprop through this)
            items_dict  : {task_name: float} for logging
        """
        losses: Dict[str, torch.Tensor] = {}

        # ── 1. Box regression: KLD or GWD ────────────────────────────────────
        if fg_mask.any():
            pred_fg   = pred_bboxes.view(-1, 5)[fg_mask]
            target_fg = target_bboxes

            if self.box_type == 'kld':
                raw_box = kld_loss(pred_fg, target_fg)
            else:
                raw_box = gwd_loss(pred_fg, target_fg)

            # Weight by per-box VFL quality score (same as ultralytics standard)
            box_weights = pred_scores.view(-1, self.nc)[fg_mask].sum(-1)
            losses['box'] = (raw_box * box_weights).sum() / max(target_scores_sum, 1)
        else:
            losses['box'] = torch.tensor(0.0, device=pred_scores.device)

        # ── 2. Classification: Varifocal Loss ────────────────────────────────
        # Build IoU-weighted targets first
        n_anchors_total = pred_scores.shape[0] * pred_scores.shape[1]
        vfl_targets = build_vfl_targets(
            target_labels, target_ious, n_anchors_total, self.nc, fg_mask
        )
        losses['cls'] = self.vfl(
            pred_scores.view(-1, self.nc),
            vfl_targets,
        )

        # ── 3. DFL (unchanged from ultralytics — placeholder) ────────────────
        # DFL is computed by the ultralytics head internally when present;
        # we include a zero here so the weighter sees it and assigns a sigma.
        losses['dfl'] = torch.tensor(0.0, device=pred_scores.device)

        # ── 4. Auxiliary task losses (if heads are available + targets exist) ─
        if aux is not None and aux_targets is not None:

            # 4a. Crack segmentation — Focal-Tversky
            if 'crack_seg' in aux and 'crack_mask' in aux_targets:
                losses['crack_seg'] = self.focal_tversky(
                    aux['crack_seg'], aux_targets['crack_mask']
                )

            # 4b. Sleeper segmentation — Dice + BCE
            if 'sleeper_seg' in aux and 'sleeper_mask' in aux_targets:
                losses['sleeper_seg'] = self.dice_bce(
                    aux['sleeper_seg'], aux_targets['sleeper_mask']
                )

            # 4c. Crack-type classification — CrossEntropy
            if 'type_cls' in aux and 'crack_type' in aux_targets:
                losses['type_cls'] = self.type_ce(
                    aux['type_cls'],
                    aux_targets['crack_type'].long(),
                )

            # 4d. Contrastive (crack vs ballast hard negatives) — NT-Xent
            if 'contrastive_emb' in aux and 'contrastive_labels' in aux_targets:
                emb  = F.normalize(aux['contrastive_emb'], dim=-1)
                labs = aux_targets['contrastive_labels']
                losses['contrastive'] = self.contrast(emb, labs)

            # 4e. Region-conditioned gate loss — L_gate
            if 'crack_seg' in aux and 'sleeper_seg' in aux:
                crack_prob   = torch.sigmoid(aux['crack_seg'].squeeze(1))
                sleeper_prob = torch.sigmoid(aux['sleeper_seg'].squeeze(1))
                losses['gate'] = region_gate_loss(
                    crack_prob, sleeper_prob, lam=self.gate_lambda
                )

        # ── 5. Uncertainty-weighted combination ───────────────────────────────
        total_loss, log_dict = self.weighter(losses)

        # Scale by batch size — matches ultralytics convention
        total_loss = total_loss * batch_size

        return total_loss, log_dict
