"""
Phase 2 — Loss 2: Varifocal Loss (VFL) for Classification
==========================================================
Section 6.2 of the design doc.

WHY replace standard BCE with VFL:
  Standard BCE treats every positive anchor equally: if the GT box is assigned
  to an anchor, that anchor's target is simply 1.0 regardless of how well the
  predicted box overlaps the GT.

  This is wrong for two reasons in the crack/ballast context:
    (a) A crack detection where the box barely overlaps the GT should be
        penalised more than one with near-perfect overlap.  BCE cannot express
        this quality distinction.
    (b) Easy negatives (smooth concrete background far from any crack) dominate
        the loss and dilute the gradient signal on the hard negatives (ballast
        gaps near the sleeper edge) that the model actually struggles with.

  VFL (Varifocal Loss) fixes both:
    - For POSITIVES: target = IoU(pred, GT), not just 1.0
      → the model learns to predict high confidence ONLY when the box is
        also geometrically accurate.  Bad-box positives get less gradient.
    - For NEGATIVES: weight = α · p^γ  (focal down-weighting of easy negatives)
      → easy background gets near-zero gradient; hard ballast negatives that
        the model is already (wrongly) confident about get large gradient.

  Net effect: the optimizer focuses effort on the hard crack-vs-ballast
  boundary that we actually need to separate.

Reference: Zhang et al. "VarifocalNet: An IoU-aware Dense Object Detector", CVPR 2021.

QFL (Quality Focal Loss) is also implemented as a drop-in alternative
(same idea, slightly different formulation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarifocalLoss(nn.Module):
    """
    VFL: varifocal (IoU-aware) classification loss.

    For a predicted class probability p and an IoU-weighted target q:

        if q > 0 (positive anchor):
            L = -q · log(p)                          ← cross-entropy, target = IoU
        if q == 0 (negative anchor):
            L = -α · p^γ · log(1 - p)               ← focal down-weighting

    Combined (element-wise):
        weight = q * (q > 0).float() + α * p^γ * (q == 0).float()
        L      = BCE(p, q) * weight

    Args:
        alpha : focal weight for negatives (default 0.75 per VFL paper)
        gamma : focusing exponent for negatives (default 2.0)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred_score: torch.Tensor,    # (N, nc) — raw sigmoid scores (after sigmoid)
        target_q:   torch.Tensor,    # (N, nc) — IoU-weighted targets in [0,1]
    ) -> torch.Tensor:
        """
        Args:
            pred_score : predicted class probabilities (sigmoid applied)
            target_q   : per-anchor IoU-weighted targets.
                         target_q[i, c] = IoU(pred_box_i, gt_box) if anchor i is
                         assigned to a GT of class c, else 0.

        Returns:
            Scalar mean VFL loss.
        """
        # Focal weight for negatives: α · p^γ
        # For positives: weight = target_q (the IoU itself serves as weight)
        # For negatives: weight = α · p^γ   (focuses on hard negatives)
        neg_weight = self.alpha * (pred_score.detach() ** self.gamma)

        # Combine: positives use IoU as weight; negatives use focal weight
        is_pos = (target_q > 0).float()
        weight = target_q * is_pos + neg_weight * (1 - is_pos)

        # BCE loss with IoU-weighted targets
        with torch.cuda.amp.autocast(enabled=False):
            bce = F.binary_cross_entropy(
                pred_score.float(),
                target_q.float(),
                reduction='none',
            )

        return (bce * weight).mean()


class QualityFocalLoss(nn.Module):
    """
    QFL: Quality Focal Loss — closely related to VFL.

    L = -|y - p|^β · [(1-y) log(1-p) + y log(p)]

    where y = IoU-weighted target.  Beta=2 matches the original paper.
    At y=0 this becomes standard focal loss; at y>0 it uses IoU quality.

    Reference: Li et al. "Generalized Focal Loss", NeurIPS 2020.
    """

    def __init__(self, beta: float = 2.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        pred_score: torch.Tensor,    # (N, nc) predicted probs
        target_q:   torch.Tensor,    # (N, nc) IoU-quality targets
    ) -> torch.Tensor:
        # Modulating factor |y - p|^beta
        mod = (target_q - pred_score.detach()).abs().pow(self.beta)

        with torch.cuda.amp.autocast(enabled=False):
            bce = F.binary_cross_entropy(
                pred_score.float(),
                target_q.float(),
                reduction='none',
            )

        return (bce * mod).mean()


# ── Utility: build IoU-weighted targets from ultralytics assignment output ────

def build_vfl_targets(
    target_labels:  torch.Tensor,    # (n_pos,) class indices for positive anchors
    target_ious:    torch.Tensor,    # (n_pos,) IoU(pred_box, gt_box) for each positive
    n_anchors:      int,
    n_classes:      int,
    fg_mask:        torch.Tensor,    # (n_anchors,) bool — which anchors are positive
) -> torch.Tensor:
    """
    Build the (n_anchors, nc) IoU-weighted target tensor for VFL/QFL.

    For positive anchors: target[anchor_idx, class_idx] = IoU
    For negative anchors: all zeros

    This replaces the standard 0/1 one-hot target used by BCE.
    The IoU here is the rotated box IoU between the predicted box and the GT
    at the time of label assignment — computed in the trainer and passed here.
    """
    targets = torch.zeros(n_anchors, n_classes, device=fg_mask.device)
    if fg_mask.sum() == 0:
        return targets

    pos_idx = fg_mask.nonzero(as_tuple=False).squeeze(-1)   # (n_pos,)
    cls_idx = target_labels.long()                           # (n_pos,)
    targets[pos_idx, cls_idx] = target_ious.clamp(0, 1)     # IoU as quality score

    return targets
