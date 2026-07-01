"""
Phase 2 — Loss 1: Gaussian Distribution Losses for Rotated Boxes
=================================================================
Section 6.1 of the design doc.

WHY replace ProbIoU / rotated-IoU with a Gaussian loss:
  For extreme-aspect-ratio oriented boxes (thin cracks are often 50:1 or more),
  IoU-based rotated losses become numerically unstable.  A tiny error in the
  predicted angle collapses the IoU to near-zero even when the centres and
  widths are correct — the gradient signal disappears exactly where the model
  needs guidance most.

  KLD and GWD treat each OBB as a 2-D Gaussian distribution:
      N(μ, Σ)  where  μ = (cx, cy)
                       Σ = R diag(a², b²) Rᵀ        (a = w/2, b = h/2)

  This representation is continuous and scale-invariant, giving stable
  gradients for all aspect ratios.

Two losses are implemented — both are used in the ablation (Section 13):
  KLD  — symmetric Kullback-Leibler divergence between pred/gt Gaussians.
          Convex, closed-form, very stable.  Preferred first choice.
  GWD  — Gaussian Wasserstein Distance (Bures metric).
          Invariant to rotation when shapes are equal — sometimes better on
          very long thin objects.  Use as the ablation comparison.

Both return (N,) per-box loss tensors so they plug into the standard
task-aligned assignment loop (FG mask selects which N to sum over).
"""

import torch
import torch.nn as nn
from typing import Tuple


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: OBB → 2-D Gaussian parameters
# ══════════════════════════════════════════════════════════════════════════════

def obb_to_gaussian(
    boxes_xywhr: torch.Tensor,   # (..., 5)  [cx, cy, w, h, theta_rad]
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert (cx, cy, w, h, θ) OBB to 2-D Gaussian (μ, Σ).

    Covariance formula:
        Σ = R diag(a², b²) Rᵀ   where a = w/2, b = h/2
                                       R = [[cosθ, -sinθ],[sinθ, cosθ]]

    Expanding:
        Σxx = a²cos²θ + b²sin²θ
        Σxy = (a² - b²) cosθ sinθ
        Σyy = a²sin²θ + b²cos²θ

    Returns:
        mu    : (..., 2)    centre coordinates (cx, cy)
        Sigma : (..., 2, 2) covariance matrices — always PSD by construction
    """
    cx, cy, w, h, theta = boxes_xywhr.unbind(-1)

    # Half-widths squared — clamped so Sigma is strictly positive definite
    a2 = (w / 2).clamp(min=eps) ** 2
    b2 = (h / 2).clamp(min=eps) ** 2

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    Sxx = a2 * cos_t**2 + b2 * sin_t**2
    Sxy = (a2 - b2) * cos_t * sin_t
    Syy = a2 * sin_t**2 + b2 * cos_t**2

    mu = torch.stack([cx, cy], dim=-1)                                 # (..., 2)
    Sigma = torch.stack([
        torch.stack([Sxx, Sxy], dim=-1),
        torch.stack([Sxy, Syy], dim=-1),
    ], dim=-2)                                                          # (..., 2, 2)

    return mu, Sigma


def _inv_2x2(M: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Analytical inverse of a batch of 2×2 matrices.
    Formula: [[a,b],[c,d]]⁻¹ = [[d,-b],[-c,a]] / det
    """
    det = (M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]).clamp(min=eps)
    adj = torch.stack([
        torch.stack([ M[..., 1, 1], -M[..., 0, 1]], dim=-1),
        torch.stack([-M[..., 1, 0],  M[..., 0, 0]], dim=-1),
    ], dim=-2)
    return adj / det.unsqueeze(-1).unsqueeze(-1)


def _det_2x2(M: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return (M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]).clamp(min=eps)


def _sqrt_2x2(M: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Matrix square root of a batch of 2×2 symmetric PSD matrices.

    Uses the Cayley-Hamilton identity for 2×2:
        M^{1/2} = (M + √det(M) · I) / √(tr(M) + 2√det(M))

    Numerically stable for near-zero determinants (thin-crack Gaussians
    are nearly degenerate because w << h).
    """
    det  = _det_2x2(M, eps).clamp(min=0)
    sqd  = det.sqrt()                                           # √det(M)
    tr   = M[..., 0, 0] + M[..., 1, 1]
    s    = (tr + 2 * sqd).clamp(min=eps).sqrt()               # denominator

    I = torch.eye(2, device=M.device, dtype=M.dtype).expand_as(M)
    return (M + sqd.unsqueeze(-1).unsqueeze(-1) * I) / s.unsqueeze(-1).unsqueeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  KLD LOSS  (symmetric)
# ══════════════════════════════════════════════════════════════════════════════

def kld_loss(
    pred_xywhr:   torch.Tensor,   # (N, 5)
    target_xywhr: torch.Tensor,   # (N, 5)
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Symmetric KLD between pred and target 2-D Gaussians.

    KLD(p||q) = 0.5 * [tr(Σq⁻¹ Σp) + (μq-μp)ᵀ Σq⁻¹ (μq-μp) - 2 + ln(|Σq|/|Σp|)]
    Loss      = KLD(p||q) + KLD(q||p)                  (symmetric, ≥0 always)

    Returns:
        loss : (N,) per-box KLD values, mean over FG set during training
    """
    mu_p, Sp = obb_to_gaussian(pred_xywhr,   eps)
    mu_q, Sq = obb_to_gaussian(target_xywhr, eps)

    inv_Sp = _inv_2x2(Sp, eps)
    inv_Sq = _inv_2x2(Sq, eps)

    # trace terms
    tr_qp = (inv_Sq @ Sp).diagonal(dim1=-2, dim2=-1).sum(-1)   # tr(Σq⁻¹ Σp)
    tr_pq = (inv_Sp @ Sq).diagonal(dim1=-2, dim2=-1).sum(-1)   # tr(Σp⁻¹ Σq)

    # Mahalanobis terms — (μq-μp)ᵀ Σ⁻¹ (μq-μp)
    diff  = (mu_q - mu_p).unsqueeze(-1)                         # (N, 2, 1)
    maha_q = (diff.transpose(-2, -1) @ inv_Sq @ diff).squeeze(-1).squeeze(-1)
    maha_p = (diff.transpose(-2, -1) @ inv_Sp @ diff).squeeze(-1).squeeze(-1)

    # Log-det terms
    ln_det_p = _det_2x2(Sp, eps).log()
    ln_det_q = _det_2x2(Sq, eps).log()

    kld_pq = 0.5 * (tr_qp + maha_q - 2 + ln_det_q - ln_det_p)
    kld_qp = 0.5 * (tr_pq + maha_p - 2 + ln_det_p - ln_det_q)

    return (kld_pq + kld_qp).clamp(min=0)


class KLDLoss(nn.Module):
    """Wrapper module for KLD loss — drop-in replacement for RotatedBboxLoss."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_xywhr:   torch.Tensor,   # (N, 5)
        target_xywhr: torch.Tensor,   # (N, 5)
    ) -> torch.Tensor:
        """Returns mean KLD over N positive-assigned boxes."""
        return kld_loss(pred_xywhr, target_xywhr, self.eps).mean()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GWD LOSS  (Gaussian Wasserstein Distance)
# ══════════════════════════════════════════════════════════════════════════════

def gwd_loss(
    pred_xywhr:   torch.Tensor,   # (N, 5)
    target_xywhr: torch.Tensor,   # (N, 5)
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Gaussian Wasserstein Distance (squared, ≈ Bures metric) between pred/target.

    GWD² = ||μp - μq||²  +  W²(Σp, Σq)

    W²(Σp, Σq) = tr(Σp) + tr(Σq) - 2 tr( (Σp^{1/2} Σq Σp^{1/2})^{1/2} )

    WHY prefer GWD over KLD sometimes:
      GWD is invariant to rotation when the two shapes are identical, giving
      smoother gradients when the box is nearly correct in position/size but
      slightly misaligned in angle — common for near-parallel cracks.

    Returns:
        loss : (N,) per-box GWD² values
    """
    mu_p, Sp = obb_to_gaussian(pred_xywhr,   eps)
    mu_q, Sq = obb_to_gaussian(target_xywhr, eps)

    # Centre term
    centre_dist_sq = ((mu_p - mu_q) ** 2).sum(-1)   # (N,)

    # Bures metric
    sqrt_Sp = _sqrt_2x2(Sp, eps)                          # Σp^{1/2}
    M       = sqrt_Sp @ Sq @ sqrt_Sp                       # Σp^{1/2} Σq Σp^{1/2}
    sqrt_M  = _sqrt_2x2(M, eps)

    tr_p   = Sp.diagonal(dim1=-2, dim2=-1).sum(-1)        # tr(Σp)
    tr_q   = Sq.diagonal(dim1=-2, dim2=-1).sum(-1)        # tr(Σq)
    tr_sqM = sqrt_M.diagonal(dim1=-2, dim2=-1).sum(-1)    # tr(√M)

    bures = (tr_p + tr_q - 2 * tr_sqM).clamp(min=0)

    return centre_dist_sq + bures


class GWDLoss(nn.Module):
    """Wrapper module for GWD loss."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_xywhr:   torch.Tensor,
        target_xywhr: torch.Tensor,
    ) -> torch.Tensor:
        return gwd_loss(pred_xywhr, target_xywhr, self.eps).mean()


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM RotatedBboxLoss that slots into ultralytics' v8OBBLoss
# ══════════════════════════════════════════════════════════════════════════════

class KLDBboxLoss(nn.Module):
    """
    Replaces ultralytics' RotatedBboxLoss.

    ultralytics calls this as:
        self.bbox_loss(pred_dist, pred_bboxes, anchor_points,
                       target_bboxes, target_scores, target_scores_sum, fg_mask)

    We ignore pred_dist / anchor_points / DFL (those are for the distribution
    focal loss in the detection regression head, not the box coordinates).
    The pred_bboxes and target_bboxes are already decoded xywhr boxes.
    """

    def __init__(self, reg_max: int, use_dfl: bool = True, loss_type: str = 'kld'):
        super().__init__()
        self.reg_max  = reg_max
        self.use_dfl  = use_dfl
        self.loss_type = loss_type

        # DFL loss is kept from ultralytics — only the IoU→KLD swap matters
        if use_dfl:
            self.dfl_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        pred_dist,          # (B*A, 4*reg_max)  — distribution predictions for DFL
        pred_bboxes,        # (B*A, 5)          — decoded (cx,cy,w,h,theta) in stride units
        anchor_points,      # (A, 2)
        target_bboxes,      # (B*A, 5)          — GT boxes in stride units
        target_scores,      # (B*A, nc)
        target_scores_sum,  # scalar
        fg_mask,            # (B*A,) bool — True for positive-assigned anchors
    ):
        """
        Returns (box_loss, dfl_loss) matching ultralytics' RotatedBboxLoss signature.
        """
        weight = target_scores.sum(-1)[fg_mask]   # (n_pos,) per-box score weight

        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_bboxes.device), \
                   torch.tensor(0.0, device=pred_bboxes.device)

        pred_fg   = pred_bboxes[fg_mask]           # (n_pos, 5)
        target_fg = target_bboxes[fg_mask]         # (n_pos, 5)

        if self.loss_type == 'kld':
            box_loss_vals = kld_loss(pred_fg, target_fg)
        else:
            box_loss_vals = gwd_loss(pred_fg, target_fg)

        # Weight by target score (high-IoU positives are emphasised more)
        box_loss = (box_loss_vals * weight).sum() / target_scores_sum

        # DFL loss (unchanged from ultralytics)
        dfl_loss = torch.tensor(0.0, device=pred_bboxes.device)
        if self.use_dfl and pred_dist is not None:
            dist_fg = pred_dist[fg_mask].view(-1, self.reg_max + 1)
            # Target for DFL comes from the target box; kept unchanged
            dfl_loss = self._dfl(dist_fg, target_fg, weight, target_scores_sum)

        return box_loss, dfl_loss

    def _dfl(self, pred_dist, target_bboxes, weight, norm):
        """
        Distribution Focal Loss — same as ultralytics default.
        Only the box reg part changes; DFL stays the same.
        """
        # Placeholder: in the trainer we pull pred_dist from the YOLO head's
        # raw output and compute DFL here.  For now return zero if not wired.
        return torch.tensor(0.0, device=pred_dist.device)
