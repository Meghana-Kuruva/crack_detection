"""
Phase 2 — Loss 3: Segmentation Losses
======================================
Section 6.5 of the design doc.

TWO losses for two different heads:

A. FOCAL-TVERSKY LOSS  — for the crack segmentation head
   WHY: Dice loss is dominated by the large background region.  On a 640×640
   tile with a 2-pixel-wide crack, the background occupies >99.9% of pixels.
   Dice alone drives the model to predict everything as background (trivially
   high Dice if it never fires).

   Tversky generalises Dice by separately weighting FP and FN:
       TI = TP / (TP + α·FP + β·FN)
   Setting β > 0.5 penalises FN more than FP → favours recall on thin structures
   → the model is pushed to NOT miss even faint cracks.
   Setting β < 0.5 → favours precision (fewer FPs) → use for ablation.

   Focal Tversky adds a (1 - TI)^γ modulation:
       FTL = (1 - TI)^γ
   This further up-weights examples where the model is completely wrong
   (TI ≈ 0 → FTL ≈ 1; TI ≈ 1 → FTL ≈ 0) — analogous to focal loss.

   Reference: Abraham & Khan, "A novel focal Tversky loss function...", ISBI 2019.

B. DICE + BCE LOSS  — for the sleeper segmentation head
   WHY: The sleeper segmentation task is less imbalanced (sleeper is a large
   region, not a 1-px line), so Dice is not dominated by background.  Dice + BCE
   is the standard reliable choice.  BCE provides per-pixel gradient everywhere
   (Dice has no gradient when TI=1 — plateau issue); Dice handles class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# A.  FOCAL TVERSKY LOSS (crack segmentation)
# ══════════════════════════════════════════════════════════════════════════════

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss for pixel-wise crack segmentation.

    Hyper-parameter guide:
        alpha = 0.3, beta = 0.7  →  penalises FN heavily → high recall (RECOMMENDED
                                      for thin cracks: missing a crack is worse
                                      than a few extra positive pixels)
        alpha = 0.5, beta = 0.5  →  standard Dice
        alpha = 0.7, beta = 0.3  →  penalises FP heavily → high precision
        gamma = 4/3              →  original paper recommendation

    Args:
        alpha : FP weight in Tversky index denominator
        beta  : FN weight in Tversky index denominator  (α + β = 1 is conventional)
        gamma : focusing exponent on hard examples
        smooth: small constant to avoid division by zero on empty GT masks
    """

    def __init__(
        self,
        alpha:  float = 0.3,   # FP penalty — lower = more tolerant of FPs
        beta:   float = 0.7,   # FN penalty — higher = recall-oriented
        gamma:  float = 1.333, # focusing exponent (4/3 from original paper)
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth

    def forward(
        self,
        pred:   torch.Tensor,   # (B, 1, H, W) or (B, H, W) — raw logits or probs
        target: torch.Tensor,   # (B, 1, H, W) or (B, H, W) — binary GT mask
    ) -> torch.Tensor:
        """
        Args:
            pred   : unnormalised logits (sigmoid applied inside) OR probabilities
            target : binary {0, 1} mask (float)

        Returns:
            Scalar Focal Tversky loss.
        """
        # Ensure same shape and apply sigmoid if needed
        pred   = pred.squeeze(1)    if pred.dim()   == 4 else pred
        target = target.squeeze(1)  if target.dim() == 4 else target
        target = target.float()

        # Apply sigmoid to convert logits → probabilities
        # If already probabilities (0..1) this is a no-op at the extremes
        prob = torch.sigmoid(pred)

        # Flatten to (B, H*W) for vectorised computation
        B   = prob.shape[0]
        p   = prob.view(B, -1)
        g   = target.view(B, -1)

        # True positives, false positives, false negatives — per image
        TP = (p * g).sum(dim=1)
        FP = (p * (1 - g)).sum(dim=1)
        FN = ((1 - p) * g).sum(dim=1)

        # Tversky index per image
        TI = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Focal Tversky loss: hard examples (TI≈0) get weight ≈1; easy (TI≈1) get ≈0
        FTL = (1.0 - TI) ** self.gamma

        return FTL.mean()


# ══════════════════════════════════════════════════════════════════════════════
# B.  DICE + BCE LOSS (sleeper segmentation)
# ══════════════════════════════════════════════════════════════════════════════

class DiceBCELoss(nn.Module):
    """
    Dice loss + Binary Cross-Entropy for sleeper region segmentation.

    WHY this combination:
      - BCE: per-pixel gradient everywhere, even when Dice is saturated (TI≈1).
        Prevents the gradient-vanishing plateau that pure Dice suffers at high TI.
      - Dice: handles class imbalance (sleeper vs background) without needing
        explicit pos_weight tuning.

    Args:
        bce_weight  : weight of BCE term relative to Dice (0.5 → equal)
        pos_weight  : optional positive class weight for BCE (if sleeper is rare)
        smooth      : Dice smoothing constant
    """

    def __init__(
        self,
        bce_weight:  float         = 0.5,
        pos_weight:  torch.Tensor  = None,
        smooth:      float         = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth     = smooth

        # pos_weight handles class imbalance in BCE if needed
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None

    def forward(
        self,
        pred:   torch.Tensor,   # (B, 1, H, W) logits
        target: torch.Tensor,   # (B, 1, H, W) binary GT mask
    ) -> torch.Tensor:
        pred   = pred.squeeze(1)    if pred.dim()   == 4 else pred
        target = target.squeeze(1)  if target.dim() == 4 else target
        target = target.float()

        prob = torch.sigmoid(pred)
        B    = prob.shape[0]
        p    = prob.view(B, -1)
        g    = target.view(B, -1)

        # ── Dice term ────────────────────────────────────────────────────────
        TP   = (p * g).sum(dim=1)
        dice = (2 * TP + self.smooth) / (p.sum(dim=1) + g.sum(dim=1) + self.smooth)
        dice_loss = (1 - dice).mean()

        # ── BCE term ─────────────────────────────────────────────────────────
        bce_loss = F.binary_cross_entropy_with_logits(
            pred.float(),
            target.float(),
            pos_weight=self.pos_weight,
            reduction='mean',
        )

        return dice_loss + self.bce_weight * bce_loss


# ══════════════════════════════════════════════════════════════════════════════
# C.  CONTRASTIVE LOSS (crack-vs-ballast, Section 6.3)
# ══════════════════════════════════════════════════════════════════════════════

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss that pulls crack embeddings together and
    pushes them apart from ballast-gap embeddings.  (Section 6.3)

    WHY: failure mode (c) says the crack/ballast decision boundary was never
    trained.  This loss explicitly CARVES that boundary in feature space by
    maximising the distance between crack and ballast embeddings relative to
    the intra-class spread.

    Two versions are implemented:
      'ntxent'  — NT-Xent (normalised temperature-scaled cross-entropy):
                   standard for self-supervised; here adapted for supervised use.
      'triplet' — Triplet margin loss (simpler, easier to tune).

    The projection head (a small 2-layer MLP) maps backbone features to an
    L2-normalised embedding space where this loss operates.  The projection
    head is attached to P5 neck features (highest semantic level).

    Args:
        temperature : temperature for NT-Xent (lower → harder boundary)
        margin      : margin for triplet (only used in 'triplet' mode)
        loss_type   : 'ntxent' or 'triplet'
    """

    def __init__(
        self,
        temperature: float = 0.07,
        margin:      float = 1.0,
        loss_type:   str   = 'ntxent',
    ):
        super().__init__()
        self.temperature = temperature
        self.margin      = margin
        self.loss_type   = loss_type

    def forward(
        self,
        embeddings: torch.Tensor,   # (N, D) L2-normalised feature vectors
        labels:     torch.Tensor,   # (N,) int — 0=crack, 1=ballast-gap hard-negative
    ) -> torch.Tensor:
        """
        Args:
            embeddings : L2-normalised projection vectors (call F.normalize first)
            labels     : 0 for crack samples, 1 for hard-mined ballast negatives

        Returns:
            Scalar contrastive loss.
        """
        if len(embeddings) < 2:
            return torch.tensor(0.0, device=embeddings.device)

        if self.loss_type == 'ntxent':
            return self._ntxent(embeddings, labels)
        else:
            return self._triplet(embeddings, labels)

    def _ntxent(
        self,
        z: torch.Tensor,   # (N, D) normalised
        y: torch.Tensor,   # (N,) labels
    ) -> torch.Tensor:
        """
        Supervised NT-Xent: for each anchor, pull positives (same class)
        and push negatives (different class).

        Loss per anchor i:
            L_i = -log( Σ_{j:y_j=y_i} exp(sim/τ) / Σ_{k≠i} exp(sim/τ) )
        """
        N   = z.shape[0]
        sim = torch.matmul(z, z.T) / self.temperature      # (N, N) cosine sim / tau

        # Mask out self-similarity on the diagonal
        mask_self = torch.eye(N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask_self, float('-inf'))

        # Positive mask: same class, different sample
        same_class = (y.unsqueeze(0) == y.unsqueeze(1))    # (N, N)
        pos_mask   = same_class & ~mask_self

        # log-sum-exp over all non-self (denominator)
        log_denom = torch.logsumexp(sim, dim=1)             # (N,)

        # For each anchor, sum log-prob of all positives
        loss = 0.0
        for i in range(N):
            pos_idx = pos_mask[i].nonzero(as_tuple=False).squeeze(-1)
            if len(pos_idx) == 0:
                continue
            log_nums = sim[i][pos_idx]
            loss -= (log_nums - log_denom[i]).mean()

        return loss / N

    def _triplet(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Online hard triplet mining: for each anchor, pick the hardest
        positive (most distant same-class) and hardest negative (most
        similar different-class).
        """
        dist = torch.cdist(z, z, p=2)   # (N, N) Euclidean distances
        loss = torch.tensor(0.0, device=z.device)
        count = 0

        for i in range(len(z)):
            pos_mask = (y == y[i])
            pos_mask[i] = False
            neg_mask = (y != y[i])

            if not pos_mask.any() or not neg_mask.any():
                continue

            # Hardest positive = furthest same-class sample
            d_pos = dist[i][pos_mask].max()
            # Hardest negative = closest different-class sample
            d_neg = dist[i][neg_mask].min()

            loss  += F.relu(d_pos - d_neg + self.margin)
            count += 1

        return loss / max(count, 1)


# ══════════════════════════════════════════════════════════════════════════════
# D.  REGION-CONDITIONED PENALTY  L_gate  (Section 6.4)
# ══════════════════════════════════════════════════════════════════════════════

def region_gate_loss(
    crack_prob:    torch.Tensor,   # (B, H, W) predicted crack probability (sigmoid)
    sleeper_prob:  torch.Tensor,   # (B, H, W) predicted sleeper probability
    lam: float = 1.0,
) -> torch.Tensor:
    """
    Penalises predicted crack activations that fall outside the sleeper region.

    L_gate = λ · Σ p(crack) · (1 - p(sleeper | pixel))

    This trains the model END-TO-END to avoid firing on ballast regions,
    rather than correcting it post-hoc (Section 8.3 vs 8.1).

    Both maps must be at the same spatial resolution.  In practice, we
    downsample the crack prediction to match the sleeper segmentation output.

    Args:
        crack_prob   : (B, H, W) per-pixel crack probability from crack seg head
        sleeper_prob : (B, H, W) per-pixel sleeper probability from sleeper seg head
        lam          : loss scale (tune via uncertainty weighting in practice)

    Returns:
        Scalar loss.
    """
    if crack_prob.shape != sleeper_prob.shape:
        # Align spatial resolution — bilinear downsample crack_prob to sleeper size
        crack_prob = F.interpolate(
            crack_prob.unsqueeze(1),
            size=sleeper_prob.shape[-2:],
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)

    # Each pixel: if the model thinks it's a crack AND not on the sleeper,
    # that is the error we want to penalise.
    gate = crack_prob * (1.0 - sleeper_prob.detach())   # detach sleeper to avoid conflicting gradients

    return lam * gate.mean()
