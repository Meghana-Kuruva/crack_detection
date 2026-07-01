"""
Phase 2 — Loss 4: Uncertainty-Based Multi-Task Loss Balancing
=============================================================
Section 6.6 of the design doc.

WHY uncertainty weighting instead of hand-tuned weights:
  We have 7 loss terms (box, cls, dfl, crack_seg, sleeper_seg, type_cls,
  contrastive, L_gate).  Manually searching the weight space 7-dimensionally
  is impractical and sensitive to dataset/scale changes.

  Kendall, Gal & Cipolla (NeurIPS 2018) showed that the optimal weight for
  task i is inversely proportional to its HOMOSCEDASTIC UNCERTAINTY (σ_i):

      L_total = Σ_i   (1 / 2σ_i²) · L_i   +   log σ_i

  σ_i is a learnable parameter per task:
    - Large σ_i → task i is uncertain → its loss is down-weighted automatically
    - Small σ_i → task i is confident → its loss carries more weight
    - log σ_i acts as a regulariser that prevents σ_i → ∞

  In practice we learn log(σ_i²) = 2·log(σ_i) for numerical stability:

      w_i   = exp(-s_i)            where s_i = log σ_i²
      L_i'  = 0.5 · w_i · L_i  +  0.5 · s_i

  The 0.5 factors fold in naturally from the 1/(2σ²) formula.

Reference: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh
           Losses for Scene Geometry and Semantics", CVPR 2018.
"""

import torch
import torch.nn as nn
from typing import Dict, List


class UncertaintyWeighting(nn.Module):
    """
    Learnable homoscedastic uncertainty weighting for N tasks.

    Usage:
        weighter = UncertaintyWeighting(n_tasks=7)
        # register as part of the model so it is optimised
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(weighter.parameters()), ...
        )

        # at each step:
        losses = {'box': L_box, 'cls': L_cls, ..., 'contrastive': L_con}
        total, per_task = weighter(losses)
        total.backward()

    The weighter can be inspected during training:
        print(weighter.log_sigma_sq)
        # positive s_i → task is uncertain (down-weighted)
        # negative s_i → task is confident (up-weighted)
    """

    def __init__(self, task_names: List[str]):
        """
        Args:
            task_names : ordered list of task name strings.
                         E.g. ['box', 'cls', 'dfl', 'crack_seg', 'sleeper_seg',
                               'type_cls', 'contrastive', 'gate']
        """
        super().__init__()
        self.task_names = task_names
        n = len(task_names)

        # Initialise log σ² to 0 (σ = 1, weight = 0.5) for all tasks.
        # The optimizer will tune these during training.
        self.log_sigma_sq = nn.Parameter(torch.zeros(n))

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> tuple:
        """
        Compute uncertainty-weighted total loss.

        Args:
            losses : dict mapping task name → scalar loss tensor.
                     Tasks not in the dict are ignored (allows partial
                     multi-task training when some targets are unavailable).

        Returns:
            total_loss   : scalar weighted sum (to backward)
            per_task_log : dict of {task_name: float} raw (unweighted) losses
                           for logging to wandb / console
        """
        total = torch.tensor(0.0, device=self.log_sigma_sq.device)
        per_task_log = {}

        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue

            L_i = losses[name]
            s_i = self.log_sigma_sq[i]     # log σ_i²

            # Uncertainty-weighted contribution:
            #   0.5 · exp(-s_i) · L_i  +  0.5 · s_i
            # = 0.5 · (L_i / σ_i²  +  log σ_i²)
            weight = torch.exp(-s_i)
            total  = total + 0.5 * weight * L_i + 0.5 * s_i

            per_task_log[name] = float(L_i.detach())

        per_task_log['_total_weighted'] = float(total.detach())
        return total, per_task_log

    def get_effective_weights(self) -> Dict[str, float]:
        """
        Return the current effective weight for each task (for logging).
        effective_weight_i = 0.5 · exp(-log_sigma_sq[i])
        """
        weights = {}
        with torch.no_grad():
            for i, name in enumerate(self.task_names):
                weights[name] = float(0.5 * torch.exp(-self.log_sigma_sq[i]))
        return weights


class FixedWeightCombiner(nn.Module):
    """
    Fallback combiner with manually-specified fixed weights.

    Use this in the ablation to compare against uncertainty weighting.
    The design doc Section 13 includes "compare uncertainty vs fixed weights"
    as an implicit ablation.

    Args:
        weights : dict mapping task name → float weight
    """

    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> tuple:
        total = torch.tensor(0.0)
        per_task_log = {}

        for name, L_i in losses.items():
            w_i   = self.weights.get(name, 1.0)
            device = L_i.device
            total  = total.to(device) + w_i * L_i
            per_task_log[name] = float(L_i.detach())

        per_task_log['_total_weighted'] = float(total.detach())
        return total, per_task_log
