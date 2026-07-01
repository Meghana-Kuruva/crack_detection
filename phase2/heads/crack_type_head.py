"""
Phase 2 — Head 3: Crack-Type Classification Head
=================================================
Section 7 of the design doc.

WHY crack-type classification:
  "Crack-type classification adds fine-grained supervision (regularises features)
   and is itself a publishable axis — engineering severity differs by crack type,
   so a type-aware detector has practical value beyond the paper."

  From an FP-reduction standpoint, forcing the model to classify crack TYPE
  regularises the shared features to encode crack-specific texture/geometry
  rather than generic "elongated dark region" (which ballast gaps also are).
  A feature space that distinguishes longitudinal from transverse cracks
  is already implicitly ruling out non-crack structures.

Four crack-type classes (standard in railway engineering):
  0 — Longitudinal crack   (runs along the sleeper length)
  1 — Transverse crack     (runs across the sleeper width)
  2 — Map cracking         (network/web of cracks over a wider area)
  3 — Spalling             (surface concrete breaking away, visible texture loss)

Note on training targets:
  The current Roboflow dataset does NOT have crack-type labels (it uses
  "cracks in midportion", "cracks under rail seat", "damage").  Two options:
    A. Map existing classes → type approximations and train with soft labels.
    B. Use the head as a pre-training target with unlabelled data and add
       manual type labels to a subset.

  The PseudoLabelGenerator (pseudo_labels.py) implements Option A:
    - "damage"            → class 3 (spalling)
    - "cracks in midportion"  → class 0 or 1 (unknown orientation → soft)
    - "cracks under rail seat" → class 1 (transverse, from engineering knowledge)

Architecture:
  Input: P5 neck features (stride 32, semantically rich).
  Global average pool → MLP → 4-way softmax.
  Very small — all the hard work is done by the shared neck.
  One type label per IMAGE (not per detection) — train on images where
  the dominant crack type is known.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


CRACK_TYPE_NAMES = [
    'longitudinal',
    'transverse',
    'map_cracking',
    'spalling',
]


class CrackTypeHead(nn.Module):
    """
    Image-level crack-type classifier operating on P5 neck features.

    Loss: standard CrossEntropyLoss (in multitask_loss.py).

    At inference this outputs a 4-class probability distribution per tile.
    The dominant prediction across tiles of a single sleeper gives the
    image-level crack-type label — useful as an engineering severity output.
    """

    def __init__(
        self,
        in_channels: int = 512,    # P5 channels (512 for yolov8s, 256 for yolov8n)
        hidden_dim:  int = 256,
        n_types:     int = 4,
        dropout:     float = 0.3,  # regularise because crack-type labels will be sparse
    ):
        super().__init__()
        self.n_types = n_types

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # global average pool → (B, C, 1, 1)
            nn.Flatten(),              # → (B, C)
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),                 # GELU slightly better than ReLU for MLP classifiers
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, n_types),
        )

    def forward(self, p5_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p5_features : (B, in_channels, H/32, W/32) P5 neck features

        Returns:
            logits : (B, n_types) — pass to CrossEntropyLoss (no softmax here)
        """
        return self.head(p5_features)

    def predict_type(self, p5_features: torch.Tensor) -> tuple:
        """
        Inference: returns (predicted_class_idx, probabilities).

        Returns:
            class_idx : (B,) int tensor
            probs     : (B, n_types) float tensor
        """
        logits = self.forward(p5_features)
        probs  = torch.softmax(logits, dim=-1)
        return probs.argmax(dim=-1), probs

    def type_name(self, idx: int) -> str:
        return CRACK_TYPE_NAMES[idx] if idx < len(CRACK_TYPE_NAMES) else f'type_{idx}'
