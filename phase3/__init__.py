# Phase 3 — Architecture ceiling: hybrid backbone + AFPN + FiLM + DyHead + enhanced OBB head
#
# Sections covered:
#   2.1 — FiLM global-context conditioning (film.py)
#   2.3 — P2 detection level (obb_head.py → MultiLevelOBBHead)
#   2.4 — Asymmetric strip convolutions in head (obb_head.py → StripConvStem)
#   2.5 — Texture-contrast side feature (obb_head.py → TextureContrastBranch)
#   3   — Hybrid backbone: RepViT stem + EfficientViT attention (backbone/)
#   4   — AFPN(P2-P5) + DyHead (neck/)
#   5   — Coordinate Attention + strip pooling (neck/)

from .backbone.hybrid_backbone import HybridBackbone
from .backbone.rep_vit         import RepVitBlock, RepVitStage, SPPF
from .backbone.efficient_vit   import EfficientViTBlock, EfficientViTStage, LinearMHSA

from .neck.afpn            import AFPN
from .neck.dyhead          import DyHead
from .neck.coord_attention import CoordinateAttention
from .neck.strip_pooling   import StripPooling, StripAttention

from .film        import GlobalContextEncoder, FiLMNeck, FiLMGenerator
from .obb_head    import EnhancedOBBHead, MultiLevelOBBHead, StripConvStem, TextureContrastBranch

from .phase3_model import Phase3Model, load_partial_weights
from .train_phase3 import (
    run_phase3_training,
    run_phase3_ablation,
    evaluate_phase3,
    load_phase3_checkpoint,
    Phase3TrainArgs,
)
