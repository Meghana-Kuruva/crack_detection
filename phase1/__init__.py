# Phase 1: Root-cause FP fixes for Railway Sleeper Crack Detection
# Import order matches the pipeline execution order.

from .sleeper_segmentation import (
    ClassicalSleeperSegmenter,
    LightweightSleeperUNet,
    extract_sleeper_rois,
)
from .roi_sahi_slicer  import ROISAHISlicer, SlicedTile
from .mask_gating      import MaskGating, compute_box_sleeper_overlap
from .nmm_fusion       import MaskAwareNMM
from .hard_negative_mining import (
    BootstrappedFPMiner,
    BoundaryTileOversampler,
    SyntheticNegativeGenerator,
    run_mining_loop,
)
from .pipeline import Phase1Pipeline, Detection, run_baseline
