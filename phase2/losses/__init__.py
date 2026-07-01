from .kld_gwd        import KLDLoss, GWDLoss, KLDBboxLoss, kld_loss, gwd_loss
from .varifocal      import VarifocalLoss, QualityFocalLoss, build_vfl_targets
from .focal_tversky  import (FocalTverskyLoss, DiceBCELoss,
                             SupervisedContrastiveLoss, region_gate_loss)
from .uncertainty    import UncertaintyWeighting, FixedWeightCombiner
from .multitask_loss import Phase2Loss, TASK_NAMES
