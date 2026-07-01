# Phase 2: Training-side improvements
# Loss functions → auxiliary heads → multi-task model → custom trainer

from .losses  import (Phase2Loss, KLDLoss, GWDLoss, VarifocalLoss,
                      FocalTverskyLoss, DiceBCELoss, UncertaintyWeighting)
from .heads   import CrackSegHead, SleeperSegHead, CrackTypeHead
from .pseudo_labels    import PseudoLabelGenerator, OBBDatasetWithPseudoLabels
from .multitask_model  import MultiTaskModel
from .custom_trainer   import Phase2Trainer, Phase2TrainArgs
from .train_phase2     import run_phase2_training, run_phase2_ablation, evaluate_phase2
