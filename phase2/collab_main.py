import sys
sys.path.insert(0, '/content/phase2')
sys.path.insert(0, '/content/phase1')

from train_phase2 import run_phase2_training

best = run_phase2_training(
    yolo_weights         = '/content/best.pt',   # Phase 1 model
    data_yaml            = '/content/BTP-1/data.yaml',
    epochs               = 80,
    box_type             = 'kld',    # 'gwd' for ablation
    tversky_beta         = 0.7,      # recall-oriented for thin cracks
    use_uncertainty      = True,     # Kendall et al. balancing
    aux_loss_start_epoch = 5,        # let detection stabilise first
    device               = 0,
)
