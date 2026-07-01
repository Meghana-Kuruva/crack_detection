import sys
sys.path.insert(0, '/content/phase3')
sys.path.insert(0, '/content/phase2')
sys.path.insert(0, '/content/phase1')

from train_phase3 import run_phase3_training

best = run_phase3_training(
    phase2_weights  = '/content/phase2_best.pt',  # Phase 2 model (optional)
    data_yaml       = '/content/BTP-1/data.yaml',
    epochs          = 120,
    batch           = 6,                           # P2 feature map is 4× larger
    lr0             = 5e-4,
    device          = '0',                         # GPU
    project         = '/content/drive/MyDrive/BTP/phase3',
    name            = 'hybrid_afpn_dyhead_film',
    width_mult      = 1.0,
    afpn_out_ch     = 256,
    dyhead_blocks   = 2,
    strip_k         = 9,
    use_film        = True,
    use_uncertainty = True,
    box_type        = 'kld',
    use_wandb       = False,
)
print(f"Phase 3 best model: {best}")
