
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    # Data params
    config.root_dir = "/media/samuele/data/LIDC-IDRI/version20251209"
    config.num_vols = 50
    config.axis = 0
    config.slice_idx = 10
    config.gray_value_scaling = 20.0
    
    # Training params
    config.lr = 5e-4
    config.epochs = 1000
    config.batch_size = 1 # Not used in projection loop directly (we use batch_size_angles)
    config.experiment_root = "/home/samuele/code/cbct_torch/results/rff_net_projection"
    config.val_interval = 10
    
    # Projection specific params
    config.num_samples = 64      # Integrations steps for training
    config.num_samples_gt = 256   # Integration steps for GT generation (high quality)
    config.num_angles = 256       # Total views in sinogram
    config.batch_size_angles = 16 # Views per step
    
    # Model params
    config.hidden_dim = 64
    config.num_layers = 4         # Slightly deeper for reconstruction?
    config.encoding_size = 256
    config.scale = 10.0
    
    # WandB
    config.wandb = ml_collections.ConfigDict()
    config.wandb.project_name = "rffnet_projection_training"
    
    return config
