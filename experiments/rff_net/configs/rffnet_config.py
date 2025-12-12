
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    # Data params
    config.root_dir = "/media/samuele/data/LIDC-IDRI/version20251209"
    config.num_vols = 50
    config.axis = 0
    config.slice_idx = 100
    config.gray_value_scaling = 20.0
    
    # Training params
    config.lr = 1e-3
    config.epochs = 1000
    config.batch_size = 1 # Keep 1 as verified
    config.experiment_root = "/home/samuele/code/cbct_torch/results/rff_net"
    config.val_interval = 1
    
    # Model params
    config.hidden_dim = 64
    config.num_layers = 3
    config.encoding_size = 256
    config.scale = 10.0
    
    # WandB
    config.wandb = ml_collections.ConfigDict()
    config.wandb.project_name = "rffnet_single_slice"
    
    return config
