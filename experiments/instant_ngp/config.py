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
    config.epochs = 5000
    config.experiment_root = "/home/samuele/code/cbct_torch/results/latent_transformer"
    config.val_interval = 100
    # Accumulate gradients across this many micro-batches before an optimizer
    # step. Use >1 to effectively increase the batch size without extra memory.
    config.grad_accum_steps = 1

    # Loss params
    # loss_type: 'mse' (default) or 'l1'
    config.loss_type = "mse"
    # Set ssim_weight > 0.0 to add an SSIM term (only used for image-space losses)
    config.ssim_weight = 0.0
    config.ssim_window_size = 11
    config.ssim_sigma = 1.5

    # Training mode: 'meta_learning' or 'joint'
    # 'meta_learning': Inner loop optimizes latents, outer loop optimizes model (default)
    # 'joint': Optimize both latents and model together with same optimizer
    config.training_mode = "joint"  # or 'joint'

    # Meta-learning params
    config.inner_steps = 3  # SGD steps for latent optimization
    config.inner_lr = 0.01  # Learning rate for latent SGD
    config.outer_lr = 0.00001  # Learning rate for transformer Adam
    config.warmup_epochs = 50  # Number of epochs for learning rate warmup

    # Stability params
    config.grad_clip_norm = 1.0  # Gradient clipping norm (None to disable)
    config.inner_grad_clip_norm = 10.0  # Gradient clipping for inner loop latents
    config.latent_clip_value = None  # Clip latent values to [-value, value] (None to disable)
    config.loss_clip_max = None  # Clip loss to max value (None to disable)
    config.check_nan = False  # Check for NaN/Inf values

    # Projection specific params
    config.num_samples = 256  # Integration steps for rendering
    config.num_samples_gt = 256  # Integration steps for GT generation
    config.num_angles = 400  # Total views in sinogram
    config.batch_size_angles = 400  # Views per step

    # Fitting mode: 'projection' or 'image'
    # 'projection': fit to sinogram projections (current method)
    # 'image': fit directly to image reconstruction
    config.fitting_mode = "projection"  # or 'image'

    # Model params (LatentTransformer)
    config.coord_dim = 2
    config.latent_dim = 128  # Dimension of latent vectors
    config.hidden_dim = 128  # Dimension of transformer hidden states
    config.num_latents = 64
    config.num_decoder_layers = 1
    config.num_cross_attn_layers = 4
    config.num_heads = 4
    config.mlp_hidden_dim = 64
    config.output_dim = 1
    config.dropout = 0.1
    config.rope_base_freq = 10000.0  # Base frequency for RoPE (used for latents)
    config.rope_learnable_freq = True  # Whether to use learnable frequency scaling
    config.rope_coord_freq_multiplier = (
        100.0  # Multiplier for coordinate RoPE frequency (higher = more high-freq)
    )
    config.rff_encoding_size = 128  # RFF encoding size
    config.rff_scale = 7.0  # RFF scale parameter

    config.num_levels = 8
    config.hashmap_size = 2**21
    config.features_per_level = 8
    config.base_resolution = 2
    config.per_level_scale = 1.5
    config.num_latent_tokens = 32
    # WandB
    config.wandb = ml_collections.ConfigDict()
    config.wandb.project_name = "instant_ngp_training"

    return config
