import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from absl import app, flags
from ml_collections.config_flags import config_flags
from skimage.transform import iradon  # type: ignore
from tqdm import tqdm

# Paths
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataset.load_slices import get_slice_dataloader
from experiments.instant_ngp.model import (
    HashGridEncoder,
    HashLatentField,
    HashLatentMLP,
    HashLatentTransformer,
)
from metrics import psnr
from rendering import (
    ImageModel,
    get_parallel_rays_2d,
    get_ray_aabb_intersection_2d,
    render_parallel_projection,
)
from utils.logging_utils import setup_logger

torch.set_float32_matmul_precision("high")

FLAGS = flags.FLAGS


def _normalize_for_logging(t: torch.Tensor) -> np.ndarray:
    """Min-max normalize a tensor to [0, 1] for visualization only."""
    x = t.detach().cpu().float().numpy()
    x_min = x.min()
    x_max = x.max()
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.zeros_like(x)
    return np.clip(x, 0.0, 1.0)


def _gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian window for SSIM."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]
    return window_2d.view(1, 1, window_size, window_size)


def _ssim(
    pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    """
    Compute SSIM between two images.
    Expects tensors in shape [B, C, H, W] or [1, H, W] / [H, W].
    Returns a scalar tensor.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"SSIM expects tensors of the same shape, got {pred.shape} and {target.shape}"
        )

    # Ensure 4D tensors
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    window = _gaussian_window(window_size, sigma, pred.device)
    padding = window_size // 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=pred.shape[1])
    mu_y = F.conv2d(target, window, padding=padding, groups=target.shape[1])

    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_x_mu_y = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, window, padding=padding, groups=pred.shape[1]) - mu_x_sq
    sigma_y_sq = (
        F.conv2d(target * target, window, padding=padding, groups=target.shape[1]) - mu_y_sq
    )
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=pred.shape[1]) - mu_x_mu_y

    # Constants from the original SSIM paper
    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2) + 1e-8
    )
    return ssim_map.mean()


config_flags.DEFINE_config_file(
    "config",
    "/home/samuele/code/cbct_torch/experiments/instant_ngp/config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def train(argv):
    config = FLAGS.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_loss(pred: torch.Tensor, target: torch.Tensor, is_image: bool) -> torch.Tensor:
        """
        Compute training loss with optional L1 and SSIM components.
        - loss_type: 'mse' (default) or 'l1'
        - ssim_weight: >0 to add SSIM term (only used for image losses)
        """
        loss_type = getattr(config, "loss_type", "mse")
        if loss_type == "l1":
            base_loss = F.l1_loss(pred, target)
        else:
            base_loss = F.mse_loss(pred, target)

        if is_image:
            ssim_weight = float(getattr(config, "ssim_weight", 0.0))
            if ssim_weight > 0.0:
                window_size = int(getattr(config, "ssim_window_size", 11))
                sigma = float(getattr(config, "ssim_sigma", 1.5))
                ssim_val = _ssim(pred, target, window_size=window_size, sigma=sigma)
                # Convert SSIM (higher is better) into a loss term
                base_loss = base_loss + ssim_weight * (1.0 - ssim_val)

        return base_loss

    # Experiment setup
    experiment_root = Path(config.experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)
    wandb.init(project=config.wandb.project_name, config=config, dir=str(experiment_root))
    experiment_dir = experiment_root / wandb.run.id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = experiment_dir / "training.log"
    logger = setup_logger(__name__, log_file)

    logger.info(f"Using device: {device}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Logging to: {log_file}")

    # --- 1. Load Ground Truth Data ---
    full_dataset = get_slice_dataloader(
        root_dir=config.root_dir,
        stage="train",
        num_vols=config.num_vols,
        batch_size=1,
        shuffle=False,
        gray_value_scaling=config.gray_value_scaling,
    ).dataset

    target_slice_idx = config.slice_idx if config.slice_idx < len(full_dataset) else 0
    sample = full_dataset[target_slice_idx]
    gt_image = sample["image"].unsqueeze(0).to(device)  # [1, H, W]
    vol_bbox = sample["vol_bbox"]

    # Ensure shape for grid_sample: [1, 1, H, W]
    if gt_image.dim() == 3:
        gt_image = gt_image.unsqueeze(0)

    logger.info(f"Loaded GT Image: {gt_image.shape}")

    # --- 2. Generate Ground Truth Projections (Sinogram) ---
    # Define Projection Geometry
    num_angles = config.num_angles
    angles = torch.linspace(0, np.pi, num_angles, device=device)  # 0 to 180 degrees

    # Detector setup
    H, W = gt_image.shape[-2:]
    num_det_pixels = max(H, W)

    # Use AABB intersection for rays
    # Use vol_bbox from sample to define physical extent
    vol_bbox = vol_bbox.to(device)  # [2, 3]
    axis = full_dataset.axis
    if axis == 0:
        aabb_min = vol_bbox[0, [1, 2]]
        aabb_max = vol_bbox[1, [1, 2]]
    elif axis == 1:
        aabb_min = vol_bbox[0, [0, 2]]
        aabb_max = vol_bbox[1, [0, 2]]
    elif axis == 2:
        aabb_min = vol_bbox[0, [0, 1]]
        aabb_max = vol_bbox[1, [0, 1]]
    else:
        raise ValueError(f"Invalid axis {axis}")

    # Calculate diagonal of actual AABB
    diag = torch.norm(aabb_max - aabb_min).item()
    detector_width = diag * 1.0

    logger.info(f"Generating Sinogram: {num_angles} angles, {num_det_pixels} pixels")
    logger.info(f"AABB: {aabb_min.cpu().numpy()} to {aabb_max.cpu().numpy()}")

    # Create wrapper model for GT generation
    gt_model = ImageModel(gt_image)

    # Generate rays for ALL angles (for GT)
    rays_o_all, rays_d_all = get_parallel_rays_2d(angles, num_det_pixels, detector_width, device)

    # Compute AABB intersections for GT
    t_min, t_max, hits = get_ray_aabb_intersection_2d(rays_o_all, rays_d_all, aabb_min, aabb_max)
    hits_mask = hits.squeeze(-1)
    # Only integrate over ray segments that actually pass through the image
    t_min = torch.where(hits, t_min.clamp_min(0.0), torch.zeros_like(t_min))
    t_max = torch.where(hits, t_max.clamp_min(0.0), torch.zeros_like(t_max))

    # Render GT sinogram (raw physical units)
    gt_sinogram = render_parallel_projection(
        gt_model,
        rays_o_all,
        rays_d_all,
        near=t_min,
        far=t_max,
        num_samples=config.num_samples_gt,
        rand=False,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        hits_mask=hits_mask,
    )

    # Affine scaling to roughly map GT sinogram values to [-1, 1]
    gt_sinogram_raw = gt_sinogram  # keep a copy for logging / PSNR
    sino_min = gt_sinogram_raw.min()
    sino_max = gt_sinogram_raw.max()
    sino_center = 0.5 * (sino_min + sino_max)
    sino_half_range = 0.5 * (sino_max - sino_min) + 1e-8
    gt_sinogram = (gt_sinogram_raw - sino_center) / sino_half_range  # ~[-1, 1] target for training

    # FBP reconstruction from ground-truth projections
    fbp_gt_image = None
    angles_deg = angles.detach().cpu().numpy() * (180.0 / np.pi)

    try:
        # iradon expects sinogram shape [num_detectors, num_angles]
        fbp_gt_np = iradon(
            gt_sinogram_raw.detach().cpu().numpy().T,
            theta=angles_deg,
            circle=False,
            output_size=max(H, W),
            filter_name="ramp",
        )
        fbp_gt_image = torch.from_numpy(fbp_gt_np)

        plt.figure(figsize=(6, 6))
        plt.imshow(fbp_gt_np, cmap="gray")
        plt.title("FBP Reconstruction (GT projections)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(experiment_dir / "fbp_gt.png")
        plt.close()
    except Exception as e:
        logger.warning(f"FBP reconstruction failed: {e}")

    # Log GT setup
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(gt_image[0, 0].cpu().numpy(), cmap="gray")
    axes[0].set_title("Ground Truth Image")
    axes[1].imshow(gt_sinogram.cpu().numpy(), cmap="gray", aspect="auto")
    axes[1].set_title("Ground Truth Sinogram")
    plt.tight_layout()
    plt.savefig(experiment_dir / "gt_setup.png")
    plt.close()

    wandb.log({"gt_image": wandb.Image(_normalize_for_logging(gt_image[0, 0]), caption="GT Image")})
    # Log raw sinogram (normalized for visualization), not the scaled training target
    wandb.log(
        {"gt_sinogram": wandb.Image(_normalize_for_logging(gt_sinogram_raw), caption="GT Sinogram")}
    )
    if fbp_gt_image is not None:
        wandb.log(
            {
                "gt_fbp": wandb.Image(
                    _normalize_for_logging(fbp_gt_image), caption="FBP from GT projections"
                )
            }
        )

    # --- 3. Initialize Model ---
    transformer = HashLatentTransformer(
        num_levels=config.num_levels,
        hashmap_size=config.hashmap_size,
        features_per_level=config.features_per_level,
        d_model=config.latent_dim,
    ).to(device)

    hash_encoder = HashGridEncoder(
        num_levels=config.num_levels,
        features_per_level=config.features_per_level,
        hashmap_size=config.hashmap_size,
        base_resolution=config.base_resolution,
        per_level_scale=config.per_level_scale,
    ).to(device)

    mlp = torch.nn.Sequential(
        torch.nn.Linear(
            config.num_levels * config.features_per_level,
            config.mlp_hidden_dim,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(config.mlp_hidden_dim, 1),
    ).to(device)

    # model = HashLatentField(transformer, hash_encoder, mlp).to(device)
    model = HashLatentMLP(hash_encoder, mlp, config.features_per_level).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Determine training mode
    training_mode = getattr(config, "training_mode", "meta_learning")
    logger.info(f"Training mode: {training_mode}")

    # Initialize latents for joint optimization mode (persist across epochs)
    if training_mode == "joint":
        joint_latents = model.init_latents(device=device)
        joint_latents.requires_grad = True
        # Optimizer includes both model and latents
        outer_optimizer = torch.optim.Adam(
            list(model.parameters()) + [joint_latents], lr=config.outer_lr
        )
        logger.info("Joint optimization mode: optimizing model and latents together")
    else:
        # Meta-learning mode: only optimize model parameters
        outer_optimizer = torch.optim.Adam(model.parameters(), lr=config.outer_lr)
        joint_latents = None

    # --- 4. Training Loop ---
    batch_size_angles = config.batch_size_angles
    grad_accum_steps = getattr(config, "grad_accum_steps", 1)

    logger.info("Starting training...")
    iter_per_epoch = max(1, num_angles // batch_size_angles)

    # Pre-generate validation grid for image reconstruction
    # Grid in world/AABB coordinates
    y_g, x_g = torch.meshgrid(
        torch.linspace(aabb_min[0].item(), aabb_max[0].item(), H, device=device),
        torch.linspace(aabb_min[1].item(), aabb_max[1].item(), W, device=device),
        indexing="xy",
    )
    grid_coords_world = torch.stack([y_g, x_g], dim=-1).reshape(-1, 2)  # [H*W, 2]

    # Normalize grid to [-1, 1] using the same AABB normalization as ray sampling
    grid_coords = 2.0 * (grid_coords_world - aabb_min) / (aabb_max - aabb_min) - 1.0

    def run_inner_loop(
        model,
        latents,
        num_steps,
        inner_lr,
        rays_o_all,
        rays_d_all,
        t_min,
        t_max,
        hits_mask,
        gt_sinogram,
        num_angles,
        batch_size_angles,
        num_samples,
        device,
        log_stats=False,
        fitting_mode="projection",
        gt_image=None,
        grid_coords=None,
        grad_clip_norm=None,
        latent_clip_value=None,
        loss_clip_max=None,
        check_nan=False,
    ):
        """
        Run inner loop to optimize latents.

        Args:
            model: LatentTransformer model
            latents: Initial latents tensor [1, num_latents, latent_dim]
            num_steps: Number of inner loop steps
            rays_o_all: All ray origins
            rays_d_all: All ray directions
            t_min: Near bounds for all rays
            t_max: Far bounds for all rays
            gt_sinogram: Ground truth sinogram
            num_angles: Total number of angles
            batch_size_angles: Batch size for angle sampling
            num_samples: Number of samples for rendering
            device: Device to use
            log_stats: Whether to log detailed statistics
            fitting_mode: 'projection' or 'image'
            gt_image: Ground truth image (required if fitting_mode='image')
            grid_coords: Grid coordinates (required if fitting_mode='image')

        Returns:
            optimized_latents: Optimized latents
            losses: List of losses for each step
        """
        inner_optimizer = torch.optim.SGD([latents], lr=inner_lr)

        losses = []
        stats = {}

        for inner_step in range(num_steps):
            inner_optimizer.zero_grad()

            if fitting_mode == "projection":
                # Projection-based fitting (original method)
                # Sample random angles
                angle_indices = torch.randperm(num_angles, device=device)[:batch_size_angles]
                rays_o_batch = rays_o_all[angle_indices]
                rays_d_batch = rays_d_all[angle_indices]
                t_min_batch = t_min[angle_indices]
                t_max_batch = t_max[angle_indices]
                hits_batch = hits_mask[angle_indices]
                gt_sino_batch = gt_sinogram[angle_indices]

                # Forward pass
                def model_fn(x):
                    return model(x, latents)

                pred_proj = render_parallel_projection(
                    model_fn,
                    rays_o_batch,
                    rays_d_batch,
                    near=t_min_batch,
                    far=t_max_batch,
                    num_samples=num_samples,
                    rand=False,
                    aabb_min=aabb_min,
                    aabb_max=aabb_max,
                    hits_mask=hits_batch,
                )
                pred_proj = (pred_proj - sino_center) / sino_half_range
                loss = compute_loss(pred_proj, gt_sino_batch, is_image=False)

                # Stability: Clip loss value
                if loss_clip_max is not None:
                    loss = torch.clamp(loss, max=loss_clip_max)

                # Log statistics on first step
                if log_stats and inner_step == 0:
                    with torch.no_grad():
                        stats["pred_proj_min"] = pred_proj.min().item()
                        stats["pred_proj_max"] = pred_proj.max().item()
                        stats["pred_proj_mean"] = pred_proj.mean().item()
                        stats["pred_proj_std"] = pred_proj.std().item()
                        stats["gt_sino_min"] = gt_sino_batch.min().item()
                        stats["gt_sino_max"] = gt_sino_batch.max().item()
                        stats["gt_sino_mean"] = gt_sino_batch.mean().item()
                        stats["gt_sino_std"] = gt_sino_batch.std().item()

            elif fitting_mode == "image":
                # Direct image fitting
                # Forward pass on grid coordinates
                pred_image_flat = model(grid_coords, latents)  # [H*W, 1]
                pred_image = pred_image_flat.reshape(gt_image.shape)  # [1, 1, H, W]
                loss = compute_loss(pred_image, gt_image, is_image=True)

                # Stability: Clip loss value
                if loss_clip_max is not None:
                    loss = torch.clamp(loss, max=loss_clip_max)

                # Log statistics on first step
                if log_stats and inner_step == 0:
                    with torch.no_grad():
                        stats["pred_image_min"] = pred_image.min().item()
                        stats["pred_image_max"] = pred_image.max().item()
                        stats["pred_image_mean"] = pred_image.mean().item()
                        stats["pred_image_std"] = pred_image.std().item()
                        stats["gt_image_min"] = gt_image.min().item()
                        stats["gt_image_max"] = gt_image.max().item()
                        stats["gt_image_mean"] = gt_image.mean().item()
                        stats["gt_image_std"] = gt_image.std().item()
            else:
                raise ValueError(f"Unknown fitting_mode: {fitting_mode}")

            losses.append(loss.item())

            # Log latent statistics on first step
            if log_stats and inner_step == 0:
                with torch.no_grad():
                    stats["latents_norm"] = latents.norm().item()
                    stats["latents_mean"] = latents.mean().item()
                    stats["latents_std"] = latents.std().item()

            loss.backward()

            # Stability: Check for NaN/Inf
            if check_nan:
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        f"NaN/Inf loss detected at inner step {inner_step}, skipping update"
                    )
                    inner_optimizer.zero_grad()
                    continue
                if latents.grad is not None:
                    if torch.isnan(latents.grad).any() or torch.isinf(latents.grad).any():
                        logger.warning(
                            f"NaN/Inf gradients detected at inner step {inner_step}, skipping update"
                        )
                        inner_optimizer.zero_grad()
                        continue

            # Stability: Gradient clipping for latents
            if latents.grad is not None and grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_([latents], grad_clip_norm)

            # Log gradient statistics
            if log_stats and inner_step == 0:
                if latents.grad is not None:
                    stats["latents_grad_norm"] = latents.grad.norm().item()
                    stats["latents_grad_mean"] = latents.grad.mean().item()
                    stats["latents_grad_max"] = latents.grad.abs().max().item()

            inner_optimizer.step()

            # Stability: Clip latent values
            if latent_clip_value is not None:
                with torch.no_grad():
                    latents.clamp_(-latent_clip_value, latent_clip_value)

        return latents, losses, stats if log_stats else {}

    pbar = tqdm(range(config.epochs), desc="Training")

    for epoch in pbar:
        model.train()

        # Learning rate warmup
        if epoch < config.warmup_epochs:
            warmup_factor = (epoch + 1) / config.warmup_epochs
            current_lr = config.outer_lr * warmup_factor
            for param_group in outer_optimizer.param_groups:
                param_group["lr"] = current_lr
        else:
            # After warmup, use full learning rate
            for param_group in outer_optimizer.param_groups:
                param_group["lr"] = config.outer_lr

        # Initialize or use latents based on training mode
        if training_mode == "joint":
            # Use persistent latents in joint mode
            latents = joint_latents
            inner_losses = []
            inner_stats = {}
        else:
            # Meta-learning mode: initialize new latents each epoch
            latents = model.init_latents(device=device)
            latents.requires_grad = True

            # Inner loop: Optimize latents with SGD (first-order approximation)
            # Log detailed stats every 10 epochs
            log_stats = epoch % 10 == 0
            latents, inner_losses, inner_stats = run_inner_loop(
                model,
                latents,
                config.inner_steps,
                config.inner_lr,
                rays_o_all,
                rays_d_all,
                t_min,
                t_max,
                hits_mask,
                gt_sinogram,
                num_angles,
                batch_size_angles,
                config.num_samples,
                device,
                log_stats=log_stats,
                fitting_mode=config.fitting_mode,
                gt_image=gt_image,
                grid_coords=grid_coords,
                grad_clip_norm=getattr(config, "inner_grad_clip_norm", None),
                latent_clip_value=getattr(config, "latent_clip_value", None),
                loss_clip_max=getattr(config, "loss_clip_max", None),
                check_nan=getattr(config, "check_nan", False),
            )

            # Log inner loop statistics
            if log_stats and inner_stats:
                logger.info(f"Epoch {epoch} Inner Loop Stats:")

                # Group stats by meaningful prefixes
                def get_group_name(key):
                    if key.startswith("pred_proj_"):
                        return "Pred projection"
                    elif key.startswith("gt_sino_"):
                        return "GT sinogram"
                    elif key.startswith("pred_image_"):
                        return "Pred image"
                    elif key.startswith("gt_image_"):
                        return "GT image"
                    elif key.startswith("latents_grad_"):
                        return "Latent gradients"
                    elif key.startswith("latents_"):
                        return "Latents"
                    else:
                        return key.split("_")[0].capitalize()

                stats_groups = {}
                for key, value in inner_stats.items():
                    group = get_group_name(key)
                    if group not in stats_groups:
                        stats_groups[group] = {}
                    stats_groups[group][key] = value

                # Log grouped stats
                for group, group_stats in sorted(stats_groups.items()):
                    stats_str = ", ".join(
                        f"{k.split('_')[-1]}={v:.4f}" for k, v in sorted(group_stats.items())
                    )
                    logger.info(f"  {group}: {stats_str}")

                # Automatically log all stats to WandB
                wandb_log_dict = {f"inner/{k}": v for k, v in inner_stats.items()}
                wandb_log_dict["epoch"] = epoch
                wandb.log(wandb_log_dict)

        # Outer loop: Update transformer (and latents in joint mode)
        outer_optimizer.zero_grad()

        # Handle latents based on training mode
        if training_mode == "joint":
            # In joint mode, use latents directly (they're already in the optimizer)
            latents_final = latents
        else:
            # Meta-learning mode: detach latents for outer loop (first-order approximation)
            latents_final = latents.detach().requires_grad_(True)

        fitting_mode = getattr(config, "fitting_mode", "projection")

        # Gradient accumulation for outer loop
        loss_outer_total = 0.0
        for accum_step in range(grad_accum_steps):
            if fitting_mode == "projection":
                # Sample angles for outer loop
                angle_indices = torch.randperm(num_angles, device=device)[:batch_size_angles]
                rays_o_batch = rays_o_all[angle_indices]
                rays_d_batch = rays_d_all[angle_indices]
                t_min_batch = t_min[angle_indices]
                t_max_batch = t_max[angle_indices]
                hits_batch = hits_mask[angle_indices]
                gt_sino_batch = gt_sinogram[angle_indices]

                # Outer loop forward pass with latents
                def model_fn_outer(x):
                    return model(x, latents_final)

                pred_proj = render_parallel_projection(
                    model_fn_outer,
                    rays_o_batch,
                    rays_d_batch,
                    near=t_min_batch,
                    far=t_max_batch,
                    num_samples=config.num_samples,
                    rand=True,
                    aabb_min=aabb_min,
                    aabb_max=aabb_max,
                    hits_mask=hits_batch,
                )
                pred_proj = (pred_proj - sino_center) / sino_half_range
                loss_outer = compute_loss(pred_proj, gt_sino_batch, is_image=False)
            elif fitting_mode == "image":
                # Image fitting outer loss on full grid
                pred_image_flat = model(grid_coords, latents_final)  # [H*W, 1]
                pred_image = pred_image_flat.reshape(gt_image.shape)  # [1, 1, H, W]
                loss_outer = compute_loss(pred_image, gt_image, is_image=True)
            else:
                raise ValueError(f"Unknown fitting_mode: {fitting_mode}")

            # Save scalar for logging before any scaling/clamping
            loss_outer_total += loss_outer.detach().item()

            # Stability: Clip loss value
            loss_clip_max = getattr(config, "loss_clip_max", None)
            if loss_clip_max is not None:
                loss_outer = torch.clamp(loss_outer, max=loss_clip_max)

            # Scale for accumulation
            loss_outer_scaled = loss_outer / grad_accum_steps

            # Backward (only through model weights, not through inner loop)
            retain = accum_step < grad_accum_steps - 1
            loss_outer_scaled.backward(retain_graph=retain)

        # Stability: Clip loss value
        loss_clip_max = getattr(config, "loss_clip_max", None)
        if loss_clip_max is not None:
            # Clamp already-scaled loss (safe for accumulation)
            loss_outer = torch.clamp(loss_outer, max=loss_clip_max)

        # Stability: Check for NaN/Inf
        check_nan = getattr(config, "check_nan", False)
        skip_update = False
        if check_nan:
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    logger.warning(f"NaN/Inf gradients detected at epoch {epoch}, skipping update")
                    skip_update = True
                    break
            if training_mode == "joint" and not skip_update and latents.grad is not None:
                if torch.isnan(latents.grad).any() or torch.isinf(latents.grad).any():
                    logger.warning(
                        f"NaN/Inf latent gradients detected at epoch {epoch}, skipping update"
                    )
                    skip_update = True

        if skip_update:
            outer_optimizer.zero_grad()
            continue

        # Stability: Gradient clipping for model parameters (and latents in joint mode)
        grad_clip_norm = getattr(config, "grad_clip_norm", None)
        grad_norm_before_clip = 0.0
        if grad_clip_norm is not None:
            if training_mode == "joint":
                # Clip both model and latents together
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + [latents], grad_clip_norm
                )
            else:
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm
                )
        else:
            # Compute gradient norm even if not clipping
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm_before_clip += p.grad.norm().item() ** 2
            if training_mode == "joint" and latents.grad is not None:
                grad_norm_before_clip += latents.grad.norm().item() ** 2
            grad_norm_before_clip = grad_norm_before_clip**0.5

        outer_optimizer.step()

        # Stability: Clip latent values (in joint mode, latents are updated by optimizer)
        if training_mode == "joint":
            latent_clip_value = getattr(config, "latent_clip_value", None)
            if latent_clip_value is not None:
                with torch.no_grad():
                    latents.clamp_(-latent_clip_value, latent_clip_value)

        # Logging
        avg_inner_loss = np.mean(inner_losses) if inner_losses else 0.0
        avg_outer_loss = (
            loss_outer_total / grad_accum_steps if grad_accum_steps > 0 else loss_outer.item()
        )
        current_lr = outer_optimizer.param_groups[0]["lr"]
        current_inner_lr = config.inner_lr
        pbar.set_postfix(
            {
                "Inner Loss": f"{avg_inner_loss:.6f}",
                "Outer Loss": f"{avg_outer_loss:.6f}",
                "LR": f"{current_lr:.2e}",
                "Inner LR": f"{current_inner_lr:.3f}",
            }
        )

        log_dict = {
            "train_loss_outer": avg_outer_loss,
            "learning_rate": current_lr,
            "grad_norm": grad_norm_before_clip,
            "epoch": epoch,
        }
        log_dict["train_loss_inner"] = avg_inner_loss
        wandb.log(log_dict)

        # Log model parameter statistics periodically
        if epoch % 10 == 0:
            with torch.no_grad():
                # Parameter norms
                total_norm = 0.0
                for p in model.parameters():
                    if p.requires_grad:
                        total_norm += p.norm().item() ** 2
                total_norm = total_norm**0.5

                # Gradient norms (after outer loop backward)
                total_grad_norm = 0.0
                max_grad = 0.0
                grad_count = 0
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        param_grad_norm = p.grad.norm().item()
                        total_grad_norm += param_grad_norm**2
                        max_grad = max(max_grad, p.grad.abs().max().item())
                        grad_count += 1

                if grad_count > 0:
                    total_grad_norm = total_grad_norm**0.5

                logger.info(f"Epoch {epoch} Model Stats:")
                logger.info(f"  Total parameter norm: {total_norm:.4f}")
                logger.info(f"  Total gradient norm: {total_grad_norm:.4f}")
                logger.info(f"  Max gradient value: {max_grad:.4f}")

                # Log to WandB
                wandb_log_dict = {
                    "model/param_norm": total_norm,
                    "model/grad_norm": total_grad_norm,
                    "model/max_grad": max_grad,
                    "epoch": epoch,
                }

                # Log latent stats in joint mode
                if training_mode == "joint":
                    with torch.no_grad():
                        latent_norm = joint_latents.norm().item()
                        latent_mean = joint_latents.mean().item()
                        latent_std = joint_latents.std().item()
                    logger.info(
                        f"  Latents norm: {latent_norm:.4f}, mean: {latent_mean:.4f}, std: {latent_std:.4f}"
                    )
                    wandb_log_dict["model/latent_norm"] = latent_norm
                    wandb_log_dict["model/latent_mean"] = latent_mean
                    wandb_log_dict["model/latent_std"] = latent_std

                wandb.log(wandb_log_dict)

        # Validation
        if (epoch + 1) % config.val_interval == 0:
            model.eval()

            # Get latents for validation based on training mode
            if training_mode == "joint":
                # Use persistent latents in joint mode
                val_latents = joint_latents.detach()
            else:
                # Meta-learning mode: run inner loop for validation
                val_latents = model.init_latents(device=device)
                val_latents.requires_grad = True

                val_latents, _, _ = run_inner_loop(
                    model,
                    val_latents,
                    config.inner_steps,
                    config.inner_lr,
                    rays_o_all,
                    rays_d_all,
                    t_min,
                    t_max,
                    hits_mask,
                    gt_sinogram,
                    num_angles,
                    batch_size_angles,
                    config.num_samples,
                    device,
                    log_stats=False,
                    fitting_mode=config.fitting_mode,
                    gt_image=gt_image,
                    grid_coords=grid_coords,
                    grad_clip_norm=getattr(config, "inner_grad_clip_norm", None),
                    latent_clip_value=getattr(config, "latent_clip_value", None),
                    loss_clip_max=getattr(config, "loss_clip_max", None),
                    check_nan=getattr(config, "check_nan", False),
                )
                val_latents = val_latents.detach()

            # Reconstruct with final latents (no grad needed here)
            with torch.no_grad():
                rec_flat = model(grid_coords, val_latents)
                rec_image = rec_flat.reshape(H, W)

                log_payload = {
                    "val_image": wandb.Image(
                        _normalize_for_logging(rec_image), caption=f"Rec Ep {epoch}"
                    ),
                    "val_psnr": psnr(rec_image, gt_image[0]).item(),
                }

                # Only reconstruct and log sinogram in projection mode
                if getattr(config, "fitting_mode", "projection") == "projection":
                    val_sinogram_list = []
                    val_batch_size = 16
                    for i in range(0, num_angles, val_batch_size):
                        batch_rays_o = rays_o_all[i : i + val_batch_size]
                        batch_rays_d = rays_d_all[i : i + val_batch_size]
                        batch_t_min = t_min[i : i + val_batch_size]
                        batch_t_max = t_max[i : i + val_batch_size]
                        batch_hits = hits_mask[i : i + val_batch_size]

                        def model_fn_val_sino(x):
                            return model(x, val_latents)

                        batch_proj = render_parallel_projection(
                            model_fn_val_sino,
                            batch_rays_o,
                            batch_rays_d,
                            near=batch_t_min,
                            far=batch_t_max,
                            num_samples=config.num_samples,
                            rand=False,
                            aabb_min=aabb_min,
                            aabb_max=aabb_max,
                            hits_mask=batch_hits,
                        )
                        val_sinogram_list.append(batch_proj)
                    rec_sinogram_raw = torch.cat(val_sinogram_list, dim=0)
                    rec_sinogram = (rec_sinogram_raw - sino_center) / sino_half_range
                    log_payload.update(
                        {
                            "val_sinogram": wandb.Image(
                                _normalize_for_logging(rec_sinogram_raw), caption=f"Sino Ep {epoch}"
                            ),
                            "proj_psnr": psnr(rec_sinogram_raw, gt_sinogram_raw).item(),
                        }
                    )

                    # FBP of reconstructed sinogram
                    try:
                        fbp_val_np = iradon(
                            rec_sinogram_raw.detach().cpu().numpy().T,
                            theta=angles_deg,
                            circle=False,
                            output_size=max(H, W),
                            filter_name="ramp",
                        )
                        fbp_val_image = torch.from_numpy(fbp_val_np)
                        log_payload["val_fbp_pred"] = wandb.Image(
                            _normalize_for_logging(fbp_val_image),
                            caption=f"FBP Pred Ep {epoch}",
                        )
                        if fbp_gt_image is not None:
                            log_payload["val_fbp_gt"] = wandb.Image(
                                _normalize_for_logging(fbp_gt_image),
                                caption="FBP GT",
                            )
                    except Exception as e:
                        logger.warning(f"FBP validation reconstruction failed: {e}")

                wandb.log(log_payload)

                # Save checkpoint
                checkpoint_path = experiment_dir / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "outer_optimizer_state_dict": outer_optimizer.state_dict(),
                    },
                    checkpoint_path,
                )

    logger.info("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    app.run(train)
