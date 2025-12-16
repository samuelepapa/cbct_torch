import sys
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

import tinycudann as tcnn

from dataset.load_slices import get_slice_dataloader
from metrics import psnr
from rendering import (
    ImageModel,
    _normalize_to_aabb,
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
    # num_det_pixels = max(H, W)

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
    num_det_pixels = int(np.ceil(np.sqrt(H**2 + W**2)))

    logger.info(f"Generating Sinogram: {num_angles} angles, {num_det_pixels} pixels")
    logger.info(f"AABB: {aabb_min.cpu().numpy()} to {aabb_max.cpu().numpy()}")

    # Create wrapper model for GT generation
    gt_model = ImageModel(gt_image)

    # Generate rays for ALL angles (for GT)
    rays_o_all, rays_d_all = get_parallel_rays_2d(angles, num_det_pixels, detector_width, device)

    # Compute AABB intersections for GT
    t_min, t_max, hits = get_ray_aabb_intersection_2d(rays_o_all, rays_d_all, aabb_min, aabb_max)

    # Ensure valid segments before passing to the renderer:
    # - order them so near <= far
    # - require finite values
    # - require strictly positive length (far > near)
    t_near = torch.minimum(t_min, t_max)
    t_far = torch.maximum(t_min, t_max)

    is_finite = torch.isfinite(t_near) & torch.isfinite(t_far)
    has_length = t_far > (t_near + 1e-6)

    valid = hits & is_finite & has_length
    hits_mask = valid.squeeze(-1)

    # For invalid rays, set a dummy [0, 0] segment; they will be ignored via hits_mask.
    t_min = torch.where(valid, t_near, torch.zeros_like(t_near))
    t_max = torch.where(valid, t_far, torch.zeros_like(t_far))

    # --- Debug visualization: sample a larger set of rays and their 2D points ---
    # Select a set of angles and detector pixels to visualize.
    num_debug_angles = min(16, num_angles)
    num_debug_det = min(32, num_det_pixels)

    angle_indices_dbg = torch.linspace(
        0, num_angles - 1, steps=num_debug_angles, device=device
    ).long()
    det_indices_dbg = torch.linspace(
        0, num_det_pixels - 1, steps=num_debug_det, device=device
    ).long()

    # Use the same sampling strategy as in render_parallel_projection (deterministic)
    steps_dbg = torch.linspace(0, 1, config.num_samples_gt, device=device).view(
        1, 1, config.num_samples_gt
    )

    ray_points_world = []
    ray_points_norm = []

    for ai in angle_indices_dbg:
        for di in det_indices_dbg:
            if not hits_mask[ai, di]:
                continue

            ro = rays_o_all[ai, di]  # [2]
            rd = rays_d_all[ai, di]  # [2]
            t0 = t_min[ai, di]  # [1]
            t1 = t_max[ai, di]  # [1]

            # Sample along this ray segment
            z_vals_dbg = t0 + steps_dbg.squeeze(0).squeeze(0) * (t1 - t0)  # [S]
            pts_dbg = ro.unsqueeze(0) + rd.unsqueeze(0) * z_vals_dbg.unsqueeze(-1)  # [S, 2]

            # Normalize to model input space [-1, 1]
            pts_norm_dbg = _normalize_to_aabb(pts_dbg, aabb_min, aabb_max)  # [S, 2]

            ray_points_world.append(pts_dbg.detach().cpu().numpy())
            ray_points_norm.append(pts_norm_dbg.detach().cpu().numpy())

    if ray_points_world:
        # Concatenate for scatter plotting
        world_concat = np.concatenate(ray_points_world, axis=0)  # [N, 2] (y, x)
        norm_concat = np.concatenate(ray_points_norm, axis=0)  # [N, 2] (x, y) in [-1,1]

        # Optionally subsample if too many points
        max_points = 5000
        if world_concat.shape[0] > max_points:
            idx = np.random.choice(world_concat.shape[0], size=max_points, replace=False)
            world_concat = world_concat[idx]
            norm_concat = norm_concat[idx]

        # Plot world-space rays over the GT image and normalized coordinates separately.
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Left: GT image with ray samples in world coordinates (scatter)
        axes[0].imshow(
            gt_image[0, 0].detach().cpu().numpy(),
            cmap="gray",
            extent=[
                aabb_min[1].item(),
                aabb_max[1].item(),
                aabb_min[0].item(),
                aabb_max[0].item(),
            ],
            origin="lower",
        )
        axes[0].scatter(
            world_concat[:, 1],
            world_concat[:, 0],
            s=4,
            c="red",
            alpha=0.4,
        )
        axes[0].set_title("Ray samples in world space")
        axes[0].set_xlim(aabb_min[1].item(), aabb_max[1].item())
        axes[0].set_ylim(aabb_min[0].item(), aabb_max[0].item())

        # Right: normalized coordinates that go into the model (scatter)
        axes[1].set_aspect("equal")
        axes[1].scatter(
            norm_concat[:, 0],
            norm_concat[:, 1],
            s=4,
            c="blue",
            alpha=0.4,
        )
        axes[1].set_title("Ray samples in normalized space (model input)")
        axes[1].set_xlim(-1.1, 1.1)
        axes[1].set_ylim(-1.1, 1.1)

        plt.tight_layout()
        plt.savefig(experiment_dir / "ray_sampling_debug.png")
        plt.close(fig)

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
                    _normalize_for_logging(fbp_gt_image),
                    caption="FBP from GT projections",
                )
            }
        )

    # --- 3. Initialize Model ---
    # Standard Instant-NGP style hash-encoded MLP (no latents / meta-learning).
    # Force float32 for both encoding and network to avoid Half/Float mismatches
    # in tiny-cuda-nn custom kernels.
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=1,
        encoding_config={
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 14,
            "base_resolution": 16,
            "per_level_scale": 1.5,
            "fixed_point_pos": False,
            "dtype": "float32",
        },
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
            "dtype": "float32",
        },
    ).to(device)

    # --- 4. Training Loop ---
    batch_size_angles = config.batch_size_angles
    grad_accum_steps = getattr(config, "grad_accum_steps", 1)

    logger.info("Starting training...")
    # Number of angle batches per epoch (for potential scheduling/diagnostics)
    iter_per_epoch = max(1, num_angles // batch_size_angles)
    _ = iter_per_epoch

    # Pre-generate validation grid for image reconstruction
    # Grid in world/AABB coordinates
    pixel_size_w = (aabb_max[1].item() - aabb_min[1].item()) / W
    pixel_size_h = (aabb_max[0].item() - aabb_min[0].item()) / H
    y_g, x_g = torch.meshgrid(
        torch.linspace(
            aabb_min[0].item() + pixel_size_h / 2,
            aabb_max[0].item() - pixel_size_h / 2,
            H,
            device=device,
        ),
        torch.linspace(
            aabb_min[1].item() + pixel_size_w / 2,
            aabb_max[1].item() - pixel_size_w / 2,
            W,
            device=device,
        ),
        indexing="xy",
    )
    grid_coords_world = torch.stack([y_g, x_g], dim=-1).reshape(-1, 2)  # [H*W, 2]

    # Normalize grid to [-1, 1] using the same AABB normalization as ray sampling
    grid_coords = _normalize_to_aabb(grid_coords_world, aabb_min, aabb_max)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Simple optimizer over model parameters only.
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=config.outer_lr)

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

        # --- Single-level optimization: update model parameters directly ---
        outer_optimizer.zero_grad()

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

                pred_proj = render_parallel_projection(
                    model,
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
                loss_outer = compute_loss(
                    pred_proj, gt_sino_batch.to(pred_proj.dtype), is_image=False
                )
            elif fitting_mode == "image":
                # Image fitting outer loss on full grid
                pred_image_flat = model(grid_coords)  # [H*W, 1]
                pred_image = pred_image_flat.reshape(gt_image.shape)  # [1, 1, H, W]
                loss_outer = compute_loss(pred_image, gt_image.to(pred_image.dtype), is_image=True)
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

        if skip_update:
            outer_optimizer.zero_grad()
            continue

        # Stability: Gradient clipping for model parameters
        grad_clip_norm = getattr(config, "grad_clip_norm", None)
        grad_norm_before_clip = 0.0
        if grad_clip_norm is not None:
            grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip_norm
            )
        else:
            # Compute gradient norm even if not clipping
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm_before_clip += p.grad.norm().item() ** 2
            grad_norm_before_clip = grad_norm_before_clip**0.5

        outer_optimizer.step()

        # Logging
        avg_outer_loss = (
            loss_outer_total / grad_accum_steps if grad_accum_steps > 0 else loss_outer.item()
        )
        current_lr = outer_optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            {
                "Loss": f"{avg_outer_loss:.6f}",
                "LR": f"{current_lr:.2e}",
            }
        )

        log_dict = {
            "train_loss": avg_outer_loss,
            "learning_rate": current_lr,
            "grad_norm": grad_norm_before_clip,
            "epoch": epoch,
        }
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

                wandb.log(wandb_log_dict)

        # Validation
        if (epoch + 1) % config.val_interval == 0:
            model.eval()

            with torch.no_grad():
                rec_flat = model(grid_coords)
                rec_image = rec_flat.reshape(H, W)

                # Ensure shapes match: gt_image is [1, 1, H, W], so use gt_image[0, 0] to get [H, W]
                gt_image_2d = gt_image[0, 0] if gt_image.dim() == 4 else gt_image[0]

                log_payload = {
                    "val_image": wandb.Image(
                        _normalize_for_logging(rec_image), caption=f"Rec Ep {epoch}"
                    ),
                    "val_psnr": psnr(rec_image, gt_image_2d.to(rec_image.dtype)).item(),
                }

                # Only reconstruct and log sinogram if we're actually training on projections
                if getattr(config, "fitting_mode", "projection") == "projection":
                    val_sinogram_list = []
                    val_batch_size = 16
                    for i in range(0, num_angles, val_batch_size):
                        batch_rays_o = rays_o_all[i : i + val_batch_size]
                        batch_rays_d = rays_d_all[i : i + val_batch_size]
                        batch_t_min = t_min[i : i + val_batch_size]
                        batch_t_max = t_max[i : i + val_batch_size]
                        batch_hits = hits_mask[i : i + val_batch_size]

                        batch_proj = render_parallel_projection(
                            model,
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

                    # Compute and visualize sinogram error
                    sino_error = rec_sinogram_raw - gt_sinogram_raw

                    # Save error plot to disk
                    plt.figure(figsize=(10, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(
                        _normalize_for_logging(gt_sinogram_raw),
                        cmap="gray",
                        aspect="auto",
                    )
                    plt.title("GT Sinogram")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(
                        _normalize_for_logging(rec_sinogram_raw),
                        cmap="gray",
                        aspect="auto",
                    )
                    plt.title("Rec Sinogram")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(
                        sino_error.detach().cpu().numpy(),
                        cmap="bwr",
                        aspect="auto",
                    )
                    plt.title("Sino Error (Rec - GT)")
                    plt.axis("off")

                    plt.tight_layout()
                    plt.savefig(experiment_dir / f"sino_error_epoch_{epoch + 1}.png")
                    plt.close()

                    log_payload.update(
                        {
                            "val_sinogram": wandb.Image(
                                _normalize_for_logging(rec_sinogram_raw),
                                caption=f"Sino Ep {epoch}",
                            ),
                            "val_sinogram_error": wandb.Image(
                                _normalize_for_logging(sino_error),
                                caption=f"Sino Error Ep {epoch}",
                            ),
                            "proj_psnr": psnr(rec_sinogram_raw, gt_sinogram_raw).item(),
                            "proj_mse": torch.mean(
                                (rec_sinogram_raw - gt_sinogram_raw) ** 2
                            ).item(),
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
                checkpoint_path = experiment_dir / f"checkpoint_epoch_{epoch + 1}.pt"
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
