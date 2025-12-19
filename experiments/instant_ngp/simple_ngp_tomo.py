import random
import sys
from pathlib import Path

import ml_collections
import numpy as np
import tinycudann as tcnn
import torch
from tqdm import tqdm

import wandb

# Paths
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataset.load_slices import get_slice_dataloader
from experiments.instant_ngp.tomo_projector import (
    get_coordinates,
    intersect_aabb_2d,
    parallel_beam_rays_2d,
)
from metrics import psnr


def get_config():
    """Get configuration for training."""
    config = ml_collections.ConfigDict()

    # Data params
    config.root_dir = "/media/samuele/data/LIDC-IDRI/version20251209"
    config.num_vols = 50
    config.slice_idx = 10
    config.gray_value_scaling = 20.0

    # Model params
    config.n_levels = 7
    config.n_features_per_level = 2
    config.log2_hashmap_size = 16
    config.base_resolution = 10
    config.per_level_scale = 1.6638040352084653
    config.n_neurons = 128
    config.n_hidden_layers = 3

    # Projection params
    config.n_angles = 128
    config.batch_size_angles = 128
    config.n_samples_per_ray = 200
    config.sample_jitter = 0.3  # Random jitter for sample positions (fraction of spacing)

    # Training params
    config.seed = 42
    config.learning_rate = 0.00030952837076999
    config.ema_alpha = 0.9999
    config.max_steps = 20000
    config.log_interval = 2000

    # WandB params
    config.wandb = ml_collections.ConfigDict()
    config.wandb.project_name = "instant-ngp-simple"
    config.wandb.run_name = "simple_ngp_tomo_training"

    return config


# Global config
config = get_config()


def seed_everything(seed: int):
    """
    Set random seeds for reproducibility across different libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior for CUDA operations (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def project_to_sinogram(
    angles,
    detector_coords,
    slice_bbox,
    aabb_min,
    aabb_max,
    n_samples_per_ray,
    sample_fn,
    jitter=False,
    rng=None,
    origin_shift=None,
):
    """
    Common function to project to sinogram using tomo_projector functions.

    This function handles the common logic of:
    - Generating rays
    - Computing AABB intersections
    - Sampling coordinates along rays
    - Normalizing coordinates
    - Integrating along rays

    The actual value sampling is done by the provided sample_fn callback.

    Args:
        angles: (A,) tensor of angles
        detector_coords: (D,) tensor of detector coordinates
        slice_bbox: [2, 2] tensor [[xmin, xmax], [ymin, ymax]]
        aabb_min: [2] tensor (x_min, y_min)
        aabb_max: [2] tensor (x_max, y_max)
        n_samples_per_ray: Number of samples along each ray
        sample_fn: Callback function that takes normalized coordinates [N, 2] and returns values [N]
        jitter: Whether to add jitter to samples
        rng: Random number generator for jitter
        origin_shift: Optional shift along ray direction. If provided, origins are shifted
                     backward by this amount along the ray direction. Can be:
                     - A scalar (same shift for all angles)
                     - A tensor [A] (different shift per angle)
                     - None (no shift, default)

    Returns:
        sinogram: (A, D) tensor of projections
    """
    A = angles.shape[0]
    D = detector_coords.shape[0]

    # Generate rays for all angles
    origins, directions = parallel_beam_rays_2d(
        angles, detector_coords, origin_shift=origin_shift
    )  # [A, D, 2], [A, D, 2]

    # Prepare bboxes for intersect_aabb_2d: [A, 2, 2]
    # Format: [[xmin, xmax], [ymin, ymax]] per angle
    bboxes = slice_bbox.unsqueeze(0).expand(A, -1, -1)  # [A, 2, 2]

    # Compute AABB intersections
    s_enter, s_exit, valid = intersect_aabb_2d(
        origins, directions, bboxes
    )  # [A, D], [A, D], [A, D]

    # Get sample coordinates along rays
    coordinates = get_coordinates(
        origins=origins,
        directions=directions,
        s_enter=s_enter,
        s_exit=s_exit,
        valid=valid,
        num_samples=n_samples_per_ray,
        jitter=jitter,
        rng=rng,
        jitter_scale=config.sample_jitter if hasattr(config, "sample_jitter") else 1.0,
    )  # [A, D, n_samples_per_ray, 2]

    # Normalize coordinates to [0, 1]
    # Map from physical coordinates to [0, 1] based on slice_bbox
    aabb_min_expanded = aabb_min.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]
    aabb_max_expanded = aabb_max.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]

    # Normalize: (coords - aabb_min) / (aabb_max - aabb_min)
    coords_norm = (coordinates - aabb_min_expanded) / (aabb_max_expanded - aabb_min_expanded + 1e-8)
    coords_norm = torch.clamp(coords_norm, 0.0, 1.0)

    # Flatten for sampling: [A * D * n_samples_per_ray, 2]
    coords_norm_flat = coords_norm.reshape(-1, 2)

    # Sample values using the provided callback
    values_flat = sample_fn(coords_norm_flat)  # [A * D * n_samples_per_ray]

    # Reshape back: [A, D, n_samples_per_ray]
    values = values_flat.reshape(A, D, n_samples_per_ray)

    # Integrate along rays using trapezoidal rule
    # Calculate step sizes along each ray
    ray_lengths = s_exit - s_enter  # [A, D]
    step_sizes = ray_lengths / (n_samples_per_ray - 1) if n_samples_per_ray > 1 else ray_lengths
    step_sizes = step_sizes.unsqueeze(-1)  # [A, D, 1]

    # Trapezoidal rule: (first + last)/2 + sum of middle values
    if n_samples_per_ray == 1:
        projections = values[:, :, 0] * step_sizes.squeeze(-1)  # [A, D]
    else:
        projections = (values[:, :, 0] + values[:, :, -1]) / 2.0 + values[:, :, 1:-1].sum(dim=-1)
        projections = projections * step_sizes.squeeze(-1)  # [A, D]

    # Mask out invalid rays
    projections = torch.where(valid, projections, torch.zeros_like(projections))

    return projections


def project_image_to_sinogram(
    img,
    angles,
    detector_coords,
    slice_bbox,
    aabb_min,
    aabb_max,
    n_samples_per_ray,
    jitter=False,
    rng=None,
    origin_shift=None,
):
    """
    Project an image to sinogram using tomo_projector functions.

    Args:
        img: (H, W) image tensor
        angles: (A,) tensor of angles
        detector_coords: (D,) tensor of detector coordinates
        slice_bbox: [2, 2] tensor [[xmin, xmax], [ymin, ymax]]
        aabb_min: [2] tensor (x_min, y_min)
        aabb_max: [2] tensor (x_max, y_max)
        n_samples_per_ray: Number of samples along each ray
        jitter: Whether to add jitter to samples
        rng: Random number generator for jitter
        origin_shift: Optional shift along ray direction. If provided, origins are shifted
                     backward by this amount along the ray direction. Can be:
                     - A scalar (same shift for all angles)
                     - A tensor [A] (different shift per angle)
                     - None (no shift, default)

    Returns:
        sinogram: (A, D) tensor of projections
    """
    # Prepare image for grid_sample: [1, 1, H, W]
    img_batch = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    def sample_fn(coords_norm_flat):
        # Reshape for grid_sample: [A * D * n_samples_per_ray, 1, 2]
        coords_for_grid = coords_norm_flat.reshape(-1, 1, 2)  # [N, 1, 2]

        # normalize coords_for_grid from [0, 1] to [-1, 1]
        coords_for_grid = 2.0 * coords_for_grid - 1.0

        # Sample from image
        sampled = torch.nn.functional.grid_sample(
            img_batch,
            coords_for_grid.unsqueeze(0),  # [1, N, 1, 2]
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [1, 1, N, 1]

        return sampled.squeeze()  # [N]

    return project_to_sinogram(
        angles=angles,
        detector_coords=detector_coords,
        slice_bbox=slice_bbox,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        n_samples_per_ray=n_samples_per_ray,
        sample_fn=sample_fn,
        jitter=jitter,
        rng=rng,
        origin_shift=origin_shift,
    )


def project_model_to_sinogram(
    model,
    angles,
    detector_coords,
    slice_bbox,
    aabb_min,
    aabb_max,
    n_samples_per_ray,
    jitter=True,
    rng=None,
    origin_shift=None,
):
    """
    Project a neural network model to sinogram using tomo_projector functions.

    Args:
        model: Neural network model that takes (N, 2) coordinates and returns (N, 1) values
        angles: (A,) tensor of angles
        detector_coords: (D,) tensor of detector coordinates
        slice_bbox: [2, 2] tensor [[xmin, xmax], [ymin, ymax]]
        aabb_min: [2] tensor (x_min, y_min)
        aabb_max: [2] tensor (x_max, y_max)
        n_samples_per_ray: Number of samples along each ray
        jitter: Whether to add jitter to samples
        rng: Random number generator for jitter
        origin_shift: Optional shift along ray direction. If provided, origins are shifted
                     backward by this amount along the ray direction. Can be:
                     - A scalar (same shift for all angles)
                     - A tensor [A] (different shift per angle)
                     - None (no shift, default)

    Returns:
        sinogram: (A, D) tensor of projections
    """

    def sample_fn(coords_norm_flat):
        # Query model
        return model(coords_norm_flat).squeeze(-1)  # [N]

    return project_to_sinogram(
        angles=angles,
        detector_coords=detector_coords,
        slice_bbox=slice_bbox,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        n_samples_per_ray=n_samples_per_ray,
        sample_fn=sample_fn,
        jitter=jitter,
        rng=rng,
        origin_shift=origin_shift,
    )


def log_training_stats(step, loss, proj, target_proj, model, coords, gt_image, H, W):
    """Log training statistics and metrics."""
    # Projection error
    proj_error = torch.mean((proj - target_proj) ** 2).item()

    # Reconstruct image for visualization stats
    with torch.no_grad():
        reconstructed_img = model(coords).reshape(H, W)
        img_mse = torch.mean((reconstructed_img - gt_image) ** 2).item()
        img_mae = torch.mean(torch.abs(reconstructed_img - gt_image)).item()

    # Gradient statistics
    total_grad_norm = 0.0
    param_count = 0
    for param in model.parameters():
        if param.grad is not None:
            param_grad_norm = param.grad.data.norm(2).item()
            total_grad_norm += param_grad_norm**2
            param_count += 1
    total_grad_norm = total_grad_norm**0.5

    # Compute difference for logging
    diff = torch.abs(reconstructed_img - gt_image)

    # Log to wandb
    wandb_log_dict = {
        "step": step,
        "train_loss": loss.item(),
        "projection_mse": proj_error,
        "image_mse": img_mse,
        "image_mae": img_mae,
        "image_psnr": psnr(reconstructed_img, gt_image).item(),
        "proj_psnr": psnr(proj, target_proj).item(),
        "grad_norm": total_grad_norm,
        "reconstruction": wandb.Image(
            _normalize_for_logging(reconstructed_img), caption=f"Reconstruction Step {step}"
        ),
        "difference": wandb.Image(_normalize_for_logging(diff), caption=f"Difference Step {step}"),
        "sinogram": wandb.Image(_normalize_for_logging(proj), caption=f"Sinogram Step {step}"),
        "sinogram_error": wandb.Image(
            _normalize_for_logging(proj - target_proj), caption=f"Sinogram Error Step {step}"
        ),
    }
    wandb.log(wandb_log_dict)


if __name__ == "__main__":
    # Set random seed for reproducibility
    seed_everything(config.seed)

    device = "cuda"

    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=1,
        encoding_config={
            "otype": "HashGrid",
            "n_levels": config.n_levels,
            "n_features_per_level": config.n_features_per_level,
            "log2_hashmap_size": config.log2_hashmap_size,
            "base_resolution": config.base_resolution,
            "per_level_scale": config.per_level_scale,
        },
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": config.n_neurons,
            "n_hidden_layers": config.n_hidden_layers,
        },
    ).to(device)

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
    gt_image = sample["image"].to(device)  # May be (H, W) or (1, H, W) or (C, H, W)
    vol_bbox = sample["vol_bbox"].to(device)  # [2, 3] - normalized bbox

    # Ensure gt_image is 2D (H, W)
    if gt_image.dim() > 2:
        gt_image = gt_image.squeeze()
        if gt_image.dim() != 2:
            raise ValueError(f"Expected 2D image after squeezing, got shape {gt_image.shape}")

    H = gt_image.shape[-2]
    W = gt_image.shape[-1]

    # Extract 2D slice bbox from vol_bbox based on axis
    axis = full_dataset.axis
    if axis == 0:
        slice_bbox = vol_bbox[:, [1, 2]]  # [2, 2] - (x, y) bbox
    elif axis == 1:
        slice_bbox = vol_bbox[:, [0, 2]]  # [2, 2] - (x, y) bbox
    elif axis == 2:
        slice_bbox = vol_bbox[:, [0, 1]]  # [2, 2] - (x, y) bbox
    else:
        raise ValueError(f"Invalid axis {axis}")

    # slice_bbox is [2, 2] where first dim is [min, max] and second dim is [x, y]
    # Convert to format expected by intersect_aabb_2d: [S, 2, 2] where S is number of samples (angles)
    # Format: [[xmin, xmax], [ymin, ymax]] per sample
    aabb_min = slice_bbox[0]  # [2] - (x_min, y_min)
    aabb_max = slice_bbox[1]  # [2] - (x_max, y_max)
    pixel_size = ((aabb_max[0] - aabb_min[0]) / H, (aabb_max[1] - aabb_min[1]) / W)

    # Create coordinate grid for model input (normalized to [0, 1])
    ys, xs = torch.meshgrid(
        torch.linspace(
            aabb_min[0] + pixel_size[0] / 2, aabb_max[0] - pixel_size[0] / 2, H, device=device
        ),
        torch.linspace(
            aabb_min[1] + pixel_size[1] / 2, aabb_max[1] - pixel_size[1] / 2, W, device=device
        ),
        indexing="ij",
    )
    coords = torch.stack([xs, ys], dim=-1).reshape(-1, 2)

    # normalize coords to [0, 1]
    coords = (coords - aabb_min) / (aabb_max - aabb_min)
    coords = torch.clamp(coords, 0.0, 1.0)

    # Normalize gt_image
    eps = 1e-6
    gt_image_min = gt_image.min()
    gt_image_max = gt_image.max()
    gt_image = (gt_image - gt_image_min) / (gt_image_max - gt_image_min + eps)

    # --- 2. Setup Detector Geometry ---
    # Calculate detector dimension and spacing
    # Use the larger dimension to ensure we cover the full slice
    detector_dim = 512  # int(max(H, W))

    # Calculate detector width based on slice bbox size
    # The detector should cover the diagonal of the slice bbox
    diag = torch.norm(aabb_max - aabb_min).item()
    detector_width = diag * 1.0  # Make detector slightly larger than diagonal

    # Calculate detector spacing
    detector_spacing = detector_width / detector_dim

    # Calculate detector coordinates in the same coordinate system as slice_bbox
    # Center the detector at origin and span detector_width
    detector_bbox = (-0.5 * detector_width, 0.5 * detector_width)
    first_pixel_coord = detector_bbox[0] + detector_spacing / 2
    last_pixel_coord = detector_bbox[1] - detector_spacing / 2
    detector_coords = torch.linspace(
        first_pixel_coord, last_pixel_coord, detector_dim, device=device, dtype=torch.float32
    )

    # --- 3. Generate Angles ---
    angles = torch.linspace(0, np.pi, config.n_angles, device=device, dtype=torch.float32)

    # --- 4. Generate Ground Truth Projections ---
    # Generate ground truth sinogram
    target_proj = project_image_to_sinogram(
        img=gt_image,
        angles=angles,
        detector_coords=detector_coords,
        slice_bbox=slice_bbox,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        n_samples_per_ray=config.n_samples_per_ray,
        jitter=False,
        rng=None,
    )

    # Initialize wandb
    wandb.init(project=config.wandb.project_name, name=config.wandb.run_name, config=config)

    # Log ground truth images
    wandb.log(
        {
            "gt_image": wandb.Image(_normalize_for_logging(gt_image), caption="Ground Truth Image"),
            "gt_sinogram": wandb.Image(
                _normalize_for_logging(target_proj), caption="Ground Truth Sinogram"
            ),
        }
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Initialize EMA (Exponential Moving Average) of model weights
    ema_alpha = config.ema_alpha
    ema_state_dict = {name: param.data.clone() for name, param in model.named_parameters()}

    # Split angles and target projections into chunks
    # first randomize the angles and target projections
    rng_perm = torch.Generator(device=device).manual_seed(config.seed + 1)
    randperm = torch.randperm(angles.shape[0], device=device, generator=rng_perm)
    angles = angles[randperm]
    target_proj = target_proj[randperm]
    angles_chunks = torch.chunk(angles, config.batch_size_angles)
    target_proj_chunks = torch.chunk(target_proj, config.batch_size_angles)

    step = 0
    rng = torch.Generator(device=device).manual_seed(config.seed)

    # Create progress bar
    pbar = tqdm(total=config.max_steps, desc="Training", unit="step")

    while step < config.max_steps:
        for angles_chunk, target_proj_chunk in zip(angles_chunks, target_proj_chunks):
            # Project model to sinogram using the same method
            proj_chunk = project_model_to_sinogram(
                model=model,
                angles=angles_chunk,
                detector_coords=detector_coords,
                slice_bbox=slice_bbox,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
                n_samples_per_ray=config.n_samples_per_ray,
                jitter=True,  # Add randomness to sample positions
                rng=rng,
            )

            loss = torch.mean((proj_chunk - target_proj_chunk) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA weights: ema = alpha * ema + (1 - alpha) * current
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in ema_state_dict:
                        ema_state_dict[name].mul_(ema_alpha).add_(param.data, alpha=1 - ema_alpha)

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            pbar.update(1)

            # Log basic metrics every step
            wandb.log(
                {
                    "step": step,
                    "train_loss": loss.item(),
                }
            )

            if step % config.log_interval == 0:
                log_training_stats(
                    step, loss, proj_chunk, target_proj_chunk, model, coords, gt_image, H, W
                )
            step += 1

    pbar.close()

    # After training: get final reconstruction using EMA weights
    with torch.no_grad():
        # Save current model state
        original_state_dict = model.state_dict()
        # Create full state dict with EMA weights for parameters
        ema_full_state_dict = original_state_dict.copy()
        ema_full_state_dict.update(ema_state_dict)
        # Load EMA weights (strict=False to allow only updating parameters)
        model.load_state_dict(ema_full_state_dict, strict=False)
        # Get reconstruction with EMA weights
        reconstructed_img = model(coords).reshape(H, W)
        # Restore original weights (optional, for consistency)
        model.load_state_dict(original_state_dict)

    # Compute difference
    diff = torch.abs(reconstructed_img - gt_image)

    # Compute final statistics
    final_loss = torch.mean((reconstructed_img - gt_image) ** 2).item()
    final_mae = torch.mean(diff).item()
    final_psnr = psnr(reconstructed_img, gt_image).item()

    # Log final results
    wandb.log(
        {
            "final_image_mse": final_loss,
            "final_image_mae": final_mae,
            "final_image_psnr": final_psnr,
            "final_reconstruction": wandb.Image(
                _normalize_for_logging(reconstructed_img), caption="Final Reconstruction"
            ),
            "final_difference": wandb.Image(
                _normalize_for_logging(diff), caption="Final Difference"
            ),
        }
    )

    wandb.finish()
