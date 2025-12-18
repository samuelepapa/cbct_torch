import sys
from pathlib import Path

import ml_collections
import numpy as np
import tinycudann as tcnn
import torch

import wandb

# Paths
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataset.load_slices import get_slice_dataloader
from experiments.instant_ngp.tomo_projector import (
    get_coordinates,
    get_coordinates_fixed_step,
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
    config.n_angles = 256
    config.batch_size_angles = 16
    config.n_samples_per_ray = 209
    config.detector_spacing = (
        0.003  # Physical spacing between detector pixels (in normalized units)
    )
    config.use_fixed_step = (
        False  # If True, use get_coordinates_fixed_step; if False, use get_coordinates
    )
    config.step_size = 0.01  # Step size for fixed_step mode (only used if use_fixed_step=True)
    config.sample_jitter = (
        0.0033  # Random jitter for sample positions (fraction of spacing or step_size)
    )

    # Training params
    config.learning_rate = 0.00030952837076999
    config.max_steps = 20000
    config.log_interval = 1000

    # WandB params
    config.wandb = ml_collections.ConfigDict()
    config.wandb.project_name = "instant-ngp-simple"
    config.wandb.run_name = "simple_ngp_tomo_training"

    return config


# Global config
config = get_config()


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

print(f"Slice bbox: min={aabb_min.cpu().numpy()}, max={aabb_max.cpu().numpy()}")
print(f"Image shape: {H}x{W}")

# Create coordinate grid for model input (normalized to [0, 1])
ys, xs = torch.meshgrid(
    torch.linspace(0, 1, H, device=device), torch.linspace(0, 1, W, device=device), indexing="ij"
)
coords = torch.stack([xs, ys], dim=-1).reshape(-1, 2)

# Normalize gt_image
eps = 1e-6
gt_image_min = gt_image.min()
gt_image_max = gt_image.max()
gt_image = (gt_image - gt_image_min) / (gt_image_max - gt_image_min + eps)

print(f"GT image stats: mean={gt_image.mean().item():.6f}, std={gt_image.std().item():.6f}")
print(f"GT image range: [{gt_image.min().item():.6f}, {gt_image.max().item():.6f}]")

# --- 2. Setup Detector Geometry ---
# Calculate detector dimension and spacing
# Use the larger dimension to ensure we cover the full slice
detector_dim = int(max(H, W))

# Calculate detector width based on slice bbox size
# The detector should cover the diagonal of the slice bbox
diag = torch.norm(aabb_max - aabb_min).item()
detector_width = diag * 1.0  # Make detector slightly larger than diagonal

# Calculate detector spacing
detector_spacing = detector_width / detector_dim

# Alternative: use config detector_spacing if provided (in same coordinate system as slice_bbox)
if hasattr(config, "detector_spacing") and config.detector_spacing is not None:
    detector_spacing = config.detector_spacing
    detector_width = detector_spacing * detector_dim

print(f"Detector: dim={detector_dim}, spacing={detector_spacing:.6f}, width={detector_width:.6f}")

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
def project_image_to_sinogram(
    img,
    angles,
    detector_coords,
    slice_bbox,
    n_samples_per_ray,
    use_fixed_step=False,
    step_size=0.01,
    jitter=False,
    rng=None,
):
    """
    Project an image to sinogram using tomo_projector functions.

    Args:
        img: (H, W) image tensor
        angles: (A,) tensor of angles
        detector_coords: (D,) tensor of detector coordinates
        slice_bbox: [2, 2] tensor [[xmin, xmax], [ymin, ymax]]
        n_samples_per_ray: Number of samples along each ray
        use_fixed_step: If True, use fixed step size; if False, use adaptive sampling
        step_size: Step size for fixed_step mode
        jitter: Whether to add jitter to samples
        rng: Random number generator for jitter

    Returns:
        sinogram: (A, D) tensor of projections
    """
    A = angles.shape[0]
    D = detector_coords.shape[0]

    # Generate rays for all angles
    origins, directions = parallel_beam_rays_2d(angles, detector_coords)  # [A, D, 2], [A, D, 2]

    # Prepare bboxes for intersect_aabb_2d: [A, 2, 2]
    # Format: [[xmin, xmax], [ymin, ymax]] per angle
    bboxes = slice_bbox.unsqueeze(0).expand(A, -1, -1)  # [A, 2, 2]

    # Compute AABB intersections
    s_enter, s_exit, valid = intersect_aabb_2d(
        origins, directions, bboxes
    )  # [A, D], [A, D], [A, D]

    # Get sample coordinates along rays
    if use_fixed_step:
        coordinates = get_coordinates_fixed_step(
            origins=origins,
            directions=directions,
            s_enter=s_enter,
            s_exit=s_exit,
            valid=valid,
            step_size=step_size,
            num_samples=n_samples_per_ray,
            jitter=jitter,
            rng=rng,
            jitter_scale=config.sample_jitter if hasattr(config, "sample_jitter") else 1.0,
        )  # [A, D, n_samples_per_ray, 2]
    else:
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

    # Normalize coordinates to [0, 1] for grid_sample
    # Map from physical coordinates to [0, 1] based on slice_bbox
    aabb_min_expanded = aabb_min.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]
    aabb_max_expanded = aabb_max.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]

    # Normalize: (coords - aabb_min) / (aabb_max - aabb_min)
    coords_norm = (coordinates - aabb_min_expanded) / (aabb_max_expanded - aabb_min_expanded + 1e-8)
    coords_norm = torch.clamp(coords_norm, 0.0, 1.0)

    # Reshape for grid_sample: [A * D * n_samples_per_ray, 1, 2]
    coords_norm_flat = coords_norm.reshape(-1, 1, 2)  # [A * D * n_samples_per_ray, 1, 2]

    # Prepare image for grid_sample: [1, 1, H, W]
    img_batch = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Sample from image
    sampled = torch.nn.functional.grid_sample(
        img_batch,
        coords_norm_flat.unsqueeze(0),  # [1, A * D * n_samples_per_ray, 1, 2]
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # [1, 1, A * D * n_samples_per_ray, 1]

    # Reshape back: [A, D, n_samples_per_ray]
    values = sampled.squeeze().reshape(A, D, n_samples_per_ray)

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


# Generate ground truth sinogram
print("Generating ground truth sinogram...")
with torch.no_grad():
    target_proj = project_image_to_sinogram(
        gt_image,
        angles,
        detector_coords,
        slice_bbox,
        config.n_samples_per_ray,
        use_fixed_step=config.use_fixed_step,
        step_size=config.step_size if config.use_fixed_step else None,
        jitter=False,
        rng=None,
    )

print("Target projection statistics:")
print(f"Mean: {target_proj.mean().item():.6f}")
print(f"Std: {target_proj.std().item():.6f}")
print(f"Min: {target_proj.min().item():.6f}")
print(f"Max: {target_proj.max().item():.6f}")

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

# Create RNG for jitter during training
rng = torch.Generator(device=device)
rng.manual_seed(42)


def project_model_to_sinogram(
    model,
    angles,
    detector_coords,
    slice_bbox,
    n_samples_per_ray,
    use_fixed_step=False,
    step_size=0.01,
    jitter=True,
    rng=None,
):
    """
    Project a neural network model to sinogram using tomo_projector functions.

    Args:
        model: Neural network model that takes (N, 2) coordinates and returns (N, 1) values
        angles: (A,) tensor of angles
        detector_coords: (D,) tensor of detector coordinates
        slice_bbox: [2, 2] tensor [[xmin, xmax], [ymin, ymax]]
        n_samples_per_ray: Number of samples along each ray
        use_fixed_step: If True, use fixed step size; if False, use adaptive sampling
        step_size: Step size for fixed_step mode
        jitter: Whether to add jitter to samples
        rng: Random number generator for jitter

    Returns:
        sinogram: (A, D) tensor of projections
    """
    A = angles.shape[0]
    D = detector_coords.shape[0]

    # Generate rays for all angles
    origins, directions = parallel_beam_rays_2d(angles, detector_coords)  # [A, D, 2], [A, D, 2]

    # Prepare bboxes for intersect_aabb_2d: [A, 2, 2]
    bboxes = slice_bbox.unsqueeze(0).expand(A, -1, -1)  # [A, 2, 2]

    # Compute AABB intersections
    s_enter, s_exit, valid = intersect_aabb_2d(
        origins, directions, bboxes
    )  # [A, D], [A, D], [A, D]

    # Get sample coordinates along rays
    if use_fixed_step:
        coordinates = get_coordinates_fixed_step(
            origins=origins,
            directions=directions,
            s_enter=s_enter,
            s_exit=s_exit,
            valid=valid,
            step_size=step_size,
            num_samples=n_samples_per_ray,
            jitter=jitter,
            rng=rng,
            jitter_scale=config.sample_jitter if hasattr(config, "sample_jitter") else 1.0,
        )  # [A, D, n_samples_per_ray, 2]
    else:
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

    # Normalize coordinates to [0, 1] for model input
    aabb_min_expanded = aabb_min.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]
    aabb_max_expanded = aabb_max.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]

    # Normalize: (coords - aabb_min) / (aabb_max - aabb_min)
    coords_norm = (coordinates - aabb_min_expanded) / (aabb_max_expanded - aabb_min_expanded + 1e-8)
    coords_norm = torch.clamp(coords_norm, 0.0, 1.0)

    # Flatten for batch query: [A * D * n_samples_per_ray, 2]
    coords_norm_flat = coords_norm.reshape(-1, 2)

    # Query model
    values_flat = model(coords_norm_flat).squeeze(-1)  # [A * D * n_samples_per_ray]

    # Reshape back: [A, D, n_samples_per_ray]
    values = values_flat.reshape(A, D, n_samples_per_ray)

    # Integrate along rays using trapezoidal rule
    ray_lengths = s_exit - s_enter  # [A, D]
    step_sizes = ray_lengths / (n_samples_per_ray - 1) if n_samples_per_ray > 1 else ray_lengths
    step_sizes = step_sizes.unsqueeze(-1)  # [A, D, 1]

    if n_samples_per_ray == 1:
        projections = values[:, :, 0] * step_sizes.squeeze(-1)  # [A, D]
    else:
        projections = (values[:, :, 0] + values[:, :, -1]) / 2.0 + values[:, :, 1:-1].sum(dim=-1)
        projections = projections * step_sizes.squeeze(-1)  # [A, D]

    # Mask out invalid rays
    projections = torch.where(valid, projections, torch.zeros_like(projections))

    return projections


def log_training_stats(step, loss, proj, target_proj, model, coords, gt_image, H, W):
    """Log training statistics and metrics."""
    print(f"\n{'='*60}")
    print(f"Step {step:04d} - Training Statistics")
    print("=" * 60)

    # Loss information
    print(f"Loss: {loss.item():.6f}")

    # Projection statistics
    print("\nProjection Statistics:")
    print(f"  Mean: {proj.mean().item():.6f} (target: {target_proj.mean().item():.6f})")
    print(f"  Std:  {proj.std().item():.6f} (target: {target_proj.std().item():.6f})")
    print(f"  Min:  {proj.min().item():.6f} (target: {target_proj.min().item():.6f})")
    print(f"  Max:  {proj.max().item():.6f} (target: {target_proj.max().item():.6f})")

    # Projection error
    proj_error = torch.mean((proj - target_proj) ** 2).item()
    print(f"  Projection MSE: {proj_error:.6f}")

    # Reconstruct image for visualization stats
    with torch.no_grad():
        reconstructed_img = model(coords).reshape(H, W)
        img_mse = torch.mean((reconstructed_img - gt_image) ** 2).item()
        img_mae = torch.mean(torch.abs(reconstructed_img - gt_image)).item()

    print("\nReconstruction Statistics:")
    print(f"  Image MSE: {img_mse:.6f}")
    print(f"  Image MAE: {img_mae:.6f}")
    print(
        f"  Reconstructed mean: {reconstructed_img.mean().item():.6f} (target: {gt_image.mean().item():.6f})"
    )
    print(
        f"  Reconstructed std:  {reconstructed_img.std().item():.6f} (target: {gt_image.std().item():.6f})"
    )

    # Gradient statistics
    total_grad_norm = 0.0
    param_count = 0
    for param in model.parameters():
        if param.grad is not None:
            param_grad_norm = param.grad.data.norm(2).item()
            total_grad_norm += param_grad_norm**2
            param_count += 1
    total_grad_norm = total_grad_norm**0.5

    print("\nGradient Statistics:")
    print(f"  Total gradient norm: {total_grad_norm:.6f}")
    print(f"  Parameters with gradients: {param_count}")

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

    print("=" * 60 + "\n")


# Split angles into batches
angles_chunks = torch.chunk(angles, config.batch_size_angles)
target_proj_chunks = torch.chunk(target_proj, config.batch_size_angles)

step = 0
while step < config.max_steps:
    for angles_chunk, target_proj_chunk in zip(angles_chunks, target_proj_chunks):
        # Project model to sinogram
        proj_chunk = project_model_to_sinogram(
            model,
            angles_chunk,
            detector_coords,
            slice_bbox,
            config.n_samples_per_ray,
            use_fixed_step=config.use_fixed_step,
            step_size=config.step_size if config.use_fixed_step else None,
            jitter=True,
            rng=rng,
        )

        loss = torch.mean((proj_chunk - target_proj_chunk) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

# After training: get final reconstruction
with torch.no_grad():
    reconstructed_img = model(coords).reshape(H, W)

# Compute difference
diff = torch.abs(reconstructed_img - gt_image)

# Print final statistics
final_loss = torch.mean((reconstructed_img - gt_image) ** 2).item()
final_mae = torch.mean(diff).item()
final_psnr = psnr(reconstructed_img, gt_image).item()
print(f"\nFinal reconstruction MSE: {final_loss:.6f}")
print(f"Final reconstruction MAE: {final_mae:.6f}")
print(f"Final reconstruction PSNR: {final_psnr:.6f}")

# Log final results
wandb.log(
    {
        "final_image_mse": final_loss,
        "final_image_mae": final_mae,
        "final_image_psnr": final_psnr,
        "final_reconstruction": wandb.Image(
            _normalize_for_logging(reconstructed_img), caption="Final Reconstruction"
        ),
        "final_difference": wandb.Image(_normalize_for_logging(diff), caption="Final Difference"),
    }
)

wandb.finish()
