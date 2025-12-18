import sys
from pathlib import Path

import ml_collections
import numpy as np
import tinycudann as tcnn
import torch

import wandb


# Custom Gaussian blur implementation for 2D images (H, W)
def gaussian_blur(img, kernel_size, sigma=None):
    """Simple Gaussian blur using convolution for 2D images (H, W)."""
    if sigma is None:
        sigma = kernel_size / 3.0
    # Create Gaussian kernel
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Ensure image is 2D (H, W)
    original_shape = img.shape
    if img.dim() > 2:
        # Remove extra dimensions if present
        img = img.squeeze()
        if img.dim() != 2:
            raise ValueError(f"Expected 2D image (H, W), got shape {original_shape}")

    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=img.dtype, device=img.device) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Add channel and batch dimensions for conv2d: (H, W) -> (1, 1, H, W)
    img = img.unsqueeze(0).unsqueeze(0)

    # Create 2D separable kernels
    kernel_h = kernel_1d.view(1, 1, kernel_size, 1)  # for vertical convolution
    kernel_w = kernel_1d.view(1, 1, 1, kernel_size)  # for horizontal convolution

    # Apply separable convolution (more efficient)
    padding = kernel_size // 2
    img = torch.nn.functional.conv2d(img, kernel_h, padding=(padding, 0))
    img = torch.nn.functional.conv2d(img, kernel_w, padding=(0, padding))

    # Remove added dimensions: (1, 1, H, W) -> (H, W)
    img = img.squeeze(0).squeeze(0)

    return img


# Paths
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataset.load_slices import get_slice_dataloader
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
    #     Best trial:
    # Trial 128 finished with value: 32.35655212402344 and parameters: {'n_levels': 4, 'n_features_per_level': 4, 'log2_hashmap_size': 15, 'base_resolution': 11, 'per_level_scale': 2.373908218883449, 'n_neurons': 128, 'n_hidden_layers': 3, 'n_samples_per_ray': 96, 'sample_jitter': 0.0007600737759883335, 'learning_rate': 0.0004098646482382532, 'max_steps': 8000}. Best is trial 128 with value: 32.35655212402344
    # Trial 275 finished with value: 32.72503662109375 and parameters: {'n_levels': 7, 'n_features_per_level': 2, 'log2_hashmap_size': 16, 'base_resolution': 9, 'per_level_scale': 1.600807516154985, 'n_neurons': 128, 'n_hidden_layers': 3, 'n_samples_per_ray': 212, 'sample_jitter': 0.0045606420892276185, 'learning_rate': 0.0005399823715276308, 'max_steps': 7000}. Best is trial 275 with value: 32.72503662109375.
    # Trial 286 finished with value: 33.16468048095703 and parameters: {'n_levels': 7, 'n_features_per_level': 2, 'log2_hashmap_size': 16, 'base_resolution': 9, 'per_level_scale': 1.6073298978528778, 'n_neurons': 128, 'n_hidden_layers': 3, 'n_samples_per_ray': 221, 'sample_jitter': 0.004374339027976014, 'learning_rate': 0.0005335316666433174, 'max_steps': 8000}. Best is trial 286 with value: 33.16468048095703.
    # Trial 352 finished with value: 33.90443420410156 and parameters: {'n_levels': 7, 'n_features_per_level': 2, 'log2_hashmap_size': 16, 'base_resolution': 10, 'per_level_scale': 1.6756345264670869, 'n_neurons': 128, 'n_hidden_layers': 3, 'n_samples_per_ray': 205, 'sample_jitter': 0.004255569066274094, 'learning_rate': 0.0004141017850642394, 'max_steps': 8000}. Best is trial 352 with value: 33.90443420410156.
    # Trial 430 finished with value: 33.947425842285156 and parameters: {'n_levels': 7, 'n_features_per_level': 2, 'log2_hashmap_size': 16, 'base_resolution': 10, 'per_level_scale': 1.6638040352084653, 'n_neurons': 128, 'n_hidden_layers': 3, 'n_samples_per_ray': 209, 'sample_jitter': 0.0033266020658749965, 'learning_rate': 0.00050952837076999, 'max_steps': 8000}. Best is trial 430 with value: 33.947425842285156.

    # Projection params
    config.n_angles = 256
    config.batch_size_angles = 16
    config.n_samples_per_ray = 209
    config.max_dist = 1.0
    config.sample_jitter = (
        0.0033  # Random jitter for sample positions when using model (fraction of spacing)
    )

    # Training params
    config.learning_rate = 0.00030952837076999
    config.max_steps = 20000
    config.log_interval = 1000

    # WandB params
    config.wandb = ml_collections.ConfigDict()
    config.wandb.project_name = "instant-ngp-simple"
    config.wandb.run_name = "simple_ngp_training"

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

# Ensure gt_image is 2D (H, W)
if gt_image.dim() > 2:
    gt_image = gt_image.squeeze()
    if gt_image.dim() != 2:
        raise ValueError(f"Expected 2D image after squeezing, got shape {gt_image.shape}")

H = gt_image.shape[-2]
W = gt_image.shape[-1]

ys, xs = torch.meshgrid(
    torch.linspace(0, 1, H, device=device), torch.linspace(0, 1, W, device=device), indexing="ij"
)
print(ys.shape, xs.shape)

coords = torch.stack([xs, ys], dim=-1).reshape(-1, 2)


with torch.no_grad():
    # gt_image = torch.exp(-((xs-0.5)**2 + (ys-0.5)**2) / 0.02)
    gt_image = gt_image.clone()
# apply a blur to the gt_image
# torchvision's gaussian_blur expects kernel_size as a list/tuple, and doesn't support sigma directly
# Using our custom implementation that supports sigma parameter
# gt_image = gaussian_blur(gt_image, kernel_size=11, sigma=2.0)

print(gt_image.mean().item(), gt_image.std().item())
print(gt_image.min().item(), gt_image.max().item())
print(gt_image.dtype)

# normalize gt_image between eps and 1-eps
eps = 1e-6
gt_image_min = gt_image.min()
gt_image_max = gt_image.max()
gt_image = (gt_image - gt_image_min) / (gt_image_max - gt_image_min + eps)
# gt_image = gt_image*2 - 1
# gt_image = torch.clamp(gt_image, min=eps, max=1-eps)
print(gt_image.mean().item(), gt_image.std().item())
print(gt_image.min().item(), gt_image.max().item())


# Convert angles to tensor
angles_torch = torch.linspace(0, np.pi, config.n_angles, device=device)
n_detectors = int(max(H, W))  # Number of detectors


def project_parallel_beam_vectorized(
    img_or_model,
    angles,
    n_detectors,
    n_samples_per_ray=100,
    max_dist=1.0,
    is_model=False,
    sample_jitter=0.01,
):
    """
    Vectorized parallel beam projection - much faster than loop-based version.

    For images: uses grid_sample to interpolate values along rays
    For models: queries the model directly at sampled points (batched)

    img_or_model: Either a (H, W) image tensor or a model function
    angles: (A,) tensor of angles in radians
    n_detectors: Number of detector positions
    n_samples_per_ray: Number of points to sample along each ray
    max_dist: Maximum distance for ray sampling
    is_model: If True, img_or_model is a model function; if False, it's an image tensor
    sample_jitter: Random jitter for sample positions when using model (fraction of spacing, 0.0 to disable)

    Returns: (A, n_detectors) tensor of projections
    """
    device = angles.device
    A = angles.shape[0]

    # Detector positions along the projection line
    s_bins = torch.linspace(-max_dist, max_dist, n_detectors, device=device)  # (n_detectors,)

    # Sample positions along the ray (perpendicular to the projection direction)
    t_samples = torch.linspace(
        -max_dist, max_dist, n_samples_per_ray, device=device
    )  # (n_samples_per_ray,)

    # Add randomness to sample positions if using model
    if is_model and sample_jitter > 0.0:
        # Calculate spacing between samples
        dt = 2.0 * max_dist / (n_samples_per_ray - 1) if n_samples_per_ray > 1 else 1.0
        # Add random jitter: uniform noise in [-sample_jitter * dt, sample_jitter * dt]
        jitter = (torch.rand(n_samples_per_ray, device=device) * 2.0 - 1.0) * sample_jitter * dt
        t_samples = t_samples + jitter

    # Precompute cos and sin for all angles
    cos_theta = torch.cos(angles)  # (A,)
    sin_theta = torch.sin(angles)  # (A,)

    # Vectorize: create all coordinate combinations at once
    # For each (angle, detector, sample): x = s*cos(θ) - t*sin(θ), y = s*sin(θ) + t*cos(θ)
    # Shape: (A, n_detectors, n_samples_per_ray)
    s_expanded = s_bins[None, :, None]  # (1, n_detectors, 1)
    t_expanded = t_samples[None, None, :]  # (1, 1, n_samples_per_ray)
    cos_expanded = cos_theta[:, None, None]  # (A, 1, 1)
    sin_expanded = sin_theta[:, None, None]  # (A, 1, 1)

    # Compute all coordinates at once: (A, n_detectors, n_samples_per_ray)
    x_coords = s_expanded * cos_expanded - t_expanded * sin_expanded
    y_coords = s_expanded * sin_expanded + t_expanded * cos_expanded

    if is_model:
        # For neural network: normalize to [0, 1] and batch query
        x_coords_norm = (x_coords + 1.0) / 2.0
        y_coords_norm = (y_coords + 1.0) / 2.0
        x_coords_norm = torch.clamp(x_coords_norm, 0.0, 1.0)
        y_coords_norm = torch.clamp(y_coords_norm, 0.0, 1.0)

        # Flatten for batch query: (A * n_detectors * n_samples_per_ray, 2)
        ray_coords = torch.stack(
            [x_coords_norm, y_coords_norm], dim=-1
        )  # (A, n_detectors, n_samples_per_ray, 2)
        ray_coords_flat = ray_coords.reshape(-1, 2)  # (A * n_detectors * n_samples_per_ray, 2)

        # Batch query the model
        values_flat = img_or_model(ray_coords_flat).squeeze(
            -1
        )  # (A * n_detectors * n_samples_per_ray,)
        values = values_flat.reshape(
            A, n_detectors, n_samples_per_ray
        )  # (A, n_detectors, n_samples_per_ray)
    else:
        # For image: use grid_sample - batch all samples at once
        img = img_or_model
        img_batch = img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Flatten all coordinates: (A * n_detectors * n_samples_per_ray, 2)
        all_coords = torch.stack(
            [x_coords, y_coords], dim=-1
        )  # (A, n_detectors, n_samples_per_ray, 2)
        all_coords_flat = all_coords.reshape(-1, 2)  # (A * n_detectors * n_samples_per_ray, 2)

        # Create grid for grid_sample: (1, A * n_detectors * n_samples_per_ray, 1, 2)
        grid = all_coords_flat.unsqueeze(0).unsqueeze(
            2
        )  # (1, A * n_detectors * n_samples_per_ray, 1, 2)

        # Sample from image (single batch call)
        sampled = torch.nn.functional.grid_sample(
            img_batch, grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # (1, 1, A * n_detectors * n_samples_per_ray, 1)

        # Reshape back: (A, n_detectors, n_samples_per_ray)
        values = (
            sampled.squeeze(0).squeeze(0).squeeze(-1).reshape(A, n_detectors, n_samples_per_ray)
        )

    # Vectorized integration using trapezoidal rule
    dt = 2.0 * max_dist / (n_samples_per_ray - 1) if n_samples_per_ray > 1 else 1.0
    if n_samples_per_ray == 1:
        projections = values[:, :, 0] * dt  # (A, n_detectors)
    else:
        # Trapezoidal rule: (first + last)/2 + sum of middle values
        projections = (values[:, :, 0] + values[:, :, -1]) / 2.0 + values[:, :, 1:-1].sum(dim=-1)
        projections = projections * dt  # (A, n_detectors)

    return projections


# Use the vectorized function directly (compilation requires C compiler which may not be available)
# The vectorization alone provides significant speedup
project_parallel_beam = project_parallel_beam_vectorized


# Compute target projection using the same method (with grid_sample for image)
target_proj = project_parallel_beam(
    gt_image,
    angles_torch,
    n_detectors,
    n_samples_per_ray=config.n_samples_per_ray,
    max_dist=config.max_dist,
    is_model=False,
    sample_jitter=0.0,  # No jitter for ground truth
)

print("Target projection statistics:")
print(f"Mean: {target_proj.mean().item():.6f}")
print(f"Std: {target_proj.std().item():.6f}")
print(f"Min: {target_proj.min().item():.6f}")
print(f"Max: {target_proj.max().item():.6f}")
print(f"Type: {target_proj.dtype}")

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


def log_training_stats(step, loss, proj, target_proj, model, coords, gt_image, H, W):
    """Log training statistics and metrics."""
    print(f"\n{'='*60}")
    print(f"Step {step:04d} - Training Statistics")
    print(f"{'='*60}")

    # Loss information
    print(f"Loss: {loss.item():.6f}")

    # Projection statistics
    print(f"\nProjection Statistics:")
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

    print(f"\nReconstruction Statistics:")
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

    print(f"\nGradient Statistics:")
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

    print(f"{'='*60}\n")


# split angles_torch into batch_size_angles chunks
angles_torch_chunks = torch.chunk(angles_torch, config.batch_size_angles)
target_proj_chunks = torch.chunk(target_proj, config.batch_size_angles)

step = 0
while step < config.max_steps:

    for angles_torch_chunk, target_proj_chunk in zip(angles_torch_chunks, target_proj_chunks):
        # Query neural network directly along rays using the same projection method
        proj_chunk = project_parallel_beam(
            model,
            angles_torch_chunk,
            n_detectors,
            n_samples_per_ray=config.n_samples_per_ray,
            max_dist=config.max_dist,
            is_model=True,
            sample_jitter=config.sample_jitter,  # Add randomness to sample positions
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
