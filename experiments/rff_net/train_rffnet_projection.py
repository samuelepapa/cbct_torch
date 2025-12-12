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
from tqdm import tqdm

# Paths
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataset.load_slices import get_slice_dataloader
from metrics import psnr
from models.rff_net import RFFNet
from rendering import (
    ImageModel,
    get_parallel_rays_2d,
    get_ray_aabb_intersection_2d,
    render_parallel_projection,
)

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "/home/samuele/code/cbct_torch/experiments/rff_net/configs/rffnet_projection_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def train(argv):
    config = FLAGS.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging before anything else
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Experiment setup
    experiment_root = Path(config.experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)
    wandb.init(project=config.wandb.project_name, config=config, dir=str(experiment_root))
    experiment_dir = experiment_root / wandb.run.id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to file and console
    log_file = experiment_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

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
    # Num pixels: match image resolution or similar
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

    # aabb_min/max are now [2]

    roi_radius = np.sqrt(2.0)
    detector_width = 2.0 * roi_radius  # TODO: Update this if physical units used?
    # If using physical coords, detector width should likely be larger.
    # Current detector width logic assumes [-1, 1].
    # Calculate diagonal of actual AABB
    diag = torch.norm(aabb_max - aabb_min).item()
    detector_width = diag * 1.5

    logger.info(f"Generating Sinogram: {num_angles} angles, {num_det_pixels} pixels")
    logger.info(f"AABB: {aabb_min.cpu().numpy()} to {aabb_max.cpu().numpy()}")

    # Create wrapper model for GT generation
    gt_model = ImageModel(gt_image, aabb_min, aabb_max)

    # Generate rays for ALL angles (for GT)
    rays_o_all, rays_d_all = get_parallel_rays_2d(angles, num_det_pixels, detector_width, device)

    # Compute AABB intersections for GT
    # t_min, t_max: [B, P, 1]
    t_min, t_max, hits = get_ray_aabb_intersection_2d(rays_o_all, rays_d_all, aabb_min, aabb_max)

    # Filter rays that don't hit: effectively near=far=0
    # Use simple masking: set near=far where !hits
    # hits is [B, P]
    t_min = torch.where(hits, t_min, torch.zeros_like(t_min))
    t_max = torch.where(hits, t_max, torch.zeros_like(t_max))

    # Render GT Sinogram
    # We use high sample count for high quality GT
    # Pass tensor near/far
    with torch.no_grad():
        gt_sinogram = render_parallel_projection(
            gt_model,
            rays_o_all,
            rays_d_all,
            near=t_min,
            far=t_max,
            num_samples=config.num_samples_gt,
            rand=False,
        )
    # gt_sinogram: [num_angles, num_det_pixels]

    # Visualize GT
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gt_image[0, 0].cpu().numpy(), cmap="gray")
    plt.title("GT Image")
    plt.subplot(1, 2, 2)
    plt.imshow(gt_sinogram.cpu().numpy(), cmap="nipy_spectral", aspect="auto")
    plt.title("GT Sinogram")
    plt.savefig(experiment_dir / "gt_setup.png")
    wandb.log({"setup": wandb.Image(str(experiment_dir / "gt_setup.png"))})

    # --- 3. Model Setup ---
    model = RFFNet(
        in_dim=2,
        out_dim=1,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        encoding_size=config.encoding_size,
        scale=config.scale,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.MSELoss()

    # --- 4. Training Loop ---

    batch_size_angles = config.batch_size_angles  # Train on subset of views per step

    logger.info("Starting training...")
    iter_per_epoch = max(1, num_angles // batch_size_angles)

    pbar = tqdm(range(config.epochs), desc="Training")

    # Pre-generate validation grid for image reconstruction
    # Create grid of (x,y) coords matching image resolution and AABB
    y_g, x_g = torch.meshgrid(
        torch.linspace(aabb_min[0].item(), aabb_max[0].item(), H, device=device),
        torch.linspace(aabb_min[1].item(), aabb_max[1].item(), W, device=device),
        indexing="xy",
    )
    # Stack as (y, x) aka (dim1, dim2) to match load_slices coords
    grid_coords = torch.stack([y_g, x_g], dim=-1).reshape(-1, 2)  # [H*W, 2]

    for epoch in pbar:
        model.train()
        total_loss = 0

        perm = torch.randperm(num_angles)

        for i in range(0, num_angles, batch_size_angles):
            optimizer.zero_grad()

            # Select batch of angles
            batch_idx = perm[i : i + batch_size_angles]
            if len(batch_idx) == 0:
                continue

            # Get rays for these angles
            # rays_o: [B_angles, num_det, 2]
            b_rays_o = rays_o_all[batch_idx]
            b_rays_d = rays_d_all[batch_idx]
            b_gt = gt_sinogram[batch_idx]

            # Bounds
            b_t_min = t_min[batch_idx]
            b_t_max = t_max[batch_idx]

            # Render
            # Use fewer samples for training speed, randomize
            pred_proj = render_parallel_projection(
                model,
                b_rays_o,
                b_rays_d,
                near=b_t_min,
                far=b_t_max,
                num_samples=config.num_samples,
                rand=True,
            )

            loss = loss_fn(pred_proj, b_gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / iter_per_epoch
        wandb.log({"train_loss": avg_loss})
        pbar.set_postfix({"Loss": f"{avg_loss:.6f}"})

        # Validation
        if (epoch + 1) % config.val_interval == 0:
            model.eval()
            with torch.no_grad():
                # 1. Reconstruct Image (query grid)
                # Split into chunks if needed to avoid OOM
                rec_flat = model(grid_coords)
                rec_image = rec_flat.reshape(H, W)

                # 2. Reconstruct Sinogram (a few fixed angles or all?)
                # Batch rendering to avoid OOM
                val_sinogram_list = []
                val_batch_size = 16
                for i in range(0, num_angles, val_batch_size):
                    batch_rays_o = rays_o_all[i : i + val_batch_size]
                    batch_rays_d = rays_d_all[i : i + val_batch_size]
                    batch_t_min = t_min[i : i + val_batch_size]
                    batch_t_max = t_max[i : i + val_batch_size]

                    batch_proj = render_parallel_projection(
                        model,
                        batch_rays_o,
                        batch_rays_d,
                        near=batch_t_min,
                        far=batch_t_max,
                        num_samples=config.num_samples,
                        rand=False,
                    )
                    val_sinogram_list.append(batch_proj)
                rec_sinogram = torch.cat(val_sinogram_list, dim=0)

                # Metrics
                gt_np = gt_image[0, 0].cpu().numpy()
                rec_np = rec_image.cpu().numpy()

                val_psnr = psnr(rec_image, gt_image[0])

                wandb.log(
                    {
                        "val_image": wandb.Image(np.clip(rec_np, 0, 1), caption=f"Rec Ep {epoch}"),
                        "val_sinogram": wandb.Image(
                            rec_sinogram.cpu().numpy(), caption=f"Sino Ep {epoch}"
                        ),
                        "val_loss": avg_loss,
                        "val_psnr": val_psnr.item(),
                    }
                )

                # Save checkpoint
                ckpt_path = experiment_dir / f"ckpt_latest.pt"
                torch.save(model.state_dict(), ckpt_path)

    wandb.finish()


if __name__ == "__main__":
    app.run(train)
