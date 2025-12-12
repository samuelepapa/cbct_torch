import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import wandb
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import sys

# Paths
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataset.load_slices import get_slice_dataloader
from models.rff_net import RFFNet
from experiments.rff_net.rendering import get_parallel_rays_2d, render_parallel_projection, get_ray_aabb_intersection_2d

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", 
    "configs/rffnet_projection_config.py", 
    "File path to the training hyperparameter configuration.", 
    lock_config=True
)

class ImageModel(torch.nn.Module):
    """
    Wraps a 2D image tensor to act as a 'function' for ray tracing.
    Input: (x, y) coordinates in [-1, 1].
    Output: Interpolated pixel values.
    """
    def __init__(self, image):
        super().__init__()
        # image: [1, 1, H, W]
        self.register_buffer('image', image)

    def forward(self, x):
        # x: [B, 2] or [B, P, S, 2] -> needs [N, 1, 1, 2] for grid_sample
        # grid_sample expects [N, C, H_in, W_in] input and [N, H_out, W_out, 2] grid
        
        orig_shape = x.shape
        # Flatten to [1, N, 1, 2] where N is total points
        # grid_sample grids are [N, H, W, 2]
        
        # We treat the batch of query points as a "image" of size 1xN per batch?
        # Simpler: 
        # grid: [1, 1, TotalPoints, 2]
        # output: [1, C, 1, TotalPoints]
        
        x_flat = x.reshape(1, 1, -1, 2)
        
        # grid_sample coordinates: -1=left/top, 1=right/bottom
        # We assume x comes in standard cartesian [-1, 1].
        # grid_sample uses (x, y). 
        # Check alignment: standard image is (y, x) in array? 
        # Usually grid_sample(input, grid) -> grid[..., 0] is x, grid[..., 1] is y.
        
        out = F.grid_sample(self.image, x_flat, align_corners=True, mode='bilinear', padding_mode='zeros')
        # out: [1, 1, 1, TotalPoints]
        
        return out.reshape(orig_shape[:-1]) # Remove last dim 2, return [..., 1] implicitly squeezed?
        # The rendering function expects output [B, P, S] or [..., 1]
        
        # reshape returns [B, P, S] (if input was that)
        return out.view(orig_shape[:-1])


def train(argv):
    config = FLAGS.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Experiment setup
    experiment_root = Path(config.experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)
    wandb.init(project=config.wandb.project_name, config=config, dir=str(experiment_root))
    experiment_dir = experiment_root / wandb.run.id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Ground Truth Data ---
    full_dataset = get_slice_dataloader(
        root_dir=config.root_dir,
        stage="train",
        num_vols=config.num_vols,
        batch_size=1,
        shuffle=False,
        gray_value_scaling=config.gray_value_scaling
    ).dataset
    
    target_slice_idx = config.slice_idx if config.slice_idx < len(full_dataset) else 0
    sample = full_dataset[target_slice_idx]
    gt_image = sample["image"].unsqueeze(0).to(device) # [1, H, W]
    coords = sample["coords"].unsqueeze(0).to(device) # [1, H, W, 2]
    min_coords = torch.min(coords)
    max_coords = torch.max(coords)
    
    # Ensure shape for grid_sample: [1, 1, H, W]
    if gt_image.dim() == 3:
        gt_image = gt_image.unsqueeze(0)
        
    print(f"Loaded GT Image: {gt_image.shape}")

    # --- 2. Generate Ground Truth Projections (Sinogram) ---
    # Define Projection Geometry
    num_angles = config.num_angles
    angles = torch.linspace(0, np.pi, num_angles, device=device) # 0 to 180 degrees
    
    # Detector setup
    # Make detector cover the diagonal of the image to ensure full coverage
    # Image is [-1, 1] -> diagonal length approx 2.82
    # Let's say width = 3.0
    # Num pixels: match image resolution or similar
    H, W = gt_image.shape[-2:]
    num_det_pixels = max(H, W)
    
    # Use AABB intersection for rays
    # Slice is [-1, 1] x [-1, 1]
    aabb_min = torch.tensor([-1.0, -1.0], device=device)
    aabb_max = torch.tensor([1.0, 1.0], device=device)
    
    roi_radius = np.sqrt(2.0)
    detector_width = 2.0 * roi_radius
    
    print(f"Generating Sinogram: {num_angles} angles, {num_det_pixels} pixels")
    
    # Create wrapper model for GT generation
    gt_model = ImageModel(gt_image)
    
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
            gt_model, rays_o_all, rays_d_all, 
            near=t_min, far=t_max, num_samples=config.num_samples_gt, rand=False
        )
    # gt_sinogram: [num_angles, num_det_pixels]
    
    # Visualize GT
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gt_image[0, 0].cpu().numpy(), cmap='gray')
    plt.title("GT Image")
    plt.subplot(1, 2, 2)
    plt.imshow(gt_sinogram.cpu().numpy(), cmap='nipy_spectral', aspect='auto')
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
        scale=config.scale
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.MSELoss()

    # --- 4. Training Loop ---
    
    batch_size_angles = config.batch_size_angles # Train on subset of views per step
    
    print("Starting training...")
    iter_per_epoch = max(1, num_angles // batch_size_angles)
    
    pbar = tqdm(range(config.epochs), desc="Training")
    
    # Pre-generate validation grid for image reconstruction
    # Create grid of (x,y) coords in [-1, 1] matching image resolution
    y_g, x_g = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    grid_coords = torch.stack([x_g, y_g], dim=-1).reshape(-1, 2) # [H*W, 2] (x,y)

    for epoch in pbar:
        model.train()
        total_loss = 0
        
        perm = torch.randperm(num_angles)
        
        for i in range(0, num_angles, batch_size_angles):
            optimizer.zero_grad()
            
            # Select batch of angles
            batch_idx = perm[i : i + batch_size_angles]
            if len(batch_idx) == 0: continue
            
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
                model, b_rays_o, b_rays_d, 
                near=b_t_min, far=b_t_max, num_samples=config.num_samples, rand=True
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
                # Let's do all for full sinogram viz if fast enough
                # Or subsample
                rec_sinogram = render_parallel_projection(
                    model, rays_o_all, rays_d_all,
                    near=t_min, far=t_max, num_samples=config.num_samples, rand=False
                )
                
                # Metrics
                gt_np = gt_image[0, 0].cpu().numpy()
                rec_np = rec_image.cpu().numpy()
                
                wandb.log({
                    "val_image": wandb.Image(np.clip(rec_np, 0, 1), caption=f"Rec Ep {epoch}"),
                    "val_sinogram": wandb.Image(rec_sinogram.cpu().numpy(), caption=f"Sino Ep {epoch}"),
                    "val_loss": avg_loss
                })
                
                # Save checkpoint
                ckpt_path = experiment_dir / f"ckpt_latest.pt"
                torch.save(model.state_dict(), ckpt_path)

    wandb.finish()

if __name__ == "__main__":
    app.run(train)
