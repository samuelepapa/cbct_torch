import math
import sys
from pathlib import Path
from typing import Tuple

import ml_collections
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# Add parent directory to path to import modules if not installed as package
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# functa model
try:  # package-style import
    from .functa import LatentModulatedSiren
except ImportError:  # script-style import
    from functa import LatentModulatedSiren

# slice dataloader
import wandb
from dataset.load_slices import get_slice_dataloader

Tensor = torch.Tensor


# ----------------------
# Latent bank
# ----------------------


class LatentBank(nn.Module):
    """Per-slice latent codes stored as an embedding table.

    We index by slice index (global index in the dataset).
    """

    def __init__(self, num_items: int, latent_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_items, latent_dim)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)

    def reset(self):
        """Reset all latents to zero."""
        nn.init.zeros_(self.embedding.weight)


# ----------------------
# Data utilities
# ----------------------


def build_dataloaders(
    root_dir: Path,
    batch_size: int,
    num_vols_train: int,
    num_vols_val: int,
    num_workers: int = 4,
    num_points: int = 1024,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders backed by get_slice_dataloader.

    Each batch is a dict with keys:
      - 'values': [B, num_points]
      - 'coords': [B, num_points, 2]
      - 'patient_idx': [B]
      - 'slice_idx', 'vol_bbox' (unused here)
    """

    train_loader = get_slice_dataloader(
        root_dir=str(root_dir),
        stage="train",
        num_vols=num_vols_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        num_points=num_points,
    )

    val_loader = get_slice_dataloader(
        root_dir=str(root_dir),
        stage="val",
        num_vols=num_vols_val,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        num_points=num_points,
    )

    return train_loader, val_loader


# ----------------------
# Training / evaluation
# ----------------------


def train_one_epoch(
    model: LatentModulatedSiren,
    latent_bank: LatentBank,
    dataloader: DataLoader,
    model_optimizer: torch.optim.Optimizer,
    latent_optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_recon_every: int = 0,
    grid_size: int = 32,
    global_step: int = 0,
):
    model.train()
    latent_bank.train()

    mse = nn.MSELoss()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        values: Tensor = batch["values"].to(device)  # [B, num_points]
        coords: Tensor = batch["coords"].to(device)  # [B, num_points, 2]
        slice_indices: Tensor = batch["global_slice_idx"].to(device)  # [B] - global slice indices

        b, num_points = values.shape
        assert coords.shape == (
            b,
            num_points,
            2,
        ), f"Expected coords shape [B, {num_points}, 2], got {coords.shape}"

        # Targets: [B, num_points, 1]
        target = values.unsqueeze(-1)  # [B, num_points, 1]

        # Per-slice latent codes from bank
        latents = latent_bank(slice_indices)  # [B, latent_dim]

        model_optimizer.zero_grad(set_to_none=True)
        latent_optimizer.zero_grad(set_to_none=True)

        # Evaluate model for each item in the batch independently
        preds = []
        for i in range(b):
            pred = model(coords[i], latents[i])  # [num_points, C]
            preds.append(pred)
        preds = torch.stack(preds, dim=0)  # [B, num_points, C]

        loss = mse(preds, target)
        loss.backward()
        model_optimizer.step()
        latent_optimizer.step()

        batch_mse = loss.item()
        total_loss += batch_mse
        num_batches += 1

        batch_psnr = -10.0 * math.log10(batch_mse) if batch_mse > 0 else float("inf")
        current_step = global_step + batch_idx
        wandb.log(
            {
                "train/batch_loss": batch_mse,
                "train/batch_psnr": batch_psnr,
                "train/epoch": epoch,
            },
            step=current_step,
        )

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss {batch_mse:.6f} (PSNR {batch_psnr:.2f} dB)"
            )

        # Optionally log reconstructions from training batches
        if log_recon_every > 0 and batch_idx % log_recon_every == 0:
            log_reconstructions(
                model=model,
                latent_bank=latent_bank,
                batch=batch,
                device=device,
                tag="train/reconstructions_during_epoch",
                grid_size=grid_size,
                step=current_step,
            )

    return total_loss / max(1, num_batches), global_step + len(dataloader)


def fit_val_latents(
    model: LatentModulatedSiren,
    val_latent_bank: LatentBank,
    dataloader: DataLoader,
    device: torch.device,
    latent_lr: float,
    inner_epochs: int = 10,
):
    """Optimize validation latents while keeping model fixed."""

    model.eval()
    val_latent_bank.train()

    optimizer = torch.optim.Adam(val_latent_bank.parameters(), lr=latent_lr)
    mse = nn.MSELoss()

    for inner_epoch in range(inner_epochs):
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            values: Tensor = batch["values"].to(device)  # [B, num_points]
            coords: Tensor = batch["coords"].to(device)  # [B, num_points, 2]
            slice_indices: Tensor = batch["global_slice_idx"].to(
                device
            )  # [B] - global slice indices

            b, num_points = values.shape
            target = values.unsqueeze(-1)  # [B, num_points, 1]

            latents = val_latent_bank(slice_indices)

            optimizer.zero_grad(set_to_none=True)

            preds = []
            for i in range(b):
                pred = model(coords[i], latents[i])  # [num_points, C]
                preds.append(pred)
            preds = torch.stack(preds, dim=0)  # [B, num_points, C]

            loss = mse(preds, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"[Val latent opt] Epoch {inner_epoch + 1}/{inner_epochs} | Loss {avg_loss:.6f}")


@torch.no_grad()
def evaluate(
    model: LatentModulatedSiren,
    val_latent_bank: LatentBank,
    dataloader: DataLoader,
    device: torch.device,
):
    """Evaluate on subsampled points from batches."""
    model.eval()
    val_latent_bank.eval()

    mse = nn.MSELoss(reduction="sum")

    total_loss = 0.0
    total_pixels = 0

    for batch in dataloader:
        values: Tensor = batch["values"].to(device)  # [B, num_points]
        coords: Tensor = batch["coords"].to(device)  # [B, num_points, 2]
        slice_indices: Tensor = batch["global_slice_idx"].to(device)  # [B] - global slice indices

        b, num_points = values.shape
        target = values.unsqueeze(-1)  # [B, num_points, 1]

        latents = val_latent_bank(slice_indices)

        preds = []
        for i in range(b):
            pred = model(coords[i], latents[i])  # [num_points, C]
            preds.append(pred)
        preds = torch.stack(preds, dim=0)  # [B, num_points, C]

        loss = mse(preds, target)
        total_loss += loss.item()
        total_pixels += b * num_points

    return total_loss / max(1, total_pixels)


@torch.no_grad()
def evaluate_full_slices(
    model: LatentModulatedSiren,
    latent_bank: LatentBank,
    dataset,
    device: torch.device,
    num_samples: int = 10,
    indices: list = None,
):
    """Evaluate on full slices (not subsampled) for accurate metrics.

    Args:
        model: The trained model
        latent_bank: Latent bank (train or val)
        dataset: SliceDataset instance (must have get_full_slice method)
        device: Device to run on
        num_samples: Number of slices to evaluate (if indices not provided)
        indices: Specific slice indices to evaluate (if provided, overrides num_samples)

    Returns:
        dict with keys:
            - 'mse': Mean squared error
            - 'psnr': Peak signal-to-noise ratio
            - 'samples': List of dicts with per-sample metrics
    """
    model.eval()
    latent_bank.eval()

    mse = nn.MSELoss(reduction="sum")

    if indices is None:
        # Use fixed indices: first num_samples slices
        total_slices = len(dataset)
        indices = list(range(min(num_samples, total_slices)))

    total_loss = 0.0
    total_pixels = 0
    samples = []

    for idx in indices:
        # Get full slice
        full_slice = dataset.get_full_slice(idx)
        image = full_slice["image"].to(device)  # [H, W]
        coords_flat = full_slice["coords_flat"].to(device)  # [H*W, 2]
        values = full_slice["values"].to(device)  # [H*W]
        global_slice_idx = full_slice["global_slice_idx"]  # global slice index

        h, w = image.shape
        num_pixels = h * w

        # Get latent for this slice
        latent = latent_bank(torch.tensor([global_slice_idx], device=device))[0]  # [latent_dim]

        # Evaluate model on all points
        pred = model(coords_flat, latent)  # [H*W, C]

        # Compute loss - ensure shapes match
        # pred is [H*W, C] where C=1, values is [H*W]
        if pred.shape[-1] == 1:
            # Squeeze last dimension to match values: [H*W, 1] -> [H*W]
            pred_flat = pred.squeeze(-1)  # [H*W]
            loss = mse(pred_flat, values)  # [H*W] vs [H*W]
        else:
            # If multiple channels, need to handle differently
            target = values.unsqueeze(-1)  # [H*W, 1]
            loss = mse(pred, target)  # [H*W, C] vs [H*W, 1]

        loss_val = loss.item()
        total_loss += loss_val
        total_pixels += num_pixels

        # Compute PSNR
        psnr = -10.0 * math.log10(loss_val / num_pixels) if loss_val > 0 else float("inf")

        samples.append(
            {
                "idx": idx,
                "mse": loss_val / num_pixels,
                "psnr": psnr,
                "shape": (h, w),
            }
        )

    avg_mse = total_loss / max(1, total_pixels)
    avg_psnr = -10.0 * math.log10(avg_mse) if avg_mse > 0 else float("inf")

    return {
        "mse": avg_mse,
        "psnr": avg_psnr,
        "samples": samples,
    }


@torch.no_grad()
def log_reconstructions(
    model: LatentModulatedSiren,
    latent_bank: LatentBank,
    batch: dict,
    device: torch.device,
    tag: str,
    max_images: int = 8,
    grid_size: int = 32,
    step: int = None,
):
    """Log reconstructions from subsampled batches."""
    model.eval()
    latent_bank.eval()

    values: Tensor = batch["values"].to(device)[:max_images]  # [B, num_points]
    coords: Tensor = batch["coords"].to(device)[:max_images]  # [B, num_points, 2]
    slice_indices: Tensor = batch["global_slice_idx"].to(device)[
        :max_images
    ]  # [B] - global slice indices

    b, num_points = values.shape

    latents = latent_bank(slice_indices)

    recons = []
    for i in range(b):
        # Create a regular grid for visualization
        # Get coordinate bounds from the sampled points
        coords_i = coords[i].cpu()  # [num_points, 2]
        x_min, x_max = coords_i[:, 0].min().item(), coords_i[:, 0].max().item()
        y_min, y_max = coords_i[:, 1].min().item(), coords_i[:, 1].max().item()

        # Create regular grid coordinates
        x_grid = torch.linspace(x_min, x_max, grid_size, device=device)
        y_grid = torch.linspace(y_min, y_max, grid_size, device=device)
        xx, yy = torch.meshgrid(x_grid, y_grid, indexing="ij")
        grid_coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [grid_size*grid_size, 2]

        # Evaluate model on regular grid for visualization
        pred_grid_flat = model(grid_coords, latents[i])  # [grid_size*grid_size, C]
        pred_grid = (
            pred_grid_flat.view(grid_size, grid_size, -1).permute(2, 0, 1).clamp(0.0, 1.0)
        )  # [C, grid_size, grid_size]
        recons.append(pred_grid)

    recons = torch.stack(recons, dim=0)  # [B, C, grid_size, grid_size]

    grid = make_grid(recons, nrow=b)
    log_dict = {tag: wandb.Image(grid)}
    if step is not None:
        wandb.log(log_dict, step=step)
    else:
        wandb.log(log_dict)


@torch.no_grad()
def log_full_slice_reconstructions(
    model: LatentModulatedSiren,
    latent_bank: LatentBank,
    dataset,
    device: torch.device,
    tag: str,
    indices: list = None,
    max_images: int = 8,
    step: int = None,
):
    """Log reconstructions of full slices (not subsampled) for accurate visualization.

    Args:
        model: The trained model
        latent_bank: Latent bank (train or val)
        dataset: SliceDataset instance (must have get_full_slice method)
        device: Device to run on
        tag: WandB tag for logging
        indices: Specific slice indices to visualize (if None, samples randomly)
        max_images: Maximum number of images to visualize
    """
    model.eval()
    latent_bank.eval()

    if indices is None:
        # Use fixed indices: first max_images slices
        total_slices = len(dataset)
        num_samples = min(max_images, total_slices)
        indices = list(range(num_samples))
    else:
        indices = indices[:max_images]

    # First pass: collect all predictions to compute global normalization range
    all_predictions = []
    all_images = []
    all_coords = []
    all_latents = []
    all_slice_indices = []

    for idx in indices:
        # Get full slice
        full_slice = dataset.get_full_slice(idx)
        image = full_slice["image"].to(device)  # [H, W]
        coords_flat = full_slice["coords_flat"].to(device)  # [H*W, 2]
        global_slice_idx = full_slice["global_slice_idx"]  # global slice index

        h, w = image.shape

        # Get latent for this slice
        slice_idx_tensor = torch.tensor([global_slice_idx], device=device)
        latent = latent_bank(slice_idx_tensor)[0]  # [latent_dim]

        # Evaluate model on all points
        pred = model(coords_flat, latent)  # [H*W, C]

        # Reshape prediction to match image shape
        if pred.shape[-1] == 1:
            pred_image = pred.squeeze(-1).view(h, w)  # [H, W]
        else:
            pred_image = pred.view(h, w, -1)[:, :, 0]  # [H, W] (take first channel)

        all_predictions.append(pred_image)
        all_images.append(image)
        all_coords.append(coords_flat)
        all_latents.append(latent)
        all_slice_indices.append(global_slice_idx)

    # Compute global normalization range across all slices
    all_pred_tensor = torch.cat([p.flatten() for p in all_predictions])
    global_pred_min = all_pred_tensor.min().item()
    global_pred_max = all_pred_tensor.max().item()

    # Debug: print stats for first slice
    print(f"[{tag}] Global pred range: [{global_pred_min:.6f}, {global_pred_max:.6f}]")
    print(
        f"[{tag}] Slice {indices[0]}: global_slice_idx={all_slice_indices[0]}, "
        f"latent_norm={all_latents[0].norm().item():.6f}, "
        f"coords_range=[{all_coords[0].min().item():.3f}, {all_coords[0].max().item():.3f}]"
    )

    # Now normalize and log each slice
    for i, idx in enumerate(indices):
        pred_image = all_predictions[i]
        image = all_images[i]
        h, w = image.shape

        # Normalize original image (per-image normalization is fine for ground truth)
        img_min = image.min().item()
        img_max = image.max().item()
        if img_max > img_min:
            image_norm = (image - img_min) / (img_max - img_min)
        else:
            image_norm = torch.zeros_like(image)

        # Normalize prediction using GLOBAL range to preserve differences between slices
        if global_pred_max > global_pred_min:
            pred_norm = (pred_image - global_pred_min) / (global_pred_max - global_pred_min)
        else:
            pred_norm = torch.zeros_like(pred_image)

        # Clamp to [0, 1] and convert to uint8 [0, 255]
        image_norm = image_norm.clamp(0.0, 1.0)
        pred_norm = pred_norm.clamp(0.0, 1.0)

        # Convert to uint8 for wandb
        image_uint8 = (image_norm * 255).cpu().numpy().astype(np.uint8)
        pred_uint8 = (pred_norm * 255).cpu().numpy().astype(np.uint8)

        # Log each slice independently (original and reconstruction side by side)
        # Create a side-by-side comparison: [H, 2*W]
        comparison = np.concatenate([image_uint8, pred_uint8], axis=1)  # [H, 2*W]

        # Log as individual image (wandb.Image expects uint8 [0, 255])
        log_dict = {f"{tag}/slice_{idx}": wandb.Image(comparison)}
        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)


# ----------------------
# Main
# ----------------------


def get_config() -> ml_collections.ConfigDict:
    """Returns the default config for training functa on slices."""
    cfg = ml_collections.ConfigDict()

    # Data
    cfg.root_dir = "/media/samuele/data/LIDC-IDRI/version20251209"
    cfg.batch_size = 128
    cfg.num_vols_train = 10
    cfg.num_vols_val = 1
    cfg.num_workers = 4

    # Model
    cfg.latent_dim = 1024
    cfg.width = 256
    cfg.depth = 5
    cfg.w0 = 50.0

    # Optimization
    cfg.epochs = 500
    cfg.lr = 1e-4
    cfg.latent_lr = 1e-2
    cfg.val_latent_epochs = 20

    # Validation / logging cadence
    cfg.val_every = 10
    cfg.log_recon_every = 100
    cfg.log_full_recon_every = 5  # Log full slice reconstructions every N epochs
    cfg.reset_latents_every = 100  # Reset latents to zero every N epochs (0 to disable)

    # Data format
    cfg.num_points = 2048  # Fixed number of points per sample (32*32)

    # Device / IO
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.save_path = "/home/samuele/code/cbct_torch/results/functa/functa_slices.pt"

    # WandB
    cfg.wandb_project = "functa_slices"
    cfg.wandb_run_name = "functa_slices_run"

    return cfg


def main():
    cfg = get_config()

    device = torch.device(cfg.device)
    root_dir = Path(cfg.root_dir)

    wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=cfg.to_dict())

    train_loader, val_loader = build_dataloaders(
        root_dir=root_dir,
        batch_size=cfg.batch_size,
        num_vols_train=cfg.num_vols_train,
        num_vols_val=cfg.num_vols_val,
        num_points=cfg.num_points,
    )

    # Compute grid_size from num_points for visualization
    grid_size = int(np.sqrt(cfg.num_points))

    # Shared functa-style model (single-channel output for grayscale slices)
    model = LatentModulatedSiren(
        width=cfg.width,
        depth=cfg.depth,
        out_channels=1,
        latent_dim=cfg.latent_dim,
        w0=cfg.w0,
    ).to(device)

    # Latent banks: one per slice (not per volume)
    train_latent_bank = LatentBank(
        num_items=len(train_loader.dataset), latent_dim=cfg.latent_dim
    ).to(device)
    val_latent_bank = LatentBank(num_items=len(val_loader.dataset), latent_dim=cfg.latent_dim).to(
        device
    )

    # Separate optimizers for model and latents to avoid momentum mixing
    model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    latent_optimizer = torch.optim.Adam(train_latent_bank.parameters(), lr=cfg.latent_lr)

    # Grab a fixed batch for reconstruction logging
    train_sample_batch = next(iter(train_loader))
    val_sample_batch = next(iter(val_loader))

    # Define fixed slice indices for logging (use same slices every time)
    # Evenly distribute slice indices across the dataset
    def get_evenly_distributed_indices(dataset, num_slices=8):
        """Get slice indices evenly distributed across the dataset."""
        total_slices = len(dataset)
        if total_slices <= num_slices:
            # If dataset is smaller than requested, use all slices
            return list(range(total_slices))

        # Calculate step size to evenly distribute across dataset
        step = total_slices / num_slices
        indices = [int(i * step) for i in range(num_slices)]
        # Ensure we don't exceed dataset bounds
        indices = [min(idx, total_slices - 1) for idx in indices]
        return indices

    train_log_indices = get_evenly_distributed_indices(train_loader.dataset, num_slices=8)
    val_log_indices = get_evenly_distributed_indices(val_loader.dataset, num_slices=8)

    # Debug: print which volumes are being used
    print(f"Train log indices: {train_log_indices}")
    train_volumes = [
        train_loader.dataset._resolve_global_index(idx)[0] for idx in train_log_indices
    ]
    print(f"Train volumes: {train_volumes}")
    print(f"Val log indices: {val_log_indices}")
    val_volumes = [val_loader.dataset._resolve_global_index(idx)[0] for idx in val_log_indices]
    print(f"Val volumes: {val_volumes}")

    # Additional debug: check how many volumes are in each dataset
    train_unique_volumes = len(set(train_volumes))
    val_unique_volumes = len(set(val_volumes))
    print(
        f"Train dataset: {len(train_loader.dataset)} total slices, {train_unique_volumes} unique volumes"
    )
    print(
        f"Val dataset: {len(val_loader.dataset)} total slices, {val_unique_volumes} unique volumes"
    )

    # Global step counter that increments continuously
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        # Reset latents periodically if configured
        if cfg.reset_latents_every > 0 and epoch > 1 and epoch % cfg.reset_latents_every == 0:
            print(f"Resetting latents at epoch {epoch}")
            train_latent_bank.reset()
            val_latent_bank.reset()
            # Recreate optimizer to clear momentum state
            latent_optimizer = torch.optim.Adam(train_latent_bank.parameters(), lr=cfg.latent_lr)

        train_loss, global_step = train_one_epoch(
            model=model,
            latent_bank=train_latent_bank,
            dataloader=train_loader,
            model_optimizer=model_optimizer,
            latent_optimizer=latent_optimizer,
            device=device,
            epoch=epoch,
            log_recon_every=cfg.log_recon_every,
            grid_size=grid_size,
            global_step=global_step,
        )

        train_psnr = -10.0 * math.log10(train_loss) if train_loss > 0 else float("inf")

        if epoch % cfg.val_every == 0:
            # Fit validation latents for a few inner epochs with model frozen
            fit_val_latents(
                model=model,
                val_latent_bank=val_latent_bank,
                dataloader=val_loader,
                device=device,
                latent_lr=cfg.latent_lr,
                inner_epochs=cfg.val_latent_epochs,
            )

            val_mse = evaluate(
                model=model,
                val_latent_bank=val_latent_bank,
                dataloader=val_loader,
                device=device,
            )
            val_psnr = -10.0 * math.log10(val_mse) if val_mse > 0 else float("inf")
        else:
            val_mse = float("nan")
            val_psnr = float("nan")

        print(
            f"Epoch {epoch} done. "
            f"Train loss: {train_loss:.6f} (PSNR {train_psnr:.2f} dB), "
            f"Val MSE: {val_mse:.6e} (PSNR {val_psnr:.2f} dB)"
        )

        # Log epoch-level metrics
        wandb.log(
            {
                "train/loss": train_loss,
                "train/psnr": train_psnr,
                "val/mse": val_mse,
                "val/psnr": val_psnr,
                "epoch": epoch,
            },
            step=global_step,
        )

        # Log reconstructions from fixed batches (subsampled)
        log_reconstructions(
            model=model,
            latent_bank=train_latent_bank,
            batch=train_sample_batch,
            device=device,
            tag="train/reconstructions",
            grid_size=grid_size,
            step=global_step,
        )

        # Log full slice reconstructions for training
        if epoch % cfg.log_full_recon_every == 0:
            log_full_slice_reconstructions(
                model=model,
                latent_bank=train_latent_bank,
                dataset=train_loader.dataset,
                device=device,
                tag="train/full_reconstructions",
                indices=train_log_indices,
                max_images=8,
                step=global_step,
            )

        if epoch % cfg.val_every == 0:
            # Increment step for validation logging
            global_step += 1

            # Log reconstructions from fixed batches (subsampled)
            log_reconstructions(
                model=model,
                latent_bank=val_latent_bank,
                batch=val_sample_batch,
                device=device,
                grid_size=grid_size,
                tag="val/reconstructions",
                step=global_step,
            )

            # Log full slice reconstructions for validation
            log_full_slice_reconstructions(
                model=model,
                latent_bank=val_latent_bank,
                dataset=val_loader.dataset,
                device=device,
                tag="val/full_reconstructions",
                indices=val_log_indices,
                max_images=8,
                step=global_step,
            )

            # Evaluate on full slices for accurate metrics (use fixed indices)
            full_metrics = evaluate_full_slices(
                model=model,
                latent_bank=val_latent_bank,
                dataset=val_loader.dataset,
                device=device,
                num_samples=10,
                indices=val_log_indices[:10] if len(val_log_indices) >= 10 else val_log_indices,
            )

            # Log full slice metrics
            wandb.log(
                {
                    "val/full_mse": full_metrics["mse"],
                    "val/full_psnr": full_metrics["psnr"],
                    "epoch": epoch,
                },
                step=global_step,
            )

            print(
                f"Full slice metrics - MSE: {full_metrics['mse']:.6e}, "
                f"PSNR: {full_metrics['psnr']:.2f} dB"
            )

    torch.save({"model": model.state_dict()}, cfg.save_path)
    print(f"Saved model to {cfg.save_path}")
    wandb.save(cfg.save_path)


if __name__ == "__main__":
    main()
