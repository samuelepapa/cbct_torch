import logging
import sys
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
from models.latent_transformer import LatentTransformer
from utils.logging_utils import setup_logger

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "/home/samuele/code/cbct_torch/experiments/latent_transformer/configs/latent_transformer_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def train(argv):
    config = FLAGS.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    gt_image = sample["image"].unsqueeze(0).to(device)
    vol_bbox = sample["vol_bbox"]

    if gt_image.dim() == 3:
        gt_image = gt_image.unsqueeze(0)

    logger.info(f"Loaded GT Image: {gt_image.shape}")

    # --- 2. Setup AABB for coordinate normalization ---
    H, W = gt_image.shape[-2:]
    vol_bbox = vol_bbox.to(device)
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

    logger.info(f"AABB: {aabb_min.cpu().numpy()} to {aabb_max.cpu().numpy()}")

    # Log GT setup
    plt.figure(figsize=(6, 6))
    plt.imshow(gt_image[0, 0].cpu().numpy(), cmap="gray")
    plt.title("Ground Truth Image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(experiment_dir / "gt_setup.png")
    plt.close()

    wandb.log(
        {"gt_image": wandb.Image(np.clip(gt_image[0, 0].cpu().numpy(), 0, 1), caption="GT Image")}
    )

    # --- 3. Initialize Model and Latents ---
    model = LatentTransformer(
        coord_dim=config.coord_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        num_latents=config.num_latents,
        num_decoder_layers=config.num_decoder_layers,
        num_cross_attn_layers=config.num_cross_attn_layers,
        num_heads=config.num_heads,
        mlp_hidden_dim=config.mlp_hidden_dim,
        output_dim=config.output_dim,
        dropout=config.dropout,
        rope_base_freq=config.rope_base_freq,
        rope_learnable_freq=config.rope_learnable_freq,
        rope_coord_freq_multiplier=config.rope_coord_freq_multiplier,
        rff_encoding_size=config.rff_encoding_size,
        rff_scale=config.rff_scale,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Initialize latents (persist across epochs)
    latents = model.init_latents(batch_size=1, device=device)
    latents.requires_grad = True

    # Optimizer includes both model and latents
    optimizer = torch.optim.Adam(list(model.parameters()) + [latents], lr=config.outer_lr)
    logger.info("Joint optimization: optimizing model and latents together")

    # Pre-generate validation grid
    y_g, x_g = torch.meshgrid(
        torch.linspace(aabb_min[0].item(), aabb_max[0].item(), H, device=device),
        torch.linspace(aabb_min[1].item(), aabb_max[1].item(), W, device=device),
        indexing="xy",
    )
    grid_coords = torch.stack([y_g, x_g], dim=-1).reshape(-1, 2)

    # --- 4. Training Loop ---
    logger.info("Starting training...")

    pbar = tqdm(range(config.epochs), desc="Training")

    for epoch in pbar:
        model.train()

        # Learning rate warmup
        if epoch < config.warmup_epochs:
            warmup_factor = (epoch + 1) / config.warmup_epochs
            current_lr = config.outer_lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.outer_lr

        optimizer.zero_grad()

        # Forward pass: predict image directly from grid coordinates
        pred_image_flat = model(grid_coords, latents)  # [H*W, 1]
        pred_image = pred_image_flat.reshape(gt_image.shape)  # [1, 1, H, W]

        loss = F.mse_loss(pred_image, gt_image)

        # Stability: Clip loss value
        loss_clip_max = getattr(config, "loss_clip_max", None)
        if loss_clip_max is not None:
            loss = torch.clamp(loss, max=loss_clip_max)

        loss.backward()

        # Stability: Check for NaN/Inf
        check_nan = getattr(config, "check_nan", False)
        skip_update = False
        if check_nan:
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected at epoch {epoch}, skipping update")
                skip_update = True
            else:
                for p in list(model.parameters()) + [latents]:
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            logger.warning(
                                f"NaN/Inf gradients detected at epoch {epoch}, skipping update"
                            )
                            skip_update = True
                            break

        if skip_update:
            optimizer.zero_grad()
            continue

        # Stability: Gradient clipping
        grad_clip_norm = getattr(config, "grad_clip_norm", None)
        grad_norm = 0.0
        if grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + [latents], grad_clip_norm
            )
        else:
            for p in list(model.parameters()) + [latents]:
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm**0.5

        optimizer.step()

        # Stability: Clip latent values
        latent_clip_value = getattr(config, "latent_clip_value", None)
        if latent_clip_value is not None:
            with torch.no_grad():
                latents.clamp_(-latent_clip_value, latent_clip_value)

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            {"Loss": f"{loss.item():.6f}", "LR": f"{current_lr:.2e}", "Grad": f"{grad_norm:.2f}"}
        )

        wandb.log(
            {
                "train_loss": loss.item(),
                "learning_rate": current_lr,
                "grad_norm": grad_norm,
                "epoch": epoch,
            }
        )

        # Log model and latent statistics periodically
        if epoch % 10 == 0:
            with torch.no_grad():
                # Parameter norms
                param_norm = (
                    sum(p.norm().item() ** 2 for p in model.parameters() if p.requires_grad) ** 0.5
                )
                latent_norm = latents.norm().item()
                latent_mean = latents.mean().item()
                latent_std = latents.std().item()

            logger.info(
                f"Epoch {epoch}: Loss={loss.item():.6f}, ParamNorm={param_norm:.4f}, "
                f"LatentNorm={latent_norm:.4f}, LatentMean={latent_mean:.4f}, LatentStd={latent_std:.4f}"
            )

            wandb.log(
                {
                    "model/param_norm": param_norm,
                    "model/latent_norm": latent_norm,
                    "model/latent_mean": latent_mean,
                    "model/latent_std": latent_std,
                    "epoch": epoch,
                }
            )

        # Validation
        if (epoch + 1) % config.val_interval == 0:
            model.eval()

            with torch.no_grad():
                # Reconstruct image
                rec_flat = model(grid_coords, latents)
                rec_image = rec_flat.reshape(H, W)

                # Metrics
                val_psnr = psnr(rec_image, gt_image[0])

                wandb.log(
                    {
                        "val_image": wandb.Image(
                            np.clip(rec_image.cpu().numpy(), 0, 1), caption=f"Rec Ep {epoch}"
                        ),
                        "val_psnr": val_psnr.item(),
                    }
                )

                # Save checkpoint
                checkpoint_path = experiment_dir / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "latents": latents.detach().cpu(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )

    logger.info("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    app.run(train)
