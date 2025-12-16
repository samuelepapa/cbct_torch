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
from skimage.transform import iradon  # type: ignore
from tqdm import tqdm

# Paths
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataset.load_slices import get_slice_dataloader
from metrics import psnr
from models.latent_transformer import LatentTransformer
from rendering import (
    ImageModel,
    get_parallel_rays_2d,
    get_ray_aabb_intersection_2d,
    render_parallel_projection,
)
from utils.logging_utils import setup_logger

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


def run_logging(
    epoch: int,
    loss: torch.Tensor,
    grad_norm: float,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    latents: torch.Tensor,
    pred_proj: torch.Tensor | None,
    pbar,
):
    """Handle progress bar, WandB logging, and periodic model stats."""
    current_lr = optimizer.param_groups[0]["lr"]

    # Progress bar
    pbar.set_postfix(
        {
            "Loss": f"{loss.item():.6f}",
            "LR": f"{current_lr:.2e}",
            "Grad": f"{grad_norm:.2f}",
        }
    )

    # Per-step logging
    wandb.log(
        {
            "train_loss": loss.item(),
            "learning_rate": current_lr,
            "grad_norm": grad_norm,
            "epoch": epoch,
        }
    )

    # Periodic model / latent / projection statistics
    if epoch % 10 == 0:
        with torch.no_grad():
            # Parameter norms
            param_norm = (
                sum(p.norm().item() ** 2 for p in model.parameters() if p.requires_grad) ** 0.5
            )
            latent_norm = latents.norm().item()
            latent_mean = latents.mean().item()
            latent_std = latents.std().item()

            proj_stats = None
            if pred_proj is not None:
                proj_stats = {
                    "pred_min": pred_proj.min().item(),
                    "pred_max": pred_proj.max().item(),
                    "pred_mean": pred_proj.mean().item(),
                    "pred_std": pred_proj.std().item(),
                }

        logging.info(
            f"Epoch {epoch}: Loss={loss.item():.6f}, ParamNorm={param_norm:.4f}, "
            f"LatentNorm={latent_norm:.4f}, LatentMean={latent_mean:.4f}, LatentStd={latent_std:.4f}"
        )
        if proj_stats:
            logging.info(
                f"  PredProj: min={proj_stats['pred_min']:.4f}, max={proj_stats['pred_max']:.4f}, "
                f"mean={proj_stats['pred_mean']:.4f}, std={proj_stats['pred_std']:.4f}"
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
        if proj_stats:
            wandb.log(
                {
                    "projections/pred_min": proj_stats["pred_min"],
                    "projections/pred_max": proj_stats["pred_max"],
                    "projections/pred_mean": proj_stats["pred_mean"],
                    "projections/pred_std": proj_stats["pred_std"],
                    "epoch": epoch,
                }
            )


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
    detector_width = diag * 1.5

    logger.info(f"Generating Sinogram: {num_angles} angles, {num_det_pixels} pixels")
    logger.info(f"AABB: {aabb_min.cpu().numpy()} to {aabb_max.cpu().numpy()}")

    # Create wrapper model for GT generation
    gt_model = ImageModel(gt_image)

    # Generate rays for ALL angles (for GT)
    rays_o_all, rays_d_all = get_parallel_rays_2d(angles, num_det_pixels, detector_width, device)

    # Compute AABB intersections for GT
    t_min, t_max, hits = get_ray_aabb_intersection_2d(rays_o_all, rays_d_all, aabb_min, aabb_max)
    hits_mask = hits.squeeze(-1)
    # Only keep valid segments inside the image; ignore rays that miss the AABB
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
    logger.info(f"Fitting ALL {num_angles} projections simultaneously")

    # Pre-generate validation grid for image reconstruction
    y_g, x_g = torch.meshgrid(
        torch.linspace(aabb_min[0].item(), aabb_max[0].item(), H, device=device),
        torch.linspace(aabb_min[1].item(), aabb_max[1].item(), W, device=device),
        indexing="xy",
    )
    grid_coords = torch.stack([y_g, x_g], dim=-1).reshape(-1, 2)  # [H*W, 2]

    # --- 4. Validation function ---
    def run_validation(
        epoch,
        model,
        latents,
        grid_coords,
        H,
        W,
        gt_image,
        config,
        rays_o_all,
        rays_d_all,
        t_min,
        t_max,
        gt_sinogram_raw,
        sino_center,
        sino_half_range,
        experiment_dir,
    ):
        """Run validation: reconstruct image (and optionally sinogram) and log to WandB."""
        model.eval()

        with torch.no_grad():
            # Reconstruct image from grid
            rec_flat = model(grid_coords, latents)
            rec_image = rec_flat.reshape(H, W)

            log_payload = {
                "val_image": wandb.Image(
                    _normalize_for_logging(rec_image), caption=f"Rec Ep {epoch}"
                ),
                "val_psnr": psnr(rec_image, gt_image[0]).item(),
            }

            # Optionally reconstruct sinogram in projection mode
            if getattr(config, "fitting_mode", "projection") == "projection":

                def model_fn_val(x):
                    return model(x, latents)

                rec_proj_raw = render_parallel_projection(
                    model_fn_val,
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
                rec_proj = rec_proj_raw
                log_payload.update(
                    {
                        "val_sinogram": wandb.Image(
                            _normalize_for_logging(rec_proj), caption=f"Rec Sinogram Ep {epoch}"
                        ),
                        "proj_psnr": psnr(rec_proj, gt_sinogram_raw).item(),
                    }
                )

                # FBP of reconstructed sinogram
                try:
                    fbp_val_np = iradon(
                        rec_proj_raw.detach().cpu().numpy().T,
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
                    "latents": latents.detach().cpu(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

    # --- 5. Training Loop ---
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

        def model_fn(x):
            return model(x, latents)

        # Compute loss depending on fitting mode
        if getattr(config, "fitting_mode", "projection") == "projection":
            # Render all projections simultaneously
            pred_proj_raw = render_parallel_projection(
                model_fn,
                rays_o_all,
                rays_d_all,
                near=t_min,
                far=t_max,
                num_samples=config.num_samples,
                rand=True,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
                hits_mask=hits_mask,
            )
            pred_proj = (pred_proj_raw - sino_center) / sino_half_range
            loss = F.mse_loss(pred_proj, gt_sinogram)
        else:
            # Image fitting: ignore projections, match image directly
            pred_image_flat = model(grid_coords, latents)  # [H*W, 1]
            pred_image = pred_image_flat.reshape(gt_image.shape)  # [1, 1, H, W]
            loss = F.mse_loss(pred_image, gt_image)

        # Stability: Clip loss value
        loss_clip_max = getattr(config, "loss_clip_max", None)
        if loss_clip_max is not None:
            loss = torch.clamp(loss, max=loss_clip_max)

        loss.backward()

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

        # Logging and metrics
        run_logging(
            epoch=epoch,
            loss=loss,
            grad_norm=grad_norm,
            optimizer=optimizer,
            model=model,
            latents=latents,
            pred_proj=(
                pred_proj if getattr(config, "fitting_mode", "projection") == "projection" else None
            ),
            pbar=pbar,
        )

        # Validation
        if (epoch + 1) % config.val_interval == 0:
            run_validation(
                epoch,
                model,
                latents,
                grid_coords,
                H,
                W,
                gt_image,
                config,
                rays_o_all,
                rays_d_all,
                t_min,
                t_max,
                gt_sinogram_raw,
                sino_center,
                sino_half_range,
                experiment_dir,
            )

    logger.info("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    app.run(train)
