import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import ml_collections
import numpy as np
import optuna
import tinycudann as tcnn
import torch

import wandb

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys

sys.path.append(str(PROJECT_ROOT))

from dataset.load_slices import get_slice_dataloader  # noqa: E402
from metrics import psnr  # noqa: E402

# Cache for static data shared across trials (image, coords, angles, etc.)
_DATA_CACHE: Optional[Dict[str, Any]] = None


def get_base_config() -> ml_collections.ConfigDict:
    """Base configuration (copied from simple_ngp.py)."""
    config = ml_collections.ConfigDict()

    # Data params
    config.root_dir = "/media/samuele/data/LIDC-IDRI/version20251209"
    config.num_vols = 1
    config.slice_idx = 10
    config.gray_value_scaling = 20.0

    # Model params
    config.n_levels = 8
    config.n_features_per_level = 4
    config.log2_hashmap_size = 15
    config.base_resolution = 8
    config.per_level_scale = 1.5
    config.n_neurons = 64
    config.n_hidden_layers = 2

    # Projection params
    config.n_angles = 256
    config.batch_size_angles = 16
    config.n_samples_per_ray = 100
    config.max_dist = 1.0
    config.sample_jitter = 0.001

    # Training params
    config.learning_rate = 5e-3
    config.max_steps = 5000
    config.log_interval = 1000

    # WandB params
    config.wandb = ml_collections.ConfigDict()
    config.wandb.project_name = "instant-ngp-simple-optuna"
    config.wandb.run_name = "simple_ngp_optuna"

    return config


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


def prepare_static_data(config: ml_collections.ConfigDict) -> Dict[str, Any]:
    """
    Prepare data that does not depend on trial-specific hyperparameters.

    This is called once and cached, so that all Optuna trials reuse the same
    ground-truth slice, coordinates, and angle setup, avoiding repeated
    dataset loading and CPU work and keeping the GPU busier.
    """
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Ground Truth Data once ---
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
    gt_image = sample["image"].to(device)

    if gt_image.dim() > 2:
        gt_image = gt_image.squeeze()
        if gt_image.dim() != 2:
            raise ValueError(f"Expected 2D image after squeezing, got shape {gt_image.shape}")

    # Normalize once for all trials
    with torch.no_grad():
        gt_image = gt_image.clone()
        eps = 1e-6
        gt_image_min = gt_image.min()
        gt_image_max = gt_image.max()
        gt_image = (gt_image - gt_image_min) / (gt_image_max - gt_image_min + eps)

    H = gt_image.shape[-2]
    W = gt_image.shape[-1]

    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing="ij",
    )
    coords = torch.stack([xs, ys], dim=-1).reshape(-1, 2)

    # Angles do not depend on trial-specific parameters since we keep n_angles fixed
    angles_torch = torch.linspace(0, np.pi, int(config.n_angles), device=device)
    n_detectors = int(max(H, W))

    _DATA_CACHE = {
        "device": device,
        "gt_image": gt_image,
        "coords": coords,
        "angles_torch": angles_torch,
        "n_detectors": n_detectors,
        "H": H,
        "W": W,
    }
    return _DATA_CACHE


def project_parallel_beam_vectorized(
    img_or_model,
    angles: torch.Tensor,
    n_detectors: int,
    n_samples_per_ray: int = 100,
    max_dist: float = 1.0,
    is_model: bool = False,
    sample_jitter: float = 0.01,
) -> torch.Tensor:
    """
    Vectorized parallel beam projection (copied from simple_ngp.py).

    For images: uses grid_sample to interpolate values along rays.
    For models: queries the model directly at sampled points (batched).
    """
    device = angles.device
    A = angles.shape[0]

    s_bins = torch.linspace(-max_dist, max_dist, n_detectors, device=device)
    t_samples = torch.linspace(-max_dist, max_dist, n_samples_per_ray, device=device)

    if is_model and sample_jitter > 0.0:
        dt = 2.0 * max_dist / (n_samples_per_ray - 1) if n_samples_per_ray > 1 else 1.0
        jitter = (torch.rand(n_samples_per_ray, device=device) * 2.0 - 1.0) * sample_jitter * dt
        t_samples = t_samples + jitter

    cos_theta = torch.cos(angles)
    sin_theta = torch.sin(angles)

    s_expanded = s_bins[None, :, None]
    t_expanded = t_samples[None, None, :]
    cos_expanded = cos_theta[:, None, None]
    sin_expanded = sin_theta[:, None, None]

    x_coords = s_expanded * cos_expanded - t_expanded * sin_expanded
    y_coords = s_expanded * sin_expanded + t_expanded * cos_expanded

    if is_model:
        x_coords_norm = (x_coords + 1.0) / 2.0
        y_coords_norm = (y_coords + 1.0) / 2.0
        x_coords_norm = torch.clamp(x_coords_norm, 0.0, 1.0)
        y_coords_norm = torch.clamp(y_coords_norm, 0.0, 1.0)

        ray_coords = torch.stack([x_coords_norm, y_coords_norm], dim=-1)
        ray_coords_flat = ray_coords.reshape(-1, 2)

        values_flat = img_or_model(ray_coords_flat).squeeze(-1)
        values = values_flat.reshape(A, n_detectors, n_samples_per_ray)
    else:
        img = img_or_model
        img_batch = img.unsqueeze(0).unsqueeze(0)

        all_coords = torch.stack([x_coords, y_coords], dim=-1)
        all_coords_flat = all_coords.reshape(-1, 2)

        grid = all_coords_flat.unsqueeze(0).unsqueeze(2)

        sampled = torch.nn.functional.grid_sample(
            img_batch,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        values = (
            sampled.squeeze(0)
            .squeeze(0)
            .squeeze(-1)
            .reshape(
                A,
                n_detectors,
                n_samples_per_ray,
            )
        )

    dt = 2.0 * max_dist / (n_samples_per_ray - 1) if n_samples_per_ray > 1 else 1.0
    if n_samples_per_ray == 1:
        projections = values[:, :, 0] * dt
    else:
        projections = (values[:, :, 0] + values[:, :, -1]) / 2.0 + values[:, :, 1:-1].sum(dim=-1)
        projections = projections * dt

    return projections


def build_model(config: ml_collections.ConfigDict, device: torch.device) -> torch.nn.Module:
    """Create the hash-encoded MLP model."""
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=1,
        encoding_config={
            "otype": "HashGrid",
            "n_levels": int(config.n_levels),
            "n_features_per_level": int(config.n_features_per_level),
            "log2_hashmap_size": int(config.log2_hashmap_size),
            "base_resolution": int(config.base_resolution),
            "per_level_scale": float(config.per_level_scale),
        },
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": int(config.n_neurons),
            "n_hidden_layers": int(config.n_hidden_layers),
        },
    ).to(device)
    return model


def run_training(
    config: ml_collections.ConfigDict,
    data: Dict[str, Any],
    trial: Optional[optuna.Trial] = None,
    log_to_wandb: bool = False,
) -> dict:
    """
    Train the simple NGP model with given config and (optionally) Optuna trial.

    Returns a dict with final metrics (MSE, MAE, PSNR).
    """
    device = data["device"]
    gt_image = data["gt_image"]
    coords = data["coords"]
    angles_torch = data["angles_torch"]
    n_detectors = int(data["n_detectors"])
    H = int(data["H"])
    W = int(data["W"])

    project_parallel_beam = project_parallel_beam_vectorized

    target_proj = project_parallel_beam(
        gt_image,
        angles_torch,
        n_detectors,
        n_samples_per_ray=int(config.n_samples_per_ray),
        max_dist=float(config.max_dist),
        is_model=False,
        sample_jitter=0.0,
    )

    if log_to_wandb:
        wandb.init(
            project=config.wandb.project_name,
            name=config.wandb.run_name,
            config=config,
        )
        wandb.log(
            {
                "gt_image": wandb.Image(_normalize_for_logging(gt_image), caption="Ground Truth"),
                "gt_sinogram": wandb.Image(
                    _normalize_for_logging(target_proj), caption="Ground Truth Sinogram"
                ),
            }
        )

    model = build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))

    angles_torch_chunks = torch.chunk(angles_torch, int(config.batch_size_angles))
    target_proj_chunks = torch.chunk(target_proj, int(config.batch_size_angles))

    step = 0
    max_steps = int(config.max_steps)

    while step < max_steps:
        for angles_chunk, target_chunk in zip(angles_torch_chunks, target_proj_chunks):
            proj_chunk = project_parallel_beam(
                model,
                angles_chunk,
                n_detectors,
                n_samples_per_ray=int(config.n_samples_per_ray),
                max_dist=float(config.max_dist),
                is_model=True,
                sample_jitter=float(config.sample_jitter),
            )

            loss = torch.mean((proj_chunk - target_chunk) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if trial is not None:
                # Report PSNR of the currently reconstructed image to Optuna every 1000 steps (use step as timestep)
                if (step + 1) % 1000 == 0:
                    with torch.no_grad():
                        reconstructed_img = model(coords).reshape(H, W)
                    current_psnr = psnr(reconstructed_img, gt_image).item()
                    trial.report(current_psnr, step)
                    if trial.should_prune():
                        if log_to_wandb:
                            wandb.finish()
                        raise optuna.exceptions.TrialPruned()

            if log_to_wandb and step % int(config.log_interval) == 0:
                with torch.no_grad():
                    reconstructed_img = model(coords).reshape(H, W)
                    diff = torch.abs(reconstructed_img - gt_image)
                wandb.log(
                    {
                        "step": step,
                        "train_loss": loss.item(),
                        "image_mse": torch.mean((reconstructed_img - gt_image) ** 2).item(),
                        "image_mae": torch.mean(diff).item(),
                        "image_psnr": psnr(reconstructed_img, gt_image).item(),
                        "reconstruction": wandb.Image(
                            _normalize_for_logging(reconstructed_img),
                            caption=f"Reconstruction Step {step}",
                        ),
                    }
                )

            step += 1
            if step >= max_steps:
                break

    with torch.no_grad():
        reconstructed_img = model(coords).reshape(H, W)

    diff = torch.abs(reconstructed_img - gt_image)
    final_mse = torch.mean((reconstructed_img - gt_image) ** 2).item()
    final_mae = torch.mean(diff).item()
    final_psnr = psnr(reconstructed_img, gt_image).item()

    if log_to_wandb:
        wandb.log(
            {
                "final_image_mse": final_mse,
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

    return {
        "final_image_mse": final_mse,
        "final_image_mae": final_mae,
        "final_image_psnr": final_psnr,
    }


def build_trial_config(trial: optuna.Trial) -> ml_collections.ConfigDict:
    """Create a config for a given Optuna trial by sampling hyperparameters."""
    config = get_base_config()

    # Model hyperparameters
    config.n_levels = trial.suggest_int("n_levels", 4, 12)
    config.n_features_per_level = trial.suggest_categorical("n_features_per_level", [1, 2, 4, 8])
    config.log2_hashmap_size = trial.suggest_int("log2_hashmap_size", 14, 18)
    config.base_resolution = trial.suggest_int("base_resolution", 4, 32)
    config.per_level_scale = trial.suggest_float("per_level_scale", 1.2, 2.5)
    config.n_neurons = trial.suggest_categorical("n_neurons", [32, 64, 128])
    config.n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 4)

    # Projection / integration hyperparameters
    config.n_samples_per_ray = trial.suggest_int("n_samples_per_ray", 64, 256)
    config.sample_jitter = trial.suggest_float("sample_jitter", 1e-4, 2e-2, log=True)

    # Training hyperparameters
    config.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    config.max_steps = trial.suggest_int("max_steps", 1000, 8000, step=1000)

    # For logging clarity
    config.wandb.run_name = f"simple_ngp_optuna_trial_{trial.number}"

    return config


def make_objective(log_to_wandb: bool = False):
    """Create an Optuna objective function."""

    # Prepare and cache static data once before running any trials so that
    # each trial spends almost all of its time on GPU-heavy computation.
    base_config = get_base_config()
    static_data = prepare_static_data(base_config)

    def objective(trial: optuna.Trial) -> float:
        config = build_trial_config(trial)
        metrics = run_training(config, static_data, trial=trial, log_to_wandb=log_to_wandb)
        # We maximize the final image PSNR
        return metrics["final_image_psnr"]

    return objective


def parse_args() -> argparse.Namespace:
    """
python -m experiments.instant_ngp.optimize_simple_ngp \
  --n-trials 1000 \
  --study-name simple_ngp_opt \
  --storage sqlite:////tmp/simple_ngp_opt.db
  """
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for simple_ngp.")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials.")
    parser.add_argument(
        "--study-name",
        type=str,
        default="simple_ngp_optuna_study",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "Optuna storage URL (e.g. sqlite:///simple_ngp_optuna.db). "
            "If omitted, an in-memory study is used."
        ),
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["minimize", "maximize"],
        default="maximize",
        help="Optimization direction (default: minimize MSE).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log each trial as a separate WandB run (slower, but detailed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.storage is None:
        study = optuna.create_study(direction=args.direction, study_name=args.study_name)
    else:
        study = optuna.create_study(
            direction=args.direction,
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
        )

    objective = make_objective(log_to_wandb=args.wandb)
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (final_image_mse): {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
