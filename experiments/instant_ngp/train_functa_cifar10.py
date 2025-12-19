import argparse
import math
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# Support running both as a module and as a standalone script
try:  # package-style import
    from .functa import LatentModulatedSiren, get_coordinate_grid
except ImportError:  # script-style import
    from functa import LatentModulatedSiren, get_coordinate_grid

import wandb

# ----------------------
# Data utilities
# ----------------------


class IndexedCIFAR10(datasets.CIFAR10):
    """CIFAR10 that also returns the sample index.

    Returns (image, index) so we can maintain per-image latents.
    """

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img, index


def build_dataloaders(data_root: Path, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [0, 1]
        ]
    )

    train_ds_full = IndexedCIFAR10(
        root=str(data_root), train=True, download=True, transform=transform
    )
    full_test_ds = IndexedCIFAR10(
        root=str(data_root), train=False, download=True, transform=transform
    )

    # Use only 10% of the training data
    train_size = max(1, int(0.25 * len(train_ds_full)))
    train_indices = list(range(train_size))
    train_ds = Subset(train_ds_full, train_indices)

    # Use only a small validation subset: two batches worth of data
    val_size = min(2 * batch_size, len(full_test_ds))
    val_indices = list(range(val_size))
    test_ds = Subset(full_test_ds, val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader


class LatentBank(nn.Module):
    """Per-image latent codes stored as an embedding table."""

    def __init__(self, num_items: int, latent_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_items, latent_dim)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)


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
    coord_grid: torch.Tensor,
    epoch: int,
    log_recon_every: int = 0,
):
    model.train()
    latent_bank.train()

    mse = nn.MSELoss()

    total_loss = 0.0
    num_batches = 0

    coord_grid_flat = coord_grid.view(-1, 2)  # [H*W, 2]

    for batch_idx, (images, indices) in enumerate(dataloader):
        images = images.to(device)  # [B, 3, H, W]
        indices = indices.to(device, non_blocking=True)
        b, c, h, w = images.shape
        assert h == coord_grid.shape[0] and w == coord_grid.shape[1], "CIFAR-10 should be 32x32."

        # Flatten images to [B, H*W, C]
        target = images.permute(0, 2, 3, 1).reshape(b, -1, c)

        # Per-image latent codes from bank
        latents = latent_bank(indices)  # [B, latent_dim]

        model_optimizer.zero_grad(set_to_none=True)
        latent_optimizer.zero_grad(set_to_none=True)

        # Evaluate model for each item in the batch independently
        preds = []
        for i in range(b):
            pred = model(coord_grid_flat, latents[i])  # [H, W, C]
            preds.append(pred.view(-1, c))
        preds = torch.stack(preds, dim=0)  # [B, H*W, C]

        loss = mse(preds, target)
        loss.backward()
        model_optimizer.step()
        latent_optimizer.step()

        batch_mse = loss.item()
        total_loss += batch_mse
        num_batches += 1

        # Per-batch logging
        batch_psnr = -10.0 * math.log10(batch_mse) if batch_mse > 0 else float("inf")
        wandb.log(
            {
                "train/batch_loss": batch_mse,
                "train/batch_psnr": batch_psnr,
                "train/epoch": epoch,
            }
        )

        # Optionally log reconstructions from training batches
        if log_recon_every > 0 and batch_idx % log_recon_every == 0:
            log_reconstructions(
                model=model,
                latent_bank=latent_bank,
                images=images,
                indices=indices,
                coord_grid=coord_grid,
                device=device,
                tag="train/reconstructions_during_epoch",
            )

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss {loss.item():.6f}")

    return total_loss / max(1, num_batches)


def fit_val_latents(
    model: LatentModulatedSiren,
    val_latent_bank: LatentBank,
    dataloader: DataLoader,
    device: torch.device,
    coord_grid: torch.Tensor,
    latent_lr: float,
    inner_epochs: int = 10,
):
    """Optimize validation latents while keeping model fixed."""

    model.eval()
    val_latent_bank.train()

    optimizer = torch.optim.Adam(val_latent_bank.parameters(), lr=latent_lr)
    mse = nn.MSELoss()
    coord_grid_flat = coord_grid.view(-1, 2)

    for inner_epoch in range(inner_epochs):
        total_loss = 0.0
        num_batches = 0
        for images, indices in dataloader:
            images = images.to(device)
            indices = indices.to(device, non_blocking=True)
            b, c, h, w = images.shape

            target = images.permute(0, 2, 3, 1).reshape(b, -1, c)

            latents = val_latent_bank(indices)

            optimizer.zero_grad(set_to_none=True)

            preds = []
            for i in range(b):
                pred = model(coord_grid_flat, latents[i])
                preds.append(pred.view(-1, c))
            preds = torch.stack(preds, dim=0)

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
    coord_grid: torch.Tensor,
):
    model.eval()
    val_latent_bank.eval()

    mse = nn.MSELoss(reduction="sum")

    total_loss = 0.0
    total_pixels = 0

    coord_grid_flat = coord_grid.view(-1, 2)

    for images, indices in dataloader:
        images = images.to(device)
        indices = indices.to(device, non_blocking=True)
        b, c, h, w = images.shape

        target = images.permute(0, 2, 3, 1).reshape(b, -1, c)

        latents = val_latent_bank(indices)

        preds = []
        for i in range(b):
            pred = model(coord_grid_flat, latents[i])
            preds.append(pred.view(-1, c))
        preds = torch.stack(preds, dim=0)

        loss = mse(preds, target)
        total_loss += loss.item()
        total_pixels += b * h * w * c

    return total_loss / max(1, total_pixels)


@torch.no_grad()
def log_reconstructions(
    model: LatentModulatedSiren,
    latent_bank: LatentBank,
    images: torch.Tensor,
    indices: torch.Tensor,
    coord_grid: torch.Tensor,
    device: torch.device,
    tag: str,
    max_images: int = 8,
):
    model.eval()
    latent_bank.eval()

    images = images.to(device)[:max_images]
    indices = indices.to(device)[:max_images]
    b, c, h, w = images.shape

    coord_grid_flat = coord_grid.view(-1, 2)
    latents = latent_bank(indices)

    recons = []
    for i in range(b):
        pred = model(coord_grid_flat, latents[i])  # [H, W, C]
        # Model output is [..., C]; here coords were flattened, so reshape back
        pred = pred.view(h, w, -1).permute(2, 0, 1).clamp(0.0, 1.0)  # [C, H, W]
        recons.append(pred)

    recons = torch.stack(recons, dim=0)

    # Build side-by-side grid: originals on top, reconstructions on bottom
    grid = make_grid(torch.cat([images, recons], dim=0), nrow=b)
    wandb.log({tag: wandb.Image(grid)})


# ----------------------
# Main
# ----------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train LatentModulatedSiren (functa-style) on CIFAR-10."
    )
    parser.add_argument(
        "--data_root", type=str, default="./data", help="CIFAR-10 data root directory"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--latent_lr", type=float, default=1e-2, help="Learning rate for validation latents"
    )
    parser.add_argument(
        "--val_latent_epochs", type=int, default=5, help="Inner epochs to fit validation latents"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--save_path", type=str, default="functa_cifar10.pt")
    parser.add_argument(
        "--val_every",
        type=int,
        default=10,
        help="Run validation (latent fitting + metrics/logging) every K epochs.",
    )
    parser.add_argument("--wandb_project", type=str, default="functa_cifar10")
    parser.add_argument("--wandb_run_name", type=str, default="functa_cifar10_run")
    parser.add_argument(
        "--log_recon_every",
        type=int,
        default=100,
        help="If > 0, log training reconstructions every N batches.",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    data_root = Path(args.data_root)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    train_loader, test_loader = build_dataloaders(data_root, args.batch_size)

    # Shared functa-style model
    model = LatentModulatedSiren(
        width=args.width,
        depth=args.depth,
        out_channels=3,
        latent_dim=args.latent_dim,
        w0=25.0,
    ).to(device)

    # Coordinate grid for CIFAR-10 images (32x32)
    coord_grid = get_coordinate_grid(32, centered=True, device=device)

    # Latent banks: one for training set, one for validation set
    train_latent_bank = LatentBank(
        num_items=len(train_loader.dataset), latent_dim=args.latent_dim
    ).to(device)
    val_latent_bank = LatentBank(num_items=len(test_loader.dataset), latent_dim=args.latent_dim).to(
        device
    )

    # Separate optimizers for model and latents to avoid momentum mixing
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    latent_optimizer = torch.optim.Adam(train_latent_bank.parameters(), lr=10 * args.lr)

    # Grab a fixed batch for reconstruction logging
    train_sample_images, train_sample_indices = next(iter(train_loader))
    val_sample_images, val_sample_indices = next(iter(test_loader))

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            latent_bank=train_latent_bank,
            dataloader=train_loader,
            model_optimizer=model_optimizer,
            latent_optimizer=latent_optimizer,
            device=device,
            coord_grid=coord_grid,
            epoch=epoch,
            log_recon_every=args.log_recon_every,
        )

        # Always log train summary; run validation only every `val_every` epochs
        train_psnr = -10.0 * math.log10(train_loss) if train_loss > 0 else float("inf")

        if epoch % args.val_every == 0:
            # Fit validation latents for a few inner epochs with model frozen
            fit_val_latents(
                model=model,
                val_latent_bank=val_latent_bank,
                dataloader=test_loader,
                device=device,
                coord_grid=coord_grid,
                latent_lr=args.latent_lr,
                inner_epochs=args.val_latent_epochs,
            )

            val_mse = evaluate(
                model=model,
                val_latent_bank=val_latent_bank,
                dataloader=test_loader,
                device=device,
                coord_grid=coord_grid,
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

        wandb.log(
            {
                "train/loss": train_loss,
                "train/psnr": train_psnr,
                "val/mse": val_mse,
                "val/psnr": val_psnr,
                "epoch": epoch,
            }
        )

        # Log reconstructions from fixed samples
        log_reconstructions(
            model=model,
            latent_bank=train_latent_bank,
            images=train_sample_images,
            indices=train_sample_indices,
            coord_grid=coord_grid,
            device=device,
            tag="train/reconstructions",
        )

        if epoch % args.val_every == 0:
            log_reconstructions(
                model=model,
                latent_bank=val_latent_bank,
                images=val_sample_images,
                indices=val_sample_indices,
                coord_grid=coord_grid,
                device=device,
                tag="val/reconstructions",
            )

    torch.save({"model": model.state_dict()}, args.save_path)
    print(f"Saved model to {args.save_path}")
    wandb.save(args.save_path)


if __name__ == "__main__":
    main()
