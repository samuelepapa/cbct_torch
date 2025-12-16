import argparse
from pathlib import Path

import tinycudann as tcnn
import torch
import torchvision
import torchvision.transforms as T


class CIFARImage(torch.nn.Module):
    """Image wrapper with bilinear sampling, similar to NVIDIA's example."""

    def __init__(self, device: torch.device):
        super().__init__()

        transform = T.Compose(
            [
                T.ToTensor(),  # [C, H, W] in [0, 1]
            ]
        )

        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )

        img, _ = dataset[0]  # [3, 32, 32]
        # Store as [H, W, C] to match the NVIDIA sample's layout
        self.data = img.permute(1, 2, 0).contiguous().to(device)  # [H, W, C]
        self.shape = self.data.shape  # (H, W, C)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        xs: [N, 2] in [0, 1]^2 (x, y).
        Returns bilinearly sampled colors [N, C] in [0, 1].
        """
        with torch.no_grad():
            H, W, _ = self.shape

            # Map [0,1] -> pixel space
            xs_pix = xs * torch.tensor([W, H], device=xs.device, dtype=torch.float32)
            indices = xs_pix.long()
            lerp_weights = xs_pix - indices.float()

            x0 = indices[:, 0].clamp(min=0, max=W - 1)
            y0 = indices[:, 1].clamp(min=0, max=H - 1)
            x1 = (x0 + 1).clamp(max=W - 1)
            y1 = (y0 + 1).clamp(max=H - 1)

            c00 = self.data[y0, x0]  # [N, C]
            c10 = self.data[y0, x1]
            c01 = self.data[y1, x0]
            c11 = self.data[y1, x1]

            wx = lerp_weights[:, 0:1]
            wy = lerp_weights[:, 1:2]

            return (
                c00 * (1.0 - wx) * (1.0 - wy)
                + c10 * wx * (1.0 - wy)
                + c01 * (1.0 - wx) * wy
                + c11 * wx * wy
            )


def build_tcnn_field(n_channels: int, device: torch.device):
    """Tiny-cuda-nn field, similar to NVIDIA's mlp_learning_an_image_pytorch.py."""
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=n_channels,
        encoding_config={
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 15,
            "base_resolution": 16,
            "per_level_scale": 1.5,
            "fixed_point_pos": False,
        },
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        },
    ).to(device)

    # Enable JIT fusion if available (as in NVIDIA's example)
    model.jit_fusion = tcnn.supports_jit_fusion()
    return model


def main():
    parser = argparse.ArgumentParser(description="Tiny-cuda-nn CIFAR10 fitting (NVIDIA-style)")
    parser.add_argument("--steps", type=int, default=20000, help="Number of optimization steps")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--out_dir", type=str, default="results/tcnn_cifar10", help="Output dir")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load CIFAR10 image wrapper
    image = CIFARImage(device)
    H, W, C = image.shape  # [H, W, C]

    # 2. Build tiny-cuda-nn model
    model = build_tcnn_field(C, device)

    # 3. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4. Prepare output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5. Set up full-image grid for periodic visualization
    half_dx = 0.5 / W
    half_dy = 0.5 / H
    xs = torch.linspace(half_dx, 1.0 - half_dx, W, device=device)
    ys = torch.linspace(half_dy, 1.0 - half_dy, H, device=device)
    xv, yv = torch.meshgrid(ys, xs, indexing="ij")  # (y, x)
    xy = torch.stack((xv.flatten(), yv.flatten()), dim=-1)  # [H*W, 2] in [0,1]

    # Save reference CIFAR image for comparison
    with torch.no_grad():
        ref = image(xy).reshape(H, W, C).permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
        torchvision.utils.save_image(ref, out_dir / "target.png")

    batch_size = 2**15
    interval = 100

    print(f"Beginning optimization with {args.steps} training steps.")

    # Try to trace the target image module for performance (optional)
    try:
        batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
        traced_image = torch.jit.trace(image, batch)
    except Exception:
        traced_image = image

    for step in range(1, args.steps + 1):
        batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
        targets = traced_image(batch)  # [B, C]
        output = model(batch)  # [B, C]

        # Relative L2 loss as in NVIDIA's sample, with dtype match
        relative_l2_error = (output - targets.to(output.dtype)) ** 2 / (output.detach() ** 2 + 0.01)
        loss = relative_l2_error.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % interval == 0 or step == 1:
            print(f"[{step}/{args.steps}] loss = {loss.item():.6f}")

            with torch.no_grad():
                pred = model(xy).reshape(H, W, C).permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
                torchvision.utils.save_image(
                    pred,
                    out_dir / f"pred_step_{step:06d}.png",
                )

    print("Done. Check", out_dir)


if __name__ == "__main__":
    main()
