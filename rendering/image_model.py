"""Image model for ray tracing with 2D images."""

import torch
import torch.nn.functional as F


class ImageModel(torch.nn.Module):
    """
    Wraps a 2D image tensor to act as a 'function' for ray tracing.
    Input: (x, y) coordinates in AABB range.
    Output: Interpolated pixel values.
    """

    def __init__(self, image, aabb_min, aabb_max):
        super().__init__()
        # image: [1, 1, H, W]
        self.register_buffer("image", image)
        self.register_buffer("aabb_min", aabb_min)
        self.register_buffer("aabb_max", aabb_max)

    def forward(self, x):
        # x: [B, 2] or [B, P, S, 2] -> needs [N, 1, 1, 2] for grid_sample
        # grid_sample expects [N, C, H_in, W_in] input and [N, H_out, W_out, 2] grid

        orig_shape = x.shape
        x_flat = x.reshape(1, 1, -1, 2)

        # Normalize from [aabb_min, aabb_max] to [-1, 1]
        # x_norm = 2 * (x - min) / (max - min) - 1
        aabb_min = self.aabb_min.reshape(1, 1, 1, 2)
        aabb_max = self.aabb_max.reshape(1, 1, 1, 2)

        x_norm = 2.0 * (x_flat - aabb_min) / (aabb_max - aabb_min) - 1.0

        # grid_sample coordinates: -1=left/top, 1=right/bottom
        out = F.grid_sample(
            self.image, x_norm, align_corners=True, mode="bilinear", padding_mode="zeros"
        )
        # out: [1, 1, 1, TotalPoints]

        # Remove last dim 2, return [...] with original batch dims
        return out.view(orig_shape[:-1])
