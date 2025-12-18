"""Image model for ray tracing with 2D images."""

import torch
import torch.nn.functional as F


class ImageModel(torch.nn.Module):
    """
    Wraps a 2D image tensor to act as a 'function' for ray tracing.
    Input: (x, y) coordinates in AABB range.
    Output: Interpolated pixel values.
    """

    def __init__(self, image):
        super().__init__()
        # image: [1, 1, H, W]
        self.register_buffer("image", image)

    def forward(self, x):
        # x: [B, 2] or [B, P, S, 2] in normalized coordinates [-1, 1]
        # grid_sample expects [N, C, H_in, W_in] input and [N, H_out, W_out, 2] grid

        orig_shape = x.shape
        x_flat = x.reshape(1, 1, -1, 2)

        # Coordinates are already in [-1, 1] range
        out = F.grid_sample(
            self.image, x_flat, align_corners=True, mode="bilinear", padding_mode="zeros"
        )
        return out.view(orig_shape[:-1])
