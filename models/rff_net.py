import numpy as np
import torch
import torch.nn as nn


class GaussianEncoding(nn.Module):
    def __init__(self, in_features, mapping_size, scale):
        super().__init__()
        self.mapping_size = mapping_size
        self.scale = scale
        self.register_buffer("B", torch.randn((mapping_size, in_features)) * scale)

    def forward(self, x):
        # x: [..., in_features]
        # B: [mapping_size, in_features]
        # x @ B.T : [..., mapping_size]
        x_proj = (2.0 * np.pi * x) @ self.B.t()
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class RFFNet(nn.Module):
    def __init__(
        self, in_dim=3, out_dim=1, hidden_dim=64, num_layers=3, encoding_size=256, scale=10.0
    ):
        super().__init__()

        self.encoding = GaussianEncoding(in_dim, encoding_size, scale)

        # Input to MLP is 2 * encoding_size
        current_dim = 2 * encoding_size

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

        # Dummy conditioning to match interface if needed, or we can ignore it for now.
        # train.py expects model.conditioning.codes or similar if it optimizes it,
        # but for a "simple RFFNet" this might not be needed unless integrated into the full pipeline.
        # I will leave out complex conditioning for now as requested "simple".

    def forward(self, x, patient_idx=None):
        # x: [batch, in_dim] or [batch, rays, in_dim]
        encoded = self.encoding(x)
        out = self.mlp(encoded)
        return out
