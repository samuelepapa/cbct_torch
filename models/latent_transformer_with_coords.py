"""Latent Transformer variant that ties latents to 2D coordinates.

Key differences vs. `models/latent_transformer.py`:
- Each latent is associated with a 2D position generated from the AABB
  (aabb_min, aabb_max) that defines the image extent.
- The same coordinate embedding pipeline is used for both query coordinates and
  latent coordinates, so positional information is shared consistently.
"""

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.rff_net import GaussianEncoding
from models.rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb


def _prefer_flash_attention():
    """
    Hint PyTorch to use flash attention kernels when available.
    If flash is unavailable, fall back to math kernels to avoid runtime errors.
    """
    if not torch.cuda.is_available():
        return

    try:
        # Prefer flash; allow math fallback to prevent "Invalid backend" errors
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        # In newer PyTorch versions, the consolidated API is available
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_mem_efficient=False, enable_math=True
            )
    except Exception as exc:  # pragma: no cover - safety fallback
        warnings.warn(f"Could not configure SDP backends; falling back to defaults: {exc}")


def _make_latent_grid(
    aabb_min: torch.Tensor, aabb_max: torch.Tensor, num_latents: int
) -> torch.Tensor:
    """
    Create a 2D grid of latent positions spanning the AABB.
    The grid is as square as possible; extra points are truncated.

    Args:
        aabb_min: [2] min corner
        aabb_max: [2] max corner
        num_latents: number of latent positions to generate

    Returns:
        latent_positions: [num_latents, 2] in world/AABB coordinates
    """
    device = aabb_min.device
    side = math.ceil(num_latents**0.5)
    ys = torch.linspace(aabb_min[0].item(), aabb_max[0].item(), side, device=device)
    xs = torch.linspace(aabb_min[1].item(), aabb_max[1].item(), side, device=device)
    y_g, x_g = torch.meshgrid(ys, xs, indexing="xy")
    grid = torch.stack([y_g, x_g], dim=-1).reshape(-1, 2)
    return grid[:num_latents]


class LatentTransformerWithCoords(nn.Module):
    """
    Latent Transformer where each latent is tied to a 2D position derived from the AABB.

    Forward signature adds aabb_min/aabb_max so latent positions are anchored to
    the same spatial frame as the query coordinates.
    """

    def __init__(
        self,
        coord_dim: int = 2,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        num_latents: int = 64,
        num_decoder_layers: int = 4,
        num_cross_attn_layers: int = 2,
        num_heads: int = 8,
        mlp_hidden_dim: int = 512,
        output_dim: int = 1,
        dropout: float = 0.1,
        rope_base_freq: float = 10000.0,
        rope_learnable_freq: bool = False,
        rope_coord_freq_multiplier: float = 100.0,  # kept for parity with base class
        rff_encoding_size: int = 128,
        rff_scale: float = 10.0,
    ):
        super().__init__()
        _prefer_flash_attention()

        assert hidden_dim % 2 == 0, f"hidden_dim must be even for RoPE, got {hidden_dim}"

        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_latents = num_latents

        # RoPE for coordinate queries
        self.rope_coords = RotaryEmbedding(
            dim=hidden_dim,
            freqs_for="pixel",
            max_freq=10.0,
            theta=rope_base_freq,
            learned_freq=rope_learnable_freq,
            cache_if_possible=False,
        )

        # RoPE for latents
        self.rope_latents = RotaryEmbedding(
            dim=hidden_dim,
            freqs_for="lang",
            theta=rope_base_freq,
            learned_freq=rope_learnable_freq,
            cache_if_possible=True,
        )

        # Shared RFF encoding and coordinate embedding for both queries and latent positions
        self.rff_encoding = GaussianEncoding(coord_dim, rff_encoding_size, rff_scale)
        rff_output_dim = 2 * rff_encoding_size
        self.coord_embed = nn.Sequential(
            nn.Linear(rff_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Project latents to hidden_dim if they differ
        self.latent_proj = (
            nn.Linear(latent_dim, hidden_dim) if latent_dim != hidden_dim else nn.Identity()
        )

        # Optional latent positional embedding (kept for compatibility; added to coord features)
        self.latent_pos_embed = nn.Parameter(torch.randn(num_latents, hidden_dim))

        # Decoder-only transformer on latents (self-attention)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)

        # Cross-attention layers (latents as K/V, coords as Q)
        self.cross_attn_layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    latent_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    rope_coords=self.rope_coords,
                )
                for _ in range(num_cross_attn_layers)
            ]
        )

        # Output head
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 2, output_dim),
        )

        self.inner_lr = nn.Parameter(torch.tensor(0.01))
        self._init_weights()

    def init_latents(self, batch_size: int, device) -> torch.Tensor:
        """Initialize latents for a batch (content only; coords are generated per forward)."""
        # Deterministic initialization at ones for stability/consistency
        return torch.ones(batch_size, self.num_latents, self.latent_dim, device=device)

    def get_inner_lr(self):
        return self.inner_lr

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Small init for final layer
        last_linear = None
        for module in reversed(list(self.output_mlp.modules())):
            if isinstance(module, nn.Linear):
                last_linear = module
                break
        if last_linear is not None:
            nn.init.xavier_uniform_(last_linear.weight, gain=0.01)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

        nn.init.normal_(self.latent_pos_embed, mean=0.0, std=0.02)

    def forward(
        self,
        coords: torch.Tensor,
        latents: torch.Tensor,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        latent_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            coords: [num_points, coord_dim] or [B, num_points, coord_dim]
            latents: [B, num_latents, latent_dim]
            aabb_min: [2] min corner of AABB
            aabb_max: [2] max corner of AABB
            latent_positions: optional [num_latents, 2] in world coords; if None, generated on a grid.

        Returns:
            [num_points, output_dim]
        """
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)

        B = coords.shape[0]

        if latent_positions is None:
            latent_positions = _make_latent_grid(aabb_min, aabb_max, self.num_latents)
        if latent_positions.dim() == 2:
            latent_positions = latent_positions.unsqueeze(0).expand(
                B, -1, -1
            )  # [B, num_latents, 2]

        # Encode coordinates (shared embedding)
        coords_encoded = self.rff_encoding(coords)  # [B, N, 2*rff_size]
        coord_embed = self.coord_embed(coords_encoded)  # [B, N, hidden_dim]

        # Encode latent positions with the same pipeline
        latent_pos_encoded = self.rff_encoding(latent_positions)  # [B, L, 2*rff_size]
        latent_pos_embed = self.coord_embed(latent_pos_encoded)  # [B, L, hidden_dim]

        # Project content latents and add positional embedding
        latents = self.latent_proj(latents)  # [B, L, hidden_dim]
        latents = latents + latent_pos_embed + self.latent_pos_embed.unsqueeze(0)

        # RoPE on latents (self-attn)
        latents = self.rope_latents.rotate_queries_or_keys(latents, seq_dim=-2)
        latents = self.decoder(latents)

        # Cross attention: queries = coord_embed, keys/values = latents
        for cross_attn in self.cross_attn_layers:
            coord_embed = cross_attn(coord_embed, latents, coords)

        output = self.output_mlp(coord_embed)  # [B, N, output_dim]
        return output.squeeze(0)


class CrossAttentionBlock(nn.Module):
    """Cross-attention with RoPE applied to queries derived from coordinates."""

    def __init__(self, latent_dim, num_heads, dropout=0.1, rope_coords=None):
        super().__init__()
        self.rope_coords = rope_coords
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, key_value, coords):
        if self.rope_coords is not None:
            B, N, _ = coords.shape
            coord_positions = torch.norm(coords, dim=-1)
            freqs = self.rope_coords.forward(coord_positions, seq_len=N)
            query = apply_rotary_emb(freqs, query, seq_dim=-2)

        attn_output, _ = self.cross_attn(query=query, key=key_value, value=key_value)
        query = self.norm1(query + attn_output)
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)
        return query
