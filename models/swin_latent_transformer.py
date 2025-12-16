"""Swin-style Latent Transformer with windowed attention on a 2D latent grid.

This model places latents on a 2D grid and uses windowed self-attention (Swin-style)
instead of full self-attention. This is more efficient for larger grids and provides
better locality inductive bias.

Key features:
- Latents arranged on a 2D grid derived from AABB
- Windowed self-attention with window shifting for cross-window communication
- Shared coordinate embedding for queries and latent positions
- Cross-attention between processed latents and query coordinates
"""

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

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
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        if hasattr(torch.backends.cuda, "sdp_kernel"):
            torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_mem_efficient=False, enable_math=True
            )
    except Exception as exc:
        warnings.warn(f"Could not configure SDP backends; falling back to defaults: {exc}")


def _make_latent_grid(
    aabb_min: torch.Tensor, aabb_max: torch.Tensor, grid_h: int, grid_w: int
) -> torch.Tensor:
    """
    Create a 2D grid of latent positions spanning the AABB.

    Args:
        aabb_min: [2] min corner
        aabb_max: [2] max corner
        grid_h: height of the latent grid
        grid_w: width of the latent grid

    Returns:
        latent_positions: [grid_h * grid_w, 2] in world/AABB coordinates
    """
    device = aabb_min.device
    ys = torch.linspace(aabb_min[0].item(), aabb_max[0].item(), grid_h, device=device)
    xs = torch.linspace(aabb_min[1].item(), aabb_max[1].item(), grid_w, device=device)
    y_g, x_g = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([y_g, x_g], dim=-1).reshape(-1, 2)
    return grid


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.

    Args:
        x: [B, H, W, C] feature map
        window_size: window size

    Returns:
        windows: [B * num_windows, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition.

    Args:
        windows: [B * num_windows, window_size, window_size, C]
        window_size: window size
        H: height of feature map
        W: width of feature map

    Returns:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, ws, ws]
        coords_flatten = torch.flatten(coords, 1)  # [2, ws*ws]
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, ws*ws, ws*ws]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [ws*ws, ws*ws, 2]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [ws*ws, ws*ws]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [num_windows * B, window_size * window_size, C]
            mask: [num_windows, ws*ws, ws*ws] or None

        Returns:
            [num_windows * B, window_size * window_size, C]
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B_, num_heads, N, head_dim]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B_, num_heads, N, N]

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # [num_heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with window attention and optional shifting.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def _compute_attention_mask(
        self, H: int, W: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Compute attention mask for shifted window attention."""
        if self.shift_size == 0:
            return None

        # Calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, ws, ws, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        return attn_mask

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C] flattened feature map
            H: height
            W: width

        Returns:
            [B, H*W, C]
        """
        B, L, C = x.shape
        assert L == H * W, f"Input size mismatch: {L} != {H * W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, ws, ws, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention with mask
        attn_mask = self._compute_attention_mask(H, W, x.device)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Residual + FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class SwinLatentTransformer(nn.Module):
    """
    Swin-style Latent Transformer with windowed attention on a 2D latent grid.

    Architecture:
    1. Latents placed on a 2D grid derived from AABB
    2. Swin-style windowed self-attention on the latent grid
    3. Cross-attention between latents (K/V) and coordinates (Q)
    4. Final MLP to produce output value

    Args:
        coord_dim: Dimension of input coordinates (e.g., 2 for 2D)
        latent_dim: Dimension of latent vectors
        hidden_dim: Transformer hidden dimension
        grid_size: Size of the latent grid (grid_size x grid_size)
        window_size: Window size for windowed attention
        num_swin_layers: Number of Swin transformer layers
        num_cross_attn_layers: Number of cross-attention layers
        num_heads: Number of attention heads
        mlp_hidden_dim: Hidden dimension for final MLP
        output_dim: Output dimension
        dropout: Dropout probability
        rff_encoding_size: RFF encoding size
        rff_scale: RFF scale parameter
    """

    def __init__(
        self,
        coord_dim: int = 2,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        grid_size: int = 8,  # 8x8 = 64 latents
        window_size: int = 4,
        num_swin_layers: int = 4,
        num_cross_attn_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mlp_hidden_dim: int = 512,
        output_dim: int = 1,
        dropout: float = 0.1,
        rope_base_freq: float = 10000.0,
        rope_learnable_freq: bool = False,
        rope_coord_freq_multiplier: float = 100.0,
        rff_encoding_size: int = 128,
        rff_scale: float = 10.0,
    ):
        super().__init__()
        _prefer_flash_attention()

        assert hidden_dim % 2 == 0, f"hidden_dim must be even for RoPE, got {hidden_dim}"
        assert (
            grid_size % window_size == 0
        ), f"grid_size ({grid_size}) must be divisible by window_size ({window_size})"

        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.num_latents = grid_size * grid_size
        self.window_size = window_size

        # RoPE for coordinate queries
        self.rope_coords = RotaryEmbedding(
            dim=hidden_dim,
            freqs_for="pixel",
            max_freq=10.0,
            theta=rope_base_freq,
            learned_freq=rope_learnable_freq,
            cache_if_possible=False,
        )

        # Shared RFF encoding and coordinate embedding
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

        # Learnable 2D position embedding for the grid (added to coordinate-based embedding)
        self.latent_pos_embed = nn.Parameter(torch.randn(1, self.num_latents, hidden_dim))

        # Swin Transformer layers (alternating W-MSA and SW-MSA)
        self.swin_layers = nn.ModuleList()
        for i in range(num_swin_layers):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            self.swin_layers.append(
                SwinTransformerBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    dropout=dropout,
                    attn_dropout=dropout,
                )
            )

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
        """Initialize latents for a batch."""
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
        H, W = self.grid_size, self.grid_size

        if latent_positions is None:
            latent_positions = _make_latent_grid(aabb_min, aabb_max, H, W)
        if latent_positions.dim() == 2:
            latent_positions = latent_positions.unsqueeze(0).expand(
                B, -1, -1
            )  # [B, num_latents, 2]

        # Encode query coordinates (shared embedding)
        coords_encoded = self.rff_encoding(coords)  # [B, N, 2*rff_size]
        coord_embed = self.coord_embed(coords_encoded)  # [B, N, hidden_dim]

        # Encode latent positions with the same pipeline
        latent_pos_encoded = self.rff_encoding(latent_positions)  # [B, L, 2*rff_size]
        latent_coord_embed = self.coord_embed(latent_pos_encoded)  # [B, L, hidden_dim]

        # Project content latents and add positional embeddings
        latents = self.latent_proj(latents)  # [B, L, hidden_dim]
        latents = latents + latent_coord_embed + self.latent_pos_embed

        # Apply Swin Transformer layers
        for swin_layer in self.swin_layers:
            latents = swin_layer(latents, H, W)

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


if __name__ == "__main__":
    # Test the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SwinLatentTransformer(
        coord_dim=2,
        latent_dim=128,
        hidden_dim=128,
        grid_size=8,  # 8x8 = 64 latents
        window_size=4,
        num_swin_layers=4,
        num_cross_attn_layers=2,
        num_heads=4,
        mlp_hidden_dim=256,
        output_dim=1,
    ).to(device)

    # Initialize latents
    latents = model.init_latents(batch_size=1, device=device)

    # Test with coordinates
    coords = torch.randn(100, 2, device=device)
    aabb_min = torch.tensor([-1.0, -1.0], device=device)
    aabb_max = torch.tensor([1.0, 1.0], device=device)

    output = model(coords, latents, aabb_min, aabb_max)
    print(f"Coord input: {coords.shape} + latents {latents.shape} -> {output.shape}")
    assert output.shape == (100, 1), f"Expected (100, 1), got {output.shape}"

    # Test with many points
    coords_many = torch.randn(10000, 2, device=device)
    output_many = model(coords_many, latents, aabb_min, aabb_max)
    print(f"Many coords: {coords_many.shape} + latents {latents.shape} -> {output_many.shape}")
    assert output_many.shape == (10000, 1)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    # Test gradient flow
    print("\nTesting gradient flow:")
    latents_opt = model.init_latents(1, device)
    latents_opt.requires_grad = True

    target = torch.ones(100, 1, device=device)
    coords_test = torch.randn(100, 2, device=device)

    output = model(coords_test, latents_opt, aabb_min, aabb_max)
    loss = F.mse_loss(output, target)
    loss.backward()

    print(f"Loss: {loss.item():.6f}")
    print(
        f"Latent gradients: mean={latents_opt.grad.mean().item():.6f}, std={latents_opt.grad.std().item():.6f}"
    )

    print("\nAll tests passed!")
