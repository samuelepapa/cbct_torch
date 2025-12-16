"""Latent Transformer model for coordinate-based field representation."""

import math
import warnings

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


class LatentTransformer(nn.Module):
    """
    Takes a coordinate and a set of latents, and returns a value for the field.

    Architecture:
    1. Decoder-only transformer on latents (self-attention)
    2. Cross-attention between latents (key/value) and coordinates (query)
    3. Final MLP to produce output value

    Args:
        coord_dim: Dimension of input coordinates (e.g., 2 for 2D, 3 for 3D)
        latent_dim: Dimension of latent vectors
        num_latents: Number of latent vectors
        num_decoder_layers: Number of decoder-only transformer layers
        num_cross_attn_layers: Number of cross-attention layers
        num_heads: Number of attention heads
        mlp_hidden_dim: Hidden dimension for final MLP
        output_dim: Output dimension (e.g., 1 for scalar field)
        dropout: Dropout probability
    """

    def __init__(
        self,
        coord_dim=2,
        latent_dim=256,
        hidden_dim=256,  # Transformer hidden dimension (can be different from latent_dim)
        num_latents=64,
        num_decoder_layers=4,
        num_cross_attn_layers=2,
        num_heads=8,
        mlp_hidden_dim=512,
        output_dim=1,
        dropout=0.1,
        rope_base_freq=10000.0,
        rope_learnable_freq=False,
        rope_coord_freq_multiplier=100.0,  # Multiplier for coordinate RoPE frequency (higher = more high-freq)
        rff_encoding_size=128,  # RFF encoding size
        rff_scale=10.0,  # RFF scale parameter
    ):
        super().__init__()
        _prefer_flash_attention()

        assert hidden_dim % 2 == 0, f"hidden_dim must be even for RoPE, got {hidden_dim}"

        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_latents = num_latents

        # NOTE: Latents are NOT learnable parameters - they will be passed as input
        # This allows for meta-learning where latents are optimized per-image

        # RoPE for coordinate queries - uses much higher frequency
        # For coordinates, we use 'lang' mode with high theta for high-frequency encoding
        self.rope_coords = RotaryEmbedding(
            dim=hidden_dim,
            freqs_for="pixel",
            max_freq=10.0,
            theta=rope_base_freq,
            learned_freq=rope_learnable_freq,
            cache_if_possible=False,  # Can't cache for arbitrary coordinates
        )

        # RoPE for latents - standard positional embedding frequency
        self.rope_latents = RotaryEmbedding(
            dim=hidden_dim,
            freqs_for="lang",
            theta=rope_base_freq,  # Standard frequency for positional embeddings
            learned_freq=rope_learnable_freq,
            cache_if_possible=True,
        )

        # RFF encoding for coordinates
        self.rff_encoding = GaussianEncoding(coord_dim, rff_encoding_size, rff_scale)
        rff_output_dim = 2 * rff_encoding_size  # cos and sin

        # Coordinate embedding - projects RFF-encoded coordinates to hidden_dim
        self.coord_embed = nn.Sequential(
            nn.Linear(rff_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Project latents to hidden_dim if they differ
        if latent_dim != hidden_dim:
            self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        else:
            self.latent_proj = nn.Identity()

        # Positional encoding for latents (in latent_dim space) - kept for backward compatibility
        # RoPE will be applied on top of this
        self.latent_pos_embed = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Decoder-only transformer on latents (self-attention) - uses hidden_dim
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # Cross-attention layers (latents as K/V, coords as Q) - uses hidden_dim
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

        # Final MLP to produce output - uses hidden_dim
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

        # Learnable inner learning rate (meta-learned)
        # Initialize with log-space for better optimization
        self.inner_lr = nn.Parameter(torch.tensor(0.01))  # exp(-2) ≈ 0.135

        self._init_weights()

    def init_latents(self, batch_size, device):
        """
        Initialize latents for a batch.

        Args:
            batch_size: Number of images in batch
            device: Device to create latents on

        Returns:
            Tensor of shape [batch_size, num_latents, latent_dim]
        """
        # Deterministic initialization at ones for stability/consistency
        return torch.ones(batch_size, self.num_latents, self.latent_dim, device=device)

    def get_inner_lr(self):
        """
        Get the current inner learning rate.

        Returns:
            Positive learning rate value (exp of log_inner_lr)
        """
        return self.inner_lr

    def _init_weights(self):
        """Initialize weights with careful scaling to prevent extreme outputs."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier/Glorot initialization with gain for better stability
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Special initialization for output layer - very small weights
        # This prevents extreme initial predictions
        if hasattr(self.output_mlp, "__iter__"):
            for module in self.output_mlp:
                if isinstance(module, nn.Linear):
                    # Last linear layer gets extra small initialization
                    pass  # Already initialized above
            # Get the last linear layer
            last_linear = None
            for module in reversed(list(self.output_mlp.modules())):
                if isinstance(module, nn.Linear):
                    last_linear = module
                    break
            if last_linear is not None:
                # Initialize final layer with very small weights
                nn.init.xavier_uniform_(last_linear.weight, gain=0.01)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)

        # Initialize positional embeddings with smaller values
        nn.init.normal_(self.latent_pos_embed, mean=0.0, std=0.02)

    def forward(self, coords, latents):
        """
        Forward pass.

        Args:
            coords: Tensor of shape [num_points, coord_dim] - all points treated as sequence
            latents: Tensor of shape [1, num_latents, latent_dim] - single set of latents

        Returns:
            Tensor of shape [num_points, output_dim]
        """
        # Coords are [num_points, coord_dim] - treat as sequence
        # Latents are [1, num_latents, latent_dim] - shared across all points

        if coords.dim() == 2:
            # [num_points, coord_dim] -> add batch dim -> [1, num_points, coord_dim]
            coords = coords.unsqueeze(0)

        # Project latents to hidden_dim if needed
        latents = self.latent_proj(latents)  # [B, num_latents, hidden_dim]

        # Apply RoPE to latents using standard positional embedding (automatic indices)
        latents = self.rope_latents.rotate_queries_or_keys(
            latents, seq_dim=-2
        )  # [B, num_latents, hidden_dim]

        # Process latents through decoder-only transformer
        latents = self.decoder(latents)  # [B, num_latents, hidden_dim]

        # Apply RFF encoding to coordinates
        coords_encoded = self.rff_encoding(coords)  # [B, num_points, 2*rff_encoding_size]

        # Embed RFF-encoded coordinates
        coord_embed = self.coord_embed(coords_encoded)  # [B, num_points, hidden_dim]

        # Cross-attention: coords query latents (RoPE is applied inside CrossAttentionBlock)
        for cross_attn in self.cross_attn_layers:
            coord_embed = cross_attn(coord_embed, latents, coords)  # [B, num_points, hidden_dim]

        # Generate output
        output = self.output_mlp(coord_embed)  # [B, num_points, output_dim]

        # Remove batch dimension and return [num_points, output_dim]
        return output.squeeze(0)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block where query comes from coordinates and key/value from latents.
    Applies RoPE to queries before attention using coordinate values.
    """

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
        """
        Args:
            query: [B, num_query, latent_dim] - from coordinates
            key_value: [B, num_kv, latent_dim] - from latents
            coords: [B, num_query, coord_dim] - coordinate values for RoPE

        Returns:
            [B, num_query, latent_dim]
        """
        # Apply RoPE to queries using coordinate values
        # For multi-dimensional coordinates, we use a learned linear combination
        # or sum across dimensions as a simple approach
        if self.rope_coords is not None:
            B, N, coord_dim = coords.shape
            # Combine coordinate dimensions: use L2 norm or sum
            # This gives a single position value per coordinate point
            coord_positions = torch.norm(
                coords, dim=-1
            )  # [B, N] - L2 norm across coordinate dimensions
            # Alternative: coord_positions = coords.sum(dim=-1)  # Simple sum

            # Get frequencies for these positions
            freqs = self.rope_coords.forward(coord_positions, seq_len=N)  # [B, N, dim]

            # Apply rotary embedding
            query_rope = apply_rotary_emb(freqs, query, seq_dim=-2)
        else:
            query_rope = query

        # Cross-attention with residual
        attn_output, _ = self.cross_attn(
            query=query_rope,
            key=key_value,
            value=key_value,
        )
        query = self.norm1(query + attn_output)

        # FFN with residual
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)

        return query


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Test the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LatentTransformer(
        coord_dim=2,
        latent_dim=128,
        hidden_dim=128,  # Must be even for RoPE
        num_latents=32,
        num_decoder_layers=2,
        num_cross_attn_layers=2,
        num_heads=4,
        mlp_hidden_dim=256,
        output_dim=1,
    ).to(device)

    # Initialize latents (single set, batch_size=1)
    latents = model.init_latents(batch_size=1, device=device)

    # Test with multiple coordinates (treated as sequence)
    coords = torch.randn(100, 2, device=device)  # [num_points, coord_dim]
    output = model(coords, latents)
    print(f"Coord input: {coords.shape} + latents {latents.shape} -> {output.shape}")
    assert output.shape == (100, 1), f"Expected (100, 1), got {output.shape}"

    # Test with many points (like rendering)
    coords_many = torch.randn(10000, 2, device=device)
    output_many = model(coords_many, latents)
    print(f"Many coords: {coords_many.shape} + latents {latents.shape} -> {output_many.shape}")
    assert output_many.shape == (10000, 1)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    # Test first-order meta-learning (Reptile-style)
    print("\nTesting first-order meta-learning:")

    # Test data
    target = torch.ones(100, 1, device=device)
    coords_test = torch.randn(100, 2, device=device)

    # Outer optimizer for model weights
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    for meta_step in range(500):
        # Initialize latents for this task
        latents_opt = model.init_latents(1, device)
        latents_opt.requires_grad = True

        # Inner loop: optimize latents with SGD
        inner_optimizer = torch.optim.SGD([latents_opt], lr=0.1)
        for i in range(3):
            inner_optimizer.zero_grad()
            output = model(coords_test, latents_opt)
            loss = F.mse_loss(output, target)
            loss.backward()
            inner_optimizer.step()

            if i == 2:  # Last inner step
                print(f"  Meta-step {meta_step+1}, Inner step {i+1}: loss = {loss.item():.6f}")

        # Outer loop: update model with detached optimized latents
        outer_optimizer.zero_grad()

        # Detach latents (first-order approximation)
        latents_final = latents_opt.detach().requires_grad_(True)

        final_output = model(coords_test, latents_final)
        final_loss = F.mse_loss(final_output, target)

        # Backward only through model weights
        final_loss.backward()
        outer_optimizer.step()

        print(f"  Meta-step {meta_step+1}, Final loss: {final_loss.item():.6f}")

    print("\nAll tests passed!")
