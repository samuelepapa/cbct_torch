import torch

PRIMES = torch.tensor([1, 2654435761], dtype=torch.long)


def hash_coords(coords, hashmap_size):
    """
    coords: [..., 2] integer grid coordinates
    returns: [...], hash indices
    """
    h = (coords * PRIMES.to(coords.device)).sum(dim=-1)
    return torch.remainder(h, hashmap_size)


def hashgrid_encode_level(
    x,  # [N, 2] in [0, 1]
    latents,  # [hashmap_size, F]
    resolution,  # int
):
    """
    Returns: [N, F]
    """
    N, F = x.shape[0], latents.shape[1]

    # Scale to grid
    pos = x * resolution
    pos_floor = torch.floor(pos).long()
    w = pos - pos_floor.float()  # interpolation weights

    features = torch.zeros((N, F), device=x.device)

    # Trilinear interpolation over 8 corners
    for dx in [0, 1]:
        for dy in [0, 1]:
            offset = torch.tensor([dx, dy], device=x.device)
            corner = pos_floor + offset
            h = hash_coords(corner, latents.shape[0])

            weight = (1 - w[:, 0] if dx == 0 else w[:, 0]) * (1 - w[:, 1] if dy == 0 else w[:, 1])

            features += latents[h] * weight[:, None]

    return features


class HashGridEncoder(torch.nn.Module):
    def __init__(
        self,
        num_levels=16,
        features_per_level=2,
        hashmap_size=2**19,
        base_resolution=16,
        per_level_scale=1.5,
    ):
        super().__init__()

        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.hashmap_size = hashmap_size

        self.resolutions = [int(base_resolution * (per_level_scale**i)) for i in range(num_levels)]

    def forward(self, x, hash_latents):
        """
        x: [N, 2] in [0,1]
        hash_latents:
            Either
              - list of length num_levels, each element: [hashmap_size, F]
              - or a single tensor of shape [num_levels * hashmap_size, F]
        """
        # convert coords from [-1, 1] to [0, 1]
        x = (x + 1) / 2

        # Allow passing a single 2D tensor with all levels concatenated
        if isinstance(hash_latents, torch.Tensor):
            assert (
                hash_latents.dim() == 2
            ), f"Expected 2D tensor for hash_latents, got {hash_latents.shape}"
            total_entries, F = hash_latents.shape
            expected_entries = self.num_levels * self.hashmap_size
            if total_entries != expected_entries:
                raise ValueError(
                    f"hash_latents has {total_entries} entries, but expected "
                    f"{expected_entries} (= num_levels * hashmap_size). "
                    f"Got shape {hash_latents.shape}"
                )
            hash_latents = hash_latents.view(self.num_levels, self.hashmap_size, F)

        outputs = []

        for lvl in range(self.num_levels):
            feat = hashgrid_encode_level(
                x,
                hash_latents[lvl],
                self.resolutions[lvl],
            )
            outputs.append(feat)

        return torch.cat(outputs, dim=-1)


class HashLatentTransformer(torch.nn.Module):
    def __init__(
        self,
        num_levels,
        hashmap_size,
        features_per_level,
        d_model=256,
    ):
        super().__init__()

        self.num_levels = num_levels
        self.hashmap_size = hashmap_size
        self.features_per_level = features_per_level

        self.token_dim = features_per_level
        self.num_tokens = num_levels * hashmap_size
        self.d_model = d_model

        self.input_proj = torch.nn.Linear(d_model, d_model)

        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=1024,
                batch_first=True,
            ),
            num_layers=6,
        )

        self.output_proj = torch.nn.Linear(d_model, features_per_level)

    def forward(self, z):
        """
        z: [T, d_model] latent input
        returns:
            list[num_levels] of [hashmap_size, F]
        """

        h = self.transformer(self.input_proj(z))
        h = h.mean(dim=0)  # [d_model]

        # Expand to all hash entries
        h = h.expand(self.num_tokens, -1)
        latents = self.output_proj(h)

        latents = latents.view(
            self.num_levels,
            self.hashmap_size,
            self.features_per_level,
        )

        return [latents[:, i] for i in range(self.num_levels)]


class HashLatentField(torch.nn.Module):
    def __init__(
        self,
        transformer: HashLatentTransformer,
        hash_encoder: HashGridEncoder,
        mlp: torch.nn.Module,
    ):
        super().__init__()
        self.transformer = transformer
        self.hash_encoder = hash_encoder
        self.mlp = mlp
        self.num_tokens = self.transformer.num_tokens
        self.d_model = self.transformer.d_model

    def init_latents(self, device):
        return torch.randn(self.num_tokens, self.d_model, device=device)

    def forward(self, coords, latents):
        """
        coords: [N, 2]
        latents: transformer input latents (optimized in inner loop)
        """
        # 1. Transformer predicts hash tables
        hash_latents = self.transformer(latents)

        # 2. Hash-grid encoding
        features = self.hash_encoder(coords, hash_latents)

        # 3. Decode to scalar
        return self.mlp(features)


class HashLatentMLP(torch.nn.Module):
    def __init__(
        self, hash_encoder: HashGridEncoder, mlp: torch.nn.Module, features_per_level: int
    ):
        super().__init__()
        self.hash_encoder = hash_encoder
        self.mlp = mlp
        self.num_latents = self.hash_encoder.num_levels * self.hash_encoder.hashmap_size
        self.features_per_level = features_per_level

    def init_latents(self, device):
        return torch.randn(self.num_latents, self.features_per_level, device=device)

    def forward(self, coords, latents):
        """
        coords: [N, 2]
        latents: [num_latents, features_per_level]
        """
        features = self.hash_encoder(coords, latents)
        return self.mlp(features)
