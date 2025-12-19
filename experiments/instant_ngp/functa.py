from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
from torch import nn

Tensor = torch.Tensor
Batch = Mapping[str, torch.Tensor]
OptState = Any


class Sine(nn.Module):
    """Applies a scaled sine transform to input: out = sin(w0 * in)."""

    def __init__(self, w0: float = 1.0):
        """Constructor.

        Args:
            w0: Scale factor in sine activation (omega_0 factor from SIREN).
        """

        super().__init__()
        self.w0 = w0

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.w0 * x)


class FiLM(nn.Module):
    """Applies a FiLM modulation: out = scale * in + shift.

    Notes:
        We currently initialize FiLM layers as the identity.
    """

    def __init__(
        self,
        f_in: int,
        modulate_scale: bool = True,
        modulate_shift: bool = True,
    ):
        """Constructor.

        Args:
            f_in: Number of input features.
            modulate_scale: If True, modulates scales.
            modulate_shift: If True, modulates shifts.
        """

        super().__init__()
        assert modulate_scale or modulate_shift

        self.f_in = f_in
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift

        if modulate_scale:
            self.scale = nn.Parameter(torch.ones(f_in))
        else:
            self.register_buffer("scale", torch.ones(f_in), persistent=False)

        if modulate_shift:
            self.shift = nn.Parameter(torch.zeros(f_in))
        else:
            self.register_buffer("shift", torch.zeros(f_in), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x + self.shift


class ModulatedSirenLayer(nn.Module):
    """Applies a linear layer followed by a modulation and sine activation."""

    def __init__(
        self,
        f_in: int,
        f_out: int,
        w0: float = 1.0,
        is_first: bool = False,
        is_last: bool = False,
        modulate_scale: bool = True,
        modulate_shift: bool = True,
        apply_activation: bool = True,
    ):
        """Constructor.

        Args:
            f_in: Number of input features.
            f_out: Number of output features.
            w0: Scale factor in sine activation.
            is_first: Whether this is first layer of model.
            is_last: Whether this is last layer of model.
            modulate_scale: If True, modulates scales.
            modulate_shift: If True, modulates shifts.
            apply_activation: If True, applies sine activation.
        """

        super().__init__()

        self.f_in = f_in
        self.f_out = f_out
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.apply_activation = apply_activation

        # SIREN-style weight initialization
        init_range = 1 / f_in if is_first else (6 / f_in) ** 0.5 / w0

        self.linear = nn.Linear(f_in, f_out)
        with torch.no_grad():
            self.linear.weight.uniform_(-init_range, init_range)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

        self.film = (
            FiLM(
                f_out,
                modulate_scale=modulate_scale,
                modulate_shift=modulate_shift,
            )
            if (modulate_scale or modulate_shift) and not is_last
            else None
        )
        self.activation = Sine(w0) if apply_activation and not is_last else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)

        if self.is_last:
            # We assume target data lies in [0, 1]; shift by .5 to learn
            # zero-centered features.
            return x + 0.5

        if self.film is not None:
            x = self.film(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class MetaSGDLrs(nn.Module):
    """Module storing learning rates for meta-SGD."""

    def __init__(
        self,
        num_lrs: int,
        lrs_init_range: Tuple[float, float] = (0.005, 0.1),
        lrs_clip_range: Tuple[float, float] = (-5.0, 5.0),
    ):
        """Constructor.

        Args:
            num_lrs: Number of learning rates to learn.
            lrs_init_range: Range from which initial learning rates
                will be uniformly sampled.
            lrs_clip_range: Range at which to clip learning rates.
        """

        super().__init__()
        self.num_lrs = num_lrs
        self.lrs_clip_range = lrs_clip_range

        low, high = lrs_init_range
        self.meta_sgd_lrs = nn.Parameter(torch.empty(num_lrs))
        with torch.no_grad():
            self.meta_sgd_lrs.uniform_(low, high)

    def forward(self) -> Tensor:
        low, high = self.lrs_clip_range
        return torch.clamp(self.meta_sgd_lrs, min=low, max=high)


class ModulatedSiren(nn.Module):
    """SIREN model with FiLM modulations as in pi-GAN."""

    def __init__(
        self,
        width: int = 256,
        depth: int = 5,
        out_channels: int = 3,
        w0: float = 1.0,
        modulate_scale: bool = True,
        modulate_shift: bool = True,
        use_meta_sgd: bool = False,
        meta_sgd_init_range: Tuple[float, float] = (0.005, 0.1),
        meta_sgd_clip_range: Tuple[float, float] = (-5.0, 5.0),
    ):
        """Constructor.

        Args:
            width: Width of each hidden layer in MLP.
            depth: Number of layers in MLP.
            out_channels: Number of output channels.
            w0: Scale factor in sine activation in first layer.
            modulate_scale: If True, modulates scales.
            modulate_shift: If True, modulates shifts.
            use_meta_sgd: Whether to use meta-SGD (only stores LRs here).
            meta_sgd_init_range: Init range for meta-SGD learning rates.
            meta_sgd_clip_range: Clip range for meta-SGD learning rates.
        """

        super().__init__()

        assert depth >= 2, "Depth must be at least 2 (input + output layer)."

        self.width = width
        self.depth = depth
        self.out_channels = out_channels
        self.w0 = w0
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.use_meta_sgd = use_meta_sgd

        # Layers (assume 2D coords by default; can be adapted as needed).
        self.first_layer = ModulatedSirenLayer(
            f_in=2,
            f_out=width,
            w0=w0,
            is_first=True,
            modulate_scale=modulate_scale,
            modulate_shift=modulate_shift,
        )

        hidden_layers = []
        for _ in range(1, depth - 1):
            hidden_layers.append(
                ModulatedSirenLayer(
                    f_in=width,
                    f_out=width,
                    w0=w0,
                    modulate_scale=modulate_scale,
                    modulate_shift=modulate_shift,
                )
            )
        self.hidden_layers = nn.ModuleList(hidden_layers)

        self.final_layer = ModulatedSirenLayer(
            f_in=width,
            f_out=out_channels,
            w0=w0,
            is_last=True,
            modulate_scale=modulate_scale,
            modulate_shift=modulate_shift,
        )

        if use_meta_sgd:
            modulations_per_unit = int(modulate_scale) + int(modulate_shift)
            num_modulations = width * (depth - 1) * modulations_per_unit
            self.meta_sgd_lrs_module = MetaSGDLrs(
                num_modulations,
                lrs_init_range=meta_sgd_init_range,
                lrs_clip_range=meta_sgd_clip_range,
            )
        else:
            self.meta_sgd_lrs_module = None

    def meta_sgd_lrs(self) -> Optional[Tensor]:
        if self.meta_sgd_lrs_module is None:
            return None
        return self.meta_sgd_lrs_module()

    def forward(self, coords: Tensor) -> Tensor:
        """Evaluates model at a batch of coordinates.

        Args:
            coords: [..., D] coordinates; typically D=2 for images.

        Returns:
            Output features with shape [..., out_channels].
        """
        x = coords.view(-1, coords.shape[-1])

        # Initial + hidden + final layers
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        out = self.final_layer(x)

        return out.view(*coords.shape[:-1], self.out_channels)


class LatentVector(nn.Module):
    """Module that holds a latent vector as a learnable parameter.

    NOTE: kept for parity with the original functa codebase, but
    `LatentModulatedSiren` below expects latents to be passed in
    explicitly instead of owning them internally.
    """

    def __init__(self, latent_dim: int, latent_init_scale: float = 0.0):
        """Constructor.

        Args:
            latent_dim: Dimension of latent vector.
            latent_init_scale: Scale at which to randomly initialize latent vector.
        """

        super().__init__()
        self.latent_dim = latent_dim
        low, high = -latent_init_scale, latent_init_scale
        self.latent_vector = nn.Parameter(torch.empty(latent_dim))
        with torch.no_grad():
            self.latent_vector.uniform_(low, high)

    def forward(self) -> Tensor:
        return self.latent_vector


class LatentToModulation(nn.Module):
    """Function mapping latent vector to a set of FiLM modulations."""

    def __init__(
        self,
        latent_dim: int,
        layer_sizes: Tuple[int, ...],
        width: int,
        num_modulation_layers: int,
        modulate_scale: bool = True,
        modulate_shift: bool = True,
        activation: Callable[[Tensor], Tensor] = torch.relu,
    ):
        """Constructor.

        Args:
            latent_dim: Dimension of latent vector (input of this network).
            layer_sizes: Hidden layer sizes of the MLP.
            width: Width (number of units) in each modulated SIREN layer.
            num_modulation_layers: Number of layers in MLP that contain modulations.
            modulate_scale: If True, returns scale modulations.
            modulate_shift: If True, returns shift modulations.
            activation: Activation function to use in MLP.
        """

        super().__init__()
        assert modulate_scale or modulate_shift

        self.latent_dim = latent_dim
        self.layer_sizes = tuple(layer_sizes)
        self.width = width
        self.num_modulation_layers = num_modulation_layers
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.activation = activation

        self.modulations_per_unit = int(modulate_scale) + int(modulate_shift)
        self.modulations_per_layer = width * self.modulations_per_unit
        self.output_size = num_modulation_layers * self.modulations_per_layer

        layers = []
        in_dim = latent_dim
        for h in self.layer_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, self.output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, latent_vector: Tensor) -> Dict[int, Dict[str, Tensor]]:
        # Ensure shape [latent_dim]
        if latent_vector.dim() > 1:
            latent_vector = latent_vector.view(-1)

        modulations = self.mlp(latent_vector)  # [output_size]

        outputs: Dict[int, Dict[str, Tensor]] = {}
        for i in range(self.num_modulation_layers):
            single_layer_modulations: Dict[str, Tensor] = {}

            if self.modulate_scale and self.modulate_shift:
                start = 2 * self.width * i
                single_layer_modulations["scale"] = modulations[start : start + self.width] + 1.0
                single_layer_modulations["shift"] = modulations[
                    start + self.width : start + 2 * self.width
                ]
            elif self.modulate_scale:
                start = self.width * i
                single_layer_modulations["scale"] = modulations[start : start + self.width] + 1.0
            elif self.modulate_shift:
                start = self.width * i
                single_layer_modulations["shift"] = modulations[start : start + self.width]

            outputs[i] = single_layer_modulations

        return outputs


class LatentModulatedSiren(nn.Module):
    """SIREN model with FiLM modulations generated from an external latent.

    The latent code is now **passed into `forward`** instead of being
    owned/initialized inside the module, which makes it easy to optimize
    latents in an outer loop or reuse the same network across latents.
    """

    def __init__(
        self,
        width: int = 256,
        depth: int = 5,
        out_channels: int = 3,
        latent_dim: int = 64,
        layer_sizes: Tuple[int, ...] = (256, 512),
        w0: float = 1.0,
        modulate_scale: bool = True,
        modulate_shift: bool = True,
        use_meta_sgd: bool = False,
        meta_sgd_init_range: Tuple[float, float] = (0.005, 0.1),
        meta_sgd_clip_range: Tuple[float, float] = (-5.0, 5.0),
    ):
        """Constructor.

        Args:
            width: Width of each hidden layer in MLP.
            depth: Number of layers in MLP.
            out_channels: Number of output channels.
            latent_dim: Dimension of latent vector.
            layer_sizes: Hidden layer sizes for latent-to-modulation MLP.
            w0: Scale factor in sine activation in first layer.
            modulate_scale: If True, modulates scales.
            modulate_shift: If True, modulates shifts.
            use_meta_sgd: Whether to use meta-SGD (only stores LRs here).
            meta_sgd_init_range: Init range for meta-SGD LRs.
            meta_sgd_clip_range: Clip range for meta-SGD LRs.
        """

        super().__init__()

        assert depth >= 2

        self.width = width
        self.depth = depth
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.layer_sizes = layer_sizes
        self.w0 = w0
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.use_meta_sgd = use_meta_sgd

        if use_meta_sgd:
            self.meta_sgd_lrs_module = MetaSGDLrs(
                latent_dim,
                lrs_init_range=meta_sgd_init_range,
                lrs_clip_range=meta_sgd_clip_range,
            )
        else:
            self.meta_sgd_lrs_module = None

        # Map from latent vector -> FiLM modulations
        self.latent_to_modulation = LatentToModulation(
            latent_dim=latent_dim,
            layer_sizes=layer_sizes,
            width=width,
            num_modulation_layers=depth - 1,
            modulate_scale=modulate_scale,
            modulate_shift=modulate_shift,
        )

        # Network layers: FiLM is applied externally via latent modulations
        # Assume 2D input coords; adapt f_in if needed for 3D.
        self.input_layer = ModulatedSirenLayer(
            f_in=2,
            f_out=width,
            is_first=True,
            w0=w0,
            modulate_scale=False,
            modulate_shift=False,
            apply_activation=False,
        )

        hidden_layers = []
        for _ in range(1, depth - 1):
            hidden_layers.append(
                ModulatedSirenLayer(
                    f_in=width,
                    f_out=width,
                    w0=w0,
                    modulate_scale=False,
                    modulate_shift=False,
                    apply_activation=False,
                )
            )
        self.hidden_layers = nn.ModuleList(hidden_layers)

        self.output_layer = ModulatedSirenLayer(
            f_in=width,
            f_out=out_channels,
            is_last=True,
            w0=w0,
            modulate_scale=False,
            modulate_shift=False,
        )

        self.sine = Sine(w0)

    def meta_sgd_lrs(self) -> Optional[Tensor]:
        if self.meta_sgd_lrs_module is None:
            return None
        return self.meta_sgd_lrs_module()

    @staticmethod
    def _apply_modulation(x: Tensor, modulations: Dict[str, Tensor]) -> Tensor:
        if "scale" in modulations:
            x = x * modulations["scale"]
        if "shift" in modulations:
            x = x + modulations["shift"]
        return x

    def forward(self, coords: Tensor, latent_vector: Tensor) -> Tensor:
        """Evaluates model at a batch of coordinates for a given latent.

        Args:
            coords: [..., D] coordinates; typically D=2 for images.
            latent_vector: [latent_dim] or [*, latent_dim] latent code.

        Returns:
            Output features with shape [..., out_channels].
        """
        modulations = self.latent_to_modulation(latent_vector)

        x = coords.view(-1, coords.shape[-1])

        # Input layer
        x = self.input_layer(x)
        x = self._apply_modulation(x, modulations[0])
        x = self.sine(x)

        # Hidden layers
        for i, layer in enumerate(self.hidden_layers, start=1):
            x = layer(x)
            x = self._apply_modulation(x, modulations[i])
            x = self.sine(x)

        # Output layer
        out = self.output_layer(x)
        return out.view(*coords.shape[:-1], self.out_channels)


def get_coordinate_grid(res: int, centered: bool = True, device=None) -> Tensor:
    """Returns a normalized coordinate grid for a res by res sized image.

    Args:
        res: Resolution of image.
        centered: If True assumes coordinates lie at pixel centers.

    Returns:
        Tensor of shape (height, width, 2) with values in [0, 1].
    """
    if device is None:
        device = "cpu"

    if centered:
        half_pixel = 1.0 / (2.0 * res)
        coords_one_dim = torch.linspace(half_pixel, 1.0 - half_pixel, res, device=device)
    else:
        coords_one_dim = torch.linspace(0.0, 1.0, res, device=device)

    yy, xx = torch.meshgrid(coords_one_dim, coords_one_dim, indexing="ij")
    return torch.stack((yy, xx), dim=-1)
