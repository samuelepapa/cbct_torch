from __future__ import annotations

import torch
from jaxtyping import Bool, Float
from torch import Tensor


def parallel_beam_rays_2d(
    angles: Float[Tensor, "B"],
    detector_coords: Float[Tensor, "D"],
    origin_shift: Float[Tensor, "B"] | float | None = None,
) -> tuple[Float[Tensor, "B D 2"], Float[Tensor, "B D 2"]]:
    """
    Generate parallel beam rays in 2D.

    Args:
        angles: [B] tensor of projection angles
        detector_coords: [D] tensor of detector pixel coordinates
        origin_shift: Optional shift along ray direction. If provided, origins are shifted
                     backward by this amount along the ray direction. Can be:
                     - A scalar (same shift for all angles)
                     - A tensor [B] (different shift per angle)
                     - None (no shift, default)

    Returns:
        origins:    [B, D, 2]
        directions: [B, D, 2] (unit)
    """

    t = detector_coords

    theta = angles[:, None]  # [B, 1]

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Direction
    d = torch.stack([cos_theta, sin_theta], dim=-1)  # [B, 1, 2]
    d = d.expand(-1, t.shape[0], -1)  # [B, D, 2]

    # Detector normal
    n = torch.stack([-sin_theta, cos_theta], dim=-1)  # [B, 1, 2]

    # Origins
    o = t[None, :, None] * n  # [B, D, 2]

    # Apply origin shift if provided (shift backward along ray direction)
    if origin_shift is not None:
        if isinstance(origin_shift, (int, float)):
            # Scalar: same shift for all angles
            shift = torch.tensor(origin_shift, device=o.device, dtype=o.dtype)
            o = o - shift * d  # Shift backward along ray direction
        else:
            # Tensor [B]: different shift per angle
            shift = origin_shift[:, None, None]  # [B, 1, 1]
            o = o - shift * d  # Shift backward along ray direction

    return o, d


def intersect_aabb_2d(
    origins: Float[Tensor, "S D 2"],
    directions: Float[Tensor, "S D 2"],
    bboxes: Float[Tensor, "S 2 2"],
    eps=1e-8,
) -> tuple[Float[Tensor, "S D"], Float[Tensor, "S D"], Bool[Tensor, "S D"]]:
    ox, oy = origins[:, :, 0], origins[:, :, 1]
    dx, dy = directions[:, :, 0], directions[:, :, 1]

    dx = torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
    dy = torch.where(dy.abs() < eps, torch.full_like(dy, eps), dy)

    # BBoxes -> [S, 1, 1]
    xmin = bboxes[:, 0, 0][:, None]
    ymin = bboxes[:, 0, 1][:, None]
    xmax = bboxes[:, 1, 0][:, None]
    ymax = bboxes[:, 1, 1][:, None]

    sx1 = (xmin - ox) / dx
    sx2 = (xmax - ox) / dx
    sy1 = (ymin - oy) / dy
    sy2 = (ymax - oy) / dy

    sx_min = torch.minimum(sx1, sx2)
    sx_max = torch.maximum(sx1, sx2)
    sy_min = torch.minimum(sy1, sy2)
    sy_max = torch.maximum(sy1, sy2)

    s_enter = torch.maximum(sx_min, sy_min)
    s_exit = torch.minimum(sx_max, sy_max)

    valid = s_exit > s_enter

    # set all invalid entries to s_enter = s_exit = 0
    s_enter = torch.where(valid, s_enter, torch.zeros_like(s_enter))
    s_exit = torch.where(valid, s_exit, torch.zeros_like(s_exit))

    return s_enter, s_exit, valid


def get_coordinates(
    origins: Float[Tensor, "samples rays 2"],
    directions: Float[Tensor, "samples rays 2"],
    s_enter: Float[Tensor, "samples rays"],
    s_exit: Float[Tensor, "samples rays"],
    valid: Bool[Tensor, "samples rays"],
    num_samples: int,
    jitter: bool,
    rng: torch.Generator | None = None,
    jitter_scale: float = 1.0,
    eps: float = 1e-6,
    dtype: torch.dtype = torch.float32,
) -> Float[Tensor, "samples rays num_samples 2"]:
    """
    Returns sample coordinates along rays from s_enter to s_exit.

    Samples num_samples points along each ray between the entry and exit points.
    Uses stratified sampling with jitter when jitter=True, or deterministic
    uniform sampling when jitter=False.

    Args:
        origins: Ray origins. Assumed to be on the correct device.
        directions: Ray directions (unit vectors). Assumed to be on the correct device.
        s_enter: Distance along ray where it enters the bounding box.
                 Assumed to be on the correct device.
        s_exit: Distance along ray where it exits the bounding box.
                Assumed to be on the correct device.
        valid: Boolean mask indicating which rays are valid.
               Currently unused but kept for API compatibility.
        num_samples: Number of samples along each ray.
        jitter: Whether to use stratified sampling (random jitter within bins).
                If True, samples are randomly distributed within each bin.
                If False, samples are uniformly spaced using linspace.
        rng: Random number generator used for jitter when jitter=True.
        jitter_scale: Scale factor for jitter extent. When jitter=True, controls
                      the range of random jitter within each bin. A value of 1.0
                      means full jitter (samples can be anywhere within the bin),
                      while smaller values reduce the jitter extent. Default is 1.0.
        eps: Small epsilon value (currently unused, kept for API compatibility).
        dtype: Dtype used for creating intermediate tensors.

    Returns:
        Sample coordinates along rays. Each coordinate is computed as
        origins + z_vals * directions, where z_vals are the sampled
        distances along the ray.
    """
    device = origins.device
    S, D, _ = origins.shape

    # Expand s_enter and s_exit to [S, D, 1] for broadcasting
    s_enter_expanded = s_enter[..., None]  # [S, D, 1]
    s_exit_expanded = s_exit[..., None]  # [S, D, 1]

    if num_samples > 1:
        if jitter:
            if rng is None:
                raise ValueError("rng is required in get_coordinates when jitter is True")
            # Stratified sampling: divide [s_enter, s_exit] into num_samples bins
            # and sample randomly within each bin
            steps = (
                torch.arange(num_samples, device=device, dtype=dtype).reshape(1, 1, num_samples)
                / num_samples
            )  # [1, 1, num_samples], values in [0, 1)

            # Random jitter within [0, jitter_scale/num_samples]
            bin_size = 1.0 / num_samples
            jitter_values = (
                torch.rand(S, D, num_samples, device=device, dtype=dtype, generator=rng)
                * bin_size
                * jitter_scale
            )

            t_rand = steps + jitter_values  # [S, D, num_samples], values in [0, 1]
            z_vals = s_enter_expanded + t_rand * (s_exit_expanded - s_enter_expanded)
        else:
            # Deterministic: linspace from s_enter to s_exit
            steps = torch.linspace(0, 1, num_samples, device=device, dtype=dtype).reshape(
                1, 1, num_samples
            )
            z_vals = s_enter_expanded + steps * (s_exit_expanded - s_enter_expanded)
    else:
        # Single sample at midpoint
        z_vals = (s_enter_expanded + s_exit_expanded) / 2.0

    # Compute coordinates: origins + z_vals * directions
    # origins: [S, D, 2]
    # directions: [S, D, 2]
    # z_vals: [S, D, num_samples]
    # We need to expand directions to [S, D, num_samples, 2]
    z_vals_expanded = z_vals.unsqueeze(-1)  # [S, D, num_samples, 1]

    coordinates = origins.unsqueeze(2) + directions.unsqueeze(2) * z_vals_expanded

    # set invalid coordinates to zero
    coordinates = torch.where(
        valid.unsqueeze(-1).unsqueeze(-1), coordinates, torch.zeros_like(coordinates)
    )
    # coordinates: [S, D, num_samples, 2]

    # Set coordinates for invalid rays to zero (or keep them as computed)
    # Optionally mask invalid rays - for now we'll keep them as computed
    # since the caller can filter using the valid mask if needed

    return coordinates


# Save original function and create compiled version
_get_coordinates_original = get_coordinates
_compiled_get_coordinates = torch.compile(_get_coordinates_original, mode="reduce-overhead")
# Replace with compiled version for better performance
get_coordinates = _compiled_get_coordinates


def get_coordinates_fixed_step(
    origins: Float[Tensor, "samples rays 2"],
    directions: Float[Tensor, "samples rays 2"],
    s_enter: Float[Tensor, "samples rays"],
    s_exit: Float[Tensor, "samples rays"],
    valid: Bool[Tensor, "samples rays"],
    step_size: float,
    num_samples: int,
    jitter: bool = False,
    rng: torch.Generator | None = None,
    jitter_scale: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> Float[Tensor, "samples rays num_samples 2"]:
    """
    Returns sample coordinates along rays using equally-spaced intervals with a fixed step size.

    First over-samples with a fixed step_size, then sub-samples to get exactly num_samples
    points per ray. The step_size is constant across all rays.

    Args:
        origins: Ray origins. Assumed to be on the correct device.
        directions: Ray directions (unit vectors). Assumed to be on the correct device.
        s_enter: Distance along ray where it enters the bounding box.
                 Assumed to be on the correct device.
        s_exit: Distance along ray where it exits the bounding box.
                Assumed to be on the correct device.
        valid: Boolean mask indicating which rays are valid.
               Currently unused but kept for API compatibility.
        step_size: Fixed distance between samples along each ray.
        num_samples: Target number of samples along each ray (after sub-sampling).
        jitter: Whether to add random jitter to the sample coordinates along the ray.
                If True, jitter is applied to the z_vals (positions along the ray)
                during over-sampling to add small perturbations to exact coordinates.
                If False, samples are at exact step_size intervals.
        rng: Random number generator used for jitter when jitter=True.
        jitter_scale: Scale factor for jitter applied to z_vals.
                      Jitter range: [-jitter_scale * step_size, jitter_scale * step_size]
        dtype: Dtype used for creating intermediate tensors.

    Returns:
        Sample coordinates along rays. Each coordinate is computed as
        origins + z_vals * directions, where z_vals are the sampled
        distances along the ray.
    """
    device = origins.device
    S, D, _ = origins.shape

    # Expand s_enter and s_exit to [S, D, 1] for broadcasting
    s_enter_expanded = s_enter[..., None]  # [S, D, 1]
    s_exit_expanded = s_exit[..., None]  # [S, D, 1]

    # Compute ray lengths
    ray_lengths = s_exit_expanded - s_enter_expanded  # [S, D, 1]

    # Compute number of over-samples we can fit with the fixed step_size
    # We use ceil to ensure we cover the entire ray length, then add 1 for the endpoint
    num_over_samples = torch.ceil(ray_lengths / step_size).long() + 1  # [S, D, 1]

    # Use a fixed reasonable maximum size to avoid .item() graph break
    # This upper bound should be larger than any realistic max_over_samples
    # The excess samples will be clamped/masked out later
    MAX_OVER_SAMPLES_BOUND = num_samples
    indices = torch.arange(MAX_OVER_SAMPLES_BOUND, device=device, dtype=dtype).reshape(
        1, 1, -1
    )  # [1, 1, MAX_OVER_SAMPLES_BOUND]

    # Compute z_vals for all over-samples: s_enter + i * step_size
    z_vals_over = s_enter_expanded + indices * step_size  # [S, D, MAX_OVER_SAMPLES_BOUND]

    # Mask out samples beyond the actual max_over_samples for each ray
    # Create a mask: [S, D, MAX_OVER_SAMPLES_BOUND] where True means valid sample
    sample_idx = torch.arange(MAX_OVER_SAMPLES_BOUND, device=device, dtype=torch.long).reshape(
        1, 1, -1
    )  # [1, 1, MAX_OVER_SAMPLES_BOUND]
    valid_mask = sample_idx < num_over_samples  # [S, D, MAX_OVER_SAMPLES_BOUND]

    # Set invalid samples to a large value so they get clamped out
    z_vals_over = torch.where(
        valid_mask, z_vals_over, torch.tensor(float("inf"), device=device, dtype=dtype)
    )

    # Clamp to s_exit to avoid going beyond the ray
    # This ensures that for shorter rays, samples beyond s_exit are clamped to s_exit
    z_vals_over = torch.clamp(z_vals_over, max=s_exit_expanded)

    # Now sub-sample to get exactly num_samples samples
    # We need to work with the actual max_over_samples per ray, not a global maximum
    # Use the num_over_samples tensor to determine valid range for each ray

    # Compute valid sample indices: evenly spaced from [0, num_over_samples-1] for each ray
    # We'll use linspace scaled by num_over_samples for each ray
    # Create normalized positions [0, 1] for num_samples points
    normalized_positions = torch.linspace(0, 1, num_samples, device=device, dtype=dtype).reshape(
        1, 1, -1
    )  # [1, 1, num_samples]

    # Scale by actual num_over_samples for each ray: [S, D, num_samples]
    sample_indices_float = normalized_positions * (
        num_over_samples.float() - 1
    )  # [S, D, num_samples]
    sample_indices = sample_indices_float.long()  # [S, D, num_samples]

    # Clamp indices to valid range [0, MAX_OVER_SAMPLES_BOUND-1]
    sample_indices = torch.clamp(sample_indices, 0, MAX_OVER_SAMPLES_BOUND - 1)

    # Gather the selected samples using torch.gather
    # z_vals_over: [S, D, MAX_OVER_SAMPLES_BOUND]
    # sample_indices: [S, D, num_samples]
    # We want to gather along the last dimension (dim=2)
    z_vals = torch.gather(z_vals_over, dim=2, index=sample_indices)  # [S, D, num_samples]

    # Filter out invalid samples (those beyond s_exit) by clamping
    # s_exit is [S, D], z_vals is [S, D, num_samples], so we need to expand s_exit
    z_vals = torch.clamp(z_vals, max=s_exit[..., None])

    # Apply jitter to the selected z_vals if requested
    if jitter:
        if rng is None:
            raise ValueError("rng is required in get_coordinates_fixed_step when jitter is True")
        # Add jitter to the selected z_vals
        # Jitter range: [-jitter_scale * step_size, jitter_scale * step_size]
        jitter_values = (
            (torch.rand(S, D, num_samples, device=device, dtype=dtype, generator=rng) * 2.0 - 1.0)
            * jitter_scale
            * step_size
        )  # [S, D, num_samples]
        z_vals = z_vals + jitter_values  # [S, D, num_samples]
        # Clamp to ensure we stay within [s_enter, s_exit] bounds
        # s_enter and s_exit are [S, D], which will broadcast correctly with z_vals [S, D, num_samples]
        z_vals = torch.clamp(
            z_vals, min=s_enter[..., None], max=s_exit[..., None]
        )  # [S, D, num_samples]

    # Sort z_vals along the ray dimension to ensure they're in order
    z_vals, _ = torch.sort(z_vals, dim=-1)  # [S, D, num_samples]

    # Compute coordinates: origins + z_vals * directions
    z_vals_expanded = z_vals.unsqueeze(-1)  # [S, D, num_samples, 1]
    coordinates = origins.unsqueeze(2) + directions.unsqueeze(2) * z_vals_expanded
    # coordinates: [S, D, num_samples, 2]

    return coordinates


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT = True

    num_projections = 10
    detector_dim = 16
    detector_spacing = 0.03

    angles = torch.linspace(0, np.pi, num_projections, device="cuda", dtype=torch.float32)

    detector_bbox = (-0.5 * detector_spacing * detector_dim, 0.5 * detector_spacing * detector_dim)
    first_pixel_coord = detector_bbox[0] + detector_spacing / 2
    last_pixel_coord = detector_bbox[1] - detector_spacing / 2

    detector_coords = torch.linspace(
        first_pixel_coord, last_pixel_coord, detector_dim, device="cuda", dtype=torch.float32
    )
    # plot the detector_coords
    if PLOT:
        plt.scatter(
            torch.zeros_like(detector_coords).cpu().numpy(),
            detector_coords.cpu().numpy(),
            color="r",
        )
        plt.show()

    bboxes = torch.tensor(
        [
            [[-0.3, -0.2], [0.3, 0.5]],
        ]
        * num_projections,
        device="cuda",
        dtype=torch.float32,
    )

    origin_shift = torch.maximum(
        bboxes[:, 0, 1] - bboxes[:, 0, 0], bboxes[:, 1, 1] - bboxes[:, 1, 0]
    )

    origins, directions = parallel_beam_rays_2d(angles, detector_coords, origin_shift=2.0)
    s_enter, s_exit, valid = intersect_aabb_2d(origins, directions, bboxes)

    enter_coords = origins + s_enter[..., None] * directions
    exit_coords = origins + s_exit[..., None] * directions
    if PLOT:
        plt.scatter(
            enter_coords[0, :, 0].cpu().numpy(), enter_coords[0, :, 1].cpu().numpy(), color="r"
        )
        plt.scatter(
            exit_coords[0, :, 0].cpu().numpy(), exit_coords[0, :, 1].cpu().numpy(), color="b"
        )
        # plt.xlim(2*detector_bbox[0], 2*detector_bbox[1])
        # plt.ylim(2*detector_bbox[0], 2*detector_bbox[1])
        plt.show()

    valid_enter_coords = enter_coords[0, valid[0]]
    valid_exit_coords = exit_coords[0, valid[0]]
    if PLOT:
        plt.scatter(
            valid_enter_coords[:, 0].cpu().numpy(),
            valid_enter_coords[:, 1].cpu().numpy(),
            color="r",
        )
        plt.scatter(
            valid_exit_coords[:, 0].cpu().numpy(), valid_exit_coords[:, 1].cpu().numpy(), color="b"
        )
        plt.xlim(2 * detector_bbox[0], 2 * detector_bbox[1])
        plt.ylim(2 * detector_bbox[0], 2 * detector_bbox[1])
        plt.show()

    def coordinate_function_test(
        coordinate_function_name: str, jitter: bool, num_samples: int, step_size: float = 0.1
    ):
        if coordinate_function_name == "get_coordinates":
            return get_coordinates(
                origins=origins,
                directions=directions,
                s_enter=s_enter,
                s_exit=s_exit,
                valid=valid,
                num_samples=num_samples,
                jitter=jitter,
                jitter_scale=1.0,
                rng=torch.Generator(device="cuda").manual_seed(42),
            )
        elif coordinate_function_name == "get_coordinates_fixed_step":
            return get_coordinates_fixed_step(
                origins=origins,
                directions=directions,
                s_enter=s_enter,
                s_exit=s_exit,
                valid=valid,
                step_size=step_size,
                num_samples=num_samples,
                jitter=jitter,
                jitter_scale=1.0,
                rng=torch.Generator(device="cuda").manual_seed(42),
            )
        else:
            raise ValueError(f"Invalid coordinate function name: {coordinate_function_name}")

    def plot_coordinates(coordinates: Float[Tensor, "samples rays num_samples 2"]):
        N_angles = coordinates.shape[0]
        num_samples = coordinates.shape[2]
        color_palette = plt.cm.get_cmap("viridis", N_angles)  # or 'tab20' for more colors
        plt.figure()
        for i in range(N_angles):
            coords_i = coordinates[i]  # [n_detectors, num_samples, 2]
            x = coords_i[:, :, 0].cpu().numpy().flatten()
            y = coords_i[:, :, 1].cpu().numpy().flatten()
            plt.scatter(x, y, color=color_palette(i), s=3, alpha=0.1)
        plt.title("Sample points for different angles")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    if PLOT:
        coordinates = coordinate_function_test("get_coordinates", False, 10)
        plot_coordinates(coordinates)

        coordinates = coordinate_function_test("get_coordinates_fixed_step", False, 10)
        plot_coordinates(coordinates)

        coordinates = coordinate_function_test("get_coordinates", True, 10)
        plot_coordinates(coordinates)

        coordinates = coordinate_function_test(
            "get_coordinates_fixed_step", True, 10, step_size=0.1
        )
        plot_coordinates(coordinates)

    import statistics
    import timeit

    def profile_get_coordinates(jitter: bool, num_samples: int):
        def coordinate_function_test_func():
            return get_coordinates(
                origins=origins,
                directions=directions,
                s_enter=s_enter,
                s_exit=s_exit,
                valid=valid,
                num_samples=num_samples,
                jitter=jitter,
                jitter_scale=1.0,
                rng=torch.Generator(device="cuda").manual_seed(42),
            )

        num_iterations = 20000
        total_time = timeit.timeit(lambda: coordinate_function_test_func(), number=num_iterations)
        per_iteration = total_time / num_iterations
        # Also get statistics from multiple runs
        times = timeit.repeat(
            lambda: coordinate_function_test_func(), number=num_iterations, repeat=5
        )
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        return {
            "total_time": total_time,
            "num_iterations": num_iterations,
            "per_iteration": per_iteration,
            "avg_total": avg_time,
            "std_total": std_time,
            "avg_per_iteration": avg_time / num_iterations,
        }

    def profile_get_coordinates_fixed_step(jitter: bool, num_samples: int, step_size: float = 0.1):
        def coordinate_function_test_func():
            return get_coordinates_fixed_step(
                origins=origins,
                directions=directions,
                s_enter=s_enter,
                s_exit=s_exit,
                valid=valid,
                step_size=step_size,
                num_samples=num_samples,
                jitter=jitter,
                jitter_scale=1.0,
                rng=torch.Generator(device="cuda").manual_seed(42),
            )

        num_iterations = 20000
        total_time = timeit.timeit(lambda: coordinate_function_test_func(), number=num_iterations)
        per_iteration = total_time / num_iterations
        # Also get statistics from multiple runs
        times = timeit.repeat(
            lambda: coordinate_function_test_func(), number=num_iterations, repeat=5
        )
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        return {
            "total_time": total_time,
            "num_iterations": num_iterations,
            "per_iteration": per_iteration,
            "avg_total": avg_time,
            "std_total": std_time,
            "avg_per_iteration": avg_time / num_iterations,
        }

    def print_timing_stats(name: str, stats: dict):
        print(f"\n{name}:")
        print(f"  Total time ({stats['num_iterations']} iterations): {stats['total_time']:.6f}s")
        print(f"  Per iteration: {stats['per_iteration']*1000:.6f}ms")
        print(f"  Average total (5 runs): {stats['avg_total']:.6f}s ± {stats['std_total']:.6f}s")
        print(f"  Average per iteration: {stats['avg_per_iteration']*1000:.6f}ms")

    stats = profile_get_coordinates(False, 200)
    print_timing_stats("get_coordinates (no jitter)", stats)

    stats = profile_get_coordinates(True, 200)
    print_timing_stats("get_coordinates (jitter)", stats)

    # Use original function for compilation test (since get_coordinates is already compiled)
    compiled_get_coordinates = torch.compile(_get_coordinates_original)
    compiled_get_coordinates_fixed_step = torch.compile(get_coordinates_fixed_step)

    def profile_compiled_get_coordinates(jitter: bool, num_samples: int):
        def compiled_get_coordinates_func():
            return compiled_get_coordinates(
                origins=origins,
                directions=directions,
                s_enter=s_enter,
                s_exit=s_exit,
                valid=valid,
                num_samples=num_samples,
                jitter=jitter,
                jitter_scale=1.0,
                rng=torch.Generator(device="cuda").manual_seed(42),
            )

        num_iterations = 20000
        total_time = timeit.timeit(lambda: compiled_get_coordinates_func(), number=num_iterations)
        per_iteration = total_time / num_iterations
        # Also get statistics from multiple runs
        times = timeit.repeat(
            lambda: compiled_get_coordinates_func(), number=num_iterations, repeat=5
        )
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        return {
            "total_time": total_time,
            "num_iterations": num_iterations,
            "per_iteration": per_iteration,
            "avg_total": avg_time,
            "std_total": std_time,
            "avg_per_iteration": avg_time / num_iterations,
        }

    def profile_compiled_get_coordinates_fixed_step(
        jitter: bool, num_samples: int, step_size: float = 0.1
    ):
        def compiled_get_coordinates_fixed_step_func():
            return compiled_get_coordinates_fixed_step(
                origins=origins,
                directions=directions,
                s_enter=s_enter,
                s_exit=s_exit,
                valid=valid,
                step_size=step_size,
                num_samples=num_samples,
                jitter=jitter,
                jitter_scale=1.0,
                rng=torch.Generator(device="cuda").manual_seed(42),
            )

        num_iterations = 20000
        total_time = timeit.timeit(
            lambda: compiled_get_coordinates_fixed_step_func(), number=num_iterations
        )
        per_iteration = total_time / num_iterations
        # Also get statistics from multiple runs
        times = timeit.repeat(
            lambda: compiled_get_coordinates_fixed_step_func(), number=num_iterations, repeat=5
        )
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        return {
            "total_time": total_time,
            "num_iterations": num_iterations,
            "per_iteration": per_iteration,
            "avg_total": avg_time,
            "std_total": std_time,
            "avg_per_iteration": avg_time / num_iterations,
        }

    stats = profile_compiled_get_coordinates(False, 200)
    print_timing_stats("compiled_get_coordinates (no jitter)", stats)

    stats = profile_compiled_get_coordinates(True, 200)
    print_timing_stats("compiled_get_coordinates (jitter)", stats)
