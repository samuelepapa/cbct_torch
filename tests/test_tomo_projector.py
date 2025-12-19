import math

import torch

from experiments.instant_ngp.tomo_projector import (
    _get_coordinates_original,
    get_coordinates,
    get_coordinates_fixed_step,
    intersect_aabb_2d,
    parallel_beam_rays_2d,
)


def _cpu_rng(seed: int = 0) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def test_parallel_beam_rays_2d_shapes_and_directions():
    # Setup: two angles, three detector coordinates
    angles = torch.tensor([0.0, math.pi / 2], dtype=torch.float32)
    detector_coords = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)

    origins, directions = parallel_beam_rays_2d(angles, detector_coords)

    # Shape checks
    assert origins.shape == (2, 3, 2)
    assert directions.shape == (2, 3, 2)

    # Functional checks for angle 0 (ray along +x, detector normal along +y)
    dirs_angle0 = directions[0]  # [D, 2]
    assert torch.allclose(dirs_angle0, torch.tensor([[1.0, 0.0]]).expand_as(dirs_angle0))

    # Origins lie on the detector line: x = 0, y = detector_coords
    orig_angle0 = origins[0]  # [D, 2]
    assert torch.allclose(orig_angle0[:, 0], torch.zeros_like(detector_coords))
    assert torch.allclose(orig_angle0[:, 1], detector_coords)


def test_parallel_beam_rays_2d_origin_shift_scalar_and_tensor():
    angles = torch.tensor([0.0, math.pi / 2], dtype=torch.float32)
    detector_coords = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)

    # No shift
    base_origins, base_dirs = parallel_beam_rays_2d(angles, detector_coords)

    # Scalar shift
    shift_scalar = 2.0
    origins_scalar, dirs_scalar = parallel_beam_rays_2d(
        angles, detector_coords, origin_shift=shift_scalar
    )
    assert torch.allclose(dirs_scalar, base_dirs)
    # Shift backward along ray direction
    assert torch.allclose(origins_scalar, base_origins - shift_scalar * base_dirs)

    # Per-angle shift tensor
    shift_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    origins_tensor, dirs_tensor = parallel_beam_rays_2d(
        angles, detector_coords, origin_shift=shift_tensor
    )
    assert torch.allclose(dirs_tensor, base_dirs)
    # Broadcasted shift per angle
    assert torch.allclose(
        origins_tensor,
        base_origins - shift_tensor[:, None, None] * base_dirs,
    )


def test_intersect_aabb_2d_shapes_and_simple_cases():
    # One sample, two rays
    origins = torch.tensor(
        [
            [
                [0.0, 0.0],  # Through the box center
                [0.0, 2.0],  # Above the box (miss)
            ]
        ],
        dtype=torch.float32,
    )  # [1, 2, 2]
    directions = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32
    )  # [1, 2, 2], horizontal rays
    bboxes = torch.tensor(
        [[[-1.0, -1.0], [1.0, 1.0]]], dtype=torch.float32
    )  # [1, 2, 2], centered square

    s_enter, s_exit, valid = intersect_aabb_2d(origins, directions, bboxes)

    # Shape checks
    assert s_enter.shape == (1, 2)
    assert s_exit.shape == (1, 2)
    assert valid.shape == (1, 2)

    # Functional: first ray intersects, second does not
    assert valid[0, 0]
    assert not valid[0, 1]

    # For the intersecting ray: origin (0,0), dir (1,0), box x in [-1,1]
    # Entry at x = -1 -> s = -1, exit at x = 1 -> s = 1
    assert torch.allclose(s_enter[0, 0], torch.tensor(-1.0))
    assert torch.allclose(s_exit[0, 0], torch.tensor(1.0))

    # For the non-intersecting ray, values are zeroed out
    assert s_enter[0, 1] == 0.0
    assert s_exit[0, 1] == 0.0


def test_get_coordinates_shapes_and_non_jitter_functional():
    S, D, num_samples = 1, 1, 4
    origins = torch.zeros(S, D, 2, dtype=torch.float32)
    directions = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    s_enter = torch.zeros(S, D, dtype=torch.float32)
    s_exit = torch.ones(S, D, dtype=torch.float32)
    valid = torch.ones(S, D, dtype=torch.bool)

    # Use original (uncompiled) function for a clearer, deterministic test of logic
    coords = _get_coordinates_original(
        origins=origins,
        directions=directions,
        s_enter=s_enter,
        s_exit=s_exit,
        valid=valid,
        num_samples=num_samples,
        jitter=False,
        rng=None,
    )

    # Shape check
    assert coords.shape == (S, D, num_samples, 2)

    # Functional: with jitter=False and s in [0,1], we expect evenly spaced samples
    expected_z = torch.linspace(0.0, 1.0, num_samples)
    assert torch.allclose(coords[0, 0, :, 0], expected_z)
    assert torch.allclose(coords[0, 0, :, 1], torch.zeros_like(expected_z))


def test_get_coordinates_jitter_determinism_and_bounds():
    S, D, num_samples = 2, 3, 5
    origins = torch.zeros(S, D, 2, dtype=torch.float32)
    directions = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32).expand(S, D, 2)
    s_enter = torch.zeros(S, D, dtype=torch.float32)
    s_exit = torch.ones(S, D, dtype=torch.float32)
    valid = torch.ones(S, D, dtype=torch.bool)

    rng1 = _cpu_rng(42)
    rng2 = _cpu_rng(42)

    coords1 = get_coordinates(
        origins=origins,
        directions=directions,
        s_enter=s_enter,
        s_exit=s_exit,
        valid=valid,
        num_samples=num_samples,
        jitter=True,
        rng=rng1,
        jitter_scale=1.0,
    )
    coords2 = get_coordinates(
        origins=origins,
        directions=directions,
        s_enter=s_enter,
        s_exit=s_exit,
        valid=valid,
        num_samples=num_samples,
        jitter=True,
        rng=rng2,
        jitter_scale=1.0,
    )

    # Deterministic given same seed
    assert torch.allclose(coords1, coords2)

    # All samples must lie between s_enter and s_exit along the ray
    z_vals = coords1[..., 0]  # since direction is along +x and origin is 0
    assert torch.all(z_vals >= 0.0)
    assert torch.all(z_vals <= 1.0)


def test_get_coordinates_fixed_step_shapes_and_monotonicity():
    S, D, num_samples = 1, 2, 6
    origins = torch.zeros(S, D, 2, dtype=torch.float32)
    directions = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
    s_enter = torch.zeros(S, D, dtype=torch.float32)
    s_exit = torch.tensor([[1.0, 2.0]], dtype=torch.float32)  # different lengths per ray
    valid = torch.ones(S, D, dtype=torch.bool)

    step_size = 0.25

    coords = get_coordinates_fixed_step(
        origins=origins,
        directions=directions,
        s_enter=s_enter,
        s_exit=s_exit,
        valid=valid,
        step_size=step_size,
        num_samples=num_samples,
        jitter=False,
        rng=None,
    )

    # Shape check
    assert coords.shape == (S, D, num_samples, 2)

    # Extract z-values along the ray (x-coordinate)
    z_vals = coords[..., 0]

    # Monotonic non-decreasing and within [s_enter, s_exit] per-ray
    assert torch.all(z_vals[:, :, 1:] >= z_vals[:, :, :-1] - 1e-6)
    assert torch.all(z_vals >= s_enter[..., None] - 1e-6)
    assert torch.all(z_vals <= s_exit[..., None] + 1e-6)


def test_get_coordinates_fixed_step_jitter_within_bounds_and_sorted():
    S, D, num_samples = 1, 1, 8
    origins = torch.zeros(S, D, 2, dtype=torch.float32)
    directions = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    s_enter = torch.zeros(S, D, dtype=torch.float32)
    s_exit = torch.ones(S, D, dtype=torch.float32)
    valid = torch.ones(S, D, dtype=torch.bool)
    step_size = 0.2

    rng = _cpu_rng(123)

    coords = get_coordinates_fixed_step(
        origins=origins,
        directions=directions,
        s_enter=s_enter,
        s_exit=s_exit,
        valid=valid,
        step_size=step_size,
        num_samples=num_samples,
        jitter=True,
        rng=rng,
        jitter_scale=0.5,
    )

    z_vals = coords[..., 0]

    # Still sorted after jitter + clamping
    assert torch.all(z_vals[:, :, 1:] >= z_vals[:, :, :-1] - 1e-6)

    # Within [s_enter, s_exit]
    assert torch.all(z_vals >= s_enter[..., None] - 1e-6)
    assert torch.all(z_vals <= s_exit[..., None] + 1e-6)
