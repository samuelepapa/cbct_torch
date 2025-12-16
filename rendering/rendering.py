import torch


def get_parallel_rays_2d(angles, num_pixels, detector_width, device):
    """
    Generates rays for 2D parallel beam projection.

    The coordinate system assumes the object is at (0,0).
    The detector is a 1D line centered at (0,0) in the rotated frame.

    Args:
        angles: Tensor of projection angles in radians. Shape [B] or scalar.
        num_pixels: Number of pixels on the detector.
        detector_width: Physical width of the detector.
        device: Torch device.

    Returns:
        rays_o: Ray origins. Shape [B, num_pixels, 2].
        rays_d: Ray directions. Shape [B, num_pixels, 2].
    """
    # Ensure angles is 1D tensor
    if not isinstance(angles, torch.Tensor):
        angles = torch.tensor([angles], device=device)
    if angles.dim() == 0:
        angles = angles.unsqueeze(0)

    B = angles.shape[0]

    # Detector sampling locations 's' along the detector line
    # shape: [num_pixels]
    s = torch.linspace(-detector_width / 2, detector_width / 2, num_pixels, device=device)

    # Normal and Direction vectors for each angle
    # theta in [B]

    cos_t = torch.cos(angles)
    sin_t = torch.sin(angles)

    # n: [B, 2] -> (cos, sin)
    n = torch.stack([cos_t, sin_t], dim=1)
    # d: [B, 2] -> (-sin, cos)
    d = torch.stack([-sin_t, cos_t], dim=1)

    # rays_o = s * n_per_angle (+ offset if needed, but we center at 0)
    # s: [P], n: [B, 2] -> [B, P, 2]
    # We need to broadcast s across B, and n across P.
    rays_o = s.unsqueeze(0).unsqueeze(-1) * n.unsqueeze(1)  # [1, P, 1] * [B, 1, 2] -> [B, P, 2]

    # rays_d = d
    # [B, 1, 2] -> expanded to [B, P, 2]
    rays_d = d.unsqueeze(1).expand(B, num_pixels, 2)

    return rays_o, rays_d


def get_ray_aabb_intersection_2d(rays_o, rays_d, aabb_min, aabb_max):
    """
    Computes intersection of rays with a 2D Axis Aligned Bounding Box.
    Args:
        rays_o: [..., 2] Ray origins
        rays_d: [..., 2] Ray directions
        aabb_min: [2] Box min corner
        aabb_max: [2] Box max corner

    Returns:
        t_min: [..., 1] Intersection entrance distance
        t_max: [..., 1] Intersection exit distance
        hits:  [..., 1] Bool, true if ray intersects box
    """
    # Inverse direction (avoid division by zero)
    inv_d = 1.0 / (rays_d + 1e-6)

    # t = (plane - origin) / direction

    t0 = (aabb_min - rays_o) * inv_d
    t1 = (aabb_max - rays_o) * inv_d

    # Swap so t0 is min, t1 is max per axis
    t_small = torch.min(t0, t1)
    t_big = torch.max(t0, t1)

    # Max of mins for t_enter
    t_min = torch.max(t_small[..., 0], t_small[..., 1]).unsqueeze(-1)

    # Min of maxes for t_exit
    t_max = torch.min(t_big[..., 0], t_big[..., 1]).unsqueeze(-1)

    hits = t_max > t_min

    return t_min, t_max, hits


def _normalize_to_aabb(pts, aabb_min, aabb_max):
    """
    Normalize 2D points from AABB coordinates to [-1, 1] range expected by grid_sample.

    Args:
        pts: [..., 2] points in world / AABB coordinates.
        aabb_min: [2] or broadcastable minimum corner.
        aabb_max: [2] or broadcastable maximum corner.
    """
    # Ensure tensors
    if not isinstance(aabb_min, torch.Tensor):
        aabb_min = torch.as_tensor(aabb_min, device=pts.device, dtype=pts.dtype)
    if not isinstance(aabb_max, torch.Tensor):
        aabb_max = torch.as_tensor(aabb_max, device=pts.device, dtype=pts.dtype)

    aabb_min = aabb_min.view(*([1] * (pts.ndim - 1)), 2)
    aabb_max = aabb_max.view(*([1] * (pts.ndim - 1)), 2)

    return (pts - aabb_min) / (aabb_max - aabb_min)


def render_parallel_projection(
    model,
    rays_o,
    rays_d,
    near,
    far,
    num_samples,
    aabb_min,
    aabb_max,
    rand=False,
    hits_mask=None,
):
    """
    Computes the line integral (Radon transform) of the neural field along rays.

    Args:
        model: Callable that takes inputs (..., 2) and returns (..., 1) or (...).
               Represents the attenuation coefficient mu(x, y).
        rays_o: [B, P, 2] Ray origins.
        rays_d: [B, P, 2] Ray directions (normalized).
        near: float or [B, P, 1], start of integration.
        far: float or [B, P, 1], end of integration.
        num_samples: int, number of integration steps.
        rand: bool, whether to use stratified sampling (jitter).
        hits_mask: Optional bool mask [B, P] or [B, P, 1]; when provided,
            only rays with True are sampled/evaluated. Others are skipped and
            returned as zero.

    Returns:
        projections: [B, P] Integrated values.
    """
    device = rays_o.device
    B, P, _ = rays_o.shape

    def _ensure_near_far(t, name):
        """Ensure near/far have shape [B, P, 1] and live on device."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand(B, P, 1)
        elif t.ndim == 2:
            t = t.unsqueeze(-1)
        if t.shape[:2] != (B, P):
            raise ValueError(f"{name} must broadcast to [B, P, 1]; got {t.shape}")
        return t

    near = _ensure_near_far(near, "near")
    far = _ensure_near_far(far, "far")

    # If a hits mask is provided, render only valid rays to avoid sampling
    # invalid regions and skip them in training.
    hits_mask_full = None
    if hits_mask is not None:
        hits_mask_full = hits_mask.bool()
        if hits_mask_full.ndim == 3:
            hits_mask_full = hits_mask_full.squeeze(-1)
        if hits_mask_full.shape != (B, P):
            raise ValueError(f"hits_mask must be [B, P] or [B, P, 1]; got {hits_mask.shape}")

        # Early exit: no valid rays
        if not hits_mask_full.any():
            return torch.zeros((B, P), device=device, dtype=torch.float32)

        # Select only valid rays
        rays_o = rays_o[hits_mask_full].unsqueeze(1)  # [N, 1, 2]
        rays_d = rays_d[hits_mask_full].unsqueeze(1)  # [N, 1, 2]
        near = near[hits_mask_full].unsqueeze(1)  # [N, 1, 1]
        far = far[hits_mask_full].unsqueeze(1)  # [N, 1, 1]
        B, P, _ = rays_o.shape  # Now B = N_valid, P = 1

    # 1. Generate sample steps along the ray
    steps = torch.linspace(0, 1, num_samples, device=device).reshape(1, 1, num_samples)

    z_vals = near + steps * (far - near)

    # Interval width (average)
    # [B, P, 1]
    if num_samples > 1:
        # Standard step size in t-space is (far-near)/(N-1) for linspace
        # But for integration usually we want volume elements.
        # Let's map steps [0, ..., N] exactly.
        # delta = (far - near) / (N-1) ? (Trapezoidal)
        # or delta = (far - near) / N (Midpoint / Riemann sum)

        # If we assume z_vals are sample locations.
        # If rand=True (stratified), we want bins of size (far-near)/N.

        bin_size = (far - near) / num_samples

        if rand:
            # Stratified sampling
            # Replace z_vals with stratified versions
            # z_vals currently are linspace(near, far, N). THIS IS ENDPOINTS logic (N items).
            # Stratified logic usually wants N bins.
            # Let's redefine steps for stratified:
            # steps = [0, 1, ... N-1] / N
            # z = near + (steps + rand_offset) * (far - near)

            # Recompute steps for bins
            steps = (
                torch.arange(num_samples, device=device, dtype=torch.float32).reshape(
                    1, 1, num_samples
                )
                / num_samples
            )

            # Random jitter within [0, 1/N]
            jitter = torch.rand(B, P, num_samples, device=device) / num_samples

            t_rand = steps + jitter  # values in [0, 1]
            z_vals = near + t_rand * (far - near)

            delta = bin_size  # Each sample represents one bin width

        else:
            # Deterministic
            # Using simple Riemann sum with linspace?
            # z_vals is already computed as linspace(near, far, N)

            # Simple approximation: delta = (far - near) / N?
            # Or (far - near) / (N-1) if N is small?
            # Let's use (far - near) / (num_samples) and assume midpoints roughly.
            # Or recompute z_vals to be midpoints of bins?
            # Standard NeRF without rand usually just does linspace.
            # Let's stick to simple delta.
            delta = (far - near) / num_samples

            # If using linspace endpoints, we span full range.
            # but delta should match coverage.
            # Keep z_vals, use average delta.
    else:
        delta = far - near

    pts = rays_o.unsqueeze(2) + rays_d.unsqueeze(2) * z_vals.unsqueeze(-1)

    # pts shape: [B, P, S, 2]

    # Optionally normalize coordinates to [-1, 1] using AABB
    pts = _normalize_to_aabb(pts, aabb_min, aabb_max)

    pts_flat = pts.reshape(-1, 2)
    mu_flat = model(pts_flat)

    if mu_flat.dim() > 1:
        mu_flat = mu_flat.squeeze(-1)

    mu = mu_flat.reshape(B, P, num_samples)

    if delta.ndim == 3:  # [B, P, 1]
        delta = delta.expand(B, P, num_samples)

    projections = torch.sum(mu * delta, dim=-1)

    if hits_mask_full is None:
        return projections

    # Scatter back into full canvas, zero where hits_mask was False
    full_proj = torch.zeros(
        (hits_mask.shape[0], hits_mask.shape[1]), device=device, dtype=projections.dtype
    )
    full_proj[hits_mask_full] = projections.squeeze(-1)
    return full_proj


# Try to create a compiled version of the integrator for speed (PyTorch 2.x).
# Falls back gracefully if torch.compile is unavailable (older PyTorch).
# try:
#     render_parallel_projection = torch.compile(render_parallel_projection)  # type: ignore[attr-defined]
# except (AttributeError, RuntimeError):
#     # AttributeError: torch.compile not present (older PyTorch).
#     # RuntimeError: backend not available / unsupported environment.
#     pass
