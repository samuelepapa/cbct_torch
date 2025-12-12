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


def render_parallel_projection(model, rays_o, rays_d, near, far, num_samples, rand=False):
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

    Returns:
        projections: [B, P] Integrated values.
    """
    device = rays_o.device
    B, P, _ = rays_o.shape

    # Convert scalar near/far to tensor if needed, or ensure shape
    if not isinstance(near, torch.Tensor):
        near = torch.tensor(near, device=device, dtype=torch.float32)
    if not isinstance(far, torch.Tensor):
        far = torch.tensor(far, device=device, dtype=torch.float32)

    # Broadcast to [B, P, 1]
    if near.ndim == 0:
        near = near.expand(B, P, 1)
    elif near.ndim == 2:
        near = near.unsqueeze(-1)  # [B, P] -> [B, P, 1]

    if far.ndim == 0:
        far = far.expand(B, P, 1)
    elif far.ndim == 2:
        far = far.unsqueeze(-1)

    # 1. Generate sample steps along the ray

    # Create base steps [0, 1, ..., N-1]
    steps = torch.linspace(0, 1, num_samples, device=device)  # [S]

    # Reshape to [1, 1, S]
    steps = steps.reshape(1, 1, num_samples)

    # z_vals = near + t * (far - near)
    # [B, P, 1] + [1, 1, S] * [B, P, 1] -> [B, P, S]
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

    # 2. Get sample points
    # rays_o: [B, P, 1, 2]
    # rays_d: [B, P, 1, 2]
    # z_vals: [B, P, S, 1]
    pts = rays_o.unsqueeze(2) + rays_d.unsqueeze(2) * z_vals.unsqueeze(-1)

    # pts shape: [B, P, S, 2]

    # 3. Query Model
    pts_flat = pts.reshape(-1, 2)
    mu_flat = model(pts_flat)

    if mu_flat.dim() > 1:
        mu_flat = mu_flat.squeeze(-1)

    mu = mu_flat.reshape(B, P, num_samples)

    if delta.ndim == 3:  # [B, P, 1]
        delta = delta.expand(B, P, num_samples)

    # 4. Integrate
    # If delta is broadcastable
    projections = torch.sum(mu * delta, dim=-1)

    return projections
