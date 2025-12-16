"""
Simple debug script for `get_ray_aabb_intersection_2d` in rendering/rendering.py.

It runs a few synthetic cases, prints t_min, t_max, hits, and also saves plots
showing the AABB, rays, and intersections for quick visual inspection.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Ensure project root is on PYTHONPATH so "rendering" resolves
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rendering.rendering import (
    get_parallel_rays_2d,
    get_ray_aabb_intersection_2d,
)


def plot_case(name, rays_o, rays_d, aabb_min, aabb_max, t_min, t_max, hits, out_dir):
    """Plot AABB, rays, and intersection points derived from t_min/t_max."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    # AABB rectangle
    rect = plt.Rectangle(
        (aabb_min[0].item(), aabb_min[1].item()),
        (aabb_max - aabb_min)[0].item(),
        (aabb_max - aabb_min)[1].item(),
        fill=False,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(rect)

    B, P, _ = rays_o.shape
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    for b in range(B):
        for p in range(P):
            o = rays_o[b, p]
            d = rays_d[b, p]
            c = colors[p % len(colors)]
            # Draw the ray line segment for visibility
            t_draw = torch.linspace(-2.0, 4.0, 100, device=rays_o.device)
            pts = o.unsqueeze(0) + t_draw.unsqueeze(1) * d.unsqueeze(0)
            ax.plot(
                pts[:, 0].cpu(),
                pts[:, 1].cpu(),
                color=c,
                alpha=0.5,
                label=f"ray {p}" if b == 0 else None,
            )

            if hits[b, p]:
                t0 = t_min[b, p, 0]
                t1 = t_max[b, p, 0]
                p0 = o + t0 * d
                p1 = o + t1 * d
                ax.scatter([p0[0].cpu()], [p0[1].cpu()], color=c, marker="o", label=None)
                ax.scatter([p1[0].cpu()], [p1[1].cpu()], color=c, marker="x", label=None)
                ax.plot(
                    [p0[0].cpu(), p1[0].cpu()],
                    [p0[1].cpu(), p1[1].cpu()],
                    color=c,
                    linewidth=2,
                    alpha=0.8,
                    label=None,
                )

    ax.set_title(name)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    out_path = Path(out_dir) / f"{name.replace(' ', '_')}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def run_case(name, rays_o, rays_d, aabb_min, aabb_max, out_dir):
    t_min, t_max, hits = get_ray_aabb_intersection_2d(rays_o, rays_d, aabb_min, aabb_max)
    print(f"\n=== {name} ===")
    print(f"rays_o:\n{rays_o}")
    print(f"rays_d:\n{rays_d}")
    print(f"aabb_min: {aabb_min}, aabb_max: {aabb_max}")
    print(f"t_min:\n{t_min}")
    print(f"t_max:\n{t_max}")
    print(f"hits:\n{hits}")
    plot_case(name, rays_o, rays_d, aabb_min, aabb_max, t_min, t_max, hits, out_dir)


def main():
    device = "cpu"
    out_dir = "debug_ray_aabb_plots"
    aabb_min = torch.tensor([-0.4, -0.4], device=device)
    aabb_max = torch.tensor([0.4, 0.4], device=device)

    # Use the production helpers to generate rays
    # Case 1: single angle, modest detector
    angles = torch.tensor([0.0], device=device)  # 0 rad, parallel to x
    rays_o, rays_d = get_parallel_rays_2d(angles, num_pixels=5, detector_width=1.0, device=device)
    run_case("helpers single angle", rays_o, rays_d, aabb_min, aabb_max, out_dir)

    # Case 2: multiple angles
    angles = torch.tensor([0.0, torch.pi / 4, torch.pi / 2], device=device)
    rays_o, rays_d = get_parallel_rays_2d(angles, num_pixels=9, detector_width=2.0, device=device)
    run_case("helpers multi angle", rays_o, rays_d, aabb_min, aabb_max, out_dir)

    # Case 3: shifted AABB to confirm normalization
    shifted_min = torch.tensor([0.2, -0.3], device=device)
    shifted_max = torch.tensor([1.0, 0.5], device=device)
    angles = torch.tensor([torch.pi / 6], device=device)
    rays_o, rays_d = get_parallel_rays_2d(angles, num_pixels=7, detector_width=1.5, device=device)
    run_case("helpers shifted aabb", rays_o, rays_d, shifted_min, shifted_max, out_dir)


if __name__ == "__main__":
    main()
