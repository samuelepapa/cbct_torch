import os
import sys

import numpy as np
import torch

# Add relevant paths to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))

from projectors import CBBackprojector, CBProjector
from tomo_projector_utils.scanner import ConebeamGeometry


def verify_dynamic_geometry():
    device = torch.device("cuda")

    # 1. Initialize Geometry with DEFAULT 256 size (simulating the start of make_data.py)
    # This represents the state before the loop
    initial_shape = (256, 256, 256)
    angles = np.linspace(0, 205 / 180 * np.pi, 20)
    geom = ConebeamGeometry(
        source_to_center_dst=1000,
        source_to_detector_dst=1500,
        vol_dims=np.array(initial_shape),
        det_dims=np.array([400, 400]),
        vol_spacing=np.array([2, 2, 2]),
        det_spacing=np.array([2, 2]),
        angles=angles,
        det_offset=0,
        sampling_step_size=0.1,
        device=device,
    )

    # 2. Simulate Loading a Volume with DIFFERENT size (e.g., 200x200x200)
    # This represents the inside of the loop
    actual_shape = (200, 200, 200)
    print(f"Simulating volume with shape: {actual_shape}")

    volume = torch.zeros(actual_shape, device=device, dtype=torch.float32)
    marker_pos = (50, 100, 150)
    volume[marker_pos] = 1.0
    volume_input = volume.unsqueeze(0).unsqueeze(0)

    # 3. APPLY THE FIX LOGIC: Update dimensions based on actual volume
    print("Applying geometry update (The Fix)...")
    geom.update_dims(vol_dims=np.array(actual_shape))
    projector_params = geom.get_projector_params(angles=angles)

    # 4. Project and Backproject
    print("Projecting...")
    projections = CBProjector.apply(volume_input, *projector_params)

    print("Backprojecting...")
    reconstruction = CBBackprojector.apply(projections, *projector_params)

    # 5. Verify Alignment
    recon_vol = reconstruction[0, 0]
    max_idx = (recon_vol == torch.max(recon_vol)).nonzero(as_tuple=False)[0].cpu().numpy()

    print(f"Original Marker: {marker_pos}")
    print(f"Recon Max Loc : {tuple(max_idx)}")

    diff = np.array(marker_pos) - max_idx
    if np.all(diff == 0):
        print("SUCCESS: Reconstruction matches original coordinates for non-standard size.")
    else:
        print(f"FAILURE: Mismatch detected. Diff: {diff}")
        exit(1)


if __name__ == "__main__":
    try:
        verify_dynamic_geometry()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
