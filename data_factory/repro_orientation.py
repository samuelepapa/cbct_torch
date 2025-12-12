import os
import sys

import numpy as np
import torch

# Add relevant paths to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Assuming we are in experiments/instant_ngp_torch/data_factory/
# dependent packages might be in the root or parallel folders
# Adjust as necessary if imports fail.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))

from projectors import CBBackprojector, CBProjector
from tomo_projector_utils.scanner import ConebeamGeometry


def test_orientation():
    device = torch.device("cuda")

    # 1. Create a dummy volume (Z, Y, X)
    vol_shape = (256, 256, 256)
    volume = torch.zeros(vol_shape, device=device, dtype=torch.float32)

    # Place a marker at an asymmetric location
    # Z=50, Y=100, X=150
    marker_pos = (50, 100, 150)
    # Add a small gaussian or just a point
    volume[marker_pos] = 1.0

    print(f"Original marker position (Z, Y, X): {marker_pos}")

    # Add batch and channel dims
    volume_input = volume.unsqueeze(0).unsqueeze(0)

    # 2. Setup Geometry
    # Using params similar to make_data.py
    angles = np.linspace(0, 205 / 180 * np.pi, 50)  # fewer projections for speed
    geom = ConebeamGeometry(
        source_to_center_dst=1000,
        source_to_detector_dst=1500,
        vol_dims=np.array(vol_shape),
        det_dims=np.array([400, 400]),
        vol_spacing=np.array([2, 2, 2]),
        det_spacing=np.array([2, 2]),
        angles=angles,
        det_offset=0,
        sampling_step_size=0.1,
        device=device,
    )

    projector_params = geom.get_projector_params(angles=angles)

    # 3. Project
    print("Projecting...")
    projections = CBProjector.apply(
        volume_input,
        *projector_params,
    )

    # 4. Backproject
    print("Backprojecting...")
    reconstruction = CBBackprojector.apply(projections, *projector_params)

    recon_vol = reconstruction[0, 0]

    # 5. Find max location
    max_val = torch.max(recon_vol)
    max_idx = (recon_vol == max_val).nonzero(as_tuple=False)

    print(f"Reconstruction max value: {max_val.item()}")
    print("Reconstruction max location(s) (Z, Y, X):")
    for idx in max_idx:
        print(f"  {idx.tolist()}")

    # Check closer match
    detected_pos = max_idx[0].cpu().numpy()
    original_pos = np.array(marker_pos)

    diff = detected_pos - original_pos
    print(f"Difference: {diff}")

    if np.all(diff == 0):
        print("PERFECT MATCH! Orientation is correct.")
    else:
        print("MISMATCH DETECTED!")
        # Analyze mismatch
        # Check for swaps
        z, y, x = detected_pos
        oz, oy, ox = original_pos

        if (z, y, x) == (ox, oy, oz):
            print("Detected inversion: (X, Y, Z) <-> (Z, Y, X) [Transpose 0, 2]")
        elif (z, y, x) == (oz, ox, oy):
            print("Detected swap Y-Z: (Z, Y, X) -> (Z, X, Y) ?")
        # .. add more checks ..

        # Check if flipped
        vol_sz = np.array(vol_shape)
        flipped_x = (z, y, vol_sz[2] - 1 - x)
        if np.array_equal(original_pos, flipped_x):
            print("Detected X-flip")


if __name__ == "__main__":
    try:
        test_orientation()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
