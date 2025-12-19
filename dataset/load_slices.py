import json
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from tomo_projector_utils.scanner import ConebeamGeometry
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SliceDataset(Dataset):
    """
    Dataset that returns specific 2D slices from 3D volumes.
    Useful for training 2D supervision or validating on 2D planes.
    """

    def __init__(
        self,
        root_dir: str,
        slice_selection: Union[
            str, int, List[int]
        ] = None,  # Deprecated/unused for now, kept for signature compatibility if needed
        axis: int = 0,
        stage: str = "train",
        normalize: bool = True,
        num_vols: int = 200,
        gray_value_scaling: float = 20.0,
        rendering_bbox=torch.tensor([[-300, -300, -300], [300, 300, 300]], dtype=torch.float32),
        num_points: int = 1024,  # Fixed number of points per sample (e.g., 32*32)
        rng_seed: int = 42,  # Seed for random number generator for reproducibility
    ):
        """
        Args:
            root_dir: Path to the data directory (containing 'volumes' and 'geometry').
            slice_selection: (Unused)
            axis: Axis to slice along (0, 1, or 2).
            stage: 'train', 'val', or 'test'. Used for splitting data.
            normalize: Whether to normalize coordinates to [-1, 1].
            num_vols: Limit the number of volumes to load.
        """
        self.root_dir = Path(root_dir)
        self.volumes_dir = self.root_dir / "volumes"
        self.geometry_dir = self.root_dir / "geometry"
        self.axis = axis
        self.normalize = normalize
        self.rendering_bbox = rendering_bbox
        self.gray_value_scaling = gray_value_scaling
        self.num_points = num_points
        self.rng_seed = rng_seed

        # 1. Gather Paths
        all_volume_paths = sorted(
            glob(str(self.volumes_dir / "volume_*.npy")),
            key=lambda x: int(Path(x).stem.split("_")[-1]),
        )

        # 2. Split Data
        train_paths = all_volume_paths[:200]
        val_paths = all_volume_paths[200:225]
        test_paths = all_volume_paths[225:250]

        if stage == "train":
            self.volume_paths = train_paths[:num_vols]
        elif stage == "val":
            self.volume_paths = val_paths[: min(len(val_paths), num_vols)]
        elif stage == "test":
            self.volume_paths = test_paths[: min(len(test_paths), num_vols)]
        else:
            raise ValueError(f"Unknown stage: {stage}")

        self.volumes = []
        self.geometries = []
        self.slice_counts = []

        # 3. Load Data into RAM and compute slice counts
        print(f"Loading {len(self.volume_paths)} volumes for stage '{stage}'...")
        for vol_path in tqdm(self.volume_paths, desc="Loading Volumes"):
            # Load Volume
            vol_data = np.load(vol_path)
            self.volumes.append(vol_data * self.gray_value_scaling)

            # Record slice count for this volume
            self.slice_counts.append(vol_data.shape[self.axis])

            # Load Geometry
            idx = int(Path(vol_path).stem.split("_")[-1])
            geo_path = self.geometry_dir / f"geometry_{idx}.json"
            if not geo_path.exists():
                raise FileNotFoundError(f"Geometry file not found: {geo_path}")

            geo = ConebeamGeometry.from_json(geo_path, device=torch.device("cpu"))
            # Make sure dims match loaded volume just in case
            geo.update_dims(vol_dims=np.array(vol_data.shape))
            self.geometries.append(geo)

        # Compute cumulative sum for indexing
        self.cumulative_slices = np.cumsum(self.slice_counts)
        self.total_slices = self.cumulative_slices[-1] if len(self.cumulative_slices) > 0 else 0

    def __len__(self):
        return self.total_slices

    def _resolve_global_index(self, idx):
        # Find the volume index
        # searchsorted returns the index where idx would be inserted to maintain order.
        # cumulative_slices[i] is the *end* (exclusive) global index of volume i.
        # Example: slice_counts = [10, 10] -> cumsum = [10, 20]
        # idx 0 -> vol 0 (0 < 10)
        # idx 9 -> vol 0 (9 < 10)
        # idx 10 -> vol 1 (10 < 20)

        if idx < 0 or idx >= self.total_slices:
            raise IndexError(
                f"Index {idx} out of range for dataset with {self.total_slices} slices."
            )

        vol_idx = np.searchsorted(self.cumulative_slices, idx, side="right")

        # If it returns an index equal to length, or if logic needs adjustment:
        # Actually searchsorted with side='right':
        # [10, 20]
        # idx 0: > 10? No. -> 0. Correct.
        # idx 9: > 10? No. -> 0. Correct.
        # idx 10: > 10? Yes. -> 1. Correct.
        # But wait, searchsorted([10, 20], 9) -> 0.
        # searchsorted([10, 20], 10) -> 1.

        remaining_idx = idx
        if vol_idx > 0:
            remaining_idx = idx - self.cumulative_slices[vol_idx - 1]

        return vol_idx, remaining_idx

    def _get_slice_data(self, idx):
        """Helper method to extract slice image and coordinates for a given index.

        Returns:
            slice_image: numpy array [H, W]
            coords: numpy array [H, W, 2]
            vol_bbox: numpy array [2, 3] (normalized)
            vol_idx: int
            slice_idx: int
        """
        vol_idx, slice_idx = self._resolve_global_index(idx)
        volume = self.volumes[vol_idx]
        geo = self.geometries[vol_idx]

        # Extract Slice Data
        # axis 0: volume[slice_idx, :, :]
        # axis 1: volume[:, slice_idx, :]
        # axis 2: volume[:, :, slice_idx]
        if self.axis == 0:
            slice_image = volume[slice_idx, :, :]
        elif self.axis == 1:
            slice_image = volume[:, slice_idx, :]
        else:
            slice_image = volume[:, :, slice_idx]

        # Generate Coordinates for this slice
        coords = self.get_slice_coords(geo, slice_idx, self.axis)

        if hasattr(geo, "vol_bbox") and geo.vol_bbox is not None:
            vol_bbox = geo.vol_bbox[0].cpu().numpy()
        elif hasattr(geo, "_vol_bbox") and geo._vol_bbox is not None:
            vol_bbox = geo._vol_bbox.cpu().numpy()
        else:
            # Fallback or error?
            vol_bbox = np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float32)

        # Normalize vol_bbox using rendering_bbox to match coordinate system
        # Assuming rendering_bbox is centered at 0 and symmetric
        ref_bbox = self.rendering_bbox.numpy()
        norm_vol_bbox = np.zeros_like(vol_bbox)
        # mins
        norm_vol_bbox[0] = 2 * vol_bbox[0] / (ref_bbox[1] - ref_bbox[0])
        # maxs
        norm_vol_bbox[1] = 2 * vol_bbox[1] / (ref_bbox[1] - ref_bbox[0])

        return slice_image, coords, norm_vol_bbox, vol_idx, slice_idx

    def get_full_slice(self, idx):
        """Get the full slice (without subsampling) for visualization and metrics.

        Args:
            idx: Index of the slice in the dataset

        Returns:
            dict with keys:
                - 'image': torch.Tensor [H, W] - full slice image
                - 'coords': torch.Tensor [H, W, 2] - full coordinate grid
                - 'values': torch.Tensor [H*W] - flattened image values
                - 'coords_flat': torch.Tensor [H*W, 2] - flattened coordinates
                - 'vol_bbox': torch.Tensor [2, 3] - normalized volume bounding box
                - 'slice_idx': int - slice index within the volume
                - 'patient_idx': int - volume index
        """
        slice_image, coords, norm_vol_bbox, vol_idx, slice_idx = self._get_slice_data(idx)

        h, w = slice_image.shape

        # Convert to tensors
        image_tensor = torch.tensor(slice_image, dtype=torch.float32)  # [H, W]
        coords_tensor = torch.tensor(coords, dtype=torch.float32)  # [H, W, 2]

        # Flattened versions
        values_flat = image_tensor.view(-1)  # [H*W]
        coords_flat = coords_tensor.view(-1, 2)  # [H*W, 2]

        return {
            "image": image_tensor,  # [H, W]
            "coords": coords_tensor,  # [H, W, 2]
            "values": values_flat,  # [H*W]
            "coords_flat": coords_flat,  # [H*W, 2]
            "vol_bbox": torch.tensor(norm_vol_bbox, dtype=torch.float32),
            "slice_idx": slice_idx,
            "patient_idx": vol_idx,  # kept for compatibility
            "global_slice_idx": idx,  # global slice index in the dataset (for latent indexing)
        }

    def __getitem__(self, idx):
        # Get full slice data
        slice_image, coords, norm_vol_bbox, vol_idx, slice_idx = self._get_slice_data(idx)

        # Flatten the slice image and coordinates
        h, w = slice_image.shape
        total_points = h * w

        # Flatten image and coordinates
        values_flat = slice_image.flatten()  # [H*W]
        coords_flat = coords.reshape(-1, 2)  # [H*W, 2]

        # Use a controlled RNG seeded with the sample index for reproducibility
        # This ensures the same sample always produces the same subsampling
        rng = np.random.Generator(np.random.PCG64(self.rng_seed + idx))

        # Randomly subsample num_points from the available points
        if total_points >= self.num_points:
            # Randomly sample indices without replacement
            indices = rng.choice(total_points, size=self.num_points, replace=False)
        else:
            # If we have fewer points than requested, sample with replacement
            indices = rng.choice(total_points, size=self.num_points, replace=True)

        # Select subsampled values and coordinates
        values_sampled = values_flat[indices]  # [num_points]
        coords_sampled = coords_flat[indices]  # [num_points, 2]

        # Convert to tensors
        values = torch.tensor(values_sampled, dtype=torch.float32)  # [num_points]
        coords_flat_tensor = torch.tensor(coords_sampled, dtype=torch.float32)  # [num_points, 2]

        return {
            "values": values,  # [num_points]
            "coords": coords_flat_tensor,  # [num_points, 2]
            "vol_bbox": torch.tensor(norm_vol_bbox, dtype=torch.float32),
            "slice_idx": slice_idx,
            "patient_idx": vol_idx,  # local index in the split (kept for compatibility)
            "global_slice_idx": idx,  # global slice index in the dataset (for latent indexing)
        }

    def get_slice_coords(self, geo, slice_idx, axis):
        """ """
        vol_dims = geo.vol_dims
        if hasattr(geo, "vol_bbox") and geo.vol_bbox is not None:
            vol_bbox = geo.vol_bbox[0].cpu().numpy()
        else:
            vol_bbox = geo._vol_bbox.cpu().numpy()

        # Calculate coordinate range/extents
        # We need to map discrete indices [0, D-1] to continuous space [min, max]
        # Or more accurately, voxel centers.

        # Assuming standard linspace mapping as in make_recon_data
        # But we need normalized coordinates usually for NeRF/MLP inputs?
        # make_recon_data used:
        # val_state.ref_coords = reference_coords(...)

        # Let's match typical global coordinate generation
        # If normalize=True, we map to approx [-1, 1] based on rendering_bbox

        bbox = vol_bbox
        ref_bbox = self.rendering_bbox.numpy()

        # extent maps the vol_bbox range to normalized space [-1, 1] relative to ref_bbox ??
        # In make_recon_data: extent = 2 * bbox / (ref_bbox[1] - ref_bbox[0])
        # This seems to be scaling factor? No.
        # Let's re-read make_recon_data logic carefully.

        # extent = 2 * bbox / (ref_bbox[1] - ref_bbox[0])
        # x = np.linspace(extent[0][0], extent[1][0], dim[0])
        # So it maps [bbox_min, bbox_max] to normalized range.

        # Let's compute the normalized grids for all dims
        mins = 2 * bbox[0] / (ref_bbox[1] - ref_bbox[0])  # (3,)
        maxs = 2 * bbox[1] / (ref_bbox[1] - ref_bbox[0])  # (3,)

        grids = []
        for d in range(3):
            grids.append(np.linspace(mins[d], maxs[d], vol_dims[d], dtype=np.float32))

        # Now construct the meshgrid for the slice
        # If axis=0, x is fixed to grids[0][slice_idx]
        # y varies grids[1], z varies grids[2]

        if axis == 0:
            fixed_x = grids[0][slice_idx]
            y, z = np.meshgrid(grids[1], grids[2], indexing="ij")
            coords = np.stack([y, z], axis=-1)

        elif axis == 1:
            fixed_y = grids[1][slice_idx]
            x, z = np.meshgrid(grids[0], grids[2], indexing="ij")
            coords = np.stack([x, z], axis=-1)

        elif axis == 2:
            fixed_z = grids[2][slice_idx]
            x, y = np.meshgrid(grids[0], grids[1], indexing="ij")
            coords = np.stack([x, y], axis=-1)

        return coords


def get_slice_dataloader(
    root_dir="../data_factory/data",
    slice_selection=None,
    axis=0,
    num_vols=50,
    batch_size=1,
    num_workers=4,
    stage="train",
    shuffle=True,
    gray_value_scaling=20.0,
    num_points=1024,  # Fixed number of points per sample
    rng_seed=42,  # Seed for random number generator
):
    dataset = SliceDataset(
        root_dir=root_dir,
        slice_selection=slice_selection,
        axis=axis,
        stage=stage,
        num_vols=num_vols,
        gray_value_scaling=gray_value_scaling,
        num_points=num_points,
        rng_seed=rng_seed,
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # seed everything
    torch.manual_seed(0)
    np.random.seed(0)

    # Simple Test
    root_dir = "/media/samuele/data/LIDC-IDRI/version20251209"
    loader = get_slice_dataloader(root_dir=root_dir, stage="train", num_vols=5)

    print(f"Testing Slice Dataloader... Total slices: {len(loader.dataset)}")
    for i, batch in enumerate(loader):
        img = batch["image"]
        coords = batch["coords"]
        print(f"Batch Image Shape: {img.shape}")
        print(f"Batch Coords Shape: {coords.shape}")
        print(f"Coords Sample: {coords[0, 0, 0]}")
        print(
            f"Coords extent: {coords.reshape(-1, 2).min(dim=0).values}, {coords.reshape(-1, 2).max(dim=0).values}"
        )

        plt.imshow(img[0].numpy(), cmap="gray")
        plt.show()

        if i == 10:
            break
