import argparse
import os
from glob import glob
from os import environ
from pathlib import Path

import itk
import numpy as np
import torch
import torchbeam as ttc
from projectors import CBBackprojector, CBProjector, add_gaussian_noise, add_photon_noise
from tomo_projector_utils.scanner import ConebeamGeometry
from tqdm import tqdm


def convert_DICOM(folder_path, out_path: str, spacing: float = 2.0, resize_factor: float = 1.0):
    PixelType = itk.ctype("signed short")
    Dimension = 3

    ImageType = itk.Image[PixelType, Dimension]
    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(folder_path)
    seriesUID = namesGenerator.GetSeriesUIDs()

    InterpolatorType = itk.LinearInterpolateImageFunction[ImageType, itk.D]
    ResampleFilterType = itk.ResampleImageFilter[ImageType, ImageType]
    InputNamesGeneratorType = itk.GDCMSeriesFileNames
    OutputNamesGeneratorType = itk.NumericSeriesFileNames
    TransformType = itk.IdentityTransform[itk.D, Dimension]

    interpolator = InterpolatorType.New()

    transform = TransformType.New()
    transform.SetIdentity()

    if len(seriesUID) < 1:
        print("No DICOMs in: " + folder_path)
        return False

    for uid in seriesUID:
        seriesIdentifier = uid
        fileNames = namesGenerator.GetFileNames(seriesIdentifier)

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()

        reader.Update()

        inputSpacing = reader.GetOutput().GetSpacing()
        inputRegion = reader.GetOutput().GetLargestPossibleRegion()
        inputSize = inputRegion.GetSize()
        outputSpacing = [spacing, spacing, spacing]

        outputSize = [0, 0, 0]
        outputSize[0] = int(inputSize[0] * inputSpacing[0] * resize_factor / outputSpacing[0] + 0.5)
        outputSize[1] = int(inputSize[1] * inputSpacing[1] * resize_factor / outputSpacing[1] + 0.5)
        outputSize[2] = int(inputSize[2] * inputSpacing[2] * resize_factor / outputSpacing[2] + 0.5)
        transform_type = itk.TranslationTransform[itk.D, 3]
        vector = [0, 0, 0]
        translation = transform_type.New()
        translation.Translate(vector)

        # Compute new output origin to keep center in the same place
        input_origin = reader.GetOutput().GetOrigin()
        input_spacing = reader.GetOutput().GetSpacing()
        input_size = reader.GetOutput().GetLargestPossibleRegion().GetSize()
        output_direction = reader.GetOutput().GetDirection()

        input_center = [
            input_origin[i] + 0.5 * input_spacing[i] * (input_size[i] - 1) for i in range(3)
        ]
        output_origin = [
            input_center[i] - 0.5 * outputSpacing[i] * (outputSize[i] - 1) for i in range(3)
        ]

        resampler = ResampleFilterType.New()
        resampler.SetInput(reader.GetOutput())
        resampler.SetTransform(translation)
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputOrigin(output_origin)
        resampler.SetOutputSpacing(outputSpacing)
        resampler.SetOutputDirection(output_direction)
        resampler.SetSize(outputSize)
        resampler.Update()

        itk.imwrite(resampler.GetOutput(), out_path)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # setup an argparser to get the data path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/samuele/data/LIDC-IDRI/TCIA_LIDC-IDRI_20200921",
        help="Path to the data folder that contains the 'LIDC-IDRI' folder, defaults to 'data'",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/media/samuele/data/LIDC-IDRI/version20251209",
        # default="data",
        help="Path to the data folder, defaults to 'data'",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: process only one volume and visualize results",
    )

    args = parser.parse_args()
    data_path = Path(args.data_path)
    out_path = Path(args.out_path)

    # Create subfolders
    (out_path / "volumes").mkdir(exist_ok=True, parents=True)
    (out_path / "projections").mkdir(exist_ok=True, parents=True)
    (out_path / "noisy_projections").mkdir(exist_ok=True, parents=True)
    (out_path / "geometry").mkdir(exist_ok=True, parents=True)

    patient_folders = glob(str(data_path / Path("LIDC-IDRI/*")))

    device = torch.device("cuda")
    torch_rng = torch.Generator(device=device)
    torch_rng = torch_rng.manual_seed(42)
    photon_count = 1e5
    gaussian_mean = 0.0
    gaussian_std = 0.05
    num_projs = 400
    angles = np.linspace(0, 205 / 180 * np.pi, num_projs)
    geom = ConebeamGeometry(
        source_to_center_dst=1000,
        source_to_detector_dst=1500,
        vol_dims=np.array([256, 256, 256]),
        det_dims=np.array([256, 256]),
        vol_spacing=np.array([2, 2, 2]),
        det_spacing=np.array([2, 2]),
        angles=angles,
        det_offset=0,
        sampling_step_size=0.1,
        device=device,
    )
    geom.dump_json("geometry.json")

    # Collect all cases first for a proper progress bar
    all_case_folders = []
    for patient_folder in patient_folders:
        all_case_folders.extend(glob(patient_folder + "/*/*"))

    # convert from DICOM to itk volume
    for i, case_folder in enumerate(tqdm(all_case_folders, desc="Processing Volumes")):
        # print(f"Processing volume {i}")

        if len(list(glob(case_folder + "/*"))) > 10:
            mha_path = str(out_path / "volumes" / f"volume_{i}.mha")

            # Check if MHA already exists to avoid re-conversion if possible,
            # or just overwrite as per original logic. Keeping original logic for now.
            convert_DICOM(
                case_folder,
                mha_path,
                spacing=2.0,
                resize_factor=1.0,
            )

            # Load and Process
            volume = itk.GetArrayFromImage(itk.imread(mha_path))
            volume_cuda = 0.0206 + (0.0206 - 0.0004) * (
                torch.clip(torch.tensor(volume, device="cuda", dtype=torch.float), -1024, 2000)
                / 1000
            )
            volume_cuda = volume_cuda.unsqueeze(0).unsqueeze(0)

            # Update geometry with the correct volume dimensions
            geom.update_dims(vol_dims=np.array(volume.shape))
            geom.dump_json(out_path / "geometry" / f"geometry_{i}.json")
            projector_params = geom.get_projector_params(angles=angles)

            projections = CBProjector.apply(
                volume_cuda,
                *projector_params,
            )

            original_volume = volume_cuda[0, 0].cpu().numpy()
            np_projections = projections[0, 0].cpu().numpy()

            np.save(out_path / "volumes" / f"volume_{i}.npy", original_volume)
            np.save(out_path / "projections" / f"projections_{i}.npy", np_projections)

            # Add photon noise
            noisy_projs_photon = add_photon_noise(projections[0, 0], photon_count, torch_rng)

            # Add Gaussian noise
            noisy_projs_combined = add_gaussian_noise(
                noisy_projs_photon, gaussian_mean, gaussian_std, torch_rng
            )

            np.save(
                out_path / "noisy_projections" / f"noisy_projections_{i}.npy",
                noisy_projs_combined.cpu().numpy(),
            )

            if args.debug:
                print("Debug mode enabled. Performing reconstruction and visualization...")

                # Backprojection
                # Unpack params for clarity or use *projector_params
                # projector_params tuple structure: (vol_bbox, src_points, det_center, det_frame, det_bbox, vol_sz, det_sz, sampling_step_size)
                # CBBackprojector needs: projection, vol_bbox, src_points, det_center, det_frame, det_bbox, vol_sz, det_sz, sampling_step_size

                # Note: geom.get_projector_params returns sampling_step_size as the last element.
                # We can directly unpack *projector_params into CBBackprojector.apply after the projection argument.

                reconstruction = CBBackprojector.apply(
                    noisy_projs_combined.unsqueeze(0).unsqueeze(0), *projector_params
                )

                reconstruction_np = reconstruction[0, 0].cpu().numpy()

                # Visualization
                fig, axes = plt.subplots(4, 3, figsize=(15, 25))

                # Volume slices
                mid_x, mid_y, mid_z = (
                    original_volume.shape[0] // 2,
                    original_volume.shape[1] // 2,
                    original_volume.shape[2] // 2,
                )
                axes[0, 0].imshow(original_volume[mid_x, :, :], cmap="gray")
                axes[0, 0].set_title("Original Axial")
                axes[0, 1].imshow(original_volume[:, mid_y, :], cmap="gray")
                axes[0, 1].set_title("Original Coronal")
                axes[0, 2].imshow(original_volume[:, :, mid_z], cmap="gray")
                axes[0, 2].set_title("Original Sagittal")

                # Projections
                proj_idx = 0
                axes[1, 0].imshow(np_projections[proj_idx], cmap="gray")
                axes[1, 0].set_title("Clean Projection")
                axes[1, 1].imshow(noisy_projs_photon[proj_idx].cpu().numpy(), cmap="gray")
                axes[1, 1].set_title("Photon Noise")
                axes[1, 2].imshow(noisy_projs_combined[proj_idx].cpu().numpy(), cmap="gray")
                axes[1, 2].set_title("Photon + Gaussian")

                # Reconstruction slices
                axes[2, 0].imshow(reconstruction_np[mid_x, :, :], cmap="gray")
                axes[2, 0].set_title("Recon Axial")
                axes[2, 1].imshow(reconstruction_np[:, mid_y, :], cmap="gray")
                axes[2, 1].set_title("Recon Coronal")
                axes[2, 2].imshow(reconstruction_np[:, :, mid_z], cmap="gray")
                axes[2, 2].set_title("Recon Sagittal")

                # Noisy Projections Grid
                for j, k in enumerate(range(0, num_projs, num_projs // 3)):
                    if j < 3:
                        axes[3, j].imshow(noisy_projs_combined[k].cpu().numpy(), cmap="gray")
                        axes[3, j].set_title(f"Noisy Proj {k}")

                for ax in axes.flatten():
                    ax.set_axis_off()

                plt.tight_layout()
                verification_filename = f"verification_debug_volume_{i}.png"
                plt.savefig(verification_filename)
                print(f"Verification image saved to {verification_filename}")
                plt.close(fig)  # Close figure to free memory

                # Process up to 10 volumes in debug mode
                if i >= 9:
                    print("Processed 10 volumes. Exiting debug mode.")
                    exit(0)
