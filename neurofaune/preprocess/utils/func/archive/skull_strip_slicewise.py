#!/usr/bin/env python3
"""
Slice-wise skull stripping for highly anisotropic BOLD data.

Runs BET independently on each axial slice with slice-specific COG,
then recombines into a 3D mask. Better handles ellipsoidal rat brain shape.
"""

from pathlib import Path
import subprocess
import nibabel as nib
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


def calculate_slice_cog(slice_data: np.ndarray) -> Tuple[float, float]:
    """
    Calculate center of gravity for a single 2D slice.

    Parameters
    ----------
    slice_data : np.ndarray
        2D slice data

    Returns
    -------
    Tuple[float, float]
        (x, y) center of gravity in voxel coordinates
    """
    # Threshold at 10th percentile of non-zero voxels
    nonzero = slice_data[slice_data > 0]
    if len(nonzero) == 0:
        # Empty slice - use geometric center
        return (slice_data.shape[0] / 2, slice_data.shape[1] / 2)

    threshold = np.percentile(nonzero, 10)
    mask = slice_data > threshold

    if mask.sum() == 0:
        return (slice_data.shape[0] / 2, slice_data.shape[1] / 2)

    # Calculate intensity-weighted COG
    cog = ndimage.center_of_mass(slice_data * mask)

    return (cog[0], cog[1])


def run_bet_on_slice(
    slice_data: np.ndarray,
    slice_idx: int,
    work_dir: Path,
    frac: float = 0.35,
    cog: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Run BET on a single 2D slice.

    Parameters
    ----------
    slice_data : np.ndarray
        2D slice data
    slice_idx : int
        Slice index for file naming
    work_dir : Path
        Working directory for temporary files
    frac : float
        BET fractional threshold
    cog : Tuple[float, float], optional
        Center of gravity (x, y). If None, will calculate automatically.

    Returns
    -------
    np.ndarray
        2D binary mask for this slice
    """
    # Create temporary 3D image (single slice) for BET
    slice_3d = slice_data[:, :, np.newaxis]

    # Create temporary NIfTI with proper affine
    affine = np.eye(4)
    img = nib.Nifti1Image(slice_3d, affine)

    # Save temporary input
    input_file = work_dir / f'slice_{slice_idx:03d}_input.nii.gz'
    nib.save(img, input_file)

    # Calculate COG if not provided
    if cog is None:
        cog = calculate_slice_cog(slice_data)

    # BET output (without .nii.gz extension)
    output_base = str(work_dir / f'slice_{slice_idx:03d}_brain')

    # Run BET with slice-specific COG
    cmd = [
        'bet',
        str(input_file),
        output_base,
        '-f', str(frac),
        '-m',  # Create mask
        '-n',  # No brain output (we only need mask)
        '-c', str(int(cog[0])), str(int(cog[1])), '0',  # 2D center, z=0
        '-F'   # Functional optimization
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    # Check if mask was created
    mask_file = Path(output_base + '_mask.nii.gz')
    if not mask_file.exists():
        print(f"  Warning: BET failed on slice {slice_idx}, using empty mask")
        return np.zeros_like(slice_data, dtype=bool)

    # Load mask
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()

    # Handle both 2D and 3D mask outputs
    if mask_data.ndim == 3:
        mask_2d = mask_data[:, :, 0].astype(bool)
    else:
        mask_2d = mask_data.astype(bool)

    # Clean up temporary files
    input_file.unlink()
    mask_file.unlink()
    if Path(output_base + '.nii.gz').exists():
        Path(output_base + '.nii.gz').unlink()

    return mask_2d


def skull_strip_slicewise(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    work_dir: Path,
    bet_frac: float = 0.35,
    auto_cog: bool = True
) -> Tuple[Path, Path, dict]:
    """
    Perform slice-wise skull stripping with per-slice COG calculation.

    Parameters
    ----------
    input_file : Path
        Input 3D reference volume
    output_file : Path
        Output brain-extracted volume
    mask_file : Path
        Output mask file
    work_dir : Path
        Working directory for temporary files
    bet_frac : float
        BET fractional threshold
    auto_cog : bool
        Calculate COG per slice automatically

    Returns
    -------
    Tuple[Path, Path, dict]
        (brain_file, mask_file, info_dict)
    """
    print("\n" + "="*80)
    print("SLICE-WISE SKULL STRIPPING")
    print("="*80)
    print(f"Input: {input_file.name}")
    print(f"BET frac: {bet_frac}")
    print(f"Auto COG per slice: {auto_cog}")

    # Create slice-specific work directory
    slice_work_dir = work_dir / 'slicewise'
    slice_work_dir.mkdir(parents=True, exist_ok=True)

    # Load input volume
    img = nib.load(input_file)
    data = img.get_fdata()

    n_slices = data.shape[2]
    print(f"\nProcessing {n_slices} axial slices...")

    # Initialize combined mask
    combined_mask = np.zeros_like(data, dtype=bool)

    # Process each slice
    slice_stats = []
    for z in range(n_slices):
        slice_data = data[:, :, z]

        # Calculate COG for this slice
        if auto_cog:
            cog = calculate_slice_cog(slice_data)
        else:
            cog = (slice_data.shape[0] / 2, slice_data.shape[1] / 2)

        print(f"  Slice {z}: COG = ({int(cog[0])}, {int(cog[1])})", end='')

        # Run BET on this slice
        slice_mask = run_bet_on_slice(
            slice_data, z, slice_work_dir, frac=bet_frac, cog=cog
        )

        # Add to combined mask
        combined_mask[:, :, z] = slice_mask

        # Track stats
        n_voxels = slice_mask.sum()
        slice_stats.append({
            'slice': z,
            'cog': cog,
            'voxels': n_voxels
        })
        print(f" → {n_voxels} voxels")

    # Save combined mask
    mask_img = nib.Nifti1Image(combined_mask.astype(np.uint8), img.affine, img.header)
    nib.save(mask_img, mask_file)

    # Apply mask to original data
    brain_data = data * combined_mask
    brain_img = nib.Nifti1Image(brain_data, img.affine, img.header)
    nib.save(brain_img, output_file)

    # Calculate summary stats
    total_voxels = combined_mask.sum()
    total_image_voxels = np.prod(data.shape)
    extraction_ratio = total_voxels / total_image_voxels

    print("\n" + "="*80)
    print("SLICE-WISE SKULL STRIPPING COMPLETE")
    print("="*80)
    print(f"Total mask voxels: {total_voxels:,}")
    print(f"Extraction ratio: {extraction_ratio:.3f}")
    print(f"Brain: {output_file}")
    print(f"Mask: {mask_file}")

    info = {
        'method': 'slicewise',
        'bet_frac': bet_frac,
        'n_slices': n_slices,
        'slice_stats': slice_stats,
        'total_voxels': int(total_voxels),
        'extraction_ratio': float(extraction_ratio)
    }

    return output_file, mask_file, info


if __name__ == '__main__':
    # Quick test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python skull_strip_slicewise.py <input.nii.gz>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    work_dir = Path('/tmp/slicewise_test')
    work_dir.mkdir(exist_ok=True)

    output_file = work_dir / 'brain.nii.gz'
    mask_file = work_dir / 'mask.nii.gz'

    brain, mask, info = skull_strip_slicewise(
        input_file, output_file, mask_file, work_dir
    )

    print(f"\nTest complete! Results in {work_dir}")
