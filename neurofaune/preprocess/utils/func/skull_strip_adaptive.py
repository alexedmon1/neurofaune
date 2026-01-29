#!/usr/bin/env python3
"""
Adaptive slice-wise skull stripping with per-slice frac optimization.

Automatically determines optimal BET frac per slice based on extraction ratio.
"""

from pathlib import Path
import subprocess
import nibabel as nib
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional, List
import tempfile


def calculate_slice_cog(slice_data: np.ndarray) -> Tuple[float, float]:
    """Calculate center of gravity for a 2D slice."""
    nonzero = slice_data[slice_data > 0]
    if len(nonzero) == 0:
        return (slice_data.shape[0] / 2, slice_data.shape[1] / 2)

    threshold = np.percentile(nonzero, 10)
    mask = slice_data > threshold

    if mask.sum() == 0:
        return (slice_data.shape[0] / 2, slice_data.shape[1] / 2)

    cog = ndimage.center_of_mass(slice_data * mask)
    return (cog[0], cog[1])


def test_bet_frac_on_slice(
    slice_data: np.ndarray,
    slice_idx: int,
    work_dir: Path,
    frac: float,
    cog: Tuple[float, float],
    invert_intensity: bool = False,
    use_R_flag: bool = False
) -> Tuple[Optional[np.ndarray], int]:
    """
    Test a specific BET frac value on a slice.

    Parameters
    ----------
    slice_data : np.ndarray
        2D slice data
    slice_idx : int
        Slice index
    work_dir : Path
        Working directory
    frac : float
        BET fractional threshold
    cog : Tuple[float, float]
        Center of gravity (used for reference, not used if use_R_flag=True)
    invert_intensity : bool
        If True, invert intensity (max - data) to make T2w look like T1w
    use_R_flag : bool
        If True, use BET's -R flag (robust center estimation) instead of -c flag

    Returns
    -------
    Tuple[Optional[np.ndarray], int]
        (mask, n_voxels) or (None, 0) if failed
    """
    # Optionally invert intensity (T2w → T1w-like contrast)
    if invert_intensity:
        slice_data_bet = np.max(slice_data) - slice_data
    else:
        slice_data_bet = slice_data.copy()

    # Create temporary 3D image (single slice)
    slice_3d = slice_data_bet[:, :, np.newaxis]
    affine = np.eye(4)
    img = nib.Nifti1Image(slice_3d, affine)

    # Save temporary input
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False, dir=work_dir) as f:
        input_file = Path(f.name)
    nib.save(img, input_file)

    # BET output
    with tempfile.NamedTemporaryFile(suffix='', delete=False, dir=work_dir) as f:
        output_base = str(f.name)

    # Run BET
    cmd = [
        'bet',
        str(input_file),
        output_base,
        '-f', str(frac),
        '-m', '-n'
    ]

    # Use either -R (robust) or -c (explicit COG)
    if use_R_flag:
        cmd.append('-R')
    else:
        cmd.extend(['-c', str(int(cog[0])), str(int(cog[1])), '0'])

    cmd.append('-F')

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    # Check for mask
    mask_file = Path(output_base + '_mask.nii.gz')
    if not mask_file.exists():
        # Clean up
        input_file.unlink(missing_ok=True)
        return None, 0

    # Load mask
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()

    if mask_data.ndim == 3:
        mask_2d = mask_data[:, :, 0].astype(bool)
    else:
        mask_2d = mask_data.astype(bool)

    n_voxels = mask_2d.sum()

    # Clean up
    input_file.unlink(missing_ok=True)
    mask_file.unlink(missing_ok=True)
    Path(output_base + '.nii.gz').unlink(missing_ok=True)

    return mask_2d, int(n_voxels)


def find_optimal_frac_for_slice(
    slice_data: np.ndarray,
    slice_idx: int,
    work_dir: Path,
    cog: Tuple[float, float],
    target_ratio: float = 0.15,
    frac_range: Tuple[float, float] = (0.30, 0.60),
    frac_step: float = 0.05,
    invert_intensity: bool = False,
    use_R_flag: bool = False
) -> Tuple[float, np.ndarray, int]:
    """
    Find optimal BET frac for a slice by testing multiple values.

    Parameters
    ----------
    slice_data : np.ndarray
        2D slice data
    slice_idx : int
        Slice index
    work_dir : Path
        Working directory
    cog : Tuple[float, float]
        Center of gravity
    target_ratio : float
        Target extraction ratio (0-1)
    frac_range : Tuple[float, float]
        (min_frac, max_frac) to test
    frac_step : float
        Step size for frac values
    invert_intensity : bool
        If True, invert intensity before BET
    use_R_flag : bool
        If True, use BET's -R flag instead of -c flag

    Returns
    -------
    Tuple[float, np.ndarray, int]
        (optimal_frac, optimal_mask, n_voxels)
    """
    total_voxels = np.prod(slice_data.shape)
    target_voxels = int(total_voxels * target_ratio)

    # Generate frac values to test
    frac_values = np.arange(frac_range[0], frac_range[1] + frac_step, frac_step)

    best_frac = None
    best_mask = None
    best_voxels = 0
    best_diff = float('inf')

    results = []

    for frac in frac_values:
        mask, n_voxels = test_bet_frac_on_slice(
            slice_data, slice_idx, work_dir, frac, cog, invert_intensity, use_R_flag
        )

        if mask is None:
            continue

        # Calculate difference from target
        diff = abs(n_voxels - target_voxels)
        results.append((frac, n_voxels, diff))

        if diff < best_diff:
            best_diff = diff
            best_frac = frac
            best_mask = mask
            best_voxels = n_voxels

    # If all failed, return empty mask
    if best_mask is None:
        return 0.35, np.zeros_like(slice_data, dtype=bool), 0

    return best_frac, best_mask, best_voxels


def skull_strip_adaptive(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    work_dir: Path,
    target_ratio: float = 0.15,
    frac_range: Tuple[float, float] = (0.30, 0.60),
    frac_step: float = 0.05,
    invert_intensity: bool = False,
    use_R_flag: bool = False,
    cog_offset_x: Optional[int] = None,
    cog_offset_y: Optional[int] = None
) -> Tuple[Path, Path, dict]:
    """
    Perform adaptive slice-wise skull stripping with per-slice frac optimization.

    Parameters
    ----------
    input_file : Path
        Input 3D reference volume
    output_file : Path
        Output brain-extracted volume
    mask_file : Path
        Output mask file
    work_dir : Path
        Working directory
    target_ratio : float
        Target extraction ratio per slice (0-1)
    frac_range : Tuple[float, float]
        (min_frac, max_frac) to test
    frac_step : float
        Step size for frac testing
    invert_intensity : bool
        If True, invert intensity before BET (T2w → T1w-like contrast)
    use_R_flag : bool
        If True, use BET's -R flag (robust center estimation) instead of -c flag
    cog_offset_x : int, optional
        X offset from image center for COG estimation. If None, calculate from intensity.
    cog_offset_y : int, optional
        Y offset from image center for COG estimation. If None, calculate from intensity.
        Negative values shift COG down (inferior), which is typical for brain positioning.

    Returns
    -------
    Tuple[Path, Path, dict]
        (brain_file, mask_file, info_dict)
    """
    # Determine COG estimation method
    use_estimated_cog = cog_offset_x is not None or cog_offset_y is not None

    print("\n" + "="*80)
    print("ADAPTIVE SLICE-WISE SKULL STRIPPING")
    print("="*80)
    print(f"Input: {input_file.name}")
    print(f"Target extraction per slice: {target_ratio*100:.1f}%")
    print(f"Frac search range: {frac_range[0]:.2f} - {frac_range[1]:.2f} (step: {frac_step:.2f})")
    print(f"Invert intensity (T2w→T1w): {invert_intensity}")
    if use_estimated_cog:
        print(f"COG estimation: image center + offset ({cog_offset_x or 0}, {cog_offset_y or 0})")
    else:
        print(f"COG estimation: intensity-weighted (use_R_flag={use_R_flag})")

    # Create slice-specific work directory
    slice_work_dir = work_dir / 'adaptive_slices'
    slice_work_dir.mkdir(parents=True, exist_ok=True)

    # Load input volume
    img = nib.load(input_file)
    data = img.get_fdata()

    n_slices = data.shape[2]
    print(f"\nProcessing {n_slices} axial slices with adaptive frac...")

    # Calculate estimated COG if using offset method
    if use_estimated_cog:
        img_center_x = data.shape[0] // 2
        img_center_y = data.shape[1] // 2
        estimated_cog = (
            img_center_x + (cog_offset_x or 0),
            img_center_y + (cog_offset_y or 0)
        )
        print(f"Image center: ({img_center_x}, {img_center_y}), Estimated COG: {estimated_cog}")

    # Initialize combined mask
    combined_mask = np.zeros_like(data, dtype=bool)

    # Process each slice
    slice_stats = []
    for z in range(n_slices):
        slice_data = data[:, :, z]

        # Get COG: either estimated from offset or calculated from intensity
        if use_estimated_cog:
            cog = estimated_cog
        else:
            cog = calculate_slice_cog(slice_data)

        # Find optimal frac for this slice
        print(f"  Slice {z}: COG=({int(cog[0])},{int(cog[1])}) ", end='', flush=True)

        optimal_frac, slice_mask, n_voxels = find_optimal_frac_for_slice(
            slice_data, z, slice_work_dir, cog,
            target_ratio=target_ratio,
            frac_range=frac_range,
            frac_step=frac_step,
            invert_intensity=invert_intensity,
            use_R_flag=use_R_flag
        )

        # Add to combined mask
        combined_mask[:, :, z] = slice_mask

        # Calculate extraction ratio for this slice
        slice_ratio = n_voxels / np.prod(slice_data.shape)

        slice_stats.append({
            'slice': z,
            'cog': cog,
            'optimal_frac': optimal_frac,
            'voxels': n_voxels,
            'ratio': slice_ratio
        })

        print(f"→ frac={optimal_frac:.2f}, {n_voxels} voxels ({slice_ratio*100:.1f}%)")

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

    # Calculate frac statistics
    frac_values = [s['optimal_frac'] for s in slice_stats]
    mean_frac = np.mean(frac_values)
    std_frac = np.std(frac_values)

    print("\n" + "="*80)
    print("ADAPTIVE SKULL STRIPPING COMPLETE")
    print("="*80)
    print(f"Total mask voxels: {total_voxels:,}")
    print(f"Overall extraction ratio: {extraction_ratio:.3f}")
    print(f"Mean frac: {mean_frac:.3f} ± {std_frac:.3f}")
    print(f"Frac range: {min(frac_values):.2f} - {max(frac_values):.2f}")
    print(f"Brain: {output_file}")
    print(f"Mask: {mask_file}")

    info = {
        'method': 'adaptive_slicewise',
        'target_ratio': target_ratio,
        'n_slices': n_slices,
        'slice_stats': slice_stats,
        'total_voxels': int(total_voxels),
        'extraction_ratio': float(extraction_ratio),
        'mean_frac': float(mean_frac),
        'std_frac': float(std_frac)
    }

    return output_file, mask_file, info
