"""
Two-pass adaptive skull stripping for functional BOLD data.

Pass 1: Adaptive per-slice BET with -R flag to get initial brain extraction
Pass 2: Per-slice refinement with high frac on extracted brain
"""

from pathlib import Path
from typing import Tuple, Optional
import tempfile
import subprocess
import numpy as np
import nibabel as nib
from scipy import ndimage


def calculate_slice_cog(slice_data: np.ndarray) -> Tuple[float, float]:
    """Calculate 2D center of gravity for a slice."""
    # Threshold at 10th percentile to exclude background
    nonzero = slice_data[slice_data > 0]
    if len(nonzero) == 0:
        # Fallback to geometric center
        return (slice_data.shape[0] / 2, slice_data.shape[1] / 2)

    threshold = np.percentile(nonzero, 10)
    mask = slice_data > threshold

    if mask.sum() == 0:
        return (slice_data.shape[0] / 2, slice_data.shape[1] / 2)

    # Calculate weighted COG
    cog = ndimage.center_of_mass(slice_data * mask)
    return (cog[0], cog[1])


def refine_slice_with_bet(
    slice_data: np.ndarray,
    slice_mask: np.ndarray,
    work_dir: Path,
    refinement_frac: float = 0.85
) -> Tuple[Optional[np.ndarray], int]:
    """
    Refine a single slice using BET with high frac on extracted brain.

    Parameters
    ----------
    slice_data : np.ndarray
        2D slice data (normalized)
    slice_mask : np.ndarray
        Initial mask from pass 1
    work_dir : Path
        Working directory
    refinement_frac : float
        High frac value for refinement (default 0.85)

    Returns
    -------
    Tuple[Optional[np.ndarray], int]
        (refined_mask, n_voxels) or (None, 0) if failed
    """
    # Apply pass 1 mask to get extracted brain
    brain_data = slice_data * slice_mask

    # Calculate COG on extracted brain
    brain_voxels = brain_data[slice_mask]
    if len(brain_voxels) == 0:
        return None, 0

    threshold = np.percentile(brain_voxels, 10)
    brain_mask_weighted = (brain_data > threshold) & slice_mask

    if brain_mask_weighted.sum() == 0:
        return None, 0

    cog = ndimage.center_of_mass(brain_data * brain_mask_weighted)

    # Create temporary 3D image (single slice) with extracted brain
    slice_3d = brain_data[:, :, np.newaxis]
    affine = np.eye(4)
    img = nib.Nifti1Image(slice_3d, affine)

    # Save temporary input
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False, dir=work_dir) as f:
        input_file = Path(f.name)
    nib.save(img, input_file)

    # BET output
    with tempfile.NamedTemporaryFile(suffix='', delete=False, dir=work_dir) as f:
        output_base = str(f.name)

    # Run BET with high frac and COG from extracted brain
    cmd = [
        'bet',
        str(input_file),
        output_base,
        '-f', str(refinement_frac),
        '-m', '-n',
        '-c', str(int(cog[0])), str(int(cog[1])), '0',
        '-F'
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    # Check for mask
    mask_file = Path(output_base + '_mask.nii.gz')
    if not mask_file.exists():
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


def skull_strip_two_pass(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    work_dir: Path,
    target_ratio: float = 0.15,
    frac_range: Tuple[float, float] = (0.30, 0.90),
    frac_step: float = 0.05,
    refinement_frac: float = 0.85
) -> Tuple[Path, Path, dict]:
    """
    Two-pass adaptive skull stripping.

    Pass 1: Adaptive per-slice frac optimization with -R flag
    Pass 2: Per-slice refinement with high frac on extracted brain

    Parameters
    ----------
    input_file : Path
        Input 3D reference volume (preprocessed: N4 + normalized)
    output_file : Path
        Output brain-extracted volume
    mask_file : Path
        Output mask file
    work_dir : Path
        Working directory
    target_ratio : float
        Target extraction ratio per slice for pass 1
    frac_range : Tuple[float, float]
        (min_frac, max_frac) to test in pass 1
    frac_step : float
        Step size for frac testing in pass 1
    refinement_frac : float
        High frac value for pass 2 refinement

    Returns
    -------
    Tuple[Path, Path, dict]
        (brain_file, mask_file, info_dict)
    """
    # Import the pass 1 function
    from neurofaune.preprocess.utils.func.skull_strip_adaptive import skull_strip_adaptive

    print("\n" + "="*80)
    print("TWO-PASS ADAPTIVE SKULL STRIPPING")
    print("="*80)

    # Create subdirectories
    pass1_dir = work_dir / 'pass1'
    pass2_dir = work_dir / 'pass2'
    pass1_dir.mkdir(parents=True, exist_ok=True)
    pass2_dir.mkdir(parents=True, exist_ok=True)

    # PASS 1: Adaptive skull stripping with -R flag
    print("\n" + "="*80)
    print("PASS 1: Adaptive per-slice optimization with -R flag")
    print("="*80)

    pass1_brain = work_dir / 'pass1_brain.nii.gz'
    pass1_mask = work_dir / 'pass1_mask.nii.gz'

    brain1, mask1, info1 = skull_strip_adaptive(
        input_file=input_file,
        output_file=pass1_brain,
        mask_file=pass1_mask,
        work_dir=pass1_dir,
        target_ratio=target_ratio,
        frac_range=frac_range,
        frac_step=frac_step,
        invert_intensity=False,
        use_R_flag=True
    )

    print(f"\nPass 1 Results:")
    print(f"  Extraction: {info1['extraction_ratio']*100:.1f}%")
    print(f"  Mean frac: {info1['mean_frac']:.2f} ± {info1['std_frac']:.2f}")

    # PASS 2: Per-slice refinement with high frac
    print("\n" + "="*80)
    print(f"PASS 2: Per-slice refinement with frac={refinement_frac}")
    print("="*80)

    # Load input and pass 1 results
    input_img = nib.load(input_file)
    input_data = input_img.get_fdata()

    mask1_img = nib.load(pass1_mask)
    mask1_data = mask1_img.get_fdata().astype(bool)

    n_slices = input_data.shape[2]
    print(f"\nRefining {n_slices} axial slices...")

    # Initialize pass 2 mask
    mask2_data = np.zeros_like(mask1_data, dtype=bool)

    slice_stats = []
    for z in range(n_slices):
        slice_data = input_data[:, :, z]
        slice_mask1 = mask1_data[:, :, z]

        # Skip if pass 1 didn't extract anything
        if slice_mask1.sum() == 0:
            print(f"  Slice {z}: No pass 1 extraction, skipping")
            continue

        # Refine this slice
        slice_mask2, n_voxels = refine_slice_with_bet(
            slice_data, slice_mask1, pass2_dir, refinement_frac
        )

        if slice_mask2 is not None:
            mask2_data[:, :, z] = slice_mask2

            total_voxels = np.prod(slice_data.shape)
            extraction = n_voxels / total_voxels

            pass1_voxels = slice_mask1.sum()
            reduction = pass1_voxels - n_voxels
            reduction_pct = (reduction / pass1_voxels) * 100 if pass1_voxels > 0 else 0

            print(f"  Slice {z}: {n_voxels} voxels ({extraction*100:.1f}%) | "
                  f"Reduction: {reduction} ({reduction_pct:.1f}%)")

            slice_stats.append({
                'slice': z,
                'pass1_voxels': pass1_voxels,
                'pass2_voxels': n_voxels,
                'reduction': reduction,
                'reduction_pct': reduction_pct
            })
        else:
            print(f"  Slice {z}: Refinement failed, keeping pass 1 mask")
            mask2_data[:, :, z] = slice_mask1

    # Calculate overall statistics
    total_voxels = np.prod(mask2_data.shape)
    n_voxels_pass2 = mask2_data.sum()
    extraction_pass2 = n_voxels_pass2 / total_voxels

    total_reduction = mask1_data.sum() - n_voxels_pass2
    reduction_pct = (total_reduction / mask1_data.sum()) * 100 if mask1_data.sum() > 0 else 0

    print(f"\n{'='*80}")
    print("PASS 2 COMPLETE")
    print(f"{'='*80}")
    print(f"Total mask voxels: {n_voxels_pass2:,}")
    print(f"Overall extraction ratio: {extraction_pass2:.3f}")
    print(f"Reduction from pass 1: {total_reduction:,} voxels ({reduction_pct:.1f}%)")

    # Save final mask
    mask2_img = nib.Nifti1Image(mask2_data.astype(np.float32), input_img.affine, input_img.header)
    nib.save(mask2_img, mask_file)

    # Apply mask to original data to create brain
    brain_data = input_data * mask2_data
    brain_img = nib.Nifti1Image(brain_data, input_img.affine, input_img.header)
    nib.save(brain_img, output_file)

    # Compile info
    info = {
        'pass1_extraction': info1['extraction_ratio'],
        'pass2_extraction': extraction_pass2,
        'pass1_voxels': int(mask1_data.sum()),
        'pass2_voxels': int(n_voxels_pass2),
        'reduction_voxels': int(total_reduction),
        'reduction_pct': reduction_pct,
        'pass1_mean_frac': info1['mean_frac'],
        'pass1_std_frac': info1['std_frac'],
        'refinement_frac': refinement_frac,
        'slice_stats': slice_stats
    }

    return output_file, mask_file, info
