"""
Unified skull stripping interface with automatic method selection.

Automatically chooses between:
- Adaptive slice-wise BET for partial-coverage data (<10 slices)
- Two-pass Atropos+BET for full-coverage data (≥10 slices)

The two-pass approach for full-coverage:
1. Atropos 5-component segmentation provides rough brain mask and tissue posteriors
2. BET refinement with adaptive frac uses the Atropos mask center of gravity
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import nibabel as nib


# Threshold for switching between methods
SLICE_THRESHOLD = 10


def skull_strip(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    work_dir: Path,
    method: str = 'auto',
    cohort: str = 'p60',
    # Adaptive slice-wise parameters
    target_ratio: float = 0.15,
    frac_range: Tuple[float, float] = (0.30, 0.80),
    frac_step: float = 0.05,
    invert_intensity: bool = False,
    cog_offset_x: Optional[int] = None,
    cog_offset_y: Optional[int] = None,
    # 3D method parameters
    template_file: Optional[Path] = None,
    template_mask: Optional[Path] = None,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Unified skull stripping with automatic method selection.

    Automatically selects the appropriate skull stripping method based on
    the number of slices in the input image:
    - <10 slices: adaptive slice-wise BET (for partial-coverage data like BOLD, MSME)
    - ≥10 slices: standard 3D methods (for full-coverage data like T2w)

    Parameters
    ----------
    input_file : Path
        Input 3D volume
    output_file : Path
        Output brain-extracted volume
    mask_file : Path
        Output brain mask
    work_dir : Path
        Working directory for intermediate files
    method : str
        Skull stripping method:
        - 'auto': automatically select based on slice count (default)
        - 'adaptive': force adaptive slice-wise BET (for <10 slices)
        - 'atropos_bet': force two-pass Atropos+BET (for ≥10 slices)
        - 'atropos': force Atropos-only (no BET refinement)
        - 'bet': force standard 3D BET only
    cohort : str
        Age cohort for 3D methods ('p30', 'p60', 'p90')
    target_ratio : float
        Target brain extraction ratio per slice for adaptive method (0-1)
    frac_range : Tuple[float, float]
        (min_frac, max_frac) to test for adaptive method
    frac_step : float
        Step size for frac testing in adaptive method
    invert_intensity : bool
        If True, invert intensity before BET (T2w → T1w-like contrast)
    cog_offset_x : int, optional
        X offset from image center for COG estimation in adaptive method
    cog_offset_y : int, optional
        Y offset from image center for COG estimation in adaptive method
        Negative values shift COG down (inferior), typical for brain positioning
    template_file : Path, optional
        Atlas template for ANTs brain extraction (3D methods)
    template_mask : Path, optional
        Atlas brain mask for ANTs brain extraction (3D methods)

    Returns
    -------
    Tuple[Path, Path, Dict[str, Any]]
        (brain_file, mask_file, info_dict)
        info_dict contains method-specific information about the extraction
    """
    # Load image to check slice count
    img = nib.load(input_file)
    n_slices = img.shape[2] if len(img.shape) >= 3 else 1

    # Auto-select method based on slice count
    if method == 'auto':
        if n_slices < SLICE_THRESHOLD:
            method = 'adaptive'
            print(f"  Auto-selected adaptive slice-wise BET ({n_slices} slices < {SLICE_THRESHOLD})")
        else:
            method = 'atropos_bet'
            print(f"  Auto-selected two-pass Atropos+BET ({n_slices} slices >= {SLICE_THRESHOLD})")

    # Dispatch to appropriate method
    if method == 'adaptive':
        return _skull_strip_adaptive(
            input_file=input_file,
            output_file=output_file,
            mask_file=mask_file,
            work_dir=work_dir,
            target_ratio=target_ratio,
            frac_range=frac_range,
            frac_step=frac_step,
            invert_intensity=invert_intensity,
            cog_offset_x=cog_offset_x,
            cog_offset_y=cog_offset_y,
        )
    elif method == 'atropos_bet':
        return _skull_strip_atropos_bet(
            input_file=input_file,
            output_file=output_file,
            mask_file=mask_file,
            work_dir=work_dir,
            cohort=cohort,
        )
    elif method == 'atropos':
        return _skull_strip_atropos_only(
            input_file=input_file,
            output_file=output_file,
            mask_file=mask_file,
            work_dir=work_dir,
            cohort=cohort,
        )
    elif method == 'bet':
        return _skull_strip_bet(
            input_file=input_file,
            output_file=output_file,
            mask_file=mask_file,
            work_dir=work_dir,
            cohort=cohort,
        )
    else:
        raise ValueError(f"Unknown skull stripping method: {method}")


def _skull_strip_adaptive(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    work_dir: Path,
    target_ratio: float = 0.15,
    frac_range: Tuple[float, float] = (0.30, 0.80),
    frac_step: float = 0.05,
    invert_intensity: bool = False,
    cog_offset_x: Optional[int] = None,
    cog_offset_y: Optional[int] = None,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Adaptive slice-wise BET for partial-coverage data.

    Best for images with <10 slices where 3D BET fails due to
    non-spherical geometry.
    """
    from neurofaune.preprocess.utils.func.skull_strip_adaptive import skull_strip_adaptive

    brain_file, mask_out, info = skull_strip_adaptive(
        input_file=input_file,
        output_file=output_file,
        mask_file=mask_file,
        work_dir=work_dir,
        target_ratio=target_ratio,
        frac_range=frac_range,
        frac_step=frac_step,
        invert_intensity=invert_intensity,
        use_R_flag=False,
        cog_offset_x=cog_offset_x,
        cog_offset_y=cog_offset_y,
    )

    info['method'] = 'adaptive'
    return brain_file, mask_out, info


def _skull_strip_atropos_bet(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    work_dir: Path,
    cohort: str = 'p60',
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Two-pass Atropos+BET skull stripping for full-coverage data.

    This is the recommended method for ≥10 slices:
    1. Atropos 5-component segmentation provides rough brain mask and COG
    2. BET refinement with adaptive frac uses the Atropos COG for initialization

    Returns posteriors for tissue segmentation reuse.
    """
    import subprocess
    import numpy as np
    from scipy import ndimage

    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create foreground mask (non-zero voxels)
    img = nib.load(input_file)
    img_data = img.get_fdata()
    foreground_mask = (img_data > 0).astype(np.uint8)
    foreground_mask_file = work_dir / f"{input_file.stem}_foreground.nii.gz"
    nib.save(nib.Nifti1Image(foreground_mask, img.affine, img.header), foreground_mask_file)

    # Step 2: Run Atropos 5-component segmentation
    print("  Running Atropos 5-component segmentation...")
    seg_output = work_dir / f"{input_file.stem}_seg.nii.gz"
    posteriors_prefix = work_dir / f"{input_file.stem}_posterior"

    atropos_cmd = [
        'Atropos',
        '-d', '3',
        '-a', str(input_file),
        '-i', 'KMeans[5]',
        '-x', str(foreground_mask_file),
        '-o', f'[{seg_output},{posteriors_prefix}%02d.nii.gz]',
        '-c', '[5,0.0]',
        '-m', '[0.1,1x1x1]',
    ]
    subprocess.run(atropos_cmd, check=True, capture_output=True)

    # Step 3: Analyze posteriors and select brain classes
    # Atropos with KMeans has non-deterministic ordering
    # Exclude both largest (background) and smallest (noise) posteriors by volume
    posteriors = sorted(work_dir.glob(f"{input_file.stem}_posterior*.nii.gz"))

    volumes = []
    for p in posteriors:
        p_data = nib.load(p).get_fdata()
        volumes.append((p_data > 0.5).sum())

    sorted_indices = np.argsort(volumes)
    # Exclude largest (background) and smallest (noise)
    brain_indices = sorted_indices[1:-1]  # Middle 3 classes

    print(f"  Posterior volumes: {volumes}")
    print(f"  Selected brain classes: {brain_indices.tolist()}")

    # Combine brain posteriors into rough mask
    atropos_mask = np.zeros(img_data.shape)
    for i in brain_indices:
        p_data = nib.load(posteriors[i]).get_fdata()
        atropos_mask += (p_data > 0.5).astype(float)
    atropos_mask = (atropos_mask > 0).astype(np.uint8)

    # Save Atropos rough mask
    atropos_mask_file = work_dir / f"{input_file.stem}_atropos_mask.nii.gz"
    nib.save(nib.Nifti1Image(atropos_mask, img.affine, img.header), atropos_mask_file)

    # Step 4: Calculate center of gravity from Atropos mask
    brain_coords = np.argwhere(atropos_mask > 0)
    cog = brain_coords.mean(axis=0)
    print(f"  Atropos COG: [{cog[0]:.1f}, {cog[1]:.1f}, {cog[2]:.1f}]")

    # Step 5: Calculate adaptive BET frac based on contrast
    brain_intensities = img_data[atropos_mask > 0]
    background_mask = (atropos_mask == 0) & (img_data > 0)
    background_intensities = img_data[background_mask] if background_mask.sum() > 0 else brain_intensities

    brain_median = np.median(brain_intensities)
    background_median = np.median(background_intensities)
    contrast = abs(brain_median - background_median) / (brain_median + 1e-6)

    # Map contrast to frac (higher contrast = can use higher frac)
    adaptive_frac = 0.2 + 0.3 * min(contrast, 1.0)
    adaptive_frac = np.clip(adaptive_frac, 0.15, 0.5)
    print(f"  Adaptive BET frac: {adaptive_frac:.3f} (contrast={contrast:.3f})")

    # Step 6: Apply Atropos mask to input, then run BET
    print("  Running BET refinement...")
    masked_input = work_dir / f"{input_file.stem}_atropos_masked.nii.gz"
    masked_data = img_data * atropos_mask
    nib.save(nib.Nifti1Image(masked_data, img.affine, img.header), masked_input)

    bet_output = work_dir / f"{input_file.stem}_bet.nii.gz"
    bet_cmd = [
        'bet',
        str(masked_input),
        str(bet_output),
        '-f', str(adaptive_frac),
        '-m',
        '-c', str(int(cog[0])), str(int(cog[1])), str(int(cog[2])),
    ]
    subprocess.run(bet_cmd, check=True, capture_output=True)

    # Get BET mask
    bet_mask_file = work_dir / f"{input_file.stem}_bet_mask.nii.gz"
    if bet_mask_file.exists():
        final_mask = nib.load(bet_mask_file).get_fdata() > 0
    else:
        # BET might name it differently
        alt_mask = Path(str(bet_output).replace('.nii.gz', '_mask.nii.gz'))
        if alt_mask.exists():
            final_mask = nib.load(alt_mask).get_fdata() > 0
        else:
            final_mask = atropos_mask > 0

    # Step 7: Morphological cleanup
    final_mask = ndimage.binary_fill_holes(final_mask)
    final_mask = ndimage.binary_closing(final_mask, iterations=2)

    # Save final mask
    nib.save(nib.Nifti1Image(final_mask.astype(np.float32), img.affine, img.header), mask_file)

    # Apply mask to original
    brain_data = img_data * final_mask
    nib.save(nib.Nifti1Image(brain_data.astype(np.float32), img.affine, img.header), output_file)

    # Calculate metrics
    total_voxels = np.prod(final_mask.shape)
    brain_voxels = int(final_mask.sum())
    extraction_ratio = brain_voxels / total_voxels

    info = {
        'method': 'atropos_bet',
        'extraction_ratio': extraction_ratio,
        'brain_voxels': brain_voxels,
        'total_voxels': total_voxels,
        'adaptive_frac': float(adaptive_frac),
        'cog': cog.tolist(),
        'posteriors': [str(p) for p in posteriors],
    }

    return output_file, mask_file, info


def _skull_strip_atropos_only(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    work_dir: Path,
    cohort: str = 'p60',
    n_classes: int = 3,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Atropos-only skull stripping (no BET refinement).

    Simpler but less accurate than atropos_bet. Uses ANTs Atropos
    to segment into tissue classes, then creates a brain mask.
    """
    import subprocess
    import numpy as np

    work_dir.mkdir(parents=True, exist_ok=True)

    # Run N4 bias correction first
    n4_output = work_dir / f"{input_file.stem}_n4.nii.gz"
    n4_cmd = [
        'N4BiasFieldCorrection',
        '-d', '3',
        '-i', str(input_file),
        '-o', str(n4_output),
        '-s', '2',
        '-c', '[50x50x50,0.0]',
        '-b', '[200]',
    ]
    subprocess.run(n4_cmd, check=True, capture_output=True)

    # Run Atropos segmentation
    seg_output = work_dir / f"{input_file.stem}_seg.nii.gz"
    posteriors_prefix = work_dir / f"{input_file.stem}_posterior"

    atropos_cmd = [
        'Atropos',
        '-d', '3',
        '-a', str(n4_output),
        '-i', f'KMeans[{n_classes}]',
        '-o', f'[{seg_output},{posteriors_prefix}%02d.nii.gz]',
        '-c', '[5,0.0]',
        '-m', '[0.1,1x1x1]',
        '-x', 'NULL',
    ]
    subprocess.run(atropos_cmd, check=True, capture_output=True)

    # Load segmentation and create brain mask
    # For T2w: class 1 = background/skull (dark), classes 2+ = brain tissue
    seg_img = nib.load(seg_output)
    seg_data = seg_img.get_fdata()

    # Brain = all non-background classes (label > 1)
    brain_mask = (seg_data > 1).astype(np.float32)

    # Clean up mask with morphological operations
    from scipy import ndimage
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    brain_mask = ndimage.binary_opening(brain_mask, iterations=1)
    brain_mask = ndimage.binary_closing(brain_mask, iterations=2)

    # Save mask
    mask_img = nib.Nifti1Image(brain_mask.astype(np.float32), seg_img.affine, seg_img.header)
    nib.save(mask_img, mask_file)

    # Apply mask to original image
    orig_img = nib.load(input_file)
    orig_data = orig_img.get_fdata()
    brain_data = orig_data * brain_mask

    brain_img = nib.Nifti1Image(brain_data.astype(np.float32), orig_img.affine, orig_img.header)
    nib.save(brain_img, output_file)

    # Calculate metrics
    total_voxels = np.prod(brain_mask.shape)
    brain_voxels = int(brain_mask.sum())
    extraction_ratio = brain_voxels / total_voxels

    info = {
        'method': 'atropos',
        'n_classes': n_classes,
        'extraction_ratio': extraction_ratio,
        'brain_voxels': brain_voxels,
        'total_voxels': total_voxels,
    }

    return output_file, mask_file, info


def _skull_strip_bet(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    work_dir: Path,
    cohort: str = 'p60',
    frac: float = 0.3,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Standard 3D BET skull stripping for full-coverage data.

    Uses FSL BET with rodent-appropriate parameters.
    """
    import subprocess
    import numpy as np

    work_dir.mkdir(parents=True, exist_ok=True)

    # Adjust frac based on cohort (younger = smaller brain relative to skull)
    cohort_frac = {
        'p30': 0.25,
        'p60': 0.30,
        'p90': 0.35,
    }
    frac = cohort_frac.get(cohort, frac)

    # Run BET
    bet_cmd = [
        'bet',
        str(input_file),
        str(output_file),
        '-f', str(frac),
        '-m',  # Generate mask
        '-R',  # Robust center estimation
    ]
    subprocess.run(bet_cmd, check=True, capture_output=True)

    # BET creates mask with _mask suffix
    bet_mask = Path(str(output_file).replace('.nii.gz', '_mask.nii.gz'))
    if bet_mask.exists() and bet_mask != mask_file:
        bet_mask.rename(mask_file)

    # Calculate metrics
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata() > 0
    total_voxels = np.prod(mask_data.shape)
    brain_voxels = int(mask_data.sum())
    extraction_ratio = brain_voxels / total_voxels

    info = {
        'method': 'bet',
        'frac': frac,
        'cohort': cohort,
        'extraction_ratio': extraction_ratio,
        'brain_voxels': brain_voxels,
        'total_voxels': total_voxels,
    }

    return output_file, mask_file, info


def get_recommended_method(n_slices: int) -> str:
    """
    Get the recommended skull stripping method based on slice count.

    Parameters
    ----------
    n_slices : int
        Number of slices in the image

    Returns
    -------
    str
        Recommended method name ('adaptive' or 'atropos_bet')
    """
    if n_slices < SLICE_THRESHOLD:
        return 'adaptive'
    else:
        return 'atropos_bet'
