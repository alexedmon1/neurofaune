"""
Skull stripping for functional BOLD data with preprocessing.

Applies N4 bias correction and intensity normalization to a reference volume
to improve contrast before BET skull stripping. The mask is then applied to
the original (non-preprocessed) data.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import subprocess
from scipy import ndimage


def bias_correct_bold(
    input_file: Path,
    output_file: Path
) -> Path:
    """
    N4 bias field correction on BOLD reference volume.

    Parameters
    ----------
    input_file : Path
        Input BOLD reference volume (3D)
    output_file : Path
        Output bias-corrected volume

    Returns
    -------
    Path
        Path to bias-corrected volume
    """
    print("  Running N4 bias correction...")

    # Use ANTs N4BiasFieldCorrection
    cmd = [
        'N4BiasFieldCorrection',
        '-d', '3',
        '-i', str(input_file),
        '-o', str(output_file),
        '-s', '2',  # Shrink factor (faster)
        '-b', '[50]',  # B-spline fitting (smaller for rodent)
        '-c', '[50x50x50,0.0]'  # Convergence
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"    Saved: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"    Warning: N4 failed, using original image")
        print(f"    Error: {e.stderr}")
        # If N4 fails, just copy the input
        import shutil
        shutil.copy(input_file, output_file)
        return output_file


def calculate_center_of_gravity(
    input_file: Path,
    threshold_percentile: float = 10.0
) -> Tuple[float, float, float]:
    """
    Calculate center of gravity from image intensity distribution.

    Uses thresholded intensity distribution to find brain center,
    making BET more robust across different voxel sizes.

    Parameters
    ----------
    input_file : Path
        Input image
    threshold_percentile : float
        Percentile threshold to exclude background (default: 10th percentile)

    Returns
    -------
    Tuple[float, float, float]
        Center of gravity coordinates (x, y, z) in voxel space
    """
    from scipy.ndimage import center_of_mass

    img = nib.load(input_file)
    data = img.get_fdata()

    # Threshold to exclude background
    nonzero = data[data > 0]
    if len(nonzero) > 0:
        threshold = np.percentile(nonzero, threshold_percentile)
        mask = data > threshold

        # Calculate center of gravity
        cog = center_of_mass(data * mask)
        print(f"  Calculated center of gravity: ({cog[0]:.1f}, {cog[1]:.1f}, {cog[2]:.1f})")
    else:
        # Fallback to geometric center if no nonzero voxels
        dims = data.shape
        cog = (dims[0] / 2, dims[1] / 2, dims[2] / 2)
        print(f"  Warning: No nonzero voxels, using geometric center")

    return cog


def normalize_intensity(
    input_file: Path,
    output_file: Path,
    method: str = 'percentile'
) -> Path:
    """
    Normalize intensity values to improve BET performance.

    Parameters
    ----------
    input_file : Path
        Input image
    output_file : Path
        Output normalized image
    method : str
        Normalization method: 'percentile', 'zscore', or 'range'

    Returns
    -------
    Path
        Path to normalized image
    """
    print(f"  Normalizing intensity (method: {method})...")

    img = nib.load(input_file)
    data = img.get_fdata()

    # Get non-zero voxels
    nonzero = data[data > 0]

    if method == 'percentile':
        # Robust normalization using percentiles
        p2 = np.percentile(nonzero, 2)
        p98 = np.percentile(nonzero, 98)

        # Clip and rescale to 0-1000
        data_norm = np.clip(data, p2, p98)
        data_norm = (data_norm - p2) / (p98 - p2) * 1000

    elif method == 'zscore':
        # Z-score normalization
        mean = nonzero.mean()
        std = nonzero.std()
        data_norm = (data - mean) / std * 100 + 500  # Center at 500

    elif method == 'range':
        # Simple min-max to 0-1000
        min_val = nonzero.min()
        max_val = nonzero.max()
        data_norm = (data - min_val) / (max_val - min_val) * 1000

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Preserve zero values (background)
    data_norm[data == 0] = 0

    # Save
    img_norm = nib.Nifti1Image(data_norm, img.affine, img.header)
    nib.save(img_norm, output_file)

    print(f"    Intensity range: [{data_norm[data_norm > 0].min():.1f}, {data_norm[data_norm > 0].max():.1f}]")
    print(f"    Saved: {output_file}")

    return output_file


def run_bet_skull_strip(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    method: str = 'bet',
    frac: float = 0.3,
    use_functional_flag: bool = True,
    use_bias_cleanup: bool = False,
    auto_center: bool = True
) -> Tuple[Path, Path]:
    """
    Run BET skull stripping with automatic center of gravity calculation.

    Parameters
    ----------
    input_file : Path
        Input preprocessed BOLD reference
    output_file : Path
        Output brain file
    mask_file : Path
        Output mask file
    method : str
        'bet' or 'bet4animal'
    frac : float
        Fractional intensity threshold
    use_functional_flag : bool
        Use -F flag for functional optimization (BET only)
    use_bias_cleanup : bool
        Use -B flag for bias field and neck cleanup (BET only)
    auto_center : bool
        Automatically calculate center of gravity (default: True)

    Returns
    -------
    Tuple[Path, Path]
        (brain_file, mask_file)
    """
    print(f"  Running {method} (frac={frac})...")

    if method == 'bet':
        # BET adds .nii.gz, so we need to remove both suffixes from output path
        output_base = str(output_file).replace('.nii.gz', '')

        cmd = [
            'bet',
            str(input_file),
            output_base,  # BET will add .nii.gz
            '-f', str(frac),
            '-m',  # Create mask
            '-n'   # No mesh
        ]

        if auto_center:
            # Calculate center of gravity for robust centering
            cog = calculate_center_of_gravity(input_file)
            cmd.extend(['-c', str(int(cog[0])), str(int(cog[1])), str(int(cog[2]))])
            print(f"    Using automatic center: ({int(cog[0])}, {int(cog[1])}, {int(cog[2])})")
        else:
            cmd.append('-R')  # Use BET's robust center estimation
            print("    Using -R (BET's robust center estimation)")

        if use_functional_flag:
            cmd.append('-F')
            print("    Using -F (functional optimization)")

        if use_bias_cleanup:
            cmd.append('-B')
            print("    Using -B (bias field and neck cleanup)")

    elif method == 'bet4animal':
        # bet4animal adds .nii.gz, so we need to remove both suffixes
        output_base = str(output_file).replace('.nii.gz', '')

        if auto_center:
            # Calculate center of gravity for robust centering
            center = calculate_center_of_gravity(input_file)
            print(f"    Using automatic center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
        else:
            # Use geometric center as fallback
            img = nib.load(input_file)
            dims = img.shape
            center = (dims[0] // 2, dims[1] // 2, dims[2] // 2)
            print(f"    Using geometric center: ({center[0]}, {center[1]}, {center[2]})")

        cmd = [
            'bet4animal',
            str(input_file),
            output_base,
            '-f', str(frac),
            '-c', f"{center[0]:.1f} {center[1]:.1f} {center[2]:.1f}",
            '-r', '125',
            '-x', '1,1,1.5',
            '-w', '2.5',
            '-m'
        ]

    else:
        raise ValueError(f"Unknown method: {method}")

    # Run command - BET may return non-zero even on success
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    # Find mask file (created with _mask suffix by BET)
    bet_mask = output_file.with_name(
        output_file.stem.replace('.nii', '') + '_mask.nii.gz'
    )

    # Check if output files were actually created (more reliable than exit code)
    if not output_file.exists() or not bet_mask.exists():
        print(f"\n✗ Error running {method}:")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Exit code: {result.returncode}")
        if result.stdout:
            print(f"  Stdout: {result.stdout[:500]}")
        if result.stderr:
            print(f"  Stderr: {result.stderr[:500]}")
        print(f"  Expected brain file: {output_file} (exists: {output_file.exists()})")
        print(f"  Expected mask file: {bet_mask} (exists: {bet_mask.exists()})")
        raise RuntimeError(f"{method} failed - output files not created")
    elif result.returncode != 0:
        print(f"  Note: {method} returned exit code {result.returncode} but output files exist (continuing)")

    # Rename mask to expected location if different
    if bet_mask != mask_file:
        import shutil
        shutil.move(str(bet_mask), str(mask_file))

    print(f"    Brain: {output_file}")
    print(f"    Mask: {mask_file}")

    return output_file, mask_file


def clean_mask_morphology(
    mask_file: Path,
    output_file: Path,
    erode_iterations: int = 2,
    dilate_iterations: int = 2,
    fill_holes: bool = True
) -> Path:
    """
    Clean up brain mask with morphological operations.

    Parameters
    ----------
    mask_file : Path
        Input binary mask
    output_file : Path
        Output cleaned mask
    erode_iterations : int
        Number of erosion iterations (removes thin structures)
    dilate_iterations : int
        Number of dilation iterations (restores size)
    fill_holes : bool
        Fill holes in the mask

    Returns
    -------
    Path
        Path to cleaned mask
    """
    print(f"  Cleaning mask with morphology...")

    img = nib.load(mask_file)
    mask_data = img.get_fdata().astype(bool)

    print(f"    Before: {mask_data.sum():,} voxels")

    # Erode to remove thin structures (skull fragments)
    if erode_iterations > 0:
        mask_data = ndimage.binary_erosion(mask_data, iterations=erode_iterations)
        print(f"    After erosion ({erode_iterations}x): {mask_data.sum():,} voxels")

    # Dilate to restore size
    if dilate_iterations > 0:
        mask_data = ndimage.binary_dilation(mask_data, iterations=dilate_iterations)
        print(f"    After dilation ({dilate_iterations}x): {mask_data.sum():,} voxels")

    # Fill holes
    if fill_holes:
        mask_data = ndimage.binary_fill_holes(mask_data)
        print(f"    After filling holes: {mask_data.sum():,} voxels")

    # Save cleaned mask
    img_clean = nib.Nifti1Image(mask_data.astype(np.uint8), img.affine, img.header)
    nib.save(img_clean, output_file)

    print(f"    Saved: {output_file}")

    return output_file


def skull_strip_bold_preprocessed(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    work_dir: Path,
    bet_method: str = 'bet',
    bet_frac: float = 0.3,
    use_functional_flag: bool = True,
    use_bias_cleanup: bool = False,
    auto_center: bool = True,
    norm_method: str = 'percentile',
    clean_mask: bool = True,
    erode_iter: int = 2,
    dilate_iter: int = 2
) -> Tuple[Path, Path, Dict]:
    """
    Skull strip BOLD reference volume with preprocessing.

    Workflow:
    1. N4 bias correction on reference volume
    2. Intensity normalization
    3. BET skull stripping
    4. Optional mask cleanup with morphology
    5. Apply mask to original (non-preprocessed) data

    Parameters
    ----------
    input_file : Path
        Input BOLD reference volume (3D - mean or single TR)
    output_file : Path
        Output brain-extracted file (using original intensities)
    mask_file : Path
        Output brain mask
    work_dir : Path
        Working directory for intermediate files
    bet_method : str
        'bet' or 'bet4animal'
    bet_frac : float
        BET fractional intensity threshold
    use_functional_flag : bool
        Use -F flag for BET functional optimization
    norm_method : str
        Intensity normalization method
    clean_mask : bool
        Apply morphological cleanup to mask
    erode_iter : int
        Erosion iterations for mask cleanup
    dilate_iter : int
        Dilation iterations for mask cleanup

    Returns
    -------
    Tuple[Path, Path, Dict]
        (brain_file, mask_file, info_dict)
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("BOLD SKULL STRIPPING WITH PREPROCESSING")
    print("="*80)
    print(f"Input: {input_file.name}")
    print(f"Method: {bet_method}, frac={bet_frac}")
    print()

    # Step 1: N4 bias correction
    print("Step 1: N4 Bias Correction")
    n4_file = work_dir / 'bold_ref_n4.nii.gz'
    bias_correct_bold(input_file, n4_file)

    # Step 2: Intensity normalization
    print("\nStep 2: Intensity Normalization")
    norm_file = work_dir / 'bold_ref_n4_norm.nii.gz'
    normalize_intensity(n4_file, norm_file, method=norm_method)

    # Step 3: BET skull stripping on preprocessed data
    print("\nStep 3: BET Skull Stripping")
    bet_brain = work_dir / 'bold_ref_preprocessed_brain.nii.gz'
    bet_mask = work_dir / 'bold_ref_preprocessed_mask.nii.gz'
    run_bet_skull_strip(
        norm_file,
        bet_brain,
        bet_mask,
        method=bet_method,
        frac=bet_frac,
        use_functional_flag=use_functional_flag,
        use_bias_cleanup=use_bias_cleanup,
        auto_center=auto_center
    )

    # Step 4: Optional mask cleanup
    if clean_mask:
        print("\nStep 4: Mask Cleanup (Morphology)")
        cleaned_mask = work_dir / 'bold_ref_mask_cleaned.nii.gz'
        clean_mask_morphology(
            bet_mask,
            cleaned_mask,
            erode_iterations=erode_iter,
            dilate_iterations=dilate_iter,
            fill_holes=True
        )
        final_mask = cleaned_mask
    else:
        final_mask = bet_mask

    # Step 5: Apply mask to ORIGINAL (non-preprocessed) data
    print("\nStep 5: Applying Mask to Original Data")
    print("  (Preserving original intensities)")

    orig_img = nib.load(input_file)
    orig_data = orig_img.get_fdata()

    mask_img = nib.load(final_mask)
    mask_data = mask_img.get_fdata().astype(bool)

    brain_data = orig_data * mask_data

    # Save brain-extracted original data
    brain_img = nib.Nifti1Image(brain_data, orig_img.affine, orig_img.header)
    nib.save(brain_img, output_file)

    # Copy final mask to output location
    import shutil
    shutil.copy(final_mask, mask_file)

    # Calculate statistics
    orig_nonzero = (orig_data > 0).sum()
    mask_voxels = mask_data.sum()
    extraction_ratio = mask_voxels / orig_nonzero if orig_nonzero > 0 else 0

    print("\n" + "="*80)
    print("SKULL STRIPPING COMPLETE")
    print("="*80)
    print(f"Brain: {output_file}")
    print(f"Mask: {mask_file}")
    print(f"Mask voxels: {mask_voxels:,}")
    print(f"Extraction ratio: {extraction_ratio:.3f}")

    info = {
        'method': bet_method,
        'frac': bet_frac,
        'mask_voxels': int(mask_voxels),
        'extraction_ratio': float(extraction_ratio),
        'preprocessing': {
            'n4_bias_correction': True,
            'normalization': norm_method,
            'mask_cleanup': clean_mask
        }
    }

    return output_file, mask_file, info
