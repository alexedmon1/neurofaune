"""
DWI/DTI preprocessing utilities.

This module provides functions for handling diffusion MRI data,
including format conversions, gradient table operations, and DTI fitting.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def convert_5d_to_4d(
    input_file: Path,
    output_file: Path,
    method: str = 'mean'
) -> Path:
    """
    Convert 5D DWI data to 4D by averaging or selecting across 5th dimension.

    Some Bruker acquisitions produce 5D data (x, y, z, directions, averages)
    instead of standard 4D (x, y, z, directions). This function handles the
    conversion by averaging across the 5th dimension.

    Parameters
    ----------
    input_file : Path
        Input 5D NIfTI file
    output_file : Path
        Output 4D NIfTI file
    method : str
        Method for combining 5th dimension:
        - 'mean': Average across 5th dimension (default)
        - 'first': Take first volume
        - 'sum': Sum across 5th dimension
        - 'rms': Root mean square

    Returns
    -------
    Path
        Path to output 4D file

    Examples
    --------
    >>> from pathlib import Path
    >>> dwi_4d = convert_5d_to_4d(
    ...     Path('dwi_5d.nii.gz'),
    ...     Path('dwi_4d.nii.gz'),
    ...     method='mean'
    ... )
    """
    print(f"Converting 5D to 4D using method: {method}")

    # Load 5D data
    img = nib.load(input_file)
    data = img.get_fdata()

    print(f"  Input shape: {data.shape}")

    if len(data.shape) == 4:
        print("  Data is already 4D, copying to output")
        nib.save(img, output_file)
        return output_file

    if len(data.shape) != 5:
        raise ValueError(f"Expected 4D or 5D data, got shape: {data.shape}")

    # Convert based on method
    if method == 'mean':
        data_4d = np.mean(data, axis=4)
    elif method == 'first':
        data_4d = data[..., 0]
    elif method == 'sum':
        data_4d = np.sum(data, axis=4)
    elif method == 'rms':
        data_4d = np.sqrt(np.mean(data**2, axis=4))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mean', 'first', 'sum', or 'rms'")

    print(f"  Output shape: {data_4d.shape}")

    # Create new image with 4D data
    img_4d = nib.Nifti1Image(data_4d, img.affine, img.header)

    # Update header for 4D
    img_4d.header.set_data_shape(data_4d.shape)

    # Save
    nib.save(img_4d, output_file)
    print(f"  Saved 4D data to: {output_file}")

    return output_file


def round_bvals_to_shells(
    bvals: np.ndarray,
    b0_threshold: float = 50.0,
    shell_tolerance: float = 100.0,
) -> np.ndarray:
    """
    Round b-values to discrete shells for eddy compatibility.

    Bruker effective b-values vary slightly per direction (e.g. 505, 507, 509
    instead of a uniform b=500).  FSL eddy requires discrete shells.  This
    function clusters b-values within *shell_tolerance* of each other and
    replaces each with the rounded shell mean.

    Parameters
    ----------
    bvals : np.ndarray
        Raw b-values (1-D).
    b0_threshold : float
        Values below this are set to 0 (b0 volumes).
    shell_tolerance : float
        Maximum spread within a shell (default 100 s/mm²).

    Returns
    -------
    np.ndarray
        Shelled b-values (same length as input).
    """
    shelled = bvals.copy()

    # b0 volumes
    shelled[bvals < b0_threshold] = 0.0

    # Cluster non-b0 values into shells by greedy grouping
    non_b0_mask = bvals >= b0_threshold
    if not non_b0_mask.any():
        return shelled

    unique_sorted = np.sort(np.unique(bvals[non_b0_mask]))
    shells: list[list[float]] = []
    current_shell = [unique_sorted[0]]

    for bv in unique_sorted[1:]:
        if bv - current_shell[0] <= shell_tolerance:
            current_shell.append(bv)
        else:
            shells.append(current_shell)
            current_shell = [bv]
    shells.append(current_shell)

    # Map each raw b-value to its rounded shell mean
    for shell_vals in shells:
        shell_mean = round(np.mean(shell_vals))
        for bv in shell_vals:
            shelled[bvals == bv] = shell_mean

    return shelled


def validate_gradient_table(
    bval_file: Path,
    bvec_file: Path,
    n_volumes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and load gradient table (bvals and bvecs).

    Parameters
    ----------
    bval_file : Path
        Path to bval file (FSL format)
    bvec_file : Path
        Path to bvec file (FSL format, 3xN)
    n_volumes : int
        Expected number of volumes

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        bvals (1D array) and bvecs (3xN array)

    Raises
    ------
    ValueError
        If gradient table is invalid or doesn't match number of volumes
    """
    # Load bvals
    bvals = np.loadtxt(bval_file)
    if bvals.ndim > 1:
        bvals = bvals.flatten()

    # Load bvecs
    bvecs = np.loadtxt(bvec_file)
    if bvecs.shape[0] != 3:
        if bvecs.shape[1] == 3:
            bvecs = bvecs.T
        else:
            raise ValueError(f"bvecs should be 3xN, got shape: {bvecs.shape}")

    # Check dimensions match
    if len(bvals) != n_volumes:
        raise ValueError(f"Number of bvals ({len(bvals)}) doesn't match volumes ({n_volumes})")

    if bvecs.shape[1] != n_volumes:
        raise ValueError(f"Number of bvecs ({bvecs.shape[1]}) doesn't match volumes ({n_volumes})")

    # Check for b0 volumes
    b0_threshold = 50  # Common threshold
    n_b0 = np.sum(bvals < b0_threshold)
    n_dw = np.sum(bvals >= b0_threshold)

    print(f"Gradient table validation:")
    print(f"  Total volumes: {n_volumes}")
    print(f"  b0 volumes: {n_b0} (b < {b0_threshold})")
    print(f"  DW volumes: {n_dw} (b >= {b0_threshold})")
    print(f"  b-value range: [{bvals.min():.1f}, {bvals.max():.1f}]")
    print(f"  Unique b-values: {np.unique(bvals.astype(int))}")

    if n_b0 == 0:
        raise ValueError("No b0 volumes found (b < 50)")

    # Normalize gradient vectors (should have unit length for non-b0)
    for i in range(bvecs.shape[1]):
        if bvals[i] >= b0_threshold:
            norm = np.linalg.norm(bvecs[:, i])
            if norm > 0:
                bvecs[:, i] = bvecs[:, i] / norm

    # Round b-values to discrete shells (Bruker effective b-values vary per
    # direction; eddy requires discrete shells)
    bvals_shelled = round_bvals_to_shells(bvals)
    unique_shells = sorted(set(bvals_shelled.astype(int)))
    if not np.array_equal(bvals.astype(int), bvals_shelled.astype(int)):
        print(f"  Rounded b-values to shells: {unique_shells}")
    bvals = bvals_shelled

    return bvals, bvecs


def normalize_dwi_intensity(
    input_file: Path,
    output_file: Path,
    target_max: float = 10000.0,
    percentile_max: float = 99.5,
    percentile_min: float = 0.5
) -> Tuple[Path, dict]:
    """
    Normalize DWI intensity to a consistent range for robust brain extraction.

    Different Bruker ParaVision reconstruction settings can result in vastly
    different intensity scales (e.g., 0-50 vs 0-300000) due to different
    VisuCoreDataSlope values applied during reconstruction. This causes
    BET and other tools to fail on low-intensity data.

    This function normalizes all DWI data to a consistent intensity range,
    enabling reliable brain extraction regardless of the original acquisition
    or reconstruction settings.

    Parameters
    ----------
    input_file : Path
        Input DWI NIfTI file (3D or 4D)
    output_file : Path
        Output normalized DWI NIfTI file
    target_max : float
        Target maximum intensity after normalization (default: 10000)
        This value is chosen to work well with FSL BET.
    percentile_max : float
        Upper percentile for robust scaling (default: 99.5)
        Values above this percentile are clipped to avoid outlier influence.
    percentile_min : float
        Lower percentile for robust scaling (default: 0.5)
        Values below this percentile are set to 0.

    Returns
    -------
    Tuple[Path, dict]
        Path to normalized file and dict with normalization parameters:
        - original_min, original_max: Original intensity range
        - original_p_min, original_p_max: Original percentile values
        - scale_factor: Multiplicative scaling factor applied
        - was_normalized: Whether normalization was needed

    Notes
    -----
    The normalization applies the following transformation:
    1. Compute percentile-based intensity range (robust to outliers)
    2. Clip values to percentile range
    3. Scale linearly to [0, target_max]

    This is applied identically to all volumes in 4D data to preserve
    relative signal differences between diffusion directions.

    Examples
    --------
    >>> normalized, params = normalize_dwi_intensity(
    ...     Path('dwi_raw.nii.gz'),
    ...     Path('dwi_normalized.nii.gz')
    ... )
    >>> print(f"Scale factor: {params['scale_factor']:.2f}")
    """
    img = nib.load(input_file)
    data = img.get_fdata()

    # Compute statistics on non-zero values (exclude background)
    nonzero_data = data[data > 0]

    if len(nonzero_data) == 0:
        print("WARNING: All voxels are zero, skipping normalization")
        nib.save(img, output_file)
        return output_file, {'was_normalized': False, 'reason': 'all_zeros'}

    original_min = float(data.min())
    original_max = float(data.max())
    p_min = float(np.percentile(nonzero_data, percentile_min))
    p_max = float(np.percentile(nonzero_data, percentile_max))

    print(f"DWI intensity normalization:")
    print(f"  Original range: {original_min:.2f} - {original_max:.2f}")
    print(f"  Percentile range ({percentile_min}-{percentile_max}%): {p_min:.2f} - {p_max:.2f}")

    # Check if normalization is needed
    # If data is already in a reasonable range (e.g., max > 1000), may not need it
    # But we normalize anyway for consistency
    if p_max < 1.0:
        print(f"  WARNING: Very low intensity data (p{percentile_max}={p_max:.4f})")
        print(f"  This likely indicates normalized/scaled Bruker data")

    # Apply normalization
    # Clip to percentile range, then scale to [0, target_max]
    normalized_data = data.copy()

    # Set values below p_min to 0 (background)
    normalized_data[normalized_data < p_min] = 0

    # Clip values above p_max
    normalized_data = np.clip(normalized_data, 0, p_max)

    # Scale to target range
    if p_max > p_min:
        scale_factor = target_max / (p_max - p_min)
        normalized_data = (normalized_data - p_min) * scale_factor
        normalized_data[data < p_min] = 0  # Ensure background stays at 0
    else:
        scale_factor = 1.0
        print("  WARNING: Cannot normalize - min equals max")

    print(f"  Normalized range: {normalized_data.min():.2f} - {normalized_data.max():.2f}")
    print(f"  Scale factor: {scale_factor:.4f}")

    # Save normalized image
    normalized_img = nib.Nifti1Image(
        normalized_data.astype(np.float32),
        img.affine,
        img.header
    )
    nib.save(normalized_img, output_file)
    print(f"  Saved normalized DWI to: {output_file}")

    params = {
        'was_normalized': True,
        'original_min': original_min,
        'original_max': original_max,
        'original_p_min': p_min,
        'original_p_max': p_max,
        'target_max': target_max,
        'scale_factor': scale_factor,
        'percentile_min': percentile_min,
        'percentile_max': percentile_max
    }

    return output_file, params


def extract_b0_volume(
    dwi_file: Path,
    bval_file: Path,
    output_file: Path,
    b0_threshold: float = 50.0
) -> Path:
    """
    Extract first b0 volume from DWI data.

    Parameters
    ----------
    dwi_file : Path
        4D DWI NIfTI file
    bval_file : Path
        bval file
    output_file : Path
        Output 3D b0 NIfTI file
    b0_threshold : float
        Maximum b-value to consider as b0 (default: 50)

    Returns
    -------
    Path
        Path to extracted b0 volume

    Examples
    --------
    >>> b0_vol = extract_b0_volume(
    ...     Path('dwi.nii.gz'),
    ...     Path('dwi.bval'),
    ...     Path('b0.nii.gz')
    ... )
    """
    # Load data
    img = nib.load(dwi_file)
    data = img.get_fdata()

    # Load bvals
    bvals = np.loadtxt(bval_file)
    if bvals.ndim > 1:
        bvals = bvals.flatten()

    # Find b0 volumes
    b0_indices = np.where(bvals < b0_threshold)[0]

    if len(b0_indices) == 0:
        raise ValueError(f"No b0 volumes found (b < {b0_threshold})")

    print(f"Extracting b0 volume:")
    print(f"  Found {len(b0_indices)} b0 volume(s)")
    print(f"  Using first b0 at index {b0_indices[0]}")

    # Extract first b0
    if len(data.shape) == 4:
        b0_data = data[..., b0_indices[0]]
    else:
        raise ValueError(f"Expected 4D data, got shape: {data.shape}")

    # Save
    b0_img = nib.Nifti1Image(b0_data, img.affine, img.header)
    nib.save(b0_img, output_file)

    print(f"  Saved b0 volume to: {output_file}")

    return output_file


def create_brain_mask_from_b0(
    b0_file: Path,
    output_mask: Path,
    frac: float = 0.3
) -> Path:
    """
    Create brain mask from b0 volume using FSL BET.

    Parameters
    ----------
    b0_file : Path
        b0 volume
    output_mask : Path
        Output mask file
    frac : float
        BET fractional intensity threshold (default: 0.3)

    Returns
    -------
    Path
        Path to brain mask

    Examples
    --------
    >>> mask = create_brain_mask_from_b0(
    ...     Path('b0.nii.gz'),
    ...     Path('brain_mask.nii.gz'),
    ...     frac=0.3
    ... )
    """
    from nipype.interfaces import fsl

    print(f"Creating brain mask from b0 using BET (frac={frac})")

    bet = fsl.BET()
    bet.inputs.in_file = str(b0_file)
    bet.inputs.out_file = str(output_mask.parent / output_mask.stem.replace('.nii', ''))
    bet.inputs.mask = True
    bet.inputs.frac = frac
    bet.inputs.robust = True

    result = bet.run()

    # BET creates files with _mask suffix
    mask_file = output_mask.parent / (output_mask.stem.replace('.nii', '') + '_mask.nii.gz')

    if mask_file.exists() and mask_file != output_mask:
        import shutil
        shutil.move(mask_file, output_mask)

    print(f"  Created brain mask: {output_mask}")

    return output_mask


def create_brain_mask_atropos(
    b0_file: Path,
    output_mask: Path,
    work_dir: Path,
    n_classes: int = 3,
    bet_frac: float = 0.3
) -> Path:
    """
    Create brain mask from b0 volume using Atropos + BET two-pass approach.

    This method is more robust than BET alone for DWI data with unusual intensity
    ranges (e.g., from different Bruker ParaVision reconstruction settings).

    Parameters
    ----------
    b0_file : Path
        b0 volume (should be intensity-normalized for best results)
    output_mask : Path
        Output mask file
    work_dir : Path
        Working directory for intermediate files
    n_classes : int
        Number of classes for Atropos segmentation (default: 3)
        Classes typically represent: background/noise, intermediate tissue, brain
    bet_frac : float
        BET fractional intensity threshold for refinement (default: 0.3)

    Returns
    -------
    Path
        Path to brain mask

    Notes
    -----
    The method works as follows:
    1. Create initial mask using intensity thresholding
    2. Run Atropos K-means segmentation with n_classes
    3. Use the brightest class as initial brain estimate
    4. Calculate center of gravity from Atropos brain mask
    5. Run BET on Atropos-masked image with COG to refine edges

    This two-pass approach (Atropos + BET) is particularly useful for
    partial-brain DWI acquisitions (e.g., hippocampal slices) where
    BET alone often fails.
    """
    import subprocess
    from nipype.interfaces import fsl

    print(f"Creating brain mask using Atropos + BET two-pass approach")
    print(f"  Step 1: Atropos {n_classes}-class segmentation")

    # Load b0 to create initial mask
    img = nib.load(b0_file)
    data = img.get_fdata()

    # Create initial mask using threshold (exclude obvious background)
    nonzero = data[data > 0]
    if len(nonzero) == 0:
        raise ValueError("b0 image has no non-zero voxels")

    threshold = np.percentile(nonzero, 5)  # 5th percentile of non-zero
    initial_mask = (data > threshold).astype(np.uint8)

    # Save initial mask
    initial_mask_file = work_dir / 'atropos_initial_mask.nii.gz'
    nib.save(nib.Nifti1Image(initial_mask, img.affine, img.header), initial_mask_file)
    print(f"    Initial mask (threshold): {int(np.sum(initial_mask)):,} voxels")

    # Run Atropos segmentation
    output_prefix = work_dir / 'atropos_dwi_'
    seg_file = work_dir / 'atropos_dwi_seg.nii.gz'

    cmd = [
        'Atropos',
        '-d', '3',
        '-a', str(b0_file),
        '-x', str(initial_mask_file),
        '-i', f'KMeans[{n_classes}]',
        '-o', f'[{seg_file},{output_prefix}prob%02d.nii.gz]',
        '-v', '0'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Atropos segmentation failed: {result.stderr}")

    # Load segmentation and identify brain class (brightest)
    seg_img = nib.load(seg_file)
    seg_data = seg_img.get_fdata()

    # Find mean intensity for each class to identify brain (brightest)
    class_means = {}
    for label in range(1, n_classes + 1):
        mask = seg_data == label
        if np.sum(mask) > 0:
            class_means[label] = data[mask].mean()
            n_vox = int(np.sum(mask))
            print(f"    Class {label}: {n_vox:,} voxels, mean intensity = {class_means[label]:.1f}")

    # Brain is the brightest class
    brain_class = max(class_means, key=class_means.get)
    print(f"    Using class {brain_class} as brain (brightest)")

    # Create Atropos brain mask
    atropos_mask = (seg_data == brain_class).astype(np.uint8)
    n_atropos_voxels = int(np.sum(atropos_mask))
    print(f"    Atropos brain mask: {n_atropos_voxels:,} voxels")

    # Save Atropos mask for reference
    atropos_mask_file = work_dir / 'atropos_brain_mask.nii.gz'
    nib.save(nib.Nifti1Image(atropos_mask, seg_img.affine, seg_img.header), atropos_mask_file)

    # Step 2: Calculate center of gravity from Atropos mask
    print(f"  Step 2: Calculate center of gravity from Atropos mask")
    brain_coords = np.argwhere(atropos_mask > 0)
    center_of_gravity = brain_coords.mean(axis=0)
    print(f"    COG (voxels): [{center_of_gravity[0]:.1f}, {center_of_gravity[1]:.1f}, {center_of_gravity[2]:.1f}]")

    # Step 3: Create Atropos-masked b0 for BET
    print(f"  Step 3: BET refinement (frac={bet_frac})")
    masked_b0_data = data * atropos_mask
    masked_b0_file = work_dir / 'b0_atropos_masked.nii.gz'
    nib.save(nib.Nifti1Image(masked_b0_data, img.affine, img.header), masked_b0_file)

    # Step 4: Run BET with COG from Atropos mask
    bet_output = work_dir / 'bet_refined'

    bet = fsl.BET()
    bet.inputs.in_file = str(masked_b0_file)
    bet.inputs.out_file = str(bet_output)
    bet.inputs.mask = True
    bet.inputs.frac = bet_frac
    bet.inputs.center = [int(center_of_gravity[0]), int(center_of_gravity[1]), int(center_of_gravity[2])]
    bet.inputs.robust = True

    bet_result = bet.run()

    # Get the mask file
    bet_mask_file = work_dir / 'bet_refined_mask.nii.gz'

    if bet_mask_file.exists():
        # Load and count voxels
        final_mask = nib.load(bet_mask_file)
        final_mask_data = final_mask.get_fdata()
        n_final_voxels = int(np.sum(final_mask_data > 0))
        print(f"    BET refined mask: {n_final_voxels:,} voxels")

        # Copy to output location
        import shutil
        shutil.copy(bet_mask_file, output_mask)
    else:
        # Fall back to Atropos mask if BET fails
        print(f"    WARNING: BET refinement failed, using Atropos mask")
        import shutil
        shutil.copy(atropos_mask_file, output_mask)
        n_final_voxels = n_atropos_voxels

    print(f"  Final brain mask: {output_mask}")
    print(f"  Voxel reduction: {n_atropos_voxels:,} -> {n_final_voxels:,} ({100*(n_atropos_voxels-n_final_voxels)/n_atropos_voxels:.1f}% removed)")

    return output_mask


def check_dwi_data_quality(
    dwi_file: Path,
    mask_file: Optional[Path] = None
) -> dict:
    """
    Perform basic quality checks on DWI data.

    Parameters
    ----------
    dwi_file : Path
        DWI NIfTI file
    mask_file : Path, optional
        Brain mask for computing statistics

    Returns
    -------
    dict
        Dictionary with QC metrics
    """
    img = nib.load(dwi_file)
    data = img.get_fdata()

    qc = {
        'shape': list(data.shape),
        'voxel_size': [float(v) for v in img.header.get_zooms()[:3]],
        'n_volumes': int(data.shape[3]) if len(data.shape) == 4 else 1
    }

    if mask_file and mask_file.exists():
        mask = nib.load(mask_file).get_fdata() > 0
        masked_data = data[mask]

        qc['mean_signal'] = float(np.mean(masked_data))
        qc['std_signal'] = float(np.std(masked_data))
        qc['snr_estimate'] = qc['mean_signal'] / qc['std_signal'] if qc['std_signal'] > 0 else 0

    # Check for common issues
    qc['has_nan'] = bool(np.any(np.isnan(data)))
    qc['has_inf'] = bool(np.any(np.isinf(data)))
    qc['has_negative'] = bool(np.any(data < 0))

    return qc


def pad_slices_for_eddy(
    input_file: Path,
    output_file: Path,
    n_pad: int = 2,
    method: str = 'reflect'
) -> Tuple[Path, int]:
    """
    Pad DWI volume with extra slices to prevent edge slice loss during eddy.

    Eddy's motion correction can cause edge slices to be interpolated from
    outside the volume, resulting in zeros. This function adds buffer slices
    using mirrored padding so eddy has valid data to interpolate from.

    Parameters
    ----------
    input_file : Path
        Input 4D DWI NIfTI file
    output_file : Path
        Output padded DWI NIfTI file
    n_pad : int
        Number of slices to pad on each side (default: 2)
    method : str
        Padding method: 'reflect' (mirror), 'edge' (replicate), or 'wrap'

    Returns
    -------
    Tuple[Path, int]
        Path to padded file and original number of slices (for later cropping)
    """
    img = nib.load(input_file)
    data = img.get_fdata()

    original_n_slices = data.shape[2]

    print(f"Padding DWI slices for eddy protection:")
    print(f"  Original slices: {original_n_slices}")
    print(f"  Padding: {n_pad} slices on each side")
    print(f"  Method: {method}")

    # Pad along the slice (Z) axis
    # For 4D data, pad only axis 2
    if method == 'reflect':
        pad_mode = 'reflect'
    elif method == 'edge':
        pad_mode = 'edge'
    elif method == 'wrap':
        pad_mode = 'wrap'
    else:
        raise ValueError(f"Unknown padding method: {method}")

    # np.pad with 4D: ((before_x, after_x), (before_y, after_y), (before_z, after_z), (before_t, after_t))
    padded_data = np.pad(data, ((0, 0), (0, 0), (n_pad, n_pad), (0, 0)), mode=pad_mode)

    print(f"  New shape: {padded_data.shape}")

    # Create new image with padded data
    # Need to adjust the affine for the new origin
    new_affine = img.affine.copy()
    # Shift origin by -n_pad slices in the Z direction
    # The Z shift is: affine[:3, 2] * (-n_pad)
    z_shift = new_affine[:3, 2] * (-n_pad)
    new_affine[:3, 3] = new_affine[:3, 3] + z_shift

    padded_img = nib.Nifti1Image(padded_data, new_affine, img.header)
    padded_img.header.set_data_shape(padded_data.shape)

    nib.save(padded_img, output_file)
    print(f"  Saved padded DWI to: {output_file}")

    return output_file, original_n_slices


def pad_mask_for_eddy(
    input_mask: Path,
    output_mask: Path,
    n_pad: int = 2
) -> Path:
    """
    Pad brain mask to match padded DWI for eddy.

    Parameters
    ----------
    input_mask : Path
        Input 3D brain mask
    output_mask : Path
        Output padded brain mask
    n_pad : int
        Number of slices to pad on each side

    Returns
    -------
    Path
        Path to padded mask file
    """
    img = nib.load(input_mask)
    data = img.get_fdata()

    # Pad with zeros (conservative - don't extend mask beyond original brain)
    # Or use 'edge' to extend the mask
    padded_data = np.pad(data, ((0, 0), (0, 0), (n_pad, n_pad)), mode='edge')

    # Adjust affine
    new_affine = img.affine.copy()
    z_shift = new_affine[:3, 2] * (-n_pad)
    new_affine[:3, 3] = new_affine[:3, 3] + z_shift

    padded_img = nib.Nifti1Image(padded_data, new_affine, img.header)
    padded_img.header.set_data_shape(padded_data.shape)

    nib.save(padded_img, output_mask)
    print(f"  Saved padded mask to: {output_mask}")

    return output_mask


def crop_slices_after_eddy(
    input_file: Path,
    output_file: Path,
    original_n_slices: int,
    n_pad: int = 2
) -> Path:
    """
    Crop padded DWI back to original slice count after eddy correction.

    Parameters
    ----------
    input_file : Path
        Input padded DWI (eddy output)
    output_file : Path
        Output cropped DWI
    original_n_slices : int
        Original number of slices before padding
    n_pad : int
        Number of slices that were padded on each side

    Returns
    -------
    Path
        Path to cropped file
    """
    img = nib.load(input_file)
    data = img.get_fdata()

    print(f"Cropping eddy output back to original slices:")
    print(f"  Padded shape: {data.shape}")

    # Crop along Z axis
    cropped_data = data[:, :, n_pad:n_pad + original_n_slices, :]

    print(f"  Cropped shape: {cropped_data.shape}")

    # Adjust affine back
    new_affine = img.affine.copy()
    z_shift = new_affine[:3, 2] * n_pad
    new_affine[:3, 3] = new_affine[:3, 3] + z_shift

    cropped_img = nib.Nifti1Image(cropped_data, new_affine, img.header)
    cropped_img.header.set_data_shape(cropped_data.shape)

    nib.save(cropped_img, output_file)
    print(f"  Saved cropped DWI to: {output_file}")

    return output_file
