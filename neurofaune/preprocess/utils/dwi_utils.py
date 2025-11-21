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

    return bvals, bvecs


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
        'shape': data.shape,
        'voxel_size': img.header.get_zooms()[:3],
        'n_volumes': data.shape[3] if len(data.shape) == 4 else 1
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
