"""
Image validation utilities for preprocessing workflows.

This module provides functions to validate input images before processing,
checking voxel sizes, orientations, dimensions, and other critical properties.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ImageValidationError(Exception):
    """Raised when image validation fails."""
    pass


def validate_image(
    img_file: Path,
    modality: str = "anat",
    min_voxel_size: float = 0.05,
    max_voxel_size: float = 2.0,
    min_dimension: int = 32,
    check_orientation: bool = True,
    strict: bool = False
) -> Dict[str, any]:
    """
    Validate an MRI image before preprocessing.

    Checks voxel size, orientation, dimensions, data type, and other
    critical properties to ensure the image is suitable for processing.

    Parameters
    ----------
    img_file : Path
        Path to NIfTI image file
    modality : str
        Modality type ('anat', 'dwi', 'func')
    min_voxel_size : float
        Minimum acceptable voxel size in mm (default: 0.05mm)
    max_voxel_size : float
        Maximum acceptable voxel size in mm (default: 2.0mm)
    min_dimension : int
        Minimum acceptable dimension size in voxels (default: 32)
    check_orientation : bool
        Whether to validate orientation information (default: True)
    strict : bool
        If True, raise exceptions on validation failures.
        If False, return warnings but continue (default: False)

    Returns
    -------
    dict
        Dictionary containing validation results and image properties:
        - 'valid': bool - Overall validation status
        - 'warnings': list - List of warning messages
        - 'errors': list - List of error messages
        - 'properties': dict - Image properties (shape, voxel_size, etc.)

    Raises
    ------
    ImageValidationError
        If strict=True and validation fails

    Examples
    --------
    >>> from pathlib import Path
    >>> result = validate_image(Path('T2w.nii.gz'), modality='anat')
    >>> if result['valid']:
    ...     print("Image is valid for processing")
    >>> else:
    ...     print(f"Validation warnings: {result['warnings']}")
    """
    warnings = []
    errors = []
    properties = {}

    # Check file exists
    if not img_file.exists():
        error_msg = f"Image file not found: {img_file}"
        if strict:
            raise ImageValidationError(error_msg)
        errors.append(error_msg)
        return {
            'valid': False,
            'warnings': warnings,
            'errors': errors,
            'properties': properties
        }

    try:
        # Load image
        img = nib.load(img_file)
        properties['file'] = str(img_file)
        properties['file_size_mb'] = img_file.stat().st_size / (1024 * 1024)

        # Check image dimensions
        shape = img.shape
        properties['shape'] = shape
        properties['ndim'] = len(shape)

        if len(shape) < 3:
            error_msg = f"Image has less than 3 dimensions: {shape}"
            errors.append(error_msg)
        elif len(shape) > 4:
            warning_msg = f"Image has more than 4 dimensions: {shape}"
            warnings.append(warning_msg)

        # Check minimum dimensions
        for i, dim in enumerate(shape[:3]):
            if dim < min_dimension:
                warning_msg = f"Dimension {i} is very small: {dim} voxels (minimum recommended: {min_dimension})"
                warnings.append(warning_msg)

        # Check for time series data in anatomical scans
        if modality == "anat" and len(shape) > 3:
            warning_msg = f"Anatomical image has time dimension (4D): {shape}. Using first volume only."
            warnings.append(warning_msg)

        # Get affine matrix and voxel sizes
        affine = img.affine
        properties['affine'] = affine.tolist()

        # Extract voxel sizes from affine diagonal
        voxel_sizes = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        properties['voxel_size'] = voxel_sizes.tolist()
        properties['voxel_size_mean'] = float(np.mean(voxel_sizes))

        # Check voxel sizes
        for i, voxel_size in enumerate(voxel_sizes):
            if voxel_size < min_voxel_size:
                warning_msg = f"Axis {i} voxel size very small: {voxel_size:.4f} mm (minimum: {min_voxel_size} mm)"
                warnings.append(warning_msg)
            elif voxel_size > max_voxel_size:
                warning_msg = f"Axis {i} voxel size very large: {voxel_size:.4f} mm (maximum: {max_voxel_size} mm)"
                warnings.append(warning_msg)

        # Check for isotropic voxels (for rodent MRI, often isotropic or near-isotropic)
        voxel_size_ratio = np.max(voxel_sizes) / np.min(voxel_sizes)
        properties['voxel_anisotropy'] = float(voxel_size_ratio)
        if voxel_size_ratio > 3.0:
            warning_msg = f"Highly anisotropic voxels: {voxel_sizes[0]:.3f} × {voxel_sizes[1]:.3f} × {voxel_sizes[2]:.3f} mm (ratio: {voxel_size_ratio:.2f})"
            warnings.append(warning_msg)

        # Check orientation information
        if check_orientation:
            # Check affine matrix is valid
            if np.any(np.isnan(affine)) or np.any(np.isinf(affine)):
                error_msg = "Affine matrix contains NaN or Inf values"
                errors.append(error_msg)

            # Extract orientation (sign of diagonal elements)
            orientation = np.sign(np.diag(affine)[:3])
            properties['orientation'] = orientation.tolist()
            properties['orientation_str'] = _orientation_to_string(orientation)

            # Check for zero orientation (unusual)
            if np.any(orientation == 0):
                warning_msg = f"Affine diagonal contains zero: {np.diag(affine)[:3]}"
                warnings.append(warning_msg)

            # Get origin
            origin = affine[:3, 3]
            properties['origin'] = origin.tolist()

        # Check data type
        dtype = img.get_data_dtype()
        properties['dtype'] = str(dtype)

        if not np.issubdtype(dtype, np.floating):
            warning_msg = f"Image data type is not floating point: {dtype}. May need conversion."
            warnings.append(warning_msg)

        # Check for reasonable intensity range (load data sample)
        data_sample = img.get_fdata()[::4, ::4, ::4]  # Subsample for efficiency
        properties['intensity_min'] = float(np.min(data_sample))
        properties['intensity_max'] = float(np.max(data_sample))
        properties['intensity_mean'] = float(np.mean(data_sample))
        properties['intensity_std'] = float(np.std(data_sample))

        if np.all(data_sample == 0):
            error_msg = "Image contains all zeros"
            errors.append(error_msg)
        elif np.max(data_sample) == np.min(data_sample):
            error_msg = "Image has constant intensity (no variation)"
            errors.append(error_msg)

        # Check for NaN or Inf in data
        if np.any(np.isnan(data_sample)):
            warning_msg = "Image data contains NaN values"
            warnings.append(warning_msg)
        if np.any(np.isinf(data_sample)):
            warning_msg = "Image data contains Inf values"
            warnings.append(warning_msg)

        # Modality-specific checks
        if modality == "dwi":
            if len(shape) != 4:
                warning_msg = f"DWI image should be 4D, but has shape: {shape}"
                warnings.append(warning_msg)
            elif shape[3] < 6:
                warning_msg = f"DWI has very few volumes: {shape[3]} (minimum recommended: 6 for DTI)"
                warnings.append(warning_msg)

        elif modality == "func":
            if len(shape) != 4:
                error_msg = f"Functional image should be 4D, but has shape: {shape}"
                errors.append(error_msg)
            elif shape[3] < 50:
                warning_msg = f"Functional image has few timepoints: {shape[3]} (minimum recommended: 50)"
                warnings.append(warning_msg)

        # Determine overall validity
        valid = len(errors) == 0

        # If strict mode and not valid, raise exception
        if strict and not valid:
            error_summary = "\n".join(errors)
            raise ImageValidationError(f"Image validation failed:\n{error_summary}")

        return {
            'valid': valid,
            'warnings': warnings,
            'errors': errors,
            'properties': properties
        }

    except Exception as e:
        if isinstance(e, ImageValidationError):
            raise
        error_msg = f"Error loading or validating image: {str(e)}"
        if strict:
            raise ImageValidationError(error_msg) from e
        errors.append(error_msg)
        return {
            'valid': False,
            'warnings': warnings,
            'errors': errors,
            'properties': properties
        }


def _orientation_to_string(orientation: np.ndarray) -> str:
    """
    Convert orientation vector to human-readable string.

    Parameters
    ----------
    orientation : np.ndarray
        Orientation signs [x, y, z] where 1=positive, -1=negative

    Returns
    -------
    str
        Human-readable orientation string
    """
    axis_names = ['X', 'Y', 'Z']
    directions = ['L→R' if o > 0 else 'R→L' if o < 0 else 'zero' for o in orientation[:1]] + \
                 ['P→A' if o > 0 else 'A→P' if o < 0 else 'zero' for o in orientation[1:2]] + \
                 ['I→S' if o > 0 else 'S→I' if o < 0 else 'zero' for o in orientation[2:3]]
    return ", ".join([f"{name}: {dir}" for name, dir in zip(axis_names, directions)])


def print_validation_results(result: Dict[str, any], name: str = "Image"):
    """
    Print validation results in a formatted way.

    Parameters
    ----------
    result : dict
        Validation result dictionary from validate_image()
    name : str
        Name to display (default: "Image")
    """
    print(f"\n{'='*60}")
    print(f"Validation Results: {name}")
    print(f"{'='*60}")

    # Overall status
    status = "✓ VALID" if result['valid'] else "✗ INVALID"
    print(f"\nStatus: {status}")

    # Properties
    if result['properties']:
        props = result['properties']
        print(f"\nImage Properties:")
        print(f"  File: {props.get('file', 'N/A')}")
        print(f"  Size: {props.get('file_size_mb', 0):.2f} MB")
        print(f"  Shape: {props.get('shape', 'N/A')}")
        print(f"  Data type: {props.get('dtype', 'N/A')}")
        print(f"  Voxel size: [{props.get('voxel_size', [0,0,0])[0]:.3f}, "
              f"{props.get('voxel_size', [0,0,0])[1]:.3f}, "
              f"{props.get('voxel_size', [0,0,0])[2]:.3f}] mm")
        print(f"  Anisotropy ratio: {props.get('voxel_anisotropy', 0):.2f}")
        print(f"  Orientation: {props.get('orientation_str', 'N/A')}")
        print(f"  Intensity range: [{props.get('intensity_min', 0):.1f}, {props.get('intensity_max', 0):.1f}]")
        print(f"  Intensity mean ± std: {props.get('intensity_mean', 0):.1f} ± {props.get('intensity_std', 0):.1f}")

    # Warnings
    if result['warnings']:
        print(f"\n⚠ Warnings ({len(result['warnings'])}):")
        for warning in result['warnings']:
            print(f"  - {warning}")

    # Errors
    if result['errors']:
        print(f"\n✗ Errors ({len(result['errors'])}):")
        for error in result['errors']:
            print(f"  - {error}")

    print(f"{'='*60}\n")


def validate_image_pair(
    img1_file: Path,
    img2_file: Path,
    check_dimensions: bool = True,
    check_orientation: bool = True,
    strict: bool = False
) -> Dict[str, any]:
    """
    Validate that two images are compatible for processing.

    Parameters
    ----------
    img1_file : Path
        First image file
    img2_file : Path
        Second image file
    check_dimensions : bool
        Whether to check that dimensions match (default: True)
    check_orientation : bool
        Whether to check that orientations match (default: True)
    strict : bool
        If True, raise exceptions on validation failures (default: False)

    Returns
    -------
    dict
        Dictionary with validation results
    """
    warnings = []
    errors = []
    properties = {}

    try:
        img1 = nib.load(img1_file)
        img2 = nib.load(img2_file)

        properties['img1_shape'] = img1.shape
        properties['img2_shape'] = img2.shape

        # Check dimensions
        if check_dimensions:
            if img1.shape[:3] != img2.shape[:3]:
                error_msg = f"Image dimensions do not match: {img1.shape[:3]} vs {img2.shape[:3]}"
                errors.append(error_msg)

        # Check orientation
        if check_orientation:
            affine1 = img1.affine
            affine2 = img2.affine

            # Compare orientation (sign of diagonal)
            orient1 = np.sign(np.diag(affine1)[:3])
            orient2 = np.sign(np.diag(affine2)[:3])

            properties['img1_orientation'] = orient1.tolist()
            properties['img2_orientation'] = orient2.tolist()

            if not np.array_equal(orient1, orient2):
                warning_msg = f"Orientation mismatch: {orient1} vs {orient2}"
                warnings.append(warning_msg)

            # Compare voxel sizes
            voxel1 = np.sqrt(np.sum(affine1[:3, :3] ** 2, axis=0))
            voxel2 = np.sqrt(np.sum(affine2[:3, :3] ** 2, axis=0))

            properties['img1_voxel_size'] = voxel1.tolist()
            properties['img2_voxel_size'] = voxel2.tolist()

            voxel_diff = np.abs(voxel1 - voxel2)
            if np.any(voxel_diff > 0.1):  # More than 0.1mm difference
                warning_msg = f"Voxel size mismatch: {voxel1} vs {voxel2}"
                warnings.append(warning_msg)

        valid = len(errors) == 0

        if strict and not valid:
            error_summary = "\n".join(errors)
            raise ImageValidationError(f"Image pair validation failed:\n{error_summary}")

        return {
            'valid': valid,
            'warnings': warnings,
            'errors': errors,
            'properties': properties
        }

    except Exception as e:
        if isinstance(e, ImageValidationError):
            raise
        error_msg = f"Error validating image pair: {str(e)}"
        if strict:
            raise ImageValidationError(error_msg) from e
        errors.append(error_msg)
        return {
            'valid': False,
            'warnings': warnings,
            'errors': errors,
            'properties': properties
        }
