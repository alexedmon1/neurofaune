#!/usr/bin/env python3
"""
Slice extraction utilities for modality-specific atlas registration.

Provides functions to extract relevant anatomical slices from atlas templates,
enabling more accurate registration for modalities that cover only specific
brain regions (e.g., 11 hippocampal slices for DTI).
"""

import logging
from typing import Optional, Tuple, Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def extract_slices(
    img: nib.Nifti1Image,
    slice_start: int,
    slice_end: int,
    axis: int = 2
) -> nib.Nifti1Image:
    """
    Extract contiguous slices from a 3D image along specified axis.

    Parameters
    ----------
    img : Nifti1Image
        Input 3D image
    slice_start : int
        Start slice index (inclusive)
    slice_end : int
        End slice index (exclusive). Use -1 for all slices from start
    axis : int, default=2
        Axis along which to extract (0=x, 1=y, 2=z)

    Returns
    -------
    Nifti1Image
        Image with extracted slices. Affine matrix is adjusted to reflect
        the new origin.

    Examples
    --------
    >>> # Extract hippocampal slices (z=15 to 25)
    >>> template = nib.load('SIGMA_template.nii')
    >>> hipp_template = extract_slices(template, 15, 25)
    >>>
    >>> # Extract from slice 10 to end
    >>> partial = extract_slices(template, 10, -1)
    """
    if axis not in [0, 1, 2]:
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

    # Get image data
    data = img.get_fdata()
    affine = img.affine.copy()
    header = img.header.copy()

    # Handle -1 as "to end"
    if slice_end == -1:
        slice_end = data.shape[axis]

    # Validate slice indices
    if slice_start < 0:
        raise ValueError(f"slice_start must be >= 0, got {slice_start}")
    if slice_end <= slice_start:
        raise ValueError(f"slice_end ({slice_end}) must be > slice_start ({slice_start})")
    if slice_start >= data.shape[axis]:
        raise ValueError(f"slice_start ({slice_start}) exceeds image size ({data.shape[axis]})")

    # Clip end to image size
    slice_end = min(slice_end, data.shape[axis])

    # Extract slices along specified axis
    if axis == 0:
        extracted_data = data[slice_start:slice_end, :, :]
    elif axis == 1:
        extracted_data = data[:, slice_start:slice_end, :]
    else:  # axis == 2
        extracted_data = data[:, :, slice_start:slice_end]

    # Update affine to reflect new origin
    # The translation component needs to account for removed slices
    affine[:3, 3] += affine[:3, axis] * slice_start

    # Update header dimensions
    new_shape = list(data.shape)
    new_shape[axis] = slice_end - slice_start
    header.set_data_shape(new_shape)

    # Create new image
    extracted_img = nib.Nifti1Image(extracted_data, affine, header)

    logger.info(
        f"Extracted slices {slice_start}:{slice_end} (axis={axis}), "
        f"shape: {data.shape} → {extracted_data.shape}"
    )

    return extracted_img


def get_slice_range(
    img: nib.Nifti1Image,
    modality: str,
    config: dict,
    axis: int = 2
) -> Tuple[int, int]:
    """
    Get slice range for a modality from configuration.

    Parameters
    ----------
    img : Nifti1Image
        Reference image (for validation)
    modality : str
        Modality name (e.g., 'dwi', 'func', 'anat')
    config : dict
        Configuration dictionary with atlas.slice_definitions
    axis : int, default=2
        Axis along which slices are defined

    Returns
    -------
    tuple of int
        (start_slice, end_slice) validated against image dimensions

    Raises
    ------
    ValueError
        If modality not found in config or slice range invalid
    """
    # Get slice definitions from config
    if 'atlas' not in config or 'slice_definitions' not in config['atlas']:
        raise ValueError("Configuration missing 'atlas.slice_definitions'")

    slice_defs = config['atlas']['slice_definitions']
    if modality not in slice_defs:
        raise ValueError(f"No slice definition for modality '{modality}'")

    # Get start and end
    slice_def = slice_defs[modality]
    start = slice_def.get('start', 0)
    end = slice_def.get('end', -1)

    # Validate against image
    img_size = img.shape[axis]

    if start < 0:
        raise ValueError(f"slice_start must be >= 0, got {start}")

    if end == -1:
        end = img_size
    elif end > img_size:
        logger.warning(
            f"slice_end ({end}) exceeds image size ({img_size}), "
            f"clipping to {img_size}"
        )
        end = img_size

    if start >= end:
        raise ValueError(
            f"Invalid slice range for {modality}: start={start}, end={end}"
        )

    return (start, end)


def extract_modality_slices(
    img: nib.Nifti1Image,
    modality: str,
    config: dict,
    axis: int = 2
) -> nib.Nifti1Image:
    """
    Extract slices for a specific modality based on configuration.

    Convenience function combining get_slice_range and extract_slices.

    Parameters
    ----------
    img : Nifti1Image
        Input image
    modality : str
        Modality name
    config : dict
        Configuration dictionary
    axis : int, default=2
        Extraction axis

    Returns
    -------
    Nifti1Image
        Image with extracted slices

    Examples
    --------
    >>> from neurofaune.config import load_config
    >>> import nibabel as nib
    >>>
    >>> config = load_config('config.yaml')
    >>> template = nib.load('SIGMA_template.nii')
    >>>
    >>> # Extract hippocampal slices for DTI
    >>> dwi_template = extract_modality_slices(template, 'dwi', config)
    """
    start, end = get_slice_range(img, modality, config, axis)
    return extract_slices(img, start, end, axis)


def match_slice_geometry(
    source_img: nib.Nifti1Image,
    target_img: nib.Nifti1Image,
    modality: Optional[str] = None,
    config: Optional[dict] = None
) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """
    Ensure source and target images have compatible slice ranges.

    For slice-specific registration, both images should cover the same
    anatomical region. This function extracts matching slices if needed.

    Parameters
    ----------
    source_img : Nifti1Image
        Source image (e.g., subject FA map)
    target_img : Nifti1Image
        Target image (e.g., atlas template)
    modality : str, optional
        Modality for slice extraction
    config : dict, optional
        Configuration with slice definitions

    Returns
    -------
    tuple of Nifti1Image
        (matched_source, matched_target) with compatible geometries

    Notes
    -----
    If modality and config are provided, both images are extracted to the
    same slice range. Otherwise, images are cropped to the minimum common
    z-range.

    Examples
    --------
    >>> # Match DTI to hippocampal atlas
    >>> fa_img = nib.load('subject_FA.nii.gz')
    >>> atlas = nib.load('SIGMA_template.nii')
    >>> fa_matched, atlas_matched = match_slice_geometry(fa_img, atlas, 'dwi', config)
    """
    # If modality specified, extract based on config
    if modality and config:
        try:
            source_matched = extract_modality_slices(source_img, modality, config)
            target_matched = extract_modality_slices(target_img, modality, config)
            logger.info(f"Extracted {modality} slices from both images")
            return source_matched, target_matched
        except (ValueError, KeyError) as e:
            logger.warning(f"Could not extract modality slices: {e}, falling back to geometry matching")

    # Otherwise, match based on actual z-extent
    source_shape = source_img.shape
    target_shape = target_img.shape

    # Find common z-range
    min_z = min(source_shape[2], target_shape[2])

    # Extract matching slices from both
    if source_shape[2] > min_z:
        logger.info(f"Cropping source image z-slices: {source_shape[2]} → {min_z}")
        source_matched = extract_slices(source_img, 0, min_z)
    else:
        source_matched = source_img

    if target_shape[2] > min_z:
        logger.info(f"Cropping target image z-slices: {target_shape[2]} → {min_z}")
        target_matched = extract_slices(target_img, 0, min_z)
    else:
        target_matched = target_img

    return source_matched, target_matched


def get_slice_metadata(
    img: nib.Nifti1Image,
    slice_start: int,
    slice_end: int,
    axis: int = 2
) -> dict:
    """
    Generate metadata describing slice extraction.

    Useful for transform registry to track which slices were used
    for registration.

    Parameters
    ----------
    img : Nifti1Image
        Original (full) image
    slice_start : int
        Start slice index
    slice_end : int
        End slice index
    axis : int, default=2
        Extraction axis

    Returns
    -------
    dict
        Metadata with keys:
        - slice_range: (start, end)
        - axis: extraction axis
        - original_shape: full image shape
        - extracted_shape: shape after extraction
        - n_slices: number of slices extracted

    Examples
    --------
    >>> template = nib.load('SIGMA_template.nii')
    >>> metadata = get_slice_metadata(template, 15, 25)
    >>> # Save to transform registry
    >>> registry.save_transform(..., slice_metadata=metadata)
    """
    original_shape = img.shape
    extracted_shape = list(original_shape)
    extracted_shape[axis] = slice_end - slice_start

    return {
        'slice_range': (slice_start, slice_end),
        'axis': axis,
        'original_shape': original_shape,
        'extracted_shape': tuple(extracted_shape),
        'n_slices': slice_end - slice_start,
        'description': f'Slices {slice_start}:{slice_end} along axis {axis}'
    }
