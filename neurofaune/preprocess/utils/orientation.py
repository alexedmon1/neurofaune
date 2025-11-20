"""
Orientation utilities for matching image orientations.

This module provides functions to detect and correct orientation mismatches
between images (e.g., atlas and subject scans).
"""

import nibabel as nib
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Dict


def match_orientation_to_reference(
    source_img: nib.Nifti1Image,
    reference_img: nib.Nifti1Image,
    output_file: Path
) -> Path:
    """
    Reorient source image to match reference image orientation.

    Compares the affine matrices of source and reference images,
    determines which axes need to be flipped, and creates a reoriented
    version of the source image.

    Parameters
    ----------
    source_img : nibabel.Nifti1Image
        Image to be reoriented (e.g., atlas template)
    reference_img : nibabel.Nifti1Image
        Reference image with desired orientation (e.g., subject scan)
    output_file : Path
        Path where reoriented image will be saved

    Returns
    -------
    Path
        Path to the reoriented image file

    Examples
    --------
    >>> atlas_img = nib.load('SIGMA_template.nii.gz')
    >>> subject_img = nib.load('subject_T2w.nii.gz')
    >>> reoriented_file = match_orientation_to_reference(
    ...     atlas_img, subject_img, Path('atlas_reoriented.nii.gz')
    ... )
    """
    # Get affine matrices
    source_affine = source_img.affine
    ref_affine = reference_img.affine

    # Extract direction (sign) of each axis from diagonal elements
    source_directions = np.sign(np.diag(source_affine)[:3])
    ref_directions = np.sign(np.diag(ref_affine)[:3])

    # Determine which axes need to be flipped
    flip_axes = []
    for axis in range(3):
        if source_directions[axis] != ref_directions[axis]:
            flip_axes.append(axis)

    # Load source data
    source_data = source_img.get_fdata()

    # Apply flips if needed
    if flip_axes:
        print(f"  Flipping axes: {flip_axes} to match reference orientation")
        source_data_flipped = np.flip(source_data, axis=flip_axes)
    else:
        print("  Orientations match - no flipping needed")
        source_data_flipped = source_data

    # Create new affine matrix matching reference orientation
    new_affine = source_affine.copy()
    for axis in flip_axes:
        # Flip the sign of the diagonal element
        new_affine[axis, axis] = -new_affine[axis, axis]
        # Adjust the origin for this axis
        # When we flip data, the origin needs to shift
        extent = source_data.shape[axis] * abs(source_affine[axis, axis])
        new_affine[axis, 3] = -new_affine[axis, 3] + (extent if new_affine[axis, axis] < 0 else -extent)

    # Save reoriented image
    reoriented_img = nib.Nifti1Image(source_data_flipped, new_affine, source_img.header)
    nib.save(reoriented_img, output_file)

    print(f"  Saved reoriented image to: {output_file.name}")
    return output_file


def check_orientation_match(
    img1: nib.Nifti1Image,
    img2: nib.Nifti1Image
) -> Tuple[bool, list]:
    """
    Check if two images have matching orientations.

    Parameters
    ----------
    img1 : nibabel.Nifti1Image
        First image
    img2 : nibabel.Nifti1Image
        Second image

    Returns
    -------
    bool
        True if orientations match, False otherwise
    list
        List of axes that are flipped (empty if orientations match)
    """
    affine1 = img1.affine
    affine2 = img2.affine

    # Extract direction signs
    dir1 = np.sign(np.diag(affine1)[:3])
    dir2 = np.sign(np.diag(affine2)[:3])

    # Find mismatched axes
    flip_axes = [i for i in range(3) if dir1[i] != dir2[i]]

    return len(flip_axes) == 0, flip_axes


def print_orientation_info(img: nib.Nifti1Image, name: str = "Image"):
    """
    Print orientation information for an image.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Image to analyze
    name : str
        Name for display purposes
    """
    affine = img.affine
    directions = np.sign(np.diag(affine)[:3])

    axis_names = ['X (L-R)', 'Y (P-A)', 'Z (I-S)']
    direction_map = {1: 'positive', -1: 'negative', 0: 'zero'}

    print(f"\n{name} orientation:")
    for i, axis_name in enumerate(axis_names):
        print(f"  {axis_name}: {direction_map[directions[i]]} ({affine[i,i]:.2f})")
    print(f"  Origin: [{affine[0,3]:.2f}, {affine[1,3]:.2f}, {affine[2,3]:.2f}]")


def save_orientation_metadata(
    subject: str,
    atlas_name: str,
    atlas_affine: np.ndarray,
    subject_affine: np.ndarray,
    flipped_axes: list,
    output_dir: Path
):
    """
    Save orientation metadata for use by other modality workflows.

    Parameters
    ----------
    subject : str
        Subject identifier
    atlas_name : str
        Name of atlas (e.g., 'SIGMA')
    atlas_affine : np.ndarray
        Original atlas affine matrix
    subject_affine : np.ndarray
        Subject image affine matrix
    flipped_axes : list
        List of axes that were flipped (0, 1, or 2)
    output_dir : Path
        Directory to save metadata (usually study root)
    """
    metadata = {
        'subject': subject,
        'atlas': atlas_name,
        'atlas_orientation': {
            'diagonal': list(np.diag(atlas_affine)[:3].astype(float)),
            'origin': list(atlas_affine[:3, 3].astype(float))
        },
        'subject_orientation': {
            'diagonal': list(np.diag(subject_affine)[:3].astype(float)),
            'origin': list(subject_affine[:3, 3].astype(float))
        },
        'flipped_axes': flipped_axes,
        'axis_names': ['X (L-R)', 'Y (P-A)', 'Z (I-S)']
    }

    metadata_file = output_dir / f'{subject}_preprocessing_metadata.yaml'
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    print(f"  Saved orientation metadata to: {metadata_file}")
    return metadata_file


def load_orientation_metadata(subject: str, output_dir: Path) -> Dict:
    """
    Load preprocessing metadata for a subject.

    Parameters
    ----------
    subject : str
        Subject identifier
    output_dir : Path
        Directory where metadata was saved

    Returns
    -------
    dict
        Preprocessing metadata
    """
    metadata_file = output_dir / f'{subject}_preprocessing_metadata.yaml'
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = yaml.safe_load(f)

    return metadata
