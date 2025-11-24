"""
Subject-to-template registration utilities.

This module provides functions for registering individual subjects to
age-matched study templates.
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import nibabel as nib


def register_subject_to_template(
    moving_image: Path,
    template_file: Path,
    output_prefix: Path,
    mask_file: Optional[Path] = None,
    transform_type: str = 'SyN',
    n_cores: int = 4
) -> Dict[str, Path]:
    """
    Register subject image to study template.

    Parameters
    ----------
    moving_image : Path
        Subject image to register
    template_file : Path
        Study template (fixed image)
    output_prefix : Path
        Output prefix for transform files
    mask_file : Path, optional
        Brain mask for registration
    transform_type : str
        Transform type ('Rigid', 'Affine', or 'SyN')
    n_cores : int
        Number of CPU cores

    Returns
    -------
    dict
        Dictionary with paths to:
        - 'composite_transform': Subject → template transform
        - 'inverse_composite_transform': Template → subject transform
        - 'warped': Subject in template space

    Examples
    --------
    >>> results = register_subject_to_template(
    ...     moving_image=Path('sub-Rat207_T2w.nii.gz'),
    ...     template_file=Path('tpl-BPARatp60_T2w.nii.gz'),
    ...     output_prefix=Path('sub-Rat207_to_template_'),
    ...     n_cores=4
    ... )
    """
    print(f"\nRegistering {moving_image.name} to template...")
    print(f"  Moving: {moving_image}")
    print(f"  Fixed: {template_file}")
    print(f"  Transform: {transform_type}")

    # Validate inputs
    if not moving_image.exists():
        raise FileNotFoundError(f"Moving image not found: {moving_image}")
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")

    # Create output directory
    output_dir = output_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use antsRegistrationSyN.sh for quick registration
    cmd = [
        'antsRegistrationSyN.sh',
        '-d', '3',
        '-f', str(template_file),      # Fixed (template)
        '-m', str(moving_image),        # Moving (subject)
        '-o', str(output_prefix),
        '-n', str(n_cores),
        '-t', transform_type[0].lower()  # 's' for SyN, 'a' for Affine, 'r' for Rigid
    ]

    if mask_file and mask_file.exists():
        cmd.extend(['-x', str(mask_file)])

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: Registration failed!")
        print(result.stdout)
        raise RuntimeError("ANTs registration failed")

    # Expected outputs
    composite_transform = Path(str(output_prefix) + 'Composite.h5')
    inverse_composite = Path(str(output_prefix) + 'InverseComposite.h5')
    warped = Path(str(output_prefix) + 'Warped.nii.gz')

    if not composite_transform.exists():
        raise FileNotFoundError(f"Expected transform not found: {composite_transform}")

    print(f"  ✓ Registration complete")
    print(f"    Transform: {composite_transform.name}")

    return {
        'composite_transform': composite_transform,
        'inverse_composite_transform': inverse_composite,
        'warped': warped
    }


def register_within_subject(
    moving_image: Path,
    fixed_image: Path,
    output_prefix: Path,
    moving_modality: str,
    fixed_modality: str,
    n_cores: int = 4
) -> Dict[str, Path]:
    """
    Register two modalities within the same subject (e.g., T2w ↔ FA).

    This is used for label propagation across modalities.

    Parameters
    ----------
    moving_image : Path
        Moving modality image (e.g., FA)
    fixed_image : Path
        Fixed modality image (e.g., T2w)
    output_prefix : Path
        Output prefix for transform files
    moving_modality : str
        Moving modality name ('FA', 'b0', 'bold')
    fixed_modality : str
        Fixed modality name ('T2w')
    n_cores : int
        Number of CPU cores

    Returns
    -------
    dict
        Dictionary with paths to registration outputs

    Examples
    --------
    >>> # Register FA to T2w within same subject
    >>> results = register_within_subject(
    ...     moving_image=Path('sub-Rat207_FA.nii.gz'),
    ...     fixed_image=Path('sub-Rat207_T2w.nii.gz'),
    ...     output_prefix=Path('sub-Rat207_FA_to_T2w_'),
    ...     moving_modality='FA',
    ...     fixed_modality='T2w',
    ...     n_cores=4
    ... )
    """
    print(f"\nWithin-subject registration: {moving_modality} → {fixed_modality}")
    print(f"  Moving: {moving_image.name}")
    print(f"  Fixed: {fixed_image.name}")

    # Use Rigid or Affine for within-subject (no deformation needed)
    # FA/b0 to T2w benefits from Affine to handle slight geometric differences
    results = register_subject_to_template(
        moving_image=moving_image,
        fixed_image=fixed_image,
        output_prefix=output_prefix,
        transform_type='Affine',
        n_cores=n_cores
    )

    print(f"  ✓ Within-subject registration complete")

    return results


def apply_transforms(
    input_image: Path,
    reference_image: Path,
    transforms: list,
    output_image: Path,
    interpolation: str = 'Linear',
    invert_transforms: Optional[list] = None
) -> Path:
    """
    Apply transform(s) to warp an image.

    Parameters
    ----------
    input_image : Path
        Image to transform
    reference_image : Path
        Reference image (defines output space)
    transforms : list of Path
        List of transform files (applied in reverse order)
    output_image : Path
        Output warped image
    interpolation : str
        Interpolation method ('Linear', 'NearestNeighbor', 'BSpline')
    invert_transforms : list of bool, optional
        Which transforms to invert (same length as transforms)

    Returns
    -------
    Path
        Path to output warped image

    Examples
    --------
    >>> # Warp FA to SIGMA space via T2w
    >>> apply_transforms(
    ...     input_image=Path('sub-Rat207_FA.nii.gz'),
    ...     reference_image=Path('SIGMA_template.nii.gz'),
    ...     transforms=[
    ...         Path('tpl-to-SIGMA_Composite.h5'),
    ...         Path('T2w_to_template_Composite.h5'),
    ...         Path('FA_to_T2w_Composite.h5')
    ...     ],
    ...     output_image=Path('sub-Rat207_space-SIGMA_FA.nii.gz')
    ... )
    """
    print(f"\nApplying transforms to {input_image.name}...")

    # Build command
    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(input_image),
        '-r', str(reference_image),
        '-o', str(output_image),
        '-n', interpolation
    ]

    # Add transforms (applied in reverse order by ANTs)
    for i, transform in enumerate(transforms):
        cmd.extend(['-t', str(transform)])

        # Check if this transform should be inverted
        if invert_transforms and i < len(invert_transforms) and invert_transforms[i]:
            cmd[-1] = '[' + cmd[-1] + ',1]'  # Add invert flag

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: Transform application failed!")
        print(result.stderr)
        raise RuntimeError("antsApplyTransforms failed")

    print(f"  ✓ Saved to {output_image.name}")

    return output_image


def propagate_labels_to_subject(
    labels_image: Path,
    subject_reference: Path,
    transforms: list,
    output_labels: Path,
    invert_transforms: Optional[list] = None
) -> Path:
    """
    Propagate atlas labels to subject space.

    Uses NearestNeighbor interpolation to preserve integer labels.

    Parameters
    ----------
    labels_image : Path
        Atlas labels (e.g., SIGMA parcellation)
    subject_reference : Path
        Subject space reference image (defines output geometry)
    transforms : list of Path
        List of transforms from atlas to subject space
    output_labels : Path
        Output labels in subject space
    invert_transforms : list of bool, optional
        Which transforms to invert

    Returns
    -------
    Path
        Path to output labels

    Examples
    --------
    >>> # Propagate SIGMA labels to subject FA space
    >>> propagate_labels_to_subject(
    ...     labels_image=Path('SIGMA_labels.nii.gz'),
    ...     subject_reference=Path('sub-Rat207_FA.nii.gz'),
    ...     transforms=[
    ...         Path('FA_to_T2w_Composite.h5'),
    ...         Path('T2w_to_template_Composite.h5'),
    ...         Path('tpl-to-SIGMA_Composite.h5')
    ...     ],
    ...     output_labels=Path('sub-Rat207_space-FA_labels.nii.gz'),
    ...     invert_transforms=[False, True, True]  # Invert last two
    ... )
    """
    print(f"\nPropagating labels to subject space...")

    return apply_transforms(
        input_image=labels_image,
        reference_image=subject_reference,
        transforms=transforms,
        output_image=output_labels,
        interpolation='NearestNeighbor',  # Critical for labels!
        invert_transforms=invert_transforms
    )
