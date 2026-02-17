"""
Subject-to-template registration utilities.

This module provides functions for registering individual subjects to
age-matched study templates.
"""

import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional
import nibabel as nib
import numpy as np


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


def propagate_atlas_to_dwi_direct(
    atlas_path: Path,
    fa_reference: Path,
    transforms_root: Path,
    templates_root: Path,
    subject: str,
    session: str,
    output_path: Path
) -> Path:
    """
    Propagate SIGMA atlas to DTI/FA space via direct FA→Template registration.

    Uses the simplified 2-stage chain:
        SIGMA → Template → Subject FA

    Only 3 transforms (vs 5 in the old FA→T2w→Template chain), giving
    better atlas overlap especially for 3D T2w subjects.

    Parameters
    ----------
    atlas_path : Path
        Path to SIGMA atlas parcellation
    fa_reference : Path
        Subject FA map (defines output space)
    transforms_root : Path
        Root directory for subject transforms
    templates_root : Path
        Root directory for templates
    subject : str
        Subject ID (e.g., 'sub-Rat1')
    session : str
        Session ID (e.g., 'ses-p60')
    output_path : Path
        Output path for atlas in FA space

    Returns
    -------
    Path
        Path to atlas in FA space

    Raises
    ------
    FileNotFoundError
        If required transforms are missing
    """
    cohort = session.replace('ses-', '')

    # Locate all required transforms
    subj_transforms = transforms_root / subject / session
    tpl_transforms = templates_root / 'anat' / cohort / 'transforms'

    fa_to_template = subj_transforms / 'FA_to_template_0GenericAffine.mat'
    tpl_to_sigma_affine = tpl_transforms / 'tpl-to-SIGMA_0GenericAffine.mat'
    tpl_to_sigma_inv_warp = tpl_transforms / 'tpl-to-SIGMA_1InverseWarp.nii.gz'

    if not fa_to_template.exists():
        raise FileNotFoundError(
            f"FA→Template transform not found: {fa_to_template}\n"
            "Run FA-to-template registration first."
        )

    if not tpl_to_sigma_affine.exists():
        raise FileNotFoundError(
            f"Template→SIGMA transform not found: {tpl_to_sigma_affine}\n"
            "Run template-to-SIGMA registration first."
        )

    # Build transform chain for SIGMA → FA (inverse direction)
    # ANTs applies transforms in reverse order, so list from FA to SIGMA
    transform_list = []

    # 1. Template → FA (inverse of FA → Template)
    transform_list.append(f"[{fa_to_template},1]")

    # 2. SIGMA → Template (inverse of Template → SIGMA)
    if tpl_to_sigma_inv_warp.exists():
        transform_list.append(str(tpl_to_sigma_inv_warp))
    transform_list.append(f"[{tpl_to_sigma_affine},1]")

    # Apply transforms
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(atlas_path),
        '-r', str(fa_reference),
        '-o', str(output_path),
        '-n', 'NearestNeighbor',
    ]

    for t in transform_list:
        cmd.extend(['-t', t])

    print(f"\nPropagating atlas to FA space (direct)...")
    print(f"  Atlas: {atlas_path.name}")
    print(f"  Reference: {fa_reference.name}")
    print(f"  Transform chain: {len(transform_list)} transforms")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: Atlas propagation failed!")
        print(result.stderr)
        raise RuntimeError("antsApplyTransforms failed")

    # Verify output
    atlas_img = nib.load(output_path)
    atlas_data = atlas_img.get_fdata()
    n_labels = len(set(atlas_data[atlas_data > 0].astype(int)))

    print(f"  Output: {output_path}")
    print(f"  Unique labels: {n_labels}")

    return output_path


def propagate_atlas_to_bold_direct(
    atlas_path: Path,
    bold_reference: Path,
    transforms_root: Path,
    templates_root: Path,
    subject: str,
    session: str,
    output_path: Path
) -> Path:
    """
    Propagate SIGMA atlas to BOLD/fMRI space via direct BOLD→Template registration.

    Uses the simplified 2-stage chain:
        SIGMA → Template → Subject BOLD

    Only 3 transforms (vs 5 in the old BOLD→T2w→Template chain).

    Parameters
    ----------
    atlas_path : Path
        Path to SIGMA atlas parcellation
    bold_reference : Path
        Subject BOLD reference volume (defines output space)
    transforms_root : Path
        Root directory for subject transforms
    templates_root : Path
        Root directory for templates
    subject : str
        Subject ID (e.g., 'sub-Rat1')
    session : str
        Session ID (e.g., 'ses-p60')
    output_path : Path
        Output path for atlas in BOLD space

    Returns
    -------
    Path
        Path to atlas in BOLD space
    """
    cohort = session.replace('ses-', '')

    subj_transforms = transforms_root / subject / session
    tpl_transforms = templates_root / 'anat' / cohort / 'transforms'

    bold_to_template = subj_transforms / 'BOLD_to_template_0GenericAffine.mat'
    tpl_to_sigma_affine = tpl_transforms / 'tpl-to-SIGMA_0GenericAffine.mat'
    tpl_to_sigma_inv_warp = tpl_transforms / 'tpl-to-SIGMA_1InverseWarp.nii.gz'

    if not bold_to_template.exists():
        raise FileNotFoundError(
            f"BOLD→Template transform not found: {bold_to_template}\n"
            "Run BOLD-to-template registration first."
        )

    if not tpl_to_sigma_affine.exists():
        raise FileNotFoundError(
            f"Template→SIGMA transform not found: {tpl_to_sigma_affine}\n"
            "Run template-to-SIGMA registration first."
        )

    transform_list = []

    # 1. Template → BOLD (inverse of BOLD → Template)
    transform_list.append(f"[{bold_to_template},1]")

    # 2. SIGMA → Template (inverse of Template → SIGMA)
    if tpl_to_sigma_inv_warp.exists():
        transform_list.append(str(tpl_to_sigma_inv_warp))
    transform_list.append(f"[{tpl_to_sigma_affine},1]")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(atlas_path),
        '-r', str(bold_reference),
        '-o', str(output_path),
        '-n', 'NearestNeighbor',
    ]

    for t in transform_list:
        cmd.extend(['-t', t])

    print(f"\nPropagating atlas to BOLD space (direct)...")
    print(f"  Atlas: {atlas_path.name}")
    print(f"  Reference: {bold_reference.name}")
    print(f"  Transform chain: {len(transform_list)} transforms")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: Atlas propagation failed!")
        print(result.stderr)
        raise RuntimeError("antsApplyTransforms failed")

    atlas_img = nib.load(output_path)
    atlas_data = atlas_img.get_fdata()
    n_labels = len(set(atlas_data[atlas_data > 0].astype(int)))

    print(f"  Output: {output_path}")
    print(f"  Unique labels: {n_labels}")

    return output_path


def propagate_atlas_to_msme_direct(
    atlas_path: Path,
    msme_reference: Path,
    transforms_root: Path,
    templates_root: Path,
    subject: str,
    session: str,
    output_path: Path
) -> Path:
    """
    Propagate SIGMA atlas to MSME space via direct MSME→Template registration.

    Uses the simplified 2-stage chain:
        SIGMA → Template → Subject MSME

    Only 3 transforms (vs 5 in the old MSME→T2w→Template chain).

    Parameters
    ----------
    atlas_path : Path
        Path to SIGMA atlas parcellation
    msme_reference : Path
        Subject MSME reference volume (defines output space)
    transforms_root : Path
        Root directory for subject transforms
    templates_root : Path
        Root directory for templates
    subject : str
        Subject ID (e.g., 'sub-Rat1')
    session : str
        Session ID (e.g., 'ses-p60')
    output_path : Path
        Output path for atlas in MSME space

    Returns
    -------
    Path
        Path to atlas in MSME space
    """
    cohort = session.replace('ses-', '')

    subj_transforms = transforms_root / subject / session
    tpl_transforms = templates_root / 'anat' / cohort / 'transforms'

    msme_to_template = subj_transforms / 'MSME_to_template_0GenericAffine.mat'
    tpl_to_sigma_affine = tpl_transforms / 'tpl-to-SIGMA_0GenericAffine.mat'
    tpl_to_sigma_inv_warp = tpl_transforms / 'tpl-to-SIGMA_1InverseWarp.nii.gz'

    if not msme_to_template.exists():
        raise FileNotFoundError(
            f"MSME→Template transform not found: {msme_to_template}\n"
            "Run MSME-to-template registration first."
        )

    if not tpl_to_sigma_affine.exists():
        raise FileNotFoundError(
            f"Template→SIGMA transform not found: {tpl_to_sigma_affine}\n"
            "Run template-to-SIGMA registration first."
        )

    transform_list = []

    # 1. Template → MSME (inverse of MSME → Template)
    transform_list.append(f"[{msme_to_template},1]")

    # 2. SIGMA → Template (inverse of Template → SIGMA)
    if tpl_to_sigma_inv_warp.exists():
        transform_list.append(str(tpl_to_sigma_inv_warp))
    transform_list.append(f"[{tpl_to_sigma_affine},1]")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(atlas_path),
        '-r', str(msme_reference),
        '-o', str(output_path),
        '-n', 'NearestNeighbor',
    ]

    for t in transform_list:
        cmd.extend(['-t', t])

    print(f"\nPropagating atlas to MSME space (direct)...")
    print(f"  Atlas: {atlas_path.name}")
    print(f"  Reference: {msme_reference.name}")
    print(f"  Transform chain: {len(transform_list)} transforms")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: Atlas propagation failed!")
        print(result.stderr)
        raise RuntimeError("antsApplyTransforms failed")

    atlas_img = nib.load(output_path)
    atlas_data = atlas_img.get_fdata()
    n_labels = len(set(atlas_data[atlas_data > 0].astype(int)))

    print(f"  Output: {output_path}")
    print(f"  Unique labels: {n_labels}")

    return output_path


def propagate_atlas_to_dwi(
    atlas_path: Path,
    fa_reference: Path,
    transforms_root: Path,
    templates_root: Path,
    subject: str,
    session: str,
    output_path: Path
) -> Path:
    """
    Propagate SIGMA atlas to DTI/FA space through the transform chain.

    .. deprecated::
        Use :func:`propagate_atlas_to_dwi_direct` instead for better overlap
        with fewer transforms.

    Uses the multi-stage registration chain:
        SIGMA → T2w Template → Subject T2w → Subject FA

    This handles the common case of separate ANTs transform files
    (affine .mat + warp .nii.gz) rather than composite .h5 files.

    Parameters
    ----------
    atlas_path : Path
        Path to SIGMA atlas parcellation
    fa_reference : Path
        Subject FA map (defines output space)
    transforms_root : Path
        Root directory for subject transforms
    templates_root : Path
        Root directory for templates
    subject : str
        Subject ID (e.g., 'sub-Rat1')
    session : str
        Session ID (e.g., 'ses-p60')
    output_path : Path
        Output path for atlas in FA space

    Returns
    -------
    Path
        Path to atlas in FA space

    Raises
    ------
    FileNotFoundError
        If required transforms are missing

    Examples
    --------
    >>> propagate_atlas_to_dwi(
    ...     atlas_path=Path('/path/to/SIGMA_atlas.nii'),
    ...     fa_reference=Path('/path/to/sub-Rat1_FA.nii.gz'),
    ...     transforms_root=Path('/study/transforms'),
    ...     templates_root=Path('/study/templates'),
    ...     subject='sub-Rat1',
    ...     session='ses-p60',
    ...     output_path=Path('/study/derivatives/sub-Rat1/ses-p60/dwi/atlas.nii.gz')
    ... )
    """
    cohort = session.replace('ses-', '')

    # Locate all required transforms
    subj_transforms = transforms_root / subject / session
    tpl_transforms = templates_root / 'anat' / cohort / 'transforms'

    # Required transforms (check existence)
    fa_to_t2w = subj_transforms / 'FA_to_T2w_0GenericAffine.mat'
    t2w_to_tpl_affine = subj_transforms / 'T2w_to_template_0GenericAffine.mat'
    t2w_to_tpl_inv_warp = subj_transforms / 'T2w_to_template_1InverseWarp.nii.gz'
    tpl_to_sigma_affine = tpl_transforms / 'tpl-to-SIGMA_0GenericAffine.mat'
    tpl_to_sigma_inv_warp = tpl_transforms / 'tpl-to-SIGMA_1InverseWarp.nii.gz'

    # Check FA→T2w exists
    if not fa_to_t2w.exists():
        raise FileNotFoundError(
            f"FA→T2w transform not found: {fa_to_t2w}\n"
            "Run within-subject DTI registration first."
        )

    # Check T2w→Template exists
    if not t2w_to_tpl_affine.exists():
        raise FileNotFoundError(
            f"T2w→Template transform not found: {t2w_to_tpl_affine}\n"
            "Run subject-to-template registration first."
        )

    # Check Template→SIGMA exists
    if not tpl_to_sigma_affine.exists():
        raise FileNotFoundError(
            f"Template→SIGMA transform not found: {tpl_to_sigma_affine}\n"
            "Run template-to-SIGMA registration first."
        )

    # Build transform chain for SIGMA → FA (inverse direction)
    # ANTs applies transforms in reverse order, so list from FA to SIGMA
    transform_list = []

    # 1. T2w → FA (inverse of FA → T2w)
    transform_list.append(f"[{fa_to_t2w},1]")

    # 2. Template → T2w (inverse of T2w → Template)
    if t2w_to_tpl_inv_warp.exists():
        transform_list.append(str(t2w_to_tpl_inv_warp))
    transform_list.append(f"[{t2w_to_tpl_affine},1]")

    # 3. SIGMA → Template (inverse of Template → SIGMA)
    if tpl_to_sigma_inv_warp.exists():
        transform_list.append(str(tpl_to_sigma_inv_warp))
    transform_list.append(f"[{tpl_to_sigma_affine},1]")

    # Apply transforms
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(atlas_path),
        '-r', str(fa_reference),
        '-o', str(output_path),
        '-n', 'NearestNeighbor',  # Critical for preserving integer labels
    ]

    for t in transform_list:
        cmd.extend(['-t', t])

    print(f"\nPropagating atlas to FA space...")
    print(f"  Atlas: {atlas_path.name}")
    print(f"  Reference: {fa_reference.name}")
    print(f"  Transform chain: {len(transform_list)} transforms")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: Atlas propagation failed!")
        print(result.stderr)
        raise RuntimeError("antsApplyTransforms failed")

    # Verify output
    atlas_img = nib.load(output_path)
    atlas_data = atlas_img.get_fdata()
    n_labels = len(set(atlas_data[atlas_data > 0].astype(int)))

    print(f"  Output: {output_path}")
    print(f"  Unique labels: {n_labels}")

    return output_path


def propagate_atlas_to_bold(
    atlas_path: Path,
    bold_reference: Path,
    transforms_root: Path,
    templates_root: Path,
    subject: str,
    session: str,
    output_path: Path
) -> Path:
    """
    Propagate SIGMA atlas to BOLD/fMRI space through the transform chain.

    .. deprecated::
        Use :func:`propagate_atlas_to_bold_direct` instead for better overlap
        with fewer transforms.

    Uses the multi-stage registration chain:
        SIGMA → T2w Template → Subject T2w → Subject BOLD

    Parameters
    ----------
    atlas_path : Path
        Path to SIGMA atlas parcellation
    bold_reference : Path
        Subject BOLD reference volume (defines output space)
    transforms_root : Path
        Root directory for subject transforms
    templates_root : Path
        Root directory for templates
    subject : str
        Subject ID (e.g., 'sub-Rat1')
    session : str
        Session ID (e.g., 'ses-p60')
    output_path : Path
        Output path for atlas in BOLD space

    Returns
    -------
    Path
        Path to atlas in BOLD space

    Raises
    ------
    FileNotFoundError
        If required transforms are missing
    """
    cohort = session.replace('ses-', '')

    # Locate all required transforms
    subj_transforms = transforms_root / subject / session
    tpl_transforms = templates_root / 'anat' / cohort / 'transforms'

    # Helper to find transforms with multiple naming patterns
    def _find(directory, *names):
        for name in names:
            p = directory / name
            if p.exists():
                return p
        return None

    # Required transforms (try prefixed and unprefixed patterns)
    bold_to_t2w = _find(subj_transforms, 'BOLD_to_T2w_0GenericAffine.mat')
    t2w_to_tpl_affine = _find(
        subj_transforms,
        f'{subject}_{session}_T2w_to_template_0GenericAffine.mat',
        'T2w_to_template_0GenericAffine.mat'
    )
    t2w_to_tpl_inv_warp = _find(
        subj_transforms,
        f'{subject}_{session}_T2w_to_template_1InverseWarp.nii.gz',
        'T2w_to_template_1InverseWarp.nii.gz'
    )
    tpl_to_sigma_affine = _find(tpl_transforms, 'tpl-to-SIGMA_0GenericAffine.mat')
    tpl_to_sigma_inv_warp = _find(tpl_transforms, 'tpl-to-SIGMA_1InverseWarp.nii.gz')

    # Check BOLD→T2w exists
    if not bold_to_t2w:
        raise FileNotFoundError(
            f"BOLD→T2w transform not found in: {subj_transforms}\n"
            "Run BOLD-to-T2w registration first."
        )

    # Check T2w→Template exists
    if not t2w_to_tpl_affine:
        raise FileNotFoundError(
            f"T2w→Template transform not found in: {subj_transforms}\n"
            "Run subject-to-template registration first."
        )

    # Check Template→SIGMA exists
    if not tpl_to_sigma_affine:
        raise FileNotFoundError(
            f"Template→SIGMA transform not found in: {tpl_transforms}\n"
            "Run template-to-SIGMA registration first."
        )

    # Build transform chain for SIGMA → BOLD (inverse direction)
    # ANTs applies transforms in reverse order, so list from BOLD to SIGMA
    transform_list = []

    # 1. T2w → BOLD (inverse of BOLD → T2w)
    transform_list.append(f"[{bold_to_t2w},1]")

    # 2. Template → T2w (inverse of T2w → Template)
    if t2w_to_tpl_inv_warp.exists():
        transform_list.append(str(t2w_to_tpl_inv_warp))
    transform_list.append(f"[{t2w_to_tpl_affine},1]")

    # 3. SIGMA → Template (inverse of Template → SIGMA)
    if tpl_to_sigma_inv_warp.exists():
        transform_list.append(str(tpl_to_sigma_inv_warp))
    transform_list.append(f"[{tpl_to_sigma_affine},1]")

    # Apply transforms
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(atlas_path),
        '-r', str(bold_reference),
        '-o', str(output_path),
        '-n', 'NearestNeighbor',  # Critical for preserving integer labels
    ]

    for t in transform_list:
        cmd.extend(['-t', t])

    print(f"\nPropagating atlas to BOLD space...")
    print(f"  Atlas: {atlas_path.name}")
    print(f"  Reference: {bold_reference.name}")
    print(f"  Transform chain: {len(transform_list)} transforms")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: Atlas propagation failed!")
        print(result.stderr)
        raise RuntimeError("antsApplyTransforms failed")

    # Verify output
    atlas_img = nib.load(output_path)
    atlas_data = atlas_img.get_fdata()
    n_labels = len(set(atlas_data[atlas_data > 0].astype(int)))

    print(f"  Output: {output_path}")
    print(f"  Unique labels: {n_labels}")

    return output_path


def _warp_single_volume(
    t: int,
    vol_data: np.ndarray,
    input_affine: np.ndarray,
    input_header,
    transforms: List[str],
    reference: Path,
    ref_affine: np.ndarray,
    mask_data: np.ndarray,
    tmpdir_path: Path,
) -> str:
    """Warp, mask, and save one 3D volume. Returns path to masked output."""
    vol_img = nib.Nifti1Image(vol_data, input_affine, input_header)
    vol_img.header.set_data_shape(vol_data.shape)

    vol_in = tmpdir_path / f"vol_{t:04d}_in.nii.gz"
    vol_out = tmpdir_path / f"vol_{t:04d}_out.nii.gz"
    masked_path = tmpdir_path / f"vol_{t:04d}_masked.nii.gz"
    nib.save(vol_img, vol_in)

    cmd = [
        "antsApplyTransforms",
        "-d", "3",
        "-i", str(vol_in),
        "-r", str(reference),
        "-o", str(vol_out),
        "-n", "Linear",
    ]
    for tf in transforms:
        cmd.extend(["-t", tf])

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"volume {t}: {result.stderr[:300]}")

    # Load warped volume, apply mask, save back to disk
    warped_vol = nib.load(vol_out).get_fdata(dtype=np.float32)
    warped_vol *= mask_data
    nib.save(nib.Nifti1Image(warped_vol, ref_affine), masked_path)

    # Clean up intermediate files
    vol_in.unlink(missing_ok=True)
    vol_out.unlink(missing_ok=True)

    return str(masked_path)


def _warp_4d_volumewise(
    input_path: Path,
    output_path: Path,
    transforms: List[str],
    reference: Path,
    mask_data: np.ndarray,
    n_threads: int = 6,
    work_dir: Optional[Path] = None,
) -> bool:
    """
    Warp a 4D NIfTI volume-by-volume to keep memory low.

    Extracts each 3D volume, warps it with antsApplyTransforms, masks it,
    saves it as a temp file, then concatenates all volumes with fslmerge.
    Volumes are warped in parallel using ``n_threads`` concurrent workers.

    Peak RAM ≈ n_threads × one 3D SIGMA volume (~14 MB each) rather than
    the full 4D (~5-10 GB).

    Parameters
    ----------
    input_path : Path
        4D NIfTI input file.
    output_path : Path
        Where to write the warped+masked 4D output.
    transforms : list of str
        Ordered transform files for antsApplyTransforms.
    reference : Path
        Reference image defining output geometry (e.g. SIGMA template).
    mask_data : np.ndarray
        Boolean brain mask in reference space.
    n_threads : int
        Number of volumes to warp concurrently (default 6).
    work_dir : Path, optional
        Directory for temp files. Must be on a real filesystem, not tmpfs.
        Defaults to output_path's parent directory.

    Returns
    -------
    bool
        True on success, False if any volume failed.
    """
    input_img = nib.load(input_path)
    n_timepoints = input_img.shape[3]
    ref_img = nib.load(reference)

    # Temp files go on a real filesystem — never /tmp which is often
    # tmpfs (RAM-backed) and would defeat the point of low-memory warping.
    tmpdir_base = work_dir or output_path.parent
    tmpdir_base.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="neurofaune_warp4d_", dir=tmpdir_base)
    tmpdir_path = Path(tmpdir)

    try:
        # Pre-extract all volumes from the compressed input so that
        # parallel threads don't contend on the same gzip stream.
        # Each 3D volume is small (~0.5 MB for 92×160×9 native BOLD).
        vol_arrays = []
        for t in range(n_timepoints):
            vol_arrays.append(np.asarray(input_img.dataobj[..., t]))

        # Submit all volumes to the thread pool
        futures = {}
        done_count = 0
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for t in range(n_timepoints):
                fut = pool.submit(
                    _warp_single_volume,
                    t=t,
                    vol_data=vol_arrays[t],
                    input_affine=input_img.affine,
                    input_header=input_img.header,
                    transforms=transforms,
                    reference=reference,
                    ref_affine=ref_img.affine,
                    mask_data=mask_data,
                    tmpdir_path=tmpdir_path,
                )
                futures[fut] = t

            for fut in as_completed(futures):
                done_count += 1
                t = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"    ERROR at volume {t}: {e}")
                    return False
                if done_count % 50 == 0 or done_count == n_timepoints:
                    print(f"    Warped {done_count}/{n_timepoints} volumes...")

        # Free the extracted volume arrays
        del vol_arrays

        print(f"    Merging {n_timepoints} volumes...")

        # Collect masked paths in temporal order for fslmerge
        masked_paths = [
            str(tmpdir_path / f"vol_{t:04d}_masked.nii.gz")
            for t in range(n_timepoints)
        ]

        merge_cmd = ["fslmerge", "-t", str(output_path)] + masked_paths
        result = subprocess.run(
            merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            print(f"    ERROR in fslmerge: {result.stderr[:300]}")
            return False

        return True

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def warp_bold_to_sigma(
    input_files: Dict[str, Path],
    transforms: List[str],
    sigma_template: Path,
    sigma_brain_mask: Path,
    output_dir: Path,
    subject: str,
    session: str,
    low_memory: bool = True,
    n_threads: int = 6,
    work_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Warp BOLD-space maps to SIGMA atlas space.

    Applies a pre-built transform chain (BOLD -> ... -> SIGMA), then
    masks with the SIGMA brain mask. Supports both 3D maps (fALFF, ReHo)
    and 4D timeseries (preprocessed BOLD).

    The transform chain is passed as an ordered list of transform file
    paths (ANTs convention: listed outermost-first, applied last-first).
    For the typical 3-hop chain (BOLD -> T2w -> Template -> SIGMA)::

        transforms = [
            tpl-to-SIGMA_1Warp.nii.gz,       # SyN warp
            tpl-to-SIGMA_0GenericAffine.mat,  # affine
            T2w_to_template_1Warp.nii.gz,     # SyN warp
            T2w_to_template_0GenericAffine.mat,# affine
            BOLD_to_T2w_0GenericAffine.mat,   # rigid
        ]

    For a 2-hop chain (BOLD -> Template -> SIGMA)::

        transforms = [
            tpl-to-SIGMA_1Warp.nii.gz,
            tpl-to-SIGMA_0GenericAffine.mat,
            BOLD_to_template_0GenericAffine.mat,
        ]

    Parameters
    ----------
    input_files : dict
        Mapping of description to path, e.g.
        {'desc-fALFF_bold': path, 'desc-preproc_bold': path, ...}
    transforms : list of str
        Ordered transform files for antsApplyTransforms (outermost first).
    sigma_template : Path
        SIGMA reference image (defines output geometry)
    sigma_brain_mask : Path
        SIGMA brain mask for masking warped outputs
    output_dir : Path
        Directory for SIGMA-space outputs (derivatives func dir)
    subject : str
        Subject ID
    session : str
        Session ID
    low_memory : bool, optional
        If True (default), warp 4D timeseries volume-by-volume to reduce
        peak memory. If False, pass the full 4D to antsApplyTransforms -e 3.
    n_threads : int, optional
        Number of volumes to warp in parallel (default 6). Only used when
        low_memory is True.
    work_dir : Path, optional
        Directory for temp files during volume-by-volume warping. Must be
        on a real filesystem (not tmpfs). Defaults to output_dir.

    Returns
    -------
    dict
        Mapping of description to SIGMA-space output path
    """
    print("\n" + "=" * 60)
    print("Warp BOLD Maps to SIGMA Space")
    print("=" * 60)

    print(f"\n  Transform chain ({len(transforms)} transforms):")
    for t in transforms:
        print(f"    {Path(t).name}")
    print(f"  Reference: {sigma_template.name}")
    print(f"  Brain mask: {sigma_brain_mask.name}")

    # Load brain mask once
    mask_img = nib.load(sigma_brain_mask)
    mask_data = mask_img.get_fdata().astype(bool)

    sigma_outputs = {}

    for desc, input_path in input_files.items():
        output_path = output_dir / f"{subject}_{session}_space-SIGMA_{desc}.nii.gz"

        if output_path.exists():
            print(f"\n  {desc}: already exists, skipping")
            sigma_outputs[desc] = output_path
            continue

        if not input_path.exists():
            print(f"\n  {desc}: input not found ({input_path.name}), skipping")
            continue

        print(f"\n  {desc}: warping to SIGMA space...")

        # Detect dimensionality
        img = nib.load(input_path)
        is_4d = len(img.shape) == 4 and img.shape[3] > 1

        if is_4d and low_memory:
            # Volume-by-volume warping to reduce peak memory
            print(f"    Using volume-by-volume warping ({img.shape[3]} volumes, {n_threads} threads)")
            success = _warp_4d_volumewise(
                input_path=input_path,
                output_path=output_path,
                transforms=transforms,
                reference=sigma_template,
                mask_data=mask_data,
                n_threads=n_threads,
                work_dir=work_dir,
            )
            if not success:
                print(f"    ERROR: volume-by-volume warping failed for {desc}")
                continue
        else:
            # Standard single-call warping (3D maps, or 4D with low_memory=False)
            cmd = [
                "antsApplyTransforms",
                "-d", "3",
                "-i", str(input_path),
                "-r", str(sigma_template),
                "-o", str(output_path),
                "-n", "Linear",
            ]

            if is_4d:
                cmd.extend(["-e", "3"])  # time-series mode

            for t in transforms:
                cmd.extend(["-t", t])

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if result.returncode != 0:
                print(f"    ERROR: antsApplyTransforms failed for {desc}")
                print(f"    {result.stderr[:500]}")
                continue

            # Apply brain mask
            warped_img = nib.load(output_path)
            warped_data = warped_img.get_fdata()

            if is_4d:
                warped_data *= mask_data[..., np.newaxis]
            else:
                warped_data *= mask_data

            nib.save(
                nib.Nifti1Image(
                    warped_data.astype(np.float32),
                    warped_img.affine,
                    warped_img.header,
                ),
                output_path,
            )

        sigma_outputs[desc] = output_path
        print(f"    {output_path.name}")

    print(f"\n  Warped {len(sigma_outputs)}/{len(input_files)} maps to SIGMA space")
    return sigma_outputs
