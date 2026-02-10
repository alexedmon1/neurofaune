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
