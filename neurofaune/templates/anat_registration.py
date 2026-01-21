"""
Anatomical registration utilities.

This module provides functions for:
- Registering preprocessed T2w to cohort templates
- Propagating SIGMA atlas labels to T2w space
- Direct T2w-to-SIGMA registration (optional mode)
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import nibabel as nib
import numpy as np

from neurofaune.templates.manifest import TemplateManifest, find_template_manifest
from neurofaune.templates.registration_qc import (
    compute_registration_metrics,
    generate_registration_qc_figure,
    generate_atlas_overlay_figure
)


def register_anat_to_template(
    t2w_file: Path,
    template_file: Path,
    output_dir: Path,
    subject: str,
    session: str,
    mask_file: Optional[Path] = None,
    template_mask: Optional[Path] = None,
    n_cores: int = 4,
    generate_qc: bool = True
) -> Dict[str, Path]:
    """
    Register preprocessed T2w to cohort template.

    Uses ANTs SyN registration for accurate non-linear alignment.

    Parameters
    ----------
    t2w_file : Path
        Preprocessed T2w image (skull-stripped, bias-corrected)
    template_file : Path
        Cohort template (fixed image)
    output_dir : Path
        Output directory for transforms
    subject : str
        Subject ID (e.g., 'sub-Rat1')
    session : str
        Session ID (e.g., 'ses-p60')
    mask_file : Path, optional
        Brain mask for T2w
    template_mask : Path, optional
        Brain mask for template
    n_cores : int
        Number of CPU cores
    generate_qc : bool
        Generate QC figures

    Returns
    -------
    dict
        Dictionary with paths to:
        - affine_transform: Affine transform (.mat)
        - warp_transform: Deformation field (T2w → Template)
        - inverse_warp: Inverse deformation field (Template → T2w)
        - warped_t2w: T2w in template space
        - qc_figure: QC figure path (if generate_qc=True)
        - metrics: Registration metrics dict
    """
    print(f"\n{'='*60}")
    print(f"Registering {subject} {session} to template")
    print(f"{'='*60}")
    print(f"  T2w: {t2w_file.name}")
    print(f"  Template: {template_file.name}")

    # Validate inputs
    if not t2w_file.exists():
        raise FileNotFoundError(f"T2w file not found: {t2w_file}")
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output prefix
    output_prefix = output_dir / f'{subject}_{session}_T2w_to_template_'

    # Build ANTs registration command
    cmd = [
        'antsRegistrationSyN.sh',
        '-d', '3',
        '-f', str(template_file),    # Fixed (template)
        '-m', str(t2w_file),          # Moving (subject T2w)
        '-o', str(output_prefix),
        '-n', str(n_cores),
        '-t', 's'  # SyN transform
    ]

    # Add masks if provided
    if mask_file and mask_file.exists() and template_mask and template_mask.exists():
        # Create combined mask approach - use template mask as reference
        cmd.extend(['-x', str(template_mask)])

    print(f"\nRunning ANTs registration...")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: Registration failed!")
        print(result.stdout)
        raise RuntimeError(f"ANTs registration failed for {subject} {session}")

    # Expected outputs
    affine_transform = Path(str(output_prefix) + '0GenericAffine.mat')
    warp_transform = Path(str(output_prefix) + '1Warp.nii.gz')
    inverse_warp = Path(str(output_prefix) + '1InverseWarp.nii.gz')
    warped_t2w = Path(str(output_prefix) + 'Warped.nii.gz')

    # Verify outputs exist
    if not affine_transform.exists():
        raise FileNotFoundError(f"Expected affine transform not found: {affine_transform}")

    print(f"  ✓ Affine: {affine_transform.name}")
    print(f"  ✓ Warp: {warp_transform.name}")
    print(f"  ✓ Inverse warp: {inverse_warp.name}")
    print(f"  ✓ Warped T2w: {warped_t2w.name}")

    results = {
        'affine_transform': affine_transform,
        'warp_transform': warp_transform,
        'inverse_warp': inverse_warp,
        'warped_t2w': warped_t2w,
    }

    # Compute metrics and generate QC
    if generate_qc:
        print(f"\nGenerating QC...")

        # Compute metrics
        metrics = compute_registration_metrics(
            moving_file=t2w_file,
            fixed_file=template_file,
            warped_file=warped_t2w,
            fixed_mask=template_mask
        )

        print(f"  Correlation (before): {metrics['correlation_before']:.3f}")
        print(f"  Correlation (after):  {metrics['correlation_after']:.3f}")

        results['metrics'] = metrics

        # Generate QC figure
        qc_figure = output_dir / f'{subject}_{session}_registration_qc.png'
        generate_registration_qc_figure(
            fixed_file=template_file,
            warped_file=warped_t2w,
            output_file=qc_figure,
            fixed_mask=template_mask,
            title=f"T2w-to-Template: {subject} {session}"
        )
        results['qc_figure'] = qc_figure

    return results


def propagate_atlas_to_anat(
    atlas_file: Path,
    t2w_reference: Path,
    transforms_dir: Path,
    templates_dir: Path,
    subject: str,
    session: str,
    output_file: Path,
    generate_qc: bool = True,
    qc_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Propagate SIGMA atlas to T2w space through the transform chain.

    Transform chain (inverse direction):
        SIGMA (study-space) → Template → Subject T2w

    IMPORTANT: atlas_file should be the STUDY-SPACE atlas (already reoriented
    to match study acquisition orientation), NOT the original SIGMA atlas.
    Use config['atlas']['study_space']['parcellation'] for the correct path.

    Parameters
    ----------
    atlas_file : Path
        SIGMA atlas parcellation in STUDY SPACE (reoriented to match study)
    t2w_reference : Path
        Subject T2w (defines output space)
    transforms_dir : Path
        Directory containing subject transforms
    templates_dir : Path
        Directory containing template transforms
    subject : str
        Subject ID
    session : str
        Session ID
    output_file : Path
        Output path for atlas in T2w space
    generate_qc : bool
        Generate QC overlay figure
    qc_dir : Path, optional
        QC output directory (uses output_file.parent if None)

    Returns
    -------
    dict
        Dictionary with:
        - atlas_file: Path to atlas in T2w space
        - n_labels: Number of unique labels
        - coverage: Fraction of brain covered
        - qc_figure: QC figure path (if generate_qc=True)
    """
    cohort = session.replace('ses-', '')

    print(f"\n{'='*60}")
    print(f"Propagating SIGMA atlas to {subject} {session} T2w space")
    print(f"{'='*60}")

    # Locate transforms
    subj_transforms = Path(transforms_dir) / subject / session
    tpl_transforms = Path(templates_dir) / 'anat' / cohort / 'transforms'

    # Required transforms
    t2w_to_tpl_affine = subj_transforms / f'{subject}_{session}_T2w_to_template_0GenericAffine.mat'
    t2w_to_tpl_inv_warp = subj_transforms / f'{subject}_{session}_T2w_to_template_1InverseWarp.nii.gz'
    tpl_to_sigma_affine = tpl_transforms / 'tpl-to-SIGMA_0GenericAffine.mat'
    tpl_to_sigma_inv_warp = tpl_transforms / 'tpl-to-SIGMA_1InverseWarp.nii.gz'

    # Check transforms exist
    if not t2w_to_tpl_affine.exists():
        raise FileNotFoundError(
            f"T2w→Template affine not found: {t2w_to_tpl_affine}\n"
            "Run register_anat_to_template() first."
        )
    if not tpl_to_sigma_affine.exists():
        raise FileNotFoundError(
            f"Template→SIGMA affine not found: {tpl_to_sigma_affine}\n"
            "Run template-to-SIGMA registration first."
        )

    print(f"  T2w→Template affine: {t2w_to_tpl_affine.name}")
    print(f"  T2w→Template warp: {t2w_to_tpl_inv_warp.name if t2w_to_tpl_inv_warp.exists() else 'N/A'}")
    print(f"  Template→SIGMA affine: {tpl_to_sigma_affine.name}")
    print(f"  Template→SIGMA warp: {tpl_to_sigma_inv_warp.name if tpl_to_sigma_inv_warp.exists() else 'N/A'}")

    # Build transform chain for SIGMA → T2w
    # ANTs applies transforms in reverse order
    transform_list = []

    # 1. Template → T2w (inverse of T2w → Template)
    if t2w_to_tpl_inv_warp.exists():
        transform_list.append(str(t2w_to_tpl_inv_warp))
    transform_list.append(f"[{t2w_to_tpl_affine},1]")  # Invert affine

    # 2. SIGMA → Template (inverse of Template → SIGMA)
    if tpl_to_sigma_inv_warp.exists():
        transform_list.append(str(tpl_to_sigma_inv_warp))
    transform_list.append(f"[{tpl_to_sigma_affine},1]")  # Invert affine

    # Apply transforms
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(atlas_file),
        '-r', str(t2w_reference),
        '-o', str(output_file),
        '-n', 'NearestNeighbor',  # Critical for preserving integer labels
    ]

    for t in transform_list:
        cmd.extend(['-t', t])

    print(f"\nApplying transforms ({len(transform_list)} total)...")

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

    # Verify output and compute stats
    atlas_img = nib.load(output_file)
    atlas_data = atlas_img.get_fdata()

    unique_labels = set(atlas_data[atlas_data > 0].astype(int))
    n_labels = len(unique_labels)

    # Compute coverage (fraction of reference brain with labels)
    ref_img = nib.load(t2w_reference)
    ref_data = ref_img.get_fdata()
    brain_voxels = np.sum(ref_data > 0)
    labeled_voxels = np.sum(atlas_data > 0)
    coverage = labeled_voxels / brain_voxels if brain_voxels > 0 else 0

    print(f"  ✓ Atlas saved: {output_file.name}")
    print(f"  Unique labels: {n_labels}")
    print(f"  Coverage: {coverage:.1%}")

    results = {
        'atlas_file': output_file,
        'n_labels': n_labels,
        'coverage': float(coverage),
    }

    # Generate QC figure
    if generate_qc:
        qc_dir = qc_dir or output_file.parent
        qc_figure = Path(qc_dir) / f'{subject}_{session}_atlas_overlay.png'

        generate_atlas_overlay_figure(
            anatomical_file=t2w_reference,
            atlas_file=output_file,
            output_file=qc_figure,
            title=f"SIGMA Atlas in T2w Space: {subject} {session}"
        )
        results['qc_figure'] = qc_figure

    return results


def register_anat_to_sigma_direct(
    t2w_file: Path,
    sigma_template: Path,
    output_dir: Path,
    subject: str,
    session: str,
    mask_file: Optional[Path] = None,
    sigma_mask: Optional[Path] = None,
    n_cores: int = 4,
    generate_qc: bool = True
) -> Dict[str, Path]:
    """
    Register T2w directly to SIGMA (no study template).

    WARNING: This is less accurate than template-based registration.
    Use only when template building is not feasible.

    Parameters
    ----------
    t2w_file : Path
        Preprocessed T2w image
    sigma_template : Path
        SIGMA atlas template
    output_dir : Path
        Output directory for transforms
    subject : str
        Subject ID
    session : str
        Session ID
    mask_file : Path, optional
        Brain mask for T2w
    sigma_mask : Path, optional
        SIGMA brain mask
    n_cores : int
        Number of CPU cores
    generate_qc : bool
        Generate QC figures

    Returns
    -------
    dict
        Dictionary with paths to registration outputs
    """
    print(f"\n{'='*60}")
    print(f"⚠ DIRECT-TO-SIGMA REGISTRATION")
    print(f"{'='*60}")
    print(f"WARNING: Direct T2w→SIGMA registration is less accurate than")
    print(f"template-based registration. Consider building a study template.")
    print(f"{'='*60}")
    print(f"  Subject: {subject} {session}")
    print(f"  T2w: {t2w_file.name}")
    print(f"  SIGMA: {sigma_template.name}")

    # Validate inputs
    if not t2w_file.exists():
        raise FileNotFoundError(f"T2w file not found: {t2w_file}")
    if not sigma_template.exists():
        raise FileNotFoundError(f"SIGMA template not found: {sigma_template}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output prefix
    output_prefix = output_dir / f'{subject}_{session}_T2w_to_SIGMA_'

    # Build ANTs registration command
    cmd = [
        'antsRegistrationSyN.sh',
        '-d', '3',
        '-f', str(sigma_template),   # Fixed (SIGMA)
        '-m', str(t2w_file),          # Moving (subject T2w)
        '-o', str(output_prefix),
        '-n', str(n_cores),
        '-t', 's'  # SyN transform
    ]

    if sigma_mask and Path(sigma_mask).exists():
        cmd.extend(['-x', str(sigma_mask)])

    print(f"\nRunning ANTs registration...")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: Registration failed!")
        print(result.stdout)
        raise RuntimeError(f"ANTs registration failed for {subject} {session}")

    # Expected outputs
    affine_transform = Path(str(output_prefix) + '0GenericAffine.mat')
    warp_transform = Path(str(output_prefix) + '1Warp.nii.gz')
    inverse_warp = Path(str(output_prefix) + '1InverseWarp.nii.gz')
    warped_t2w = Path(str(output_prefix) + 'Warped.nii.gz')

    if not affine_transform.exists():
        raise FileNotFoundError(f"Expected affine transform not found: {affine_transform}")

    print(f"  ✓ Affine: {affine_transform.name}")
    print(f"  ✓ Warp: {warp_transform.name}")
    print(f"  ✓ Inverse warp: {inverse_warp.name}")
    print(f"  ✓ Warped T2w: {warped_t2w.name}")

    results = {
        'affine_transform': affine_transform,
        'warp_transform': warp_transform,
        'inverse_warp': inverse_warp,
        'warped_t2w': warped_t2w,
    }

    # Generate QC
    if generate_qc:
        metrics = compute_registration_metrics(
            moving_file=t2w_file,
            fixed_file=sigma_template,
            warped_file=warped_t2w,
            fixed_mask=sigma_mask
        )

        print(f"\nQC Metrics:")
        print(f"  Correlation (before): {metrics['correlation_before']:.3f}")
        print(f"  Correlation (after):  {metrics['correlation_after']:.3f}")

        results['metrics'] = metrics

        qc_figure = output_dir / f'{subject}_{session}_direct_registration_qc.png'
        generate_registration_qc_figure(
            fixed_file=sigma_template,
            warped_file=warped_t2w,
            output_file=qc_figure,
            fixed_mask=sigma_mask,
            title=f"Direct T2w-to-SIGMA: {subject} {session}"
        )
        results['qc_figure'] = qc_figure

    return results


def propagate_atlas_direct(
    atlas_file: Path,
    t2w_reference: Path,
    transforms_dir: Path,
    subject: str,
    session: str,
    output_file: Path,
    generate_qc: bool = True,
    qc_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Propagate SIGMA atlas to T2w using direct registration transforms.

    For use with register_anat_to_sigma_direct() when no template is available.

    Parameters
    ----------
    atlas_file : Path
        SIGMA atlas parcellation
    t2w_reference : Path
        Subject T2w (defines output space)
    transforms_dir : Path
        Directory containing direct T2w→SIGMA transforms
    subject : str
        Subject ID
    session : str
        Session ID
    output_file : Path
        Output path for atlas in T2w space
    generate_qc : bool
        Generate QC overlay figure
    qc_dir : Path, optional
        QC output directory

    Returns
    -------
    dict
        Dictionary with atlas info and QC paths
    """
    print(f"\nPropagating atlas using direct transforms for {subject} {session}...")

    # Locate transforms
    subj_transforms = Path(transforms_dir) / subject / session

    t2w_to_sigma_affine = subj_transforms / f'{subject}_{session}_T2w_to_SIGMA_0GenericAffine.mat'
    t2w_to_sigma_inv_warp = subj_transforms / f'{subject}_{session}_T2w_to_SIGMA_1InverseWarp.nii.gz'

    if not t2w_to_sigma_affine.exists():
        raise FileNotFoundError(
            f"T2w→SIGMA affine not found: {t2w_to_sigma_affine}\n"
            "Run register_anat_to_sigma_direct() first."
        )

    # Build transform chain for SIGMA → T2w
    transform_list = []

    if t2w_to_sigma_inv_warp.exists():
        transform_list.append(str(t2w_to_sigma_inv_warp))
    transform_list.append(f"[{t2w_to_sigma_affine},1]")

    # Apply transforms
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(atlas_file),
        '-r', str(t2w_reference),
        '-o', str(output_file),
        '-n', 'NearestNeighbor',
    ]

    for t in transform_list:
        cmd.extend(['-t', t])

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("ERROR: Atlas propagation failed!")
        print(result.stderr)
        raise RuntimeError("antsApplyTransforms failed")

    # Compute stats
    atlas_img = nib.load(output_file)
    atlas_data = atlas_img.get_fdata()
    n_labels = len(set(atlas_data[atlas_data > 0].astype(int)))

    ref_data = nib.load(t2w_reference).get_fdata()
    brain_voxels = np.sum(ref_data > 0)
    labeled_voxels = np.sum(atlas_data > 0)
    coverage = labeled_voxels / brain_voxels if brain_voxels > 0 else 0

    print(f"  ✓ Atlas saved: {output_file.name}")
    print(f"  Unique labels: {n_labels}")
    print(f"  Coverage: {coverage:.1%}")

    results = {
        'atlas_file': output_file,
        'n_labels': n_labels,
        'coverage': float(coverage),
    }

    if generate_qc:
        qc_dir = qc_dir or output_file.parent
        qc_figure = Path(qc_dir) / f'{subject}_{session}_atlas_overlay_direct.png'

        generate_atlas_overlay_figure(
            anatomical_file=t2w_reference,
            atlas_file=output_file,
            output_file=qc_figure,
            title=f"SIGMA Atlas (Direct): {subject} {session}"
        )
        results['qc_figure'] = qc_figure

    return results
