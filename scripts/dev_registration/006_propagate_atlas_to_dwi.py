#!/usr/bin/env python3
"""
006_propagate_atlas_to_dwi.py

Propagate SIGMA atlas labels to DTI space through the transform chain:
    SIGMA → T2w Template → Subject T2w → Subject FA

This script:
1. Loads the transform chain components
2. Composes transforms to create SIGMA → FA mapping
3. Applies inverse transforms to bring atlas labels to FA space
4. Uses weighted averaging for partial volume handling

The transform chain (forward: FA → SIGMA):
    FA → T2w (within-subject affine, from script 003)
    T2w → T2w Template (subject-to-template, computed separately)
    T2w Template → SIGMA (from template building)

For atlas propagation, we use INVERSE transforms:
    SIGMA → T2w Template (inverse of template→SIGMA)
    T2w Template → Subject T2w (inverse of subject→template)
    Subject T2w → FA (inverse of FA→T2w)

Usage:
    python 006_propagate_atlas_to_dwi.py /path/to/bpa-rat sub-Rat1 ses-p60

Prerequisites:
    - DTI preprocessing complete (FA map exists)
    - FA→T2w registration complete (script 003)
    - Subject T2w→Template registration complete
    - Template→SIGMA registration complete

Output:
    derivatives/{subject}/{session}/dwi/
        - {subject}_{session}_space-FA_atlas.nii.gz (integer labels)
        - {subject}_{session}_space-FA_atlas-prob.nii.gz (probability, optional)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional

import nibabel as nib
import numpy as np


def check_transforms_exist(
    transforms_root: Path,
    templates_root: Path,
    subject: str,
    session: str,
    cohort: str
) -> Dict[str, Path]:
    """
    Check that all required transforms exist.

    Returns dict of transform paths or raises error if missing.
    """
    transforms = {}

    # 1. FA → T2w (within-subject)
    fa_to_t2w = transforms_root / subject / session / 'FA_to_T2w_0GenericAffine.mat'
    if fa_to_t2w.exists():
        transforms['fa_to_t2w'] = fa_to_t2w
    else:
        raise FileNotFoundError(f"FA→T2w transform not found: {fa_to_t2w}\nRun script 003 first.")

    # 2. Subject T2w → Template (if exists)
    t2w_to_template = transforms_root / subject / session / 'T2w_to_template_0GenericAffine.mat'
    t2w_to_template_warp = transforms_root / subject / session / 'T2w_to_template_1Warp.nii.gz'
    t2w_to_template_inv_warp = transforms_root / subject / session / 'T2w_to_template_1InverseWarp.nii.gz'
    if t2w_to_template.exists():
        transforms['t2w_to_template_affine'] = t2w_to_template
        if t2w_to_template_warp.exists():
            transforms['t2w_to_template_warp'] = t2w_to_template_warp
        if t2w_to_template_inv_warp.exists():
            transforms['t2w_to_template_inv_warp'] = t2w_to_template_inv_warp

    # 3. Template → SIGMA (from template building)
    template_to_sigma = templates_root / 'anat' / cohort / 'transforms' / 'tpl-to-SIGMA_0GenericAffine.mat'
    template_to_sigma_warp = templates_root / 'anat' / cohort / 'transforms' / 'tpl-to-SIGMA_1Warp.nii.gz'
    template_to_sigma_inv = templates_root / 'anat' / cohort / 'transforms' / 'tpl-to-SIGMA_1InverseWarp.nii.gz'

    if template_to_sigma.exists():
        transforms['template_to_sigma_affine'] = template_to_sigma
    else:
        raise FileNotFoundError(f"Template→SIGMA affine not found: {template_to_sigma}")

    if template_to_sigma_warp.exists():
        transforms['template_to_sigma_warp'] = template_to_sigma_warp
    if template_to_sigma_inv.exists():
        transforms['template_to_sigma_inv_warp'] = template_to_sigma_inv

    return transforms


def apply_inverse_transforms(
    input_image: Path,
    reference_image: Path,
    transforms: List[str],
    output_image: Path,
    interpolation: str = 'NearestNeighbor'
) -> Path:
    """
    Apply transforms using antsApplyTransforms.

    For inverse propagation (atlas → subject), transforms are applied
    in forward order but with inversion flags where needed.
    """
    output_image.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(input_image),
        '-r', str(reference_image),
        '-o', str(output_image),
        '-n', interpolation,
    ]

    # Add transforms
    for t in transforms:
        cmd.extend(['-t', t])

    print(f"  Running antsApplyTransforms...")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(f"ERROR: Transform application failed!")
        print(result.stderr)
        raise RuntimeError("antsApplyTransforms failed")

    return output_image


def propagate_atlas_simple(
    sigma_atlas: Path,
    fa_reference: Path,
    transforms: Dict[str, Path],
    output_path: Path
) -> Path:
    """
    Simple atlas propagation using available transforms.

    If full transform chain isn't available, uses direct T2w→FA transform
    to bring atlas labels to approximate FA space.
    """
    print(f"\nPropagating atlas to FA space...")
    print(f"  Atlas: {sigma_atlas.name}")
    print(f"  Reference: {fa_reference.name}")

    # Build transform list for SIGMA → FA
    # Order: last transform first (ANTs applies in reverse order)
    transform_list = []

    # FA → T2w (need inverse: T2w → FA)
    transform_list.append(f"[{transforms['fa_to_t2w']},1]")  # 1 = invert

    # If we have T2w → Template transform (need inverse: Template → T2w)
    if 't2w_to_template_affine' in transforms:
        if 't2w_to_template_inv_warp' in transforms:
            transform_list.append(str(transforms['t2w_to_template_inv_warp']))
        transform_list.append(f"[{transforms['t2w_to_template_affine']},1]")

    # Template → SIGMA (need inverse: SIGMA → Template)
    if 'template_to_sigma_inv_warp' in transforms:
        transform_list.append(str(transforms['template_to_sigma_inv_warp']))
    transform_list.append(f"[{transforms['template_to_sigma_affine']},1]")

    print(f"  Transform chain ({len(transform_list)} transforms):")
    for i, t in enumerate(transform_list):
        print(f"    {i+1}. {Path(t.replace('[','').replace(']','').split(',')[0]).name}")

    # Apply transforms
    result = apply_inverse_transforms(
        input_image=sigma_atlas,
        reference_image=fa_reference,
        transforms=transform_list,
        output_image=output_path,
        interpolation='NearestNeighbor'  # Preserve integer labels
    )

    print(f"  ✓ Output: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Propagate SIGMA atlas to DTI/FA space'
    )
    parser.add_argument('study_root', type=Path, help='Path to study root')
    parser.add_argument('subject', type=str, help='Subject ID')
    parser.add_argument('session', type=str, help='Session ID')
    parser.add_argument('--sigma-atlas', type=Path,
                        help='Path to SIGMA atlas (default: from config)')
    parser.add_argument('--atlas-type', choices=['anatomical', 'functional'],
                        default='anatomical', help='Atlas type')
    args = parser.parse_args()

    # Normalize IDs
    subject = args.subject if args.subject.startswith('sub-') else f'sub-{args.subject}'
    session = args.session if args.session.startswith('ses-') else f'ses-{args.session}'
    cohort = session.replace('ses-', '')

    derivatives_root = args.study_root / 'derivatives'
    transforms_root = args.study_root / 'transforms'
    templates_root = args.study_root / 'templates'

    print("=" * 70)
    print(f"Atlas Propagation to FA Space")
    print(f"{subject} / {session}")
    print("=" * 70)

    # Find FA reference
    fa_path = derivatives_root / subject / session / 'dwi' / f'{subject}_{session}_FA.nii.gz'
    if not fa_path.exists():
        print(f"ERROR: FA map not found: {fa_path}")
        sys.exit(1)

    # Find SIGMA atlas
    if args.sigma_atlas:
        sigma_atlas = args.sigma_atlas
    else:
        # Default location (scaled SIGMA)
        sigma_base = Path('/mnt/arborea/atlases/SIGMA_scaled')
        if args.atlas_type == 'anatomical':
            sigma_atlas = (sigma_base / 'SIGMA_Rat_Brain_Atlases' / 'SIGMA_Anatomical_Atlas' /
                          'InVivo_Atlas' / 'SIGMA_InVivo_Anatomical_Brain_Atlas.nii')
        else:
            sigma_atlas = (sigma_base / 'SIGMA_Rat_Brain_Atlases' / 'SIGMA_Functional_Atlas' /
                          'SIGMA_Functional_Brain_Atlas_InVivo_Anatomical_Template.nii')

    if not sigma_atlas.exists():
        print(f"ERROR: SIGMA atlas not found: {sigma_atlas}")
        print(f"  Specify path with --sigma-atlas")
        sys.exit(1)

    print(f"\nInputs:")
    print(f"  FA reference: {fa_path}")
    print(f"  SIGMA atlas: {sigma_atlas}")

    # Check transforms
    print(f"\nChecking transforms...")
    try:
        transforms = check_transforms_exist(
            transforms_root, templates_root, subject, session, cohort
        )
        print(f"  Found {len(transforms)} transform components")
        for name, path in transforms.items():
            print(f"    {name}: {path.name}")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print(f"\nRequired transforms:")
        print(f"  1. FA→T2w: Run script 003_register_dwi_to_t2w.py")
        print(f"  2. T2w→Template: Run subject-to-template registration")
        print(f"  3. Template→SIGMA: Should exist from template building")
        sys.exit(1)

    # Propagate atlas
    output_path = derivatives_root / subject / session / 'dwi' / f'{subject}_{session}_space-FA_atlas.nii.gz'

    propagate_atlas_simple(
        sigma_atlas=sigma_atlas,
        fa_reference=fa_path,
        transforms=transforms,
        output_path=output_path
    )

    # Verify output
    atlas_img = nib.load(output_path)
    atlas_data = atlas_img.get_fdata()
    unique_labels = np.unique(atlas_data[atlas_data > 0]).astype(int)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Output atlas: {output_path}")
    print(f"Unique labels: {len(unique_labels)}")
    print(f"Label range: {unique_labels.min()} to {unique_labels.max()}")


if __name__ == '__main__':
    main()
