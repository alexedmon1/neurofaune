#!/usr/bin/env python3
"""
007_register_subject_to_template.py

Register subject T2w to age-matched cohort template.

This creates the middle link in the registration chain:
    Subject FA → Subject T2w → Template T2w → SIGMA

Usage:
    python 007_register_subject_to_template.py /path/to/bpa-rat sub-Rat1 ses-p60

Prerequisites:
    - Subject T2w preprocessed
    - Cohort template exists (templates/anat/{cohort}/)

Output:
    transforms/{subject}/{session}/
        - T2w_to_template_0GenericAffine.mat
        - T2w_to_template_1Warp.nii.gz
        - T2w_to_template_1InverseWarp.nii.gz
        - T2w_to_template_Warped.nii.gz
"""

import argparse
import subprocess
import sys
from pathlib import Path

import nibabel as nib


def register_to_template(
    moving_image: Path,
    template_image: Path,
    output_prefix: Path,
    transform_type: str = 'SyN',
    n_cores: int = 4
):
    """
    Register subject image to template using ANTs.

    Parameters
    ----------
    moving_image : Path
        Subject image (e.g., preprocessed T2w)
    template_image : Path
        Template image (e.g., cohort T2w template)
    output_prefix : Path
        Output prefix for transform files
    transform_type : str
        'SyN' for deformable, 'Affine' for affine-only
    n_cores : int
        Number of CPU cores

    Returns
    -------
    dict
        Paths to output files
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Map transform type to antsRegistrationSyN.sh flag
    transform_flag = {'SyN': 's', 'Affine': 'a', 'Rigid': 'r'}[transform_type]

    cmd = [
        'antsRegistrationSyN.sh',
        '-d', '3',
        '-f', str(template_image),
        '-m', str(moving_image),
        '-o', str(output_prefix),
        '-n', str(n_cores),
        '-t', transform_flag
    ]

    print(f"\n  Running ANTs {transform_type} registration...")
    print(f"  Moving: {moving_image.name}")
    print(f"  Fixed: {template_image.name}")
    print(f"  Output: {output_prefix}")

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
    outputs = {
        'affine': Path(str(output_prefix) + '0GenericAffine.mat'),
        'warped': Path(str(output_prefix) + 'Warped.nii.gz'),
    }

    # SyN also produces warp fields
    if transform_type == 'SyN':
        outputs['warp'] = Path(str(output_prefix) + '1Warp.nii.gz')
        outputs['inverse_warp'] = Path(str(output_prefix) + '1InverseWarp.nii.gz')

    # Verify outputs
    for name, path in outputs.items():
        if path.exists():
            print(f"  ✓ {name}: {path.name}")
        else:
            print(f"  ✗ {name}: NOT FOUND")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description='Register subject T2w to cohort template'
    )
    parser.add_argument('study_root', type=Path, help='Path to study root')
    parser.add_argument('subject', type=str, help='Subject ID')
    parser.add_argument('session', type=str, help='Session ID')
    parser.add_argument('--transform-type', choices=['SyN', 'Affine'],
                        default='SyN', help='Registration type (default: SyN)')
    parser.add_argument('--n-cores', type=int, default=4, help='CPU cores')
    args = parser.parse_args()

    # Normalize IDs
    subject = args.subject if args.subject.startswith('sub-') else f'sub-{args.subject}'
    session = args.session if args.session.startswith('ses-') else f'ses-{args.session}'
    cohort = session.replace('ses-', '')

    derivatives_root = args.study_root / 'derivatives'
    templates_root = args.study_root / 'templates'
    transforms_root = args.study_root / 'transforms'

    print("=" * 70)
    print(f"Subject to Template Registration: {subject} / {session}")
    print("=" * 70)

    # Find subject T2w
    t2w_path = derivatives_root / subject / session / 'anat' / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
    if not t2w_path.exists():
        print(f"ERROR: Subject T2w not found: {t2w_path}")
        sys.exit(1)

    # Find cohort template
    template_path = templates_root / 'anat' / cohort / f'tpl-BPARat_{cohort}_T2w.nii.gz'
    if not template_path.exists():
        print(f"ERROR: Template not found: {template_path}")
        sys.exit(1)

    print(f"\nInputs:")
    print(f"  Subject T2w: {t2w_path}")
    print(f"  Template: {template_path}")

    # Check image properties
    subj_img = nib.load(t2w_path)
    tpl_img = nib.load(template_path)

    print(f"\nImage properties:")
    print(f"  Subject: {subj_img.shape} @ {subj_img.header.get_zooms()[:3]} mm")
    print(f"  Template: {tpl_img.shape} @ {tpl_img.header.get_zooms()[:3]} mm")

    # Run registration
    output_prefix = transforms_root / subject / session / 'T2w_to_template_'

    outputs = register_to_template(
        moving_image=t2w_path,
        template_image=template_path,
        output_prefix=output_prefix,
        transform_type=args.transform_type,
        n_cores=args.n_cores
    )

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Transforms saved to: {output_prefix.parent}")
    print()
    print("Full registration chain now available:")
    print(f"  1. FA → T2w: FA_to_T2w_0GenericAffine.mat")
    print(f"  2. T2w → Template: T2w_to_template_*")
    print(f"  3. Template → SIGMA: tpl-to-SIGMA_*")


if __name__ == '__main__':
    main()
