#!/usr/bin/env python3
"""
008_register_template_to_sigma.py

Register cohort template to SIGMA atlas in STUDY SPACE.

IMPORTANT: This script now uses the STUDY-SPACE SIGMA atlas, which has been
reoriented to match the study's native acquisition orientation. This avoids
orientation mismatches that previously caused registration failures.

The study-space atlas is created by setup_study_atlas() from:
    neurofaune.templates.slice_registration

Usage:
    python 008_register_template_to_sigma.py /path/to/bpa-rat p60

Prerequisites:
    - Cohort template exists (templates/anat/{cohort}/)
    - Study-space SIGMA atlas exists at {study_root}/atlas/SIGMA_study_space/
      (Run setup_study_atlas() first if not present)

Output:
    templates/anat/{cohort}/transforms/
        - tpl-to-SIGMA_0GenericAffine.mat
        - tpl-to-SIGMA_1Warp.nii.gz
        - tpl-to-SIGMA_1InverseWarp.nii.gz
        - tpl-to-SIGMA_Warped.nii.gz
"""

import argparse
import subprocess
import sys
from pathlib import Path

import nibabel as nib


def register_template_to_sigma(
    template_image: Path,
    sigma_template: Path,
    output_prefix: Path,
    n_cores: int = 4
):
    """
    Register cohort template to SIGMA using ANTs SyN.

    Parameters
    ----------
    template_image : Path
        Cohort T2w template
    sigma_template : Path
        SIGMA InVivo template (STANDARD orientation)
    output_prefix : Path
        Output prefix for transform files
    n_cores : int
        Number of CPU cores

    Returns
    -------
    dict
        Paths to output files
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsRegistrationSyN.sh',
        '-d', '3',
        '-f', str(sigma_template),  # Fixed = SIGMA
        '-m', str(template_image),  # Moving = our template
        '-o', str(output_prefix),
        '-n', str(n_cores),
        '-t', 's'  # SyN (deformable)
    ]

    print(f"\n  Running ANTs SyN registration...")
    print(f"  Moving: {template_image.name}")
    print(f"  Fixed: {sigma_template.name}")
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
        'warp': Path(str(output_prefix) + '1Warp.nii.gz'),
        'inverse_warp': Path(str(output_prefix) + '1InverseWarp.nii.gz'),
        'warped': Path(str(output_prefix) + 'Warped.nii.gz'),
    }

    # Verify outputs
    for name, path in outputs.items():
        if path.exists():
            print(f"  ✓ {name}: {path.name}")
        else:
            print(f"  ✗ {name}: NOT FOUND")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description='Register cohort template to SIGMA atlas (study space)'
    )
    parser.add_argument('study_root', type=Path, help='Path to study root')
    parser.add_argument('cohort', type=str, help='Cohort ID (e.g., p60)')
    parser.add_argument('--n-cores', type=int, default=4, help='CPU cores')
    parser.add_argument('--atlas-path', type=Path, default=None,
                        help='Path to study-space SIGMA atlas (default: {study_root}/atlas/SIGMA_study_space)')
    args = parser.parse_args()

    templates_root = args.study_root / 'templates'

    # Use study-space atlas (default location or specified)
    if args.atlas_path is None:
        atlas_dir = args.study_root / 'atlas' / 'SIGMA_study_space'
    else:
        atlas_dir = args.atlas_path

    print("=" * 70)
    print(f"Template to SIGMA Registration: {args.cohort}")
    print("=" * 70)

    # Find cohort template
    template_path = templates_root / 'anat' / args.cohort / f'tpl-BPARat_{args.cohort}_T2w.nii.gz'
    if not template_path.exists():
        print(f"ERROR: Template not found: {template_path}")
        sys.exit(1)

    # SIGMA template - use STUDY-SPACE version (same orientation as our data)
    sigma_template = atlas_dir / 'SIGMA_InVivo_Brain_Template_Masked.nii.gz'
    if not sigma_template.exists():
        print(f"ERROR: Study-space SIGMA template not found: {sigma_template}")
        print(f"\nPlease run setup_study_atlas() first:")
        print(f"  from neurofaune.templates.slice_registration import setup_study_atlas")
        print(f"  setup_study_atlas(")
        print(f"      sigma_base_path='/mnt/arborea/atlases/SIGMA_scaled',")
        print(f"      study_atlas_dir='{atlas_dir}'")
        print(f"  )")
        sys.exit(1)

    # Also locate the atlas for verification
    sigma_atlas = atlas_dir / 'SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz'

    print(f"\nInputs:")
    print(f"  Cohort template: {template_path}")
    print(f"  SIGMA template: {sigma_template}")
    print(f"  SIGMA atlas: {sigma_atlas}")

    # Check image properties
    tpl_img = nib.load(template_path)
    sigma_img = nib.load(sigma_template)
    atlas_img = nib.load(sigma_atlas)

    print(f"\nImage properties:")
    print(f"  Template: {tpl_img.shape} @ {tpl_img.header.get_zooms()[:3]} mm")
    print(f"  SIGMA template: {sigma_img.shape} @ {sigma_img.header.get_zooms()[:3]} mm")
    print(f"  SIGMA atlas: {atlas_img.shape} @ {atlas_img.header.get_zooms()[:3]} mm")

    # Verify atlas and template match
    if sigma_img.shape != atlas_img.shape:
        print(f"\nWARNING: SIGMA template and atlas have different shapes!")
        print(f"  This would cause atlas propagation to fail.")
        sys.exit(1)
    else:
        print(f"\n  ✓ SIGMA template and atlas geometries match")

    # Back up old transforms if they exist
    output_dir = templates_root / 'anat' / args.cohort / 'transforms'
    old_affine = output_dir / 'tpl-to-SIGMA_0GenericAffine.mat'
    if old_affine.exists():
        import shutil
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = output_dir / f'backup_old_sigma_{timestamp}'
        backup_dir.mkdir(exist_ok=True)
        print(f"\n  Backing up old transforms to {backup_dir.name}/")
        for f in output_dir.glob('tpl-to-SIGMA_*'):
            if f.is_file():
                shutil.move(str(f), str(backup_dir / f.name))

    # Run registration
    output_prefix = output_dir / 'tpl-to-SIGMA_'

    outputs = register_template_to_sigma(
        template_image=template_path,
        sigma_template=sigma_template,
        output_prefix=output_prefix,
        n_cores=args.n_cores
    )

    # Verify output geometry
    if outputs['warped'].exists():
        warped_img = nib.load(outputs['warped'])
        print(f"\n  Warped output: {warped_img.shape} @ {warped_img.header.get_zooms()[:3]} mm")
        if warped_img.shape == atlas_img.shape:
            print(f"  ✓ Output matches atlas space - registration successful!")
        else:
            print(f"  ✗ Output shape mismatch - check registration")

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Transforms saved to: {output_dir}")
    print()
    print("Atlas propagation chain now properly configured:")
    print(f"  SIGMA atlas → Template space → Subject T2w → FA")


if __name__ == '__main__':
    main()
