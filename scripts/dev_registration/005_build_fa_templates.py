#!/usr/bin/env python3
"""
005_build_fa_templates.py

Build age-specific FA (Fractional Anisotropy) templates from preprocessed DTI data.

This script:
1. Collects FA maps from preprocessed DTI for each cohort
2. Selects best subjects based on QC metrics (if available)
3. Builds cohort-specific FA templates using ANTs
4. (Optional) Registers FA template to T2w template for cross-modal alignment

Unlike T2w templates which register directly to SIGMA, FA templates are:
- Built in native FA space (same geometry as DTI acquisitions)
- Aligned to T2w template space via within-cohort registration
- SIGMA access is through: FA → T2w template → SIGMA

Usage:
    python 005_build_fa_templates.py /path/to/bpa-rat --cohort p60
    python 005_build_fa_templates.py /path/to/bpa-rat --cohort all

Output:
    templates/dwi/{cohort}/
        - tpl-BPARat_{cohort}_FA.nii.gz
        - tpl-metadata_{cohort}_FA.json
        - transforms/FA-to-T2w_0GenericAffine.mat (if T2w template exists)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict

import nibabel as nib
import numpy as np


def find_fa_maps(derivatives_root: Path, cohort: str) -> List[Dict]:
    """
    Find all FA maps for a given cohort.

    Returns list of dicts with subject info and FA path.
    """
    session = f'ses-{cohort}'
    fa_maps = []

    for subject_dir in sorted(derivatives_root.glob('sub-*')):
        subject = subject_dir.name
        fa_path = subject_dir / session / 'dwi' / f'{subject}_{session}_FA.nii.gz'

        if fa_path.exists():
            # Load to check validity
            try:
                img = nib.load(fa_path)
                data = img.get_fdata()

                # Basic QC: check FA values are reasonable (0-1 range)
                fa_mean = np.nanmean(data[data > 0])
                fa_max = np.nanmax(data)

                fa_maps.append({
                    'subject': subject,
                    'session': session,
                    'fa_path': fa_path,
                    'shape': img.shape,
                    'voxel_size': tuple(round(v, 3) for v in img.header.get_zooms()[:3]),
                    'fa_mean': round(fa_mean, 4),
                    'fa_max': round(fa_max, 4),
                    'valid': 0 < fa_mean < 1 and fa_max <= 1.5  # Allow slight overshoot
                })
            except Exception as e:
                print(f"  Warning: Could not load {fa_path}: {e}")

    return fa_maps


def select_subjects_for_template(
    fa_maps: List[Dict],
    n_subjects: int = 10,
    min_subjects: int = 5
) -> List[Dict]:
    """
    Select best subjects for template building.

    Simple selection based on FA map validity and mean FA.
    """
    # Filter valid
    valid_maps = [m for m in fa_maps if m['valid']]

    if len(valid_maps) < min_subjects:
        raise ValueError(
            f"Not enough valid FA maps ({len(valid_maps)} < {min_subjects}). "
            f"Run DTI preprocessing for more subjects."
        )

    # Sort by mean FA (higher is generally better quality)
    sorted_maps = sorted(valid_maps, key=lambda x: x['fa_mean'], reverse=True)

    # Select top N
    selected = sorted_maps[:min(n_subjects, len(sorted_maps))]

    return selected


def build_fa_template(
    fa_files: List[Path],
    output_dir: Path,
    cohort: str,
    n_iterations: int = 4,
    n_cores: int = 8
) -> Dict[str, Path]:
    """
    Build FA template using ANTs multivariate template construction.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = output_dir / f'tpl-BPARat_{cohort}_FA_'

    print(f"\nBuilding FA template for {cohort}...")
    print(f"  Input files: {len(fa_files)}")
    print(f"  Output: {output_prefix}")

    cmd = [
        'antsMultivariateTemplateConstruction.sh',
        '-d', '3',
        '-o', str(output_prefix),
        '-i', str(n_iterations),
        '-g', '0.2',  # gradient step
        '-c', '2',    # parallel SyN
        '-j', str(n_cores),
        '-n', '0',
        '-r', '1',    # float precision
    ]
    cmd.extend([str(f) for f in fa_files])

    print(f"  Command: {' '.join(cmd[:10])}...")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: Template construction failed!")
        print(result.stdout)
        raise RuntimeError("ANTs template construction failed")

    # Find output template
    template_file = Path(str(output_prefix) + 'template0.nii.gz')

    if not template_file.exists():
        raise FileNotFoundError(f"Expected template not found: {template_file}")

    # Rename to final name
    final_template = output_dir / f'tpl-BPARat_{cohort}_FA.nii.gz'
    template_file.rename(final_template)

    print(f"  ✓ Template: {final_template}")

    return {'template': final_template}


def register_fa_to_t2w_template(
    fa_template: Path,
    t2w_template: Path,
    output_prefix: Path,
    n_cores: int = 4
) -> Dict[str, Path]:
    """
    Register FA template to T2w template (for cross-modal alignment).

    This enables propagating T2w→SIGMA transforms to FA space.
    """
    print(f"\nRegistering FA template to T2w template...")
    print(f"  FA template: {fa_template.name}")
    print(f"  T2w template: {t2w_template.name}")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsRegistrationSyN.sh',
        '-d', '3',
        '-f', str(t2w_template),
        '-m', str(fa_template),
        '-o', str(output_prefix),
        '-n', str(n_cores),
        '-t', 'a'  # Affine only (templates are already in similar space)
    ]

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

    outputs = {
        'affine': Path(str(output_prefix) + '0GenericAffine.mat'),
        'warped': Path(str(output_prefix) + 'Warped.nii.gz'),
    }

    for name, path in outputs.items():
        if path.exists():
            print(f"  ✓ {name}: {path.name}")

    return outputs


def save_template_metadata(
    output_dir: Path,
    cohort: str,
    subjects_used: List[Dict],
    template_path: Path
):
    """Save metadata about template construction."""
    metadata = {
        'cohort': cohort,
        'modality': 'FA',
        'n_subjects': len(subjects_used),
        'subjects': [s['subject'] for s in subjects_used],
        'mean_fa_values': {s['subject']: s['fa_mean'] for s in subjects_used},
        'template_shape': list(nib.load(template_path).shape),
        'template_voxel_size': list(nib.load(template_path).header.get_zooms()[:3]),
    }

    metadata_file = output_dir / f'tpl-metadata_{cohort}_FA.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Metadata: {metadata_file.name}")

    return metadata_file


def main():
    parser = argparse.ArgumentParser(
        description='Build FA templates from preprocessed DTI'
    )
    parser.add_argument('study_root', type=Path, help='Path to study root')
    parser.add_argument('--cohort', required=True,
                        help='Cohort to build (p30, p60, p90, or "all")')
    parser.add_argument('--n-subjects', type=int, default=10,
                        help='Number of subjects for template (default: 10)')
    parser.add_argument('--n-iterations', type=int, default=4,
                        help='ANTs iterations (default: 4)')
    parser.add_argument('--n-cores', type=int, default=8,
                        help='CPU cores (default: 8)')
    parser.add_argument('--skip-t2w-registration', action='store_true',
                        help='Skip FA→T2w template registration')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show what would be done')
    args = parser.parse_args()

    derivatives_root = args.study_root / 'derivatives'
    templates_root = args.study_root / 'templates'

    if not derivatives_root.exists():
        print(f"ERROR: Derivatives not found: {derivatives_root}")
        sys.exit(1)

    # Determine cohorts to process
    if args.cohort.lower() == 'all':
        cohorts = ['p30', 'p60', 'p90']
    else:
        cohorts = [args.cohort]

    print("=" * 70)
    print("FA Template Building")
    print("=" * 70)

    for cohort in cohorts:
        print(f"\n{'='*70}")
        print(f"Cohort: {cohort}")
        print(f"{'='*70}")

        # Find FA maps
        fa_maps = find_fa_maps(derivatives_root, cohort)
        print(f"\nFound {len(fa_maps)} FA maps for {cohort}")

        if not fa_maps:
            print(f"  No FA maps found. Run DTI preprocessing first.")
            continue

        # Select subjects
        try:
            selected = select_subjects_for_template(
                fa_maps,
                n_subjects=args.n_subjects,
                min_subjects=5
            )
        except ValueError as e:
            print(f"  {e}")
            continue

        print(f"\nSelected {len(selected)} subjects for template:")
        for s in selected[:5]:
            print(f"  {s['subject']}: mean FA = {s['fa_mean']:.4f}")
        if len(selected) > 5:
            print(f"  ... and {len(selected) - 5} more")

        if args.dry_run:
            print("\n[DRY RUN] Would build template from these subjects")
            continue

        # Build template
        output_dir = templates_root / 'dwi' / cohort
        fa_files = [s['fa_path'] for s in selected]

        template_result = build_fa_template(
            fa_files=fa_files,
            output_dir=output_dir,
            cohort=cohort,
            n_iterations=args.n_iterations,
            n_cores=args.n_cores
        )

        # Save metadata
        save_template_metadata(
            output_dir=output_dir,
            cohort=cohort,
            subjects_used=selected,
            template_path=template_result['template']
        )

        # Register to T2w template if it exists
        if not args.skip_t2w_registration:
            t2w_template = templates_root / 'anat' / cohort / f'tpl-BPARat_{cohort}_T2w.nii.gz'

            if t2w_template.exists():
                fa_to_t2w_prefix = output_dir / 'transforms' / 'FA-to-T2w_'
                register_fa_to_t2w_template(
                    fa_template=template_result['template'],
                    t2w_template=t2w_template,
                    output_prefix=fa_to_t2w_prefix,
                    n_cores=args.n_cores
                )
            else:
                print(f"\n  Note: T2w template not found at {t2w_template}")
                print(f"  FA→T2w registration skipped")

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
