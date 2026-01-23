#!/usr/bin/env python3
"""
Batch warp preprocessed BOLD to SIGMA atlas space for group-level analysis.

Chains transforms: BOLD → T2w → Template → SIGMA
Requires: BOLD_to_T2w registration, T2w_to_template registration, tpl-to-SIGMA registration.

Usage:
    python batch_warp_bold_to_sigma.py
    python batch_warp_bold_to_sigma.py --dry-run
    python batch_warp_bold_to_sigma.py --force
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))


def _find_transform(directory: Path, *filenames: str) -> Optional[Path]:
    """Find transform file trying multiple naming patterns."""
    for name in filenames:
        path = directory / name
        if path.exists():
            return path
    return None


def find_warpable_subjects(study_root: Path):
    """Find subjects with all required transforms and preprocessed BOLD."""
    derivatives = study_root / 'derivatives'
    transforms_root = study_root / 'transforms'
    templates_root = study_root / 'templates'
    subjects = []

    # Find preprocessed BOLD files
    for bold_file in sorted(derivatives.glob('sub-*/ses-*/func/*_desc-preproc_bold.nii.gz')):
        parts = bold_file.parts
        func_idx = parts.index('func')
        subject = parts[func_idx - 2]
        session = parts[func_idx - 1]
        cohort = session.replace('ses-', '')

        subj_transforms = transforms_root / subject / session
        tpl_transforms = templates_root / 'anat' / cohort / 'transforms'

        # Check BOLD → T2w
        bold_to_t2w = _find_transform(subj_transforms, 'BOLD_to_T2w_0GenericAffine.mat')

        # Check T2w → Template (try prefixed and unprefixed)
        t2w_to_tpl_affine = _find_transform(
            subj_transforms,
            f'{subject}_{session}_T2w_to_template_0GenericAffine.mat',
            'T2w_to_template_0GenericAffine.mat'
        )
        t2w_to_tpl_warp = _find_transform(
            subj_transforms,
            f'{subject}_{session}_T2w_to_template_1Warp.nii.gz',
            'T2w_to_template_1Warp.nii.gz'
        )

        # Check Template → SIGMA
        tpl_to_sigma_affine = _find_transform(tpl_transforms, 'tpl-to-SIGMA_0GenericAffine.mat')
        tpl_to_sigma_warp = _find_transform(tpl_transforms, 'tpl-to-SIGMA_1Warp.nii.gz')

        # Check which transforms are missing
        missing = []
        if not bold_to_t2w:
            missing.append('BOLD_to_T2w')
        if not t2w_to_tpl_affine:
            missing.append('T2w_to_template')
        if not tpl_to_sigma_affine:
            missing.append('tpl_to_SIGMA')

        # Check for existing output
        output_file = derivatives / subject / session / 'func' / f'{subject}_{session}_space-SIGMA_bold.nii.gz'

        subjects.append({
            'subject': subject,
            'session': session,
            'cohort': cohort,
            'bold_file': bold_file,
            'bold_to_t2w': bold_to_t2w,
            't2w_to_tpl_affine': t2w_to_tpl_affine,
            't2w_to_tpl_warp': t2w_to_tpl_warp,
            'tpl_to_sigma_affine': tpl_to_sigma_affine,
            'tpl_to_sigma_warp': tpl_to_sigma_warp,
            'missing_transforms': missing,
            'output_exists': output_file.exists(),
            'output_file': output_file,
        })

    return subjects


def warp_bold_to_sigma(
    bold_file: Path,
    subject_info: dict,
    sigma_template: Path,
    output_file: Path
) -> Path:
    """Warp preprocessed 4D BOLD to SIGMA space using transform chain."""
    # Build transform chain (listed in ANTs order: last-applied first)
    # Goal: BOLD-space → SIGMA-space
    # Application order: BOLD→T2w → T2w→Template → Template→SIGMA
    # ANTs list order: Template→SIGMA, T2w→Template, BOLD→T2w
    transforms = []

    # 3. Template → SIGMA (applied last, listed first)
    if subject_info['tpl_to_sigma_warp']:
        transforms.append(str(subject_info['tpl_to_sigma_warp']))
    transforms.append(str(subject_info['tpl_to_sigma_affine']))

    # 2. T2w → Template
    if subject_info['t2w_to_tpl_warp']:
        transforms.append(str(subject_info['t2w_to_tpl_warp']))
    transforms.append(str(subject_info['t2w_to_tpl_affine']))

    # 1. BOLD → T2w (applied first, listed last)
    transforms.append(str(subject_info['bold_to_t2w']))

    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-e', '3',  # 3D timeseries (applies transform to each volume)
        '-i', str(bold_file),
        '-r', str(sigma_template),
        '-o', str(output_file),
        '-n', 'Linear'
    ]
    for t in transforms:
        cmd.extend(['-t', t])

    print(f"  Applying {len(transforms)} transforms to 4D BOLD...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        raise RuntimeError("antsApplyTransforms failed")

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Batch warp BOLD to SIGMA space')
    parser.add_argument('--study-root', type=Path, default=Path('/mnt/arborea/bpa-rat'))
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--force', action='store_true', help='Re-run even if output exists')
    args = parser.parse_args()

    print("=" * 70)
    print("Batch Warp BOLD to SIGMA Space")
    print("=" * 70)
    print(f"Study root: {args.study_root}")
    print()

    # Find SIGMA template
    sigma_template = args.study_root / 'atlas' / 'SIGMA_study_space' / 'SIGMA_InVivo_Brain_Template_Masked.nii.gz'
    if not sigma_template.exists():
        print(f"ERROR: SIGMA template not found: {sigma_template}")
        sys.exit(1)

    # Find subjects
    subjects = find_warpable_subjects(args.study_root)
    print(f"Found {len(subjects)} subjects with preprocessed BOLD")

    # Filter: must have all transforms
    ready = [s for s in subjects if not s['missing_transforms']]
    missing = [s for s in subjects if s['missing_transforms']]

    if missing:
        print(f"\n  Missing transforms ({len(missing)} subjects):")
        for s in missing[:5]:
            print(f"    {s['subject']}/{s['session']}: {', '.join(s['missing_transforms'])}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    # Filter: skip already done
    if args.force:
        to_process = ready
    else:
        to_process = [s for s in ready if not s['output_exists']]
        already_done = len(ready) - len(to_process)
        if already_done > 0:
            print(f"  Already warped: {already_done} (use --force to redo)")

    print(f"  Ready to process: {len(to_process)}")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for s in to_process:
            print(f"  {s['subject']}/{s['session']} ({s['cohort']})")
        return

    if not to_process:
        print("\nNothing to do.")
        return

    # Process
    print(f"\nWarping {len(to_process)} subjects to SIGMA space...")
    results = {'success': 0, 'error': 0, 'errors': []}
    start_time = datetime.now()

    for i, s in enumerate(to_process, 1):
        subject = s['subject']
        session = s['session']
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(to_process)}] {subject}/{session}")

        try:
            warp_bold_to_sigma(
                bold_file=s['bold_file'],
                subject_info=s,
                sigma_template=sigma_template,
                output_file=s['output_file']
            )
            print(f"  Output: {s['output_file'].name}")
            results['success'] += 1

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results['error'] += 1
            results['errors'].append({'subject': subject, 'session': session, 'error': str(e)})

    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'=' * 70}")
    print("Warping Complete")
    print(f"{'=' * 70}")
    print(f"  Success: {results['success']}")
    print(f"  Errors:  {results['error']}")
    print(f"  Elapsed: {elapsed}")

    if results['errors']:
        print("\nFailed subjects:")
        for err in results['errors']:
            print(f"  {err['subject']}/{err['session']}: {err['error']}")


if __name__ == '__main__':
    main()
