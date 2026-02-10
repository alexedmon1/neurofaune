#!/usr/bin/env python3
"""
Batch FA-to-Template registration for all subjects with existing FA maps.

Runs ANTs affine registration of FA directly to the cohort template for each
subject/session, skipping those that already have the transform. This replaces
the old FA-to-T2w approach with better SIGMA atlas overlap.

Usage:
    python batch_register_fa_to_template.py
    python batch_register_fa_to_template.py --dry-run
    python batch_register_fa_to_template.py --force  # Re-run even if transform exists
    python batch_register_fa_to_template.py --n-cores 8
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.preprocess.workflows.dwi_preprocess import register_fa_to_template


def find_registration_subjects(study_root: Path):
    """Find all subject/sessions with FA maps and cohort templates."""
    derivatives = study_root / 'derivatives'
    templates_root = study_root / 'templates'
    subjects = []

    for fa_file in sorted(derivatives.glob('sub-*/ses-*/dwi/*_FA.nii.gz')):
        # Skip space-SIGMA FA files
        if 'space-' in fa_file.name:
            continue

        # Parse subject/session from path
        parts = fa_file.parts
        dwi_idx = parts.index('dwi')
        subject = parts[dwi_idx - 2]
        session = parts[dwi_idx - 1]
        cohort = session.replace('ses-', '')

        # Skip unknown cohort (no template)
        if cohort == 'unknown':
            continue

        # Check for cohort template
        template_file = templates_root / 'anat' / cohort / f'tpl-BPARat_{cohort}_T2w.nii.gz'
        if not template_file.exists():
            continue

        # Check for existing transform
        transform_file = study_root / 'transforms' / subject / session / 'FA_to_template_0GenericAffine.mat'

        subjects.append({
            'subject': subject,
            'session': session,
            'cohort': cohort,
            'fa_file': fa_file,
            'template_file': template_file,
            'transform_exists': transform_file.exists(),
        })

    return subjects


def main():
    parser = argparse.ArgumentParser(description='Batch FA-to-Template registration')
    parser.add_argument('--study-root', type=Path, default=Path('/mnt/arborea/bpa-rat'))
    parser.add_argument('--n-cores', type=int, default=4, help='Cores for ANTs registration')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--force', action='store_true', help='Re-run even if transform exists')
    args = parser.parse_args()

    print("=" * 70)
    print("Batch FA-to-Template Registration")
    print("=" * 70)
    print(f"Study root: {args.study_root}")
    print(f"Cores: {args.n_cores}")
    print()

    # Find all subjects
    subjects = find_registration_subjects(args.study_root)
    print(f"Found {len(subjects)} subject/sessions with FA + cohort template")

    # Filter based on existing transforms
    if args.force:
        to_process = subjects
    else:
        to_process = [s for s in subjects if not s['transform_exists']]
        already_done = len(subjects) - len(to_process)
        if already_done > 0:
            print(f"  Already registered: {already_done} (use --force to redo)")

    print(f"  To process: {len(to_process)}")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for s in to_process:
            print(f"  {s['subject']}/{s['session']} ({s['cohort']})")
        return

    if not to_process:
        print("\nNothing to do.")
        return

    # Process
    print(f"\nStarting registration of {len(to_process)} subjects...")
    results = {'success': 0, 'error': 0, 'errors': []}
    start_time = datetime.now()

    for i, s in enumerate(to_process, 1):
        subject = s['subject']
        session = s['session']
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(to_process)}] {subject}/{session} ({s['cohort']})")

        work_dir = args.study_root / 'work' / subject / session / 'dwi_registration'
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            registration_results = register_fa_to_template(
                fa_file=s['fa_file'],
                template_file=s['template_file'],
                output_dir=args.study_root,
                subject=subject,
                session=session,
                work_dir=work_dir,
                n_cores=args.n_cores
            )

            # Save metadata JSON
            derivatives_dir = args.study_root / 'derivatives' / subject / session / 'dwi'
            reg_metadata_file = derivatives_dir / f'{subject}_{session}_FA_to_template_registration.json'
            reg_metadata = {
                'fa_file': str(s['fa_file']),
                'template_file': str(registration_results['template_file']),
                'affine_transform': str(registration_results['affine_transform']),
                'warped_fa': str(registration_results['warped_fa']) if registration_results.get('warped_fa') else None,
                'fa_shape': list(registration_results['fa_shape']),
                'template_shape': list(registration_results['template_shape']),
                'timestamp': datetime.now().isoformat(),
            }
            with open(reg_metadata_file, 'w') as f:
                json.dump(reg_metadata, f, indent=2)

            results['success'] += 1

        except Exception as e:
            print(f"\n  FAILED: {e}")
            traceback.print_exc()
            results['error'] += 1
            results['errors'].append({'subject': subject, 'session': session, 'error': str(e)})

    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'=' * 70}")
    print("Registration Complete")
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
