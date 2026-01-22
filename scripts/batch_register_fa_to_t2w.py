#!/usr/bin/env python3
"""
Batch FA-to-T2w registration for all subjects with existing FA and preprocessed T2w.

Runs ANTs affine registration of FA to T2w for each subject/session,
skipping those that already have the transform.

Usage:
    python batch_register_fa_to_t2w.py
    python batch_register_fa_to_t2w.py --dry-run
    python batch_register_fa_to_t2w.py --force  # Re-run even if transform exists
    python batch_register_fa_to_t2w.py --n-cores 8
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.preprocess.workflows.dwi_preprocess import register_fa_to_t2w


def find_registration_pairs(study_root: Path):
    """Find all subject/sessions with both FA and preprocessed T2w."""
    derivatives = study_root / 'derivatives'
    pairs = []

    for fa_file in sorted(derivatives.glob('sub-*/ses-*/dwi/*_FA.nii.gz')):
        # Skip space-SIGMA FA files
        if 'space-' in fa_file.name:
            continue

        # Parse subject/session from path
        parts = fa_file.parts
        dwi_idx = parts.index('dwi')
        subject = parts[dwi_idx - 2]
        session = parts[dwi_idx - 1]

        # Check for preprocessed T2w
        t2w_file = derivatives / subject / session / 'anat' / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
        if not t2w_file.exists():
            continue

        # Check for existing transform
        transform_file = study_root / 'transforms' / subject / session / 'FA_to_T2w_0GenericAffine.mat'

        pairs.append({
            'subject': subject,
            'session': session,
            'fa_file': fa_file,
            't2w_file': t2w_file,
            'transform_exists': transform_file.exists(),
        })

    return pairs


def main():
    parser = argparse.ArgumentParser(description='Batch FA-to-T2w registration')
    parser.add_argument('--study-root', type=Path, default=Path('/mnt/arborea/bpa-rat'))
    parser.add_argument('--n-cores', type=int, default=4, help='Cores for ANTs registration')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--force', action='store_true', help='Re-run even if transform exists')
    args = parser.parse_args()

    print("=" * 70)
    print("Batch FA-to-T2w Registration")
    print("=" * 70)
    print(f"Study root: {args.study_root}")
    print(f"Cores: {args.n_cores}")
    print()

    # Find all pairs
    pairs = find_registration_pairs(args.study_root)
    print(f"Found {len(pairs)} subject/sessions with both FA and T2w")

    # Filter based on existing transforms
    if args.force:
        to_process = pairs
    else:
        to_process = [p for p in pairs if not p['transform_exists']]
        already_done = len(pairs) - len(to_process)
        if already_done > 0:
            print(f"  Already registered: {already_done} (use --force to redo)")

    print(f"  To process: {len(to_process)}")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for p in to_process:
            print(f"  {p['subject']}/{p['session']}")
        return

    if not to_process:
        print("\nNothing to do.")
        return

    # Process
    print(f"\nStarting registration of {len(to_process)} subjects...")
    results = {'success': 0, 'error': 0, 'errors': []}
    start_time = datetime.now()

    for i, p in enumerate(to_process, 1):
        subject = p['subject']
        session = p['session']
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(to_process)}] {subject}/{session}")

        work_dir = args.study_root / 'work' / subject / session / 'dwi_registration'
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            registration_results = register_fa_to_t2w(
                fa_file=p['fa_file'],
                t2w_file=p['t2w_file'],
                output_dir=args.study_root,
                subject=subject,
                session=session,
                work_dir=work_dir,
                n_cores=args.n_cores
            )

            # Save metadata JSON
            derivatives_dir = args.study_root / 'derivatives' / subject / session / 'dwi'
            reg_metadata_file = derivatives_dir / f'{subject}_{session}_FA_to_T2w_registration.json'
            reg_metadata = {
                'fa_file': str(p['fa_file']),
                't2w_file': str(registration_results['t2w_file']),
                'affine_transform': str(registration_results['affine_transform']),
                'warped_fa': str(registration_results['warped_fa']) if registration_results.get('warped_fa') else None,
                'fa_shape': list(registration_results['fa_shape']),
                't2w_shape': list(registration_results['t2w_shape']),
                'timestamp': datetime.now().isoformat(),
            }
            with open(reg_metadata_file, 'w') as f:
                json.dump(reg_metadata, f, indent=2)

            results['success'] += 1

        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
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
