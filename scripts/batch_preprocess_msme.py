#!/usr/bin/env python3
"""
Batch preprocessing script for MSME (Multi-Slice Multi-Echo) T2 mapping data.

Processes all MSME scans in the BPA-Rat BIDS dataset with parallel execution,
error handling, and progress tracking.
"""

import argparse
from pathlib import Path
import sys
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Add neurofaune to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.preprocess.workflows.msme_preprocess import run_msme_preprocessing
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.utils.select_anatomical import is_3d_only_subject


def find_all_msme_scans(bids_root: Path):
    """
    Find all MSME scans in BIDS directory.

    Returns
    -------
    list
        List of scan dictionaries with path, subject, session, key
    """
    import nibabel as nib

    msme_scans = list(bids_root.glob('sub-*/ses-*/msme/*MSME*.nii.gz'))

    # Parse into structured list
    scans = []
    seen_keys = set()

    for msme_path in msme_scans:
        # Extract subject and session from directory structure
        # Path looks like: .../sub-Rat110/ses-p90/msme/filename.nii.gz
        try:
            msme_idx = msme_path.parts.index('msme')
            session = msme_path.parts[msme_idx - 1]
            subject = msme_path.parts[msme_idx - 2]
        except (ValueError, IndexError):
            print(f"  Warning: Could not parse subject/session from {msme_path}")
            continue

        if not subject.startswith('sub-') or not session.startswith('ses-'):
            print(f"  Warning: Invalid subject/session format: {subject}/{session}")
            continue

        key = f"{subject}_{session}"

        # Skip duplicates (same subject/session with different runs)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Check file is valid
        try:
            img = nib.load(msme_path)
            shape = img.shape
            zooms = img.header.get_zooms()
        except Exception as e:
            print(f"  Warning: Could not load {msme_path}: {e}")
            continue

        scans.append({
            'path': msme_path,
            'subject': subject,
            'session': session,
            'key': key,
            'shape': shape,
            'zooms': zooms[:3]
        })

    return scans


def find_t2w_file(output_root: Path, subject: str, session: str) -> Path:
    """
    Find preprocessed T2w file for MSME registration.

    Looks for skull-stripped T2w in derivatives.
    """
    derivatives = output_root / 'derivatives' / subject / session / 'anat'

    if not derivatives.exists():
        return None

    # Look for skull-stripped T2w (preferred)
    for pattern in ['*desc-skullstrip_T2w.nii.gz', '*T2w_brain.nii.gz', '*desc-preproc_T2w.nii.gz']:
        matches = list(derivatives.glob(pattern))
        if matches:
            return matches[0]

    # Fallback to any T2w
    matches = list(derivatives.glob('*T2w*.nii.gz'))
    if matches:
        return matches[0]

    return None


def process_single_scan(scan_info: dict, config_path: Path, output_root: Path, force: bool = False):
    """
    Process a single MSME scan.

    Returns
    -------
    dict
        Processing result with status, subject/session info, and any errors
    """
    import os
    import sys

    # Suppress stdout in worker processes to avoid broken pipe errors
    # when running with ProcessPoolExecutor
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    subject = scan_info['subject']
    session = scan_info['session']
    msme_file = scan_info['path']
    key = scan_info['key']

    # Check if already processed (look for T2 map or MWF)
    output_dir = output_root / 'derivatives' / subject / session / 'msme'
    t2_map = output_dir / f'{subject}_{session}_T2map.nii.gz'
    mwf_map = output_dir / f'{subject}_{session}_MWF.nii.gz'

    if (t2_map.exists() or mwf_map.exists()) and not force:
        return {
            'status': 'skipped',
            'key': key,
            'subject': subject,
            'session': session,
            'message': 'Already processed'
        }

    try:
        # Load config
        config = load_config(config_path)

        # Extract cohort from session (p30, p60, p90)
        cohort = session.split('-')[1] if '-' in session else 'unknown'

        # Create transform registry
        registry = create_transform_registry(config, subject, cohort=cohort)

        # Find T2w file for registration
        t2w_file = find_t2w_file(output_root, subject, session)

        # Work directory
        work_dir = output_root / 'work' / subject / session / 'msme_batch'

        # Run preprocessing
        results = run_msme_preprocessing(
            config=config,
            subject=subject,
            session=session,
            msme_file=msme_file,
            output_dir=output_root,
            transform_registry=registry,
            work_dir=work_dir,
            t2w_file=t2w_file,
            run_registration=t2w_file is not None
        )

        result = {
            'status': 'success',
            'key': key,
            'subject': subject,
            'session': session,
            't2_map': str(results.get('t2_map', '')),
            'mwf_map': str(results.get('mwf_map', '')),
            't2w_file': str(t2w_file) if t2w_file else None
        }

        # Add registration info if available
        if 'registration' in results and results['registration']:
            reg_info = results['registration']
            if isinstance(reg_info, dict):
                result['registration'] = {
                    'transform': str(reg_info.get('affine_transform', '')),
                }

        return result

    except Exception as e:
        return {
            'status': 'failed',
            'key': key,
            'subject': subject,
            'session': session,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(
        description='Batch preprocess MSME T2 mapping data'
    )
    parser.add_argument(
        '--bids-root',
        type=Path,
        default=Path('/mnt/arborea/bpa-rat/raw/bids'),
        help='BIDS root directory'
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('/mnt/arborea/bpa-rat'),
        help='Output root directory (study root)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent.parent / 'configs' / 'default.yaml',
        help='Configuration file'
    )
    parser.add_argument(
        '--n-workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing of already processed scans'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List scans without processing'
    )
    parser.add_argument(
        '--subjects',
        nargs='+',
        default=None,
        help='Specific subjects to process (e.g., sub-Rat110 sub-Rat111)'
    )
    parser.add_argument(
        '--exclude-3d',
        action='store_true',
        help='Exclude subjects that only have 3D T2w scans (no 2D available)'
    )

    args = parser.parse_args()

    # Find all scans
    print(f"Scanning for MSME images in {args.bids_root}...")
    scans = find_all_msme_scans(args.bids_root)

    # Filter by subjects if specified
    if args.subjects:
        scans = [s for s in scans if s['subject'] in args.subjects]

    # Exclude 3D-only subjects if requested
    if args.exclude_3d:
        before = len(scans)
        scans = [
            s for s in scans
            if not is_3d_only_subject(args.bids_root / s['subject'], s['session'])
        ]
        n_excluded = before - len(scans)
        if n_excluded > 0:
            print(f"Excluding {n_excluded} MSME scans from 3D-only subjects (--exclude-3d)")

    print(f"Found {len(scans)} MSME scans")

    if not scans:
        print("No MSME scans found.")
        return 0

    if args.dry_run:
        print("\nScans to process:")
        for scan in scans[:30]:
            t2w = find_t2w_file(args.output_root, scan['subject'], scan['session'])
            t2w_status = "✓ T2w" if t2w else "✗ no T2w"
            shape_str = f"{scan['shape'][0]}x{scan['shape'][1]}x{scan['shape'][2]}x{scan['shape'][3]}"
            zooms_str = f"{scan['zooms'][0]:.1f}x{scan['zooms'][1]:.1f}x{scan['zooms'][2]:.1f}mm"
            print(f"  {scan['key']}: {shape_str} @ {zooms_str} - {t2w_status}")
        if len(scans) > 30:
            print(f"  ... and {len(scans) - 30} more")

        # Summary by cohort
        from collections import Counter
        cohort_counts = Counter(s['session'] for s in scans)
        print(f"\nBy cohort:")
        for cohort, count in sorted(cohort_counts.items()):
            print(f"  {cohort}: {count} scans")

        # Count with T2w available
        with_t2w = sum(1 for s in scans if find_t2w_file(args.output_root, s['subject'], s['session']))
        print(f"\nWith T2w for registration: {with_t2w}/{len(scans)}")

        return 0

    # Create log directory
    log_dir = Path('/tmp/msme_batch_preprocessing')
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = log_dir / f'batch_results_{timestamp}.json'

    print(f"\nProcessing configuration:")
    print(f"  BIDS root: {args.bids_root}")
    print(f"  Output root: {args.output_root}")
    print(f"  Config: {args.config}")
    print(f"  Workers: {args.n_workers}")
    print(f"  Force reprocessing: {args.force}")
    print(f"  Results file: {results_file}")
    print()

    # Process scans in parallel
    results = []
    completed = 0
    failed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_single_scan,
                scan,
                args.config,
                args.output_root,
                args.force
            ): scan for scan in scans
        }

        # Process results as they complete
        for future in as_completed(futures):
            scan = futures[future]
            try:
                result = future.result()
                results.append(result)

                if result['status'] == 'success':
                    completed += 1
                    reg_status = "with registration" if result.get('registration') else "no registration"
                    print(f"✓ [{completed + failed + skipped}/{len(scans)}] {result['key']}: SUCCESS ({reg_status})")
                elif result['status'] == 'failed':
                    failed += 1
                    print(f"✗ [{completed + failed + skipped}/{len(scans)}] {result['key']}: FAILED - {result['error']}")
                elif result['status'] == 'skipped':
                    skipped += 1
                    print(f"→ [{completed + failed + skipped}/{len(scans)}] {result['key']}: SKIPPED (already processed)")

                # Write intermediate results
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                failed += 1
                print(f"✗ [{completed + failed + skipped}/{len(scans)}] {scan['key']}: EXCEPTION - {e}")
                results.append({
                    'status': 'exception',
                    'key': scan['key'],
                    'error': str(e)
                })

    # Final summary
    print("\n" + "="*80)
    print("BATCH MSME PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Total scans: {len(scans)}")
    print(f"Successful: {completed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"\nResults saved to: {results_file}")

    if failed > 0:
        print(f"\nFailed scans:")
        for result in results:
            if result['status'] == 'failed':
                print(f"  {result['key']}: {result['error']}")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
