#!/usr/bin/env python3
"""
Batch preprocessing script for functional fMRI data.

Processes all BOLD scans in the BPA-Rat BIDS dataset with parallel execution,
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
from neurofaune.preprocess.workflows.func_preprocess import run_functional_preprocessing
from neurofaune.utils.transforms import create_transform_registry


def find_all_bold_scans(bids_root: Path):
    """Find all BOLD scans in BIDS directory."""
    bold_scans = list(bids_root.glob('sub-*/ses-*/func/*_bold.nii.gz'))

    # Parse into structured list
    scans = []
    for bold_path in bold_scans:
        parts = bold_path.stem.replace('_bold.nii', '').split('_')
        subject = parts[0]
        session = parts[1]
        run = next((p for p in parts if p.startswith('run-')), None)

        scans.append({
            'path': bold_path,
            'subject': subject,
            'session': session,
            'run': run,
            'key': f"{subject}_{session}_{run}"
        })

    return scans


def process_single_scan(scan_info: dict, config_path: Path, output_root: Path,
                        n_discard: int = 5, force: bool = False):
    """
    Process a single BOLD scan.

    Returns
    -------
    dict
        Processing result with status, subject/session info, and any errors
    """
    subject = scan_info['subject']
    session = scan_info['session']
    bold_file = scan_info['path']
    key = scan_info['key']

    # Check if already processed
    output_file = output_root / 'derivatives' / subject / session / 'func' / f'{subject}_{session}_desc-preproc_bold.nii.gz'
    if output_file.exists() and not force:
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
        cohort = session.split('-')[1] if '-' in session else None

        # Create transform registry
        registry = create_transform_registry(config, subject, cohort=cohort)

        # Run preprocessing
        results = run_functional_preprocessing(
            config=config,
            subject=subject,
            session=session,
            bold_file=bold_file,
            output_dir=output_root,
            transform_registry=registry,
            n_discard=n_discard
        )

        return {
            'status': 'success',
            'key': key,
            'subject': subject,
            'session': session,
            'preprocessed_bold': str(results.get('preprocessed_bold', '')),
            'brain_mask': str(results.get('brain_mask', '')),
            'confounds': str(results.get('confounds', ''))
        }

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
        description='Batch preprocess functional fMRI data'
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
        default=Path('/home/edm9fd/sandbox/neurofaune/configs/default.yaml'),
        help='Configuration file'
    )
    parser.add_argument(
        '--n-workers',
        type=int,
        default=6,
        help='Number of parallel workers (default: 6)'
    )
    parser.add_argument(
        '--n-discard',
        type=int,
        default=5,
        help='Number of initial volumes to discard (default: 5)'
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

    args = parser.parse_args()

    # Find all scans
    print(f"Scanning for BOLD images in {args.bids_root}...")
    scans = find_all_bold_scans(args.bids_root)
    print(f"Found {len(scans)} BOLD scans")

    if args.dry_run:
        print("\nScans to process:")
        for scan in scans[:20]:
            print(f"  {scan['key']}: {scan['path']}")
        if len(scans) > 20:
            print(f"  ... and {len(scans) - 20} more")
        return 0

    # Create log directory
    log_dir = Path('/tmp/func_batch_preprocessing')
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'batch_preprocess_{timestamp}.log'
    results_file = log_dir / f'batch_results_{timestamp}.json'

    print(f"\nProcessing configuration:")
    print(f"  BIDS root: {args.bids_root}")
    print(f"  Output root: {args.output_root}")
    print(f"  Config: {args.config}")
    print(f"  Workers: {args.n_workers}")
    print(f"  Discard volumes: {args.n_discard}")
    print(f"  Force reprocessing: {args.force}")
    print(f"  Log file: {log_file}")
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
                args.n_discard,
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
                    print(f"✓ [{completed + failed + skipped}/{len(scans)}] {result['key']}: SUCCESS")
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
    print("BATCH PREPROCESSING COMPLETE")
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
