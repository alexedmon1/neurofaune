#!/usr/bin/env python3
"""
List subjects that only have 3D T2w acquisitions (no 2D multi-slice available).

These subjects may need special handling (resampling to 2D-like geometry) or
exclusion from cross-modal registration pipelines.

Usage:
    # Plain text output (one subject/session per line)
    python scripts/list_3d_subjects.py /path/to/bids

    # JSON output for programmatic use
    python scripts/list_3d_subjects.py /path/to/bids --json

    # Filter by cohort
    python scripts/list_3d_subjects.py /path/to/bids --cohort p60
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.utils.select_anatomical import is_3d_only_subject


def find_3d_subjects(bids_dir: Path, cohort: str = None):
    """
    Scan BIDS directory for subjects with only 3D T2w acquisitions.

    Parameters
    ----------
    bids_dir : Path
        BIDS root directory
    cohort : str, optional
        Filter by cohort (e.g., 'p60')

    Returns
    -------
    list
        List of dicts with subject, session, and T2w file info
    """
    results = []

    for subject_dir in sorted(bids_dir.glob('sub-*')):
        if not subject_dir.is_dir():
            continue

        for session_dir in sorted(subject_dir.glob('ses-*')):
            if not session_dir.is_dir():
                continue

            session = session_dir.name

            # Filter by cohort if specified
            if cohort and cohort not in session:
                continue

            # Check for anat directory
            anat_dir = session_dir / 'anat'
            if not anat_dir.exists():
                continue

            t2w_files = list(anat_dir.glob('*_T2w.nii.gz'))
            if not t2w_files:
                continue

            if is_3d_only_subject(subject_dir, session):
                results.append({
                    'subject': subject_dir.name,
                    'session': session,
                    't2w_files': [f.name for f in t2w_files]
                })

    return results


def main():
    parser = argparse.ArgumentParser(
        description='List subjects with only 3D T2w acquisitions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'bids_dir',
        type=Path,
        help='BIDS root directory'
    )
    parser.add_argument(
        '--cohort',
        type=str,
        default=None,
        help='Filter by cohort (e.g., p60)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        dest='output_json',
        help='Output as JSON instead of plain text'
    )

    args = parser.parse_args()

    if not args.bids_dir.exists():
        print(f"ERROR: BIDS directory not found: {args.bids_dir}", file=sys.stderr)
        sys.exit(1)

    subjects_3d = find_3d_subjects(args.bids_dir, args.cohort)

    if args.output_json:
        json.dump(subjects_3d, sys.stdout, indent=2)
        print()
    else:
        if not subjects_3d:
            print("No 3D-only subjects found.")
            return

        # Group by subject for cleaner output
        from collections import defaultdict
        by_subject = defaultdict(list)
        for entry in subjects_3d:
            by_subject[entry['subject']].append(entry['session'])

        print(f"3D-only subjects: {len(by_subject)} subjects, "
              f"{len(subjects_3d)} sessions")
        print("=" * 60)

        for subject in sorted(by_subject.keys()):
            sessions = sorted(by_subject[subject])
            print(f"{subject}: {', '.join(sessions)}")

        print("=" * 60)
        print(f"\nTotal: {len(subjects_3d)} subject/sessions")
        if args.cohort:
            print(f"Cohort filter: {args.cohort}")


if __name__ == '__main__':
    main()
