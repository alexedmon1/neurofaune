#!/usr/bin/env python3
"""
Initialize a new neuroimaging study.

This script sets up the directory structure, discovers BIDS data,
generates configuration, and optionally sets up the study-space atlas.

Usage:
    # Initialize a new study
    uv run python scripts/init_study.py /path/to/study --name "My Study" --code mystudy

    # Initialize with existing BIDS data
    uv run python scripts/init_study.py /path/to/study --name "My Study" --code mystudy \
        --bids-root /path/to/bids

    # Initialize with SIGMA atlas
    uv run python scripts/init_study.py /path/to/study --name "My Study" --code mystudy \
        --sigma-atlas /path/to/SIGMA

    # Just discover BIDS data (no directory creation)
    uv run python scripts/init_study.py --discover-only /path/to/bids

    # Print study summary
    uv run python scripts/init_study.py /path/to/study --summary
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.study_initialization import (
    setup_study,
    discover_bids_data,
    print_study_summary,
    get_study_subjects
)


def main():
    parser = argparse.ArgumentParser(
        description='Initialize a neuroimaging study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'study_root',
        type=Path,
        help='Root directory for the study'
    )

    parser.add_argument(
        '--name', '-n',
        type=str,
        help='Human-readable study name'
    )

    parser.add_argument(
        '--code', '-c',
        type=str,
        help='Short study code (e.g., bpa-rat)'
    )

    parser.add_argument(
        '--bids-root',
        type=Path,
        help='Path to BIDS data (if different from study_root/raw/bids)'
    )

    parser.add_argument(
        '--bruker-root',
        type=Path,
        help='Path to raw Bruker data (if different from study_root/raw/bruker)'
    )

    parser.add_argument(
        '--sigma-atlas',
        type=Path,
        help='Path to SIGMA atlas base directory'
    )

    parser.add_argument(
        '--config',
        type=Path,
        help='Path to existing config file to validate'
    )

    parser.add_argument(
        '--no-atlas',
        action='store_true',
        help='Skip atlas setup'
    )

    parser.add_argument(
        '--no-link',
        action='store_true',
        help='Do not create symlinks to data'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing files'
    )

    parser.add_argument(
        '--discover-only',
        action='store_true',
        help='Only discover BIDS data, do not create directories'
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print study summary and exit'
    )

    parser.add_argument(
        '--list-subjects',
        action='store_true',
        help='List subjects available for processing'
    )

    parser.add_argument(
        '--cohort',
        type=str,
        help='Filter by cohort (for --list-subjects)'
    )

    parser.add_argument(
        '--modality',
        type=str,
        choices=['anat', 'dwi', 'func'],
        help='Filter by modality (for --list-subjects)'
    )

    parser.add_argument(
        '--output-json',
        type=Path,
        help='Write results to JSON file'
    )

    args = parser.parse_args()

    # Handle --summary
    if args.summary:
        print_study_summary(args.study_root)
        return 0

    # Handle --list-subjects
    if args.list_subjects:
        subjects = get_study_subjects(
            args.study_root,
            cohort=args.cohort,
            modality=args.modality
        )
        print(f"Found {len(subjects)} subject/session combinations:")
        for s in subjects:
            print(f"  {s['subject']} / {s['session']} ({s['cohort']}): {', '.join(s['modalities'])}")

        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(subjects, f, indent=2)
            print(f"\nSaved to: {args.output_json}")

        return 0

    # Handle --discover-only
    if args.discover_only:
        from neurofaune.study_initialization import discover_bruker_data

        results = {}

        # Discover Bruker data if available
        bruker_root = args.bruker_root or args.study_root / 'raw' / 'bruker'
        if bruker_root.exists():
            print(f"Discovering Bruker data in: {bruker_root}")
            bruker_manifest = discover_bruker_data(bruker_root)

            print(f"\nBruker Data Summary:")
            print(f"  Sessions: {bruker_manifest.n_sessions}")
            print(f"  Scans: {bruker_manifest.n_scans}")
            if bruker_manifest.get_scan_types():
                print(f"  Scan types: {bruker_manifest.get_scan_types()}")

            results['bruker'] = bruker_manifest.to_dict()

        # Discover BIDS data if available
        bids_root = args.bids_root or args.study_root / 'raw' / 'bids'
        if bids_root.exists():
            print(f"\nDiscovering BIDS data in: {bids_root}")
            bids_manifest = discover_bids_data(bids_root)

            print(f"\nBIDS Data Summary:")
            print(f"  Subjects: {bids_manifest.n_subjects}")
            print(f"  Sessions: {bids_manifest.n_sessions}")
            print(f"  Cohorts: {bids_manifest.get_cohort_breakdown()}")
            print(f"  Modalities: {bids_manifest.get_modality_breakdown()}")

            if bids_manifest.issues:
                print(f"\nIssues ({len(bids_manifest.issues)}):")
                for issue in bids_manifest.issues[:10]:
                    print(f"  - {issue['subject']}: {issue['issue']}")
                if len(bids_manifest.issues) > 10:
                    print(f"  ... and {len(bids_manifest.issues) - 10} more")

            results['bids'] = bids_manifest.to_dict()

        if not results:
            print(f"No data found. Checked:")
            print(f"  Bruker: {bruker_root}")
            print(f"  BIDS: {bids_root}")
            return 1

        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved manifest to: {args.output_json}")

        return 0

    # Full initialization requires name and code
    if not args.name or not args.code:
        parser.error("--name and --code are required for study initialization")

    # Run full setup
    report = setup_study(
        study_root=args.study_root,
        study_name=args.name,
        study_code=args.code,
        bids_root=args.bids_root,
        bruker_root=args.bruker_root,
        sigma_atlas_path=args.sigma_atlas,
        config_path=args.config,
        link_bids=not args.no_link,
        link_bruker=not args.no_link,
        setup_atlas=not args.no_atlas,
        force=args.force
    )

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to: {args.output_json}")

    return 0 if report['status'] == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())
