#!/usr/bin/env python3
"""
Migrate QC directory structure from old to new format.

Old structure: qc/{subject}/{session}/{modality}/
New structure: qc/sub/{subject}/{session}/{modality}/

This script moves all subject QC directories under qc/sub/ for better organization.

Usage:
    uv run python scripts/migrate_qc_to_sub_dir.py /path/to/study --dry-run
    uv run python scripts/migrate_qc_to_sub_dir.py /path/to/study
"""

import argparse
import shutil
from pathlib import Path


def migrate_qc_structure(study_root: Path, dry_run: bool = False) -> dict:
    """
    Migrate QC from old structure to new structure.

    Returns dict with counts of moved items.
    """
    qc_root = study_root / 'qc'
    sub_dir = qc_root / 'sub'

    if not qc_root.exists():
        print(f"QC directory not found: {qc_root}")
        return {'moved': 0, 'skipped': 0, 'errors': 0}

    # Find all subject directories in old location (directly under qc/)
    old_subject_dirs = sorted([
        d for d in qc_root.iterdir()
        if d.is_dir() and d.name.startswith('sub-')
    ])

    print(f"Found {len(old_subject_dirs)} subject directories to migrate")

    if not old_subject_dirs:
        print("No subject directories found in old location.")
        return {'moved': 0, 'skipped': 0, 'errors': 0}

    # Create sub directory
    if not dry_run:
        sub_dir.mkdir(exist_ok=True)

    moved = 0
    skipped = 0
    errors = 0

    for old_dir in old_subject_dirs:
        subject = old_dir.name
        new_dir = sub_dir / subject

        if new_dir.exists():
            print(f"  SKIP: {subject} - already exists in new location")
            skipped += 1
            continue

        if dry_run:
            print(f"  WOULD MOVE: {old_dir} -> {new_dir}")
            moved += 1
        else:
            try:
                shutil.move(str(old_dir), str(new_dir))
                print(f"  MOVED: {subject}")
                moved += 1
            except Exception as e:
                print(f"  ERROR: {subject} - {e}")
                errors += 1

    return {'moved': moved, 'skipped': skipped, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(
        description='Migrate QC directory structure to qc/sub/{subject}/'
    )
    parser.add_argument('study_root', type=Path, help='Study root directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be moved without moving')
    args = parser.parse_args()

    study_root = args.study_root.resolve()

    print("="*60)
    print("QC Directory Structure Migration")
    print("="*60)
    print(f"Study root: {study_root}")
    print(f"Old structure: qc/sub-*/")
    print(f"New structure: qc/sub/sub-*/")
    print()

    if args.dry_run:
        print("[DRY RUN MODE - No changes will be made]\n")

    results = migrate_qc_structure(study_root, dry_run=args.dry_run)

    print()
    print("="*60)
    print("Summary")
    print("="*60)
    print(f"Moved: {results['moved']}")
    print(f"Skipped (already exists): {results['skipped']}")
    print(f"Errors: {results['errors']}")

    if args.dry_run and results['moved'] > 0:
        print(f"\nRun without --dry-run to perform migration.")


if __name__ == '__main__':
    main()
