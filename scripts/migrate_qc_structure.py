#!/usr/bin/env python3
"""
Migrate QC directory structure to the new layout.

Old structures (handled):
    qc/sub-*/ses-*/{modality}/     (flat, original)
    qc/sub/sub-*/ses-*/{modality}/ (partial migration, legacy)
    qc/{mod}_batch_summary/        (batch summaries at top level)
    qc/msme_visualization/         (MSME viz at top level)
    qc/templates/                   (template QC at top level)

New structure:
    qc/subjects/sub-*/ses-*/{modality}/   (per-subject QC)
    qc/reports/{mod}_batch_summary/       (batch summaries)
    qc/reports/msme_visualization/        (MSME viz)
    qc/reports/templates/                  (template QC)

Replaces the older migrate_qc_to_sub_dir.py script.

Usage:
    uv run python scripts/migrate_qc_structure.py /path/to/study --dry-run
    uv run python scripts/migrate_qc_structure.py /path/to/study
"""

import argparse
import shutil
from pathlib import Path


BATCH_SUMMARY_MODALITIES = ['anat', 'dwi', 'func', 'msme']


def migrate_qc_structure(study_root: Path, dry_run: bool = False) -> dict:
    """
    Migrate QC directory to new subjects/ + reports/ layout.

    Returns dict with counts of moved items.
    """
    qc_root = study_root / 'qc'

    if not qc_root.exists():
        print(f"QC directory not found: {qc_root}")
        return {'moved': 0, 'skipped': 0, 'errors': 0}

    subjects_dir = qc_root / 'subjects'
    reports_dir = qc_root / 'reports'

    moved = 0
    skipped = 0
    errors = 0

    # --- 1. Move qc/sub-* → qc/subjects/sub-* (flat structure) ---
    flat_subject_dirs = sorted([
        d for d in qc_root.iterdir()
        if d.is_dir() and d.name.startswith('sub-')
    ])
    if flat_subject_dirs:
        print(f"\n[1] Moving {len(flat_subject_dirs)} flat subject dirs → qc/subjects/")
        if not dry_run:
            subjects_dir.mkdir(exist_ok=True)
        for d in flat_subject_dirs:
            dest = subjects_dir / d.name
            m, s, e = _move_dir(d, dest, dry_run)
            moved += m; skipped += s; errors += e

    # --- 2. Move qc/sub/sub-* → qc/subjects/sub-* (legacy partial migration) ---
    old_sub_dir = qc_root / 'sub'
    if old_sub_dir.exists() and old_sub_dir.is_dir():
        legacy_dirs = sorted([
            d for d in old_sub_dir.iterdir()
            if d.is_dir() and d.name.startswith('sub-')
        ])
        if legacy_dirs:
            print(f"\n[2] Moving {len(legacy_dirs)} legacy qc/sub/ dirs → qc/subjects/")
            if not dry_run:
                subjects_dir.mkdir(exist_ok=True)
            for d in legacy_dirs:
                dest = subjects_dir / d.name
                m, s, e = _move_dir(d, dest, dry_run)
                moved += m; skipped += s; errors += e

        # Remove empty qc/sub/ directory
        if not dry_run:
            try:
                old_sub_dir.rmdir()
                print(f"  Removed empty directory: {old_sub_dir}")
            except OSError:
                pass  # Not empty, leave it

    # --- 3. Move qc/{mod}_batch_summary → qc/reports/{mod}_batch_summary ---
    batch_dirs = []
    for mod in BATCH_SUMMARY_MODALITIES:
        d = qc_root / f'{mod}_batch_summary'
        if d.exists() and d.is_dir():
            batch_dirs.append(d)

    if batch_dirs:
        print(f"\n[3] Moving {len(batch_dirs)} batch summary dirs → qc/reports/")
        if not dry_run:
            reports_dir.mkdir(exist_ok=True)
        for d in batch_dirs:
            dest = reports_dir / d.name
            m, s, e = _move_dir(d, dest, dry_run)
            moved += m; skipped += s; errors += e

    # --- 4. Move qc/msme_visualization → qc/reports/msme_visualization ---
    msme_viz = qc_root / 'msme_visualization'
    if msme_viz.exists() and msme_viz.is_dir():
        print(f"\n[4] Moving msme_visualization → qc/reports/")
        if not dry_run:
            reports_dir.mkdir(exist_ok=True)
        dest = reports_dir / 'msme_visualization'
        m, s, e = _move_dir(msme_viz, dest, dry_run)
        moved += m; skipped += s; errors += e

    # --- 5. Move qc/templates/ → qc/reports/templates/ ---
    templates_dir = qc_root / 'templates'
    if templates_dir.exists() and templates_dir.is_dir():
        print(f"\n[5] Moving templates → qc/reports/templates/")
        if not dry_run:
            reports_dir.mkdir(exist_ok=True)
        dest = reports_dir / 'templates'
        m, s, e = _move_dir(templates_dir, dest, dry_run)
        moved += m; skipped += s; errors += e

    return {'moved': moved, 'skipped': skipped, 'errors': errors}


def _move_dir(src: Path, dest: Path, dry_run: bool) -> tuple:
    """Move a directory, returning (moved, skipped, errors) counts."""
    if dest.exists():
        print(f"  SKIP: {src.name} → already exists at {dest}")
        return (0, 1, 0)
    if dry_run:
        print(f"  WOULD MOVE: {src} → {dest}")
        return (1, 0, 0)
    try:
        shutil.move(str(src), str(dest))
        print(f"  MOVED: {src.name}")
        return (1, 0, 0)
    except Exception as e:
        print(f"  ERROR: {src.name} → {e}")
        return (0, 0, 1)


def main():
    parser = argparse.ArgumentParser(
        description='Migrate QC directory structure to qc/subjects/ + qc/reports/'
    )
    parser.add_argument('study_root', type=Path, help='Study root directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be moved without moving')
    args = parser.parse_args()

    study_root = args.study_root.resolve()

    print("=" * 60)
    print("QC Directory Structure Migration")
    print("=" * 60)
    print(f"Study root: {study_root}")
    print(f"New structure:")
    print(f"  qc/subjects/sub-*/ses-*/ (per-subject QC)")
    print(f"  qc/reports/              (batch summaries, omnibus reports)")
    print()

    if args.dry_run:
        print("[DRY RUN MODE — No changes will be made]\n")

    results = migrate_qc_structure(study_root, dry_run=args.dry_run)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Moved:   {results['moved']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Errors:  {results['errors']}")

    if args.dry_run and results['moved'] > 0:
        print(f"\nRun without --dry-run to perform migration.")


if __name__ == '__main__':
    main()
