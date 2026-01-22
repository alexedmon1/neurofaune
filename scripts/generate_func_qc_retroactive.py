#!/usr/bin/env python3
"""
Generate functional QC retroactively for already-preprocessed subjects.

This script scans the derivatives directory for subjects that have preprocessed
functional outputs but no QC metrics, and generates QC reports for them.

Usage:
    uv run python scripts/generate_func_qc_retroactive.py /path/to/study
    uv run python scripts/generate_func_qc_retroactive.py /path/to/study --dry-run
    uv run python scripts/generate_func_qc_retroactive.py /path/to/study --subject sub-Rat1
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.preprocess.qc import get_subject_qc_dir
from neurofaune.preprocess.qc.func.motion_qc import (
    calculate_framewise_displacement,
    calculate_dvars,
    generate_motion_qc_report
)


def find_preprocessed_subjects(study_root: Path):
    """Find all subjects with preprocessed functional data."""
    derivatives_dir = study_root / 'derivatives'
    if not derivatives_dir.exists():
        return []

    subjects = []
    for subj_dir in sorted(derivatives_dir.glob('sub-*')):
        for sess_dir in sorted(subj_dir.glob('ses-*')):
            func_dir = sess_dir / 'func'
            if not func_dir.exists():
                continue

            subject = subj_dir.name
            session = sess_dir.name

            # Check for required preprocessed files
            confounds = func_dir / f'{subject}_{session}_desc-confounds_timeseries.tsv'
            bold = func_dir / f'{subject}_{session}_desc-preproc_bold.nii.gz'
            mask = func_dir / f'{subject}_{session}_desc-brain_mask.nii.gz'

            if confounds.exists() and bold.exists() and mask.exists():
                subjects.append({
                    'subject': subject,
                    'session': session,
                    'func_dir': func_dir,
                    'confounds': confounds,
                    'bold': bold,
                    'mask': mask,
                })

    return subjects


def check_existing_qc(study_root: Path, subject: str, session: str) -> bool:
    """Check if QC metrics already exist for a subject."""
    qc_dir = study_root / 'qc' / 'sub' / subject / session / 'func'
    metrics_file = qc_dir / f'{subject}_{session}_func_qc_metrics.json'
    return metrics_file.exists()


def extract_motion_params_from_confounds(confounds_file: Path) -> np.ndarray:
    """Extract 6-parameter motion from confounds TSV.

    Returns array of shape (n_volumes, 6) with columns:
    [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    """
    df = pd.read_csv(confounds_file, sep='\t')

    # Extract motion columns
    motion_cols = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']
    motion_params = df[motion_cols].values

    return motion_params


def generate_func_qc_from_confounds(
    subject: str,
    session: str,
    confounds_file: Path,
    bold_file: Path,
    mask_file: Path,
    output_dir: Path,
    threshold_fd: float = 5.0  # 5mm scaled = 0.5mm actual (10x voxel scaling)
) -> Path:
    """
    Generate functional QC from confounds TSV file.

    This is a wrapper that extracts motion params from confounds and
    calls the standard motion QC generator.
    """
    # Create temporary motion params file
    motion_params = extract_motion_params_from_confounds(confounds_file)

    # Save as temporary .par file (FSL format)
    temp_par = output_dir / f'{subject}_{session}_motion.par'
    np.savetxt(temp_par, motion_params, fmt='%.10e')

    try:
        # Generate full motion QC report
        report = generate_motion_qc_report(
            subject=subject,
            session=session,
            motion_params_file=temp_par,
            bold_file=bold_file,
            mask_file=mask_file,
            output_dir=output_dir,
            threshold_fd=threshold_fd
        )
        return report
    finally:
        # Clean up temporary file
        if temp_par.exists():
            temp_par.unlink()


def main():
    parser = argparse.ArgumentParser(
        description='Generate functional QC retroactively for preprocessed subjects'
    )
    parser.add_argument('study_root', type=Path, help='Study root directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='List subjects without generating QC')
    parser.add_argument('--subject', type=str,
                        help='Process specific subject only')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate QC even if it already exists')
    parser.add_argument('--fd-threshold', type=float, default=5.0,
                        help='FD threshold in mm (default: 5.0 = 0.5mm actual with 10x scaling)')
    args = parser.parse_args()

    study_root = args.study_root.resolve()

    print("="*70)
    print("Retroactive Functional QC Generation")
    print("="*70)
    print(f"Study root: {study_root}")
    print(f"FD threshold: {args.fd_threshold}mm (scaled)")
    print()

    # Find all preprocessed subjects
    subjects = find_preprocessed_subjects(study_root)
    print(f"Found {len(subjects)} preprocessed subject/sessions")

    # Filter by specific subject if requested
    if args.subject:
        subjects = [s for s in subjects if s['subject'] == args.subject]
        print(f"Filtered to {len(subjects)} sessions for {args.subject}")

    # Check which need QC
    need_qc = []
    have_qc = []

    for subj in subjects:
        has_qc = check_existing_qc(study_root, subj['subject'], subj['session'])
        if has_qc and not args.force:
            have_qc.append(subj)
        else:
            need_qc.append(subj)

    print(f"\nAlready have QC: {len(have_qc)}")
    print(f"Need QC: {len(need_qc)}")

    if args.dry_run:
        print("\n[DRY RUN] Would generate QC for:")
        for subj in need_qc:
            print(f"  - {subj['subject']} {subj['session']}")
        return

    if not need_qc:
        print("\nNo subjects need QC generation.")
        return

    # Generate QC for subjects that need it
    print(f"\nGenerating QC for {len(need_qc)} subjects...")
    print("-"*70)

    success = 0
    failed = []

    for i, subj in enumerate(need_qc, 1):
        subject = subj['subject']
        session = subj['session']

        print(f"\n[{i}/{len(need_qc)}] {subject} {session}")

        # Get QC output directory
        qc_dir = get_subject_qc_dir(study_root, subject, session, 'func')

        try:
            qc_report = generate_func_qc_from_confounds(
                subject=subject,
                session=session,
                confounds_file=subj['confounds'],
                bold_file=subj['bold'],
                mask_file=subj['mask'],
                output_dir=qc_dir,
                threshold_fd=args.fd_threshold
            )
            print(f"  -> QC report: {qc_report.name}")
            success += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append((subject, session, str(e)))

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Success: {success}/{len(need_qc)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed subjects:")
        for subject, session, error in failed:
            print(f"  - {subject} {session}: {error[:60]}...")


if __name__ == '__main__':
    main()
