#!/usr/bin/env python3
"""
Generate anatomical QC retroactively for already-preprocessed subjects.

This script scans the derivatives directory for subjects that have preprocessed
anatomical outputs but no QC metrics, and generates QC reports for them.

Usage:
    uv run python scripts/generate_anat_qc_retroactive.py /path/to/study
    uv run python scripts/generate_anat_qc_retroactive.py /path/to/study --dry-run
    uv run python scripts/generate_anat_qc_retroactive.py /path/to/study --subject sub-Rat1
"""

import argparse
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.preprocess.qc.anat import generate_anatomical_qc_report
from neurofaune.preprocess.qc import get_subject_qc_dir


def find_preprocessed_subjects(study_root: Path):
    """Find all subjects with preprocessed anatomical data."""
    derivatives_dir = study_root / 'derivatives'
    if not derivatives_dir.exists():
        return []

    subjects = []
    for subj_dir in sorted(derivatives_dir.glob('sub-*')):
        for sess_dir in sorted(subj_dir.glob('ses-*')):
            anat_dir = sess_dir / 'anat'
            if not anat_dir.exists():
                continue

            # Check for required preprocessed files
            subject = subj_dir.name
            session = sess_dir.name

            brain = anat_dir / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
            mask = anat_dir / f'{subject}_{session}_desc-brain_mask.nii.gz'
            brain_ss = anat_dir / f'{subject}_{session}_desc-skullstrip_T2w.nii.gz'

            if brain.exists() and mask.exists():
                subjects.append({
                    'subject': subject,
                    'session': session,
                    'anat_dir': anat_dir,
                    'brain': brain,
                    'brain_ss': brain_ss if brain_ss.exists() else brain,
                    'mask': mask,
                    'gm': anat_dir / f'{subject}_{session}_label-GM_probseg.nii.gz',
                    'wm': anat_dir / f'{subject}_{session}_label-WM_probseg.nii.gz',
                    'csf': anat_dir / f'{subject}_{session}_label-CSF_probseg.nii.gz',
                })

    return subjects


def check_existing_qc(study_root: Path, subject: str, session: str) -> bool:
    """Check if QC metrics already exist for a subject."""
    # Check new structure: qc/sub/{subject}/{session}/anat/
    qc_dir = study_root / 'qc' / 'sub' / subject / session / 'anat'
    metrics_file = qc_dir / f'{subject}_{session}_anat_qc_metrics.json'
    return metrics_file.exists()


def main():
    parser = argparse.ArgumentParser(
        description='Generate anatomical QC retroactively for preprocessed subjects'
    )
    parser.add_argument('study_root', type=Path, help='Study root directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='List subjects without generating QC')
    parser.add_argument('--subject', type=str,
                        help='Process specific subject only')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate QC even if it already exists')
    args = parser.parse_args()

    study_root = args.study_root.resolve()

    print("="*70)
    print("Retroactive Anatomical QC Generation")
    print("="*70)
    print(f"Study root: {study_root}")
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

        # Get QC output directory using standard function
        qc_dir = get_subject_qc_dir(study_root, subject, session, 'anat')

        try:
            qc_report = generate_anatomical_qc_report(
                subject=subject,
                session=session,
                t2w_file=subj['brain_ss'],  # Use skull-stripped T2w
                brain_file=subj['brain'],
                mask_file=subj['mask'],
                gm_file=subj['gm'] if subj['gm'].exists() else None,
                wm_file=subj['wm'] if subj['wm'].exists() else None,
                csf_file=subj['csf'] if subj['csf'].exists() else None,
                output_dir=qc_dir
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
