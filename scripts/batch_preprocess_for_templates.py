#!/usr/bin/env python3
"""
Batch preprocess subjects for template building.

This script preprocesses unprocessed subjects from each cohort (p30, p60, p90)
to build up to the minimum required for template building (10 subjects per cohort).
"""

import sys
from pathlib import Path
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing
from neurofaune.utils.exclusion import create_exclusion_marker, check_exclusion_marker
from neurofaune.utils.transforms import create_transform_registry


def find_unpreprocessed_subjects(bids_dir: Path, derivatives_dir: Path, cohort: str, limit: int = 10):
    """
    Find unpreprocessed subjects for a given cohort.

    Parameters
    ----------
    bids_dir : Path
        BIDS directory with raw data
    derivatives_dir : Path
        Derivatives directory
    cohort : str
        Cohort name (p30, p60, p90)
    limit : int
        Maximum number of subjects to return

    Returns
    -------
    list
        List of subject IDs that need preprocessing
    """
    unpreprocessed = []

    for subject_dir in sorted(bids_dir.glob('sub-Rat*')):
        if len(unpreprocessed) >= limit:
            break

        subject = subject_dir.name
        session = f'ses-{cohort}'

        # Check if session exists
        session_dir = subject_dir / session
        if not session_dir.exists():
            continue

        # Check for T2w scans
        t2w_scans = list((session_dir / 'anat').glob('*T2w.nii.gz'))
        if not t2w_scans:
            continue

        # Check if already preprocessed
        brain_file = derivatives_dir / subject / session / 'anat' / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
        gm_file = derivatives_dir / subject / session / 'anat' / f'{subject}_{session}_label-GM_probseg.nii.gz'

        if brain_file.exists() and gm_file.exists():
            continue

        # Check for exclusion marker
        excluded, marker_data = check_exclusion_marker(subject, session, derivatives_dir.parent)
        if excluded:
            print(f"  ⊘ {subject} {session} excluded (preprocessing previously failed)")
            continue

        unpreprocessed.append(subject)

    return unpreprocessed


def main():
    print("="*80)
    print("BATCH PREPROCESSING FOR TEMPLATE BUILDING")
    print("="*80)

    # Setup paths
    config_file = Path(__file__).parent.parent / 'configs' / 'bpa_rat_example.yaml'
    config = load_config(config_file)

    output_dir = Path(config['paths']['study_root'])
    bids_dir = Path(config['paths']['bids'])
    derivatives_dir = output_dir / 'derivatives'

    cohorts = ['p30', 'p60', 'p90']
    subjects_per_cohort = 10  # Preprocess 10 per cohort

    print(f"\nConfiguration:")
    print(f"  BIDS directory: {bids_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Subjects per cohort: {subjects_per_cohort}")
    print(f"  Target total per cohort: 14 (4 existing + {subjects_per_cohort} new)")

    # Find subjects to preprocess for each cohort
    subjects_to_process = {}
    for cohort in cohorts:
        print(f"\nFinding unpreprocessed subjects for {cohort}...")
        subjects = find_unpreprocessed_subjects(bids_dir, derivatives_dir, cohort, subjects_per_cohort)
        subjects_to_process[cohort] = subjects
        print(f"  Found {len(subjects)} subjects to preprocess")
        if subjects:
            print(f"  Subjects: {', '.join(subjects[:5])}", end='')
            if len(subjects) > 5:
                print(f" ... +{len(subjects) - 5} more")
            else:
                print()

    # Calculate total preprocessing runs
    total_runs = sum(len(subjects) for subjects in subjects_to_process.values())
    print(f"\nTotal preprocessing runs: {total_runs}")
    print("="*80)

    # Process each cohort
    completed = 0
    failed = 0

    for cohort in cohorts:
        subjects = subjects_to_process[cohort]
        if not subjects:
            print(f"\n{'='*80}")
            print(f"Cohort {cohort.upper()}: No subjects to process")
            print('='*80)
            continue

        print(f"\n{'='*80}")
        print(f"PROCESSING COHORT: {cohort.upper()} ({len(subjects)} subjects)")
        print('='*80)

        for i, subject in enumerate(subjects, 1):
            session = f'ses-{cohort}'
            print(f"\n[{completed + failed + 1}/{total_runs}] Processing {subject} {session} ({i}/{len(subjects)} for {cohort})...")

            try:
                # Create transform registry for this subject
                registry = create_transform_registry(config, subject, cohort)

                # Get subject directory from BIDS
                subject_bids_dir = bids_dir / subject

                results = run_anatomical_preprocessing(
                    config=config,
                    subject=subject,
                    session=session,
                    output_dir=output_dir,
                    transform_registry=registry,
                    subject_dir=subject_bids_dir
                )

                if results.get('status') == 'excluded':
                    print(f"  ⊘ {subject} {session} excluded")
                    failed += 1
                else:
                    print(f"  ✓ {subject} {session} complete!")
                    completed += 1

            except Exception as e:
                print(f"  ❌ Error: {e}")
                traceback.print_exc()

                # Create exclusion marker
                error_tb = traceback.format_exc()
                marker_file = create_exclusion_marker(
                    subject=subject,
                    session=session,
                    output_dir=output_dir,
                    reason=str(e)[:200],
                    error_traceback=error_tb
                )
                print(f"  Created exclusion marker: {marker_file}")
                failed += 1

    # Summary
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\n✓ Successful: {completed}/{total_runs}")
    print(f"❌ Failed: {failed}/{total_runs}")

    # Check template building readiness
    print("\nTemplate Building Readiness:")
    for cohort in cohorts:
        # Count preprocessed subjects
        preprocessed = []
        for subject_dir in bids_dir.glob('sub-Rat*'):
            subject = subject_dir.name
            session = f'ses-{cohort}'

            brain_file = derivatives_dir / subject / session / 'anat' / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
            gm_file = derivatives_dir / subject / session / 'anat' / f'{subject}_{session}_label-GM_probseg.nii.gz'

            if brain_file.exists() and gm_file.exists():
                preprocessed.append(subject)

        status = "✓ READY" if len(preprocessed) >= 10 else f"⊙ NEED {10 - len(preprocessed)} MORE"
        print(f"  {cohort.upper()}: {len(preprocessed)} subjects - {status}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
