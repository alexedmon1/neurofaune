#!/usr/bin/env python3
"""
Batch anatomical preprocessing for all subjects.

This script:
1. Finds all subjects in BIDS directory
2. Runs anatomical preprocessing for each subject/session
3. Automatically excludes 3D T2w scans (via scoring system)
4. Creates exclusion markers for failed subjects
5. Logs all processing to file

Usage:
    python batch_anatomical_preprocessing.py <bids_dir> <output_dir>

Example:
    python batch_anatomical_preprocessing.py /mnt/arborea/bpa-rat/raw/bids /mnt/arborea/bpa-rat
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import json

from neurofaune.config import load_config
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'batch_anatomical_preprocessing_{timestamp}.log'

    # Create logger
    logger = logging.getLogger('batch_anat')
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def find_all_subjects(bids_dir: Path) -> List[Dict[str, Path]]:
    """
    Find all subject/session combinations in BIDS directory.

    Returns
    -------
    List[Dict]
        List of dicts with 'subject', 'session', 'subject_dir', 'session_dir'
    """
    subjects = []

    # Find all subject directories
    for subject_dir in sorted(bids_dir.glob('sub-*')):
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name

        # Find all session directories
        session_dirs = sorted(subject_dir.glob('ses-*'))

        if session_dirs:
            # Multi-session dataset
            for session_dir in session_dirs:
                session = session_dir.name

                # Check if anatomical data exists
                anat_dir = session_dir / 'anat'
                if anat_dir.exists() and list(anat_dir.glob('*T2w*.nii*')):
                    subjects.append({
                        'subject': subject,
                        'session': session,
                        'subject_dir': subject_dir,
                        'session_dir': session_dir
                    })
        else:
            # Single-session dataset
            anat_dir = subject_dir / 'anat'
            if anat_dir.exists() and list(anat_dir.glob('*T2w*.nii*')):
                subjects.append({
                    'subject': subject,
                    'session': None,
                    'subject_dir': subject_dir,
                    'session_dir': subject_dir
                })

    return subjects


def check_exclusion_marker(output_dir: Path, subject: str, session: str = None) -> bool:
    """Check if subject has an exclusion marker."""
    if session:
        marker_file = output_dir / 'derivatives' / subject / session / f'{subject}_{session}_EXCLUDE.txt'
    else:
        marker_file = output_dir / 'derivatives' / subject / f'{subject}_EXCLUDE.txt'

    return marker_file.exists()


def create_exclusion_marker(output_dir: Path, subject: str, session: str, reason: str):
    """Create an exclusion marker file for failed preprocessing."""
    if session:
        derivatives_dir = output_dir / 'derivatives' / subject / session
    else:
        derivatives_dir = output_dir / 'derivatives' / subject

    derivatives_dir.mkdir(parents=True, exist_ok=True)

    if session:
        marker_file = derivatives_dir / f'{subject}_{session}_EXCLUDE.txt'
    else:
        marker_file = derivatives_dir / f'{subject}_EXCLUDE.txt'

    with open(marker_file, 'w') as f:
        f.write(f"Excluded from analysis\n")
        f.write(f"Reason: {reason}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")


def process_subject(
    config: Dict[str, Any],
    subject: str,
    session: str,
    subject_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    force: bool = False
) -> Dict[str, Any]:
    """
    Process a single subject/session.

    Returns
    -------
    dict
        Processing results with 'status': 'success', 'excluded', or 'failed'
    """
    session_str = session if session else 'no-session'
    logger.info(f"{'='*80}")
    logger.info(f"Processing: {subject} / {session_str}")
    logger.info(f"{'='*80}")

    # Check for existing exclusion marker
    if check_exclusion_marker(output_dir, subject, session) and not force:
        logger.info(f"Skipping {subject}/{session_str}: Exclusion marker found")
        return {'status': 'excluded', 'reason': 'Previous exclusion marker'}

    # Check if already processed (skip if outputs exist)
    if session:
        derivatives_dir = output_dir / 'derivatives' / subject / session / 'anat'
    else:
        derivatives_dir = output_dir / 'derivatives' / subject / 'anat'

    brain_file = derivatives_dir / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
    if brain_file.exists() and not force:
        logger.info(f"Skipping {subject}/{session_str}: Already processed")
        return {'status': 'skipped', 'reason': 'Already processed'}

    # Extract cohort from session
    if session:
        cohort = session.split('-')[1][:3]  # Extract 'p30', 'p60', or 'p90'
    else:
        cohort = 'unknown'

    # Create transform registry
    registry = create_transform_registry(config, subject, cohort=cohort)

    # Run preprocessing
    try:
        logger.info(f"Running anatomical preprocessing...")

        results = run_anatomical_preprocessing(
            config=config,
            subject=subject,
            session=session if session else '',
            output_dir=output_dir,
            transform_registry=registry,
            subject_dir=subject_dir,
            t2w_file=None,  # Let it auto-select
            prefer_orientation='axial'
        )

        # Check if subject was excluded due to 3D scan or other reasons
        if results.get('status') == 'excluded':
            reason = results.get('reason', 'Unknown')
            logger.warning(f"Subject excluded: {reason}")
            create_exclusion_marker(output_dir, subject, session, reason)
            return {'status': 'excluded', 'reason': reason}

        logger.info(f"✓ Successfully processed {subject}/{session_str}")
        return {'status': 'success', 'results': results}

    except Exception as e:
        logger.error(f"✗ Error processing {subject}/{session_str}: {e}")
        import traceback
        logger.error(traceback.format_exc())

        create_exclusion_marker(output_dir, subject, session, f"Processing error: {str(e)}")
        return {'status': 'failed', 'error': str(e)}


def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_anatomical_preprocessing.py <bids_dir> <output_dir>")
        print("\nExample:")
        print("  python batch_anatomical_preprocessing.py /mnt/arborea/bpa-rat/raw/bids /mnt/arborea/bpa-rat")
        print("\nOptions:")
        print("  --force : Reprocess subjects even if already completed")
        sys.exit(1)

    bids_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    force = '--force' in sys.argv

    if not bids_dir.exists():
        print(f"Error: BIDS directory not found: {bids_dir}")
        sys.exit(1)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("="*80)
    logger.info("Batch Anatomical Preprocessing")
    logger.info("="*80)
    logger.info(f"BIDS directory: {bids_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Force reprocessing: {force}")
    logger.info("="*80)

    # Load configuration
    config_file = Path('configs/default.yaml')
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        sys.exit(1)

    logger.info(f"Loading config: {config_file}")
    config = load_config(config_file)

    # Find all subjects
    logger.info("Finding subjects...")
    subjects = find_all_subjects(bids_dir)
    logger.info(f"Found {len(subjects)} subject/session combinations")

    if len(subjects) == 0:
        logger.error("No subjects found!")
        sys.exit(1)

    # Process each subject
    results_summary = {
        'success': [],
        'excluded': [],
        'failed': [],
        'skipped': []
    }

    for idx, subj_info in enumerate(subjects, 1):
        logger.info(f"\n[{idx}/{len(subjects)}]")

        result = process_subject(
            config=config,
            subject=subj_info['subject'],
            session=subj_info.get('session'),
            subject_dir=subj_info['subject_dir'],
            output_dir=output_dir,
            logger=logger,
            force=force
        )

        status = result['status']
        key = f"{subj_info['subject']}/{subj_info.get('session', 'no-session')}"
        results_summary[status].append(key)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total subjects: {len(subjects)}")
    logger.info(f"  ✓ Success: {len(results_summary['success'])}")
    logger.info(f"  ⊘ Excluded: {len(results_summary['excluded'])}")
    logger.info(f"  ✗ Failed: {len(results_summary['failed'])}")
    logger.info(f"  - Skipped: {len(results_summary['skipped'])}")

    if results_summary['excluded']:
        logger.info("\nExcluded subjects:")
        for subj in results_summary['excluded']:
            logger.info(f"  - {subj}")

    if results_summary['failed']:
        logger.info("\nFailed subjects:")
        for subj in results_summary['failed']:
            logger.info(f"  - {subj}")

    # Save summary to JSON
    summary_file = output_dir / 'logs' / f'preprocessing_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info("="*80)

    # Exit with error if any failures
    if results_summary['failed']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
