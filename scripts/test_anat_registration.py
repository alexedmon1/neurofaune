#!/usr/bin/env python3
"""
Test script for ANTs-based anatomical registration.

Tests on one subject per cohort (P30, P60, P90) to evaluate
registration quality and compare with FLIRT/FNIRT approach.
"""

from pathlib import Path
import sys

# Add neurofaune to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing


def test_subject(
    subject: str,
    session: str,
    t2w_file: Path,
    output_dir: Path,
    config_file: Path
):
    """Test anatomical preprocessing on a single subject."""

    print(f"\n{'='*80}")
    print(f"Testing {subject} {session}")
    print(f"{'='*80}\n")

    # Load config
    config = load_config(config_file)

    # Extract cohort from session
    cohort = session.split('-')[1]

    # Create transform registry
    registry = create_transform_registry(
        config, subject=subject, cohort=cohort
    )

    # Run preprocessing
    try:
        results = run_anatomical_preprocessing(
            config=config,
            subject=subject,
            session=session,
            t2w_file=t2w_file,
            output_dir=output_dir,
            transform_registry=registry,
            modality='anat',
            slice_range=None  # Full volume for initial test
        )

        print(f"\n✓ {subject} {session} completed successfully!")
        print(f"  Brain: {results['brain']}")
        print(f"  Warped: {results['warped']}")
        print(f"  Transform: {results['composite_transform']}")

        return results

    except Exception as e:
        print(f"\n✗ {subject} {session} FAILED:")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run tests on all three cohorts."""

    # Paths
    test_dir = Path('/mnt/arborea/bpa-rat/test')
    config_file = Path('/home/edm9fd/sandbox/neurofaune/configs/default.yaml')

    # Test subjects
    test_subjects = [
        {
            'subject': 'sub-Rat207',
            'session': 'ses-p30',
            't2w_file': test_dir / 'anat/p30/sub-Rat207_ses-p30_T2w.nii.gz'
        },
        {
            'subject': 'sub-Rat207',
            'session': 'ses-p60',
            't2w_file': test_dir / 'anat/p60/sub-Rat207_ses-p60_T2w.nii.gz'
        },
        {
            'subject': 'sub-Rat110',
            'session': 'ses-p90',
            't2w_file': test_dir / 'anat/p90/sub-Rat110_ses-p90_T2w.nii.gz'
        }
    ]

    # Run tests
    results_summary = []
    for test_case in test_subjects:
        result = test_subject(
            subject=test_case['subject'],
            session=test_case['session'],
            t2w_file=test_case['t2w_file'],
            output_dir=test_dir,
            config_file=config_file
        )
        results_summary.append({
            'subject': test_case['subject'],
            'session': test_case['session'],
            'success': result is not None
        })

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    for item in results_summary:
        status = "✓ SUCCESS" if item['success'] else "✗ FAILED"
        print(f"  {status}: {item['subject']} {item['session']}")

    print()


if __name__ == '__main__':
    main()
