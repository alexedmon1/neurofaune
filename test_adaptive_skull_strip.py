#!/usr/bin/env python3
"""
Test adaptive skull stripping on a few subjects to validate improvement.

This script runs anatomical preprocessing on selected subjects to see:
1. What adaptive frac values are calculated
2. How skull stripping quality compares to fixed frac=0.3
"""

from pathlib import Path
from neurofaune.config import load_config
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing

# Test subjects - pick a variety
TEST_SUBJECTS = [
    ('sub-Rat207', 'ses-p30'),  # Already processed - good quality
    ('sub-Rat207', 'ses-p60'),  # Different cohort, same subject
    ('sub-Rat160', 'ses-p90'),  # Recently processed
]

BIDS_DIR = Path('/mnt/arborea/bpa-rat/raw/bids')
OUTPUT_DIR = Path('/mnt/arborea/bpa-rat')

def test_subject(subject: str, session: str):
    """Test adaptive skull stripping on a single subject."""
    print("\n" + "="*80)
    print(f"TESTING: {subject} / {session}")
    print("="*80)

    # Load config
    config = load_config(Path('configs/default.yaml'))

    # Extract cohort
    cohort = session.split('-')[1][:3] if session else 'unknown'

    # Create transform registry
    registry = create_transform_registry(config, subject, cohort=cohort)

    # Find subject directory
    subject_dir = BIDS_DIR / subject

    try:
        # Run preprocessing with adaptive skull stripping
        results = run_anatomical_preprocessing(
            config=config,
            subject=subject,
            session=session,
            output_dir=OUTPUT_DIR,
            transform_registry=registry,
            subject_dir=subject_dir,
            t2w_file=None,  # Auto-select
            prefer_orientation='axial'
        )

        print(f"\n✓ Successfully processed {subject}/{session}")
        print(f"  Brain: {results.get('brain_file')}")
        print(f"  Mask: {results.get('mask_file')}")

    except Exception as e:
        print(f"\n✗ Error processing {subject}/{session}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*80)
    print("Adaptive Skull Stripping Test")
    print("="*80)
    print(f"Testing {len(TEST_SUBJECTS)} subjects with adaptive BET frac calculation")
    print()

    for subject, session in TEST_SUBJECTS:
        test_subject(subject, session)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nReview the adaptive frac values printed above.")
    print("Check skull stripping quality in:")
    print("  /mnt/arborea/bpa-rat/derivatives/<subject>/<session>/anat/")
    print("\nCompare brain masks to see if adaptive approach improves quality.")
