#!/usr/bin/env python3
"""
Test script for template building.

This script:
1. Preprocesses 3 subjects from each cohort (p30, p60, p90)
2. Builds templates for each cohort
3. Validates outputs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.utils.exclusion import create_exclusion_marker, check_exclusion_marker
from neurofaune.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing
from scripts.build_templates import build_anatomical_template
import traceback


# Test subjects (manually selected to have good quality data)
# Using 4 subjects per cohort for minimal template building
# Note: Automatic T2w selection now avoids 3D TurboRARE scans (Cohorts 1-2)
# and prefers 2D RARE scans for better skull stripping results
TEST_SUBJECTS = {
    'p30': ['sub-Rat108', 'sub-Rat116', 'sub-Rat120', 'sub-Rat207'],
    'p60': ['sub-Rat108', 'sub-Rat116', 'sub-Rat120', 'sub-Rat207'],
    'p90': ['sub-Rat108', 'sub-Rat116', 'sub-Rat120', 'sub-Rat207']
}


def preprocess_test_subjects():
    """Preprocess test subjects for template building."""

    # Load config
    config_file = Path('/home/edm9fd/sandbox/neurofaune/configs/bpa_rat_example.yaml')
    config = load_config(config_file)

    study_root = Path('/mnt/arborea/bpa-rat')
    bids_dir = study_root / 'raw' / 'bids'
    output_dir = study_root  # Study root

    print("="*80)
    print("PREPROCESSING TEST SUBJECTS")
    print("="*80)

    for cohort, subjects in TEST_SUBJECTS.items():
        session = f'ses-{cohort}'

        for subject in subjects:
            print(f"\n{'='*80}")
            print(f"Processing {subject} {session}")
            print(f"{'='*80}")

            subject_dir = bids_dir / subject

            # Check if already preprocessed
            derivatives_dir = output_dir / 'derivatives' / subject / session / 'anat'
            gm_file = derivatives_dir / f'{subject}_{session}_label-GM_probseg.nii.gz'

            if gm_file.exists():
                print(f"✓ {subject} {session} already preprocessed with tissue segmentation")
                continue

            try:
                # Create transform registry
                registry = create_transform_registry(config, subject, cohort=cohort)

                # Run preprocessing
                results = run_anatomical_preprocessing(
                    config=config,
                    subject=subject,
                    session=session,
                    subject_dir=subject_dir,
                    output_dir=output_dir,
                    transform_registry=registry,
                    prefer_orientation='axial'
                )

                # Check if subject was excluded
                if results.get('status') == 'excluded':
                    print(f"⊘ {subject} {session} excluded (preprocessing previously failed)")
                    continue

                print(f"\n✓ {subject} {session} preprocessing complete!")
                print(f"  Brain: {results['brain']}")
                print(f"  GM: {results['gm_prob']}")
                print(f"  WM: {results['wm_prob']}")
                print(f"  CSF: {results['csf_prob']}")

            except Exception as e:
                print(f"\n❌ Error processing {subject} {session}: {e}")
                traceback.print_exc()

                # Create exclusion marker so this subject is skipped in future runs
                error_tb = traceback.format_exc()
                create_exclusion_marker(
                    subject=subject,
                    session=session,
                    output_dir=output_dir,
                    reason=str(e)[:200],  # Limit reason to 200 chars
                    error_traceback=error_tb
                )
                continue


def build_test_templates():
    """Build templates from preprocessed test subjects."""

    # Load config
    config_file = Path('/home/edm9fd/sandbox/neurofaune/configs/bpa_rat_example.yaml')
    config = load_config(config_file)

    study_root = Path('/mnt/arborea/bpa-rat')
    derivatives_dir = study_root / 'derivatives'
    template_dir = study_root / 'templates'

    print("\n" + "="*80)
    print("BUILDING TEMPLATES")
    print("="*80)

    for cohort in ['p30', 'p60', 'p90']:
        print(f"\n{'='*80}")
        print(f"Building templates for cohort: {cohort}")
        print(f"{'='*80}")

        try:
            build_anatomical_template(
                config=config,
                cohort=cohort,
                derivatives_dir=derivatives_dir,
                template_dir=template_dir,
                top_percent=1.0,  # Use all available subjects
                n_cores=4
            )

            print(f"\n✓ {cohort} templates complete!")

        except Exception as e:
            print(f"\n❌ Error building {cohort} template: {e}")
            import traceback
            traceback.print_exc()
            continue


def validate_outputs():
    """Validate that all expected outputs were created."""

    study_root = Path('/mnt/arborea/bpa-rat')
    template_dir = study_root / 'templates'

    print("\n" + "="*80)
    print("VALIDATING OUTPUTS")
    print("="*80)

    expected_files = {
        'p30': [
            'tpl-BPARat_p30_T2w.nii.gz',
            'tpl-BPARat_p30_label-GM_probseg.nii.gz',
            'tpl-BPARat_p30_label-WM_probseg.nii.gz',
            'tpl-BPARat_p30_label-CSF_probseg.nii.gz',
            'tpl-BPARat_p30_space-SIGMA_T2w.nii.gz',
            'transforms/tpl-to-SIGMA_Composite.h5'
        ],
        'p60': [
            'tpl-BPARat_p60_T2w.nii.gz',
            'tpl-BPARat_p60_label-GM_probseg.nii.gz',
            'tpl-BPARat_p60_label-WM_probseg.nii.gz',
            'tpl-BPARat_p60_label-CSF_probseg.nii.gz',
            'tpl-BPARat_p60_space-SIGMA_T2w.nii.gz',
            'transforms/tpl-to-SIGMA_Composite.h5'
        ],
        'p90': [
            'tpl-BPARat_p90_T2w.nii.gz',
            'tpl-BPARat_p90_label-GM_probseg.nii.gz',
            'tpl-BPARat_p90_label-WM_probseg.nii.gz',
            'tpl-BPARat_p90_label-CSF_probseg.nii.gz',
            'tpl-BPARat_p90_space-SIGMA_T2w.nii.gz',
            'transforms/tpl-to-SIGMA_Composite.h5'
        ]
    }

    all_good = True

    for cohort, files in expected_files.items():
        cohort_dir = template_dir / cohort
        print(f"\nCohort: {cohort}")

        for file in files:
            file_path = cohort_dir / file
            if file_path.exists():
                size = file_path.stat().st_size / (1024**2)  # MB
                print(f"  ✓ {file} ({size:.1f} MB)")
            else:
                print(f"  ❌ {file} (missing)")
                all_good = False

    if all_good:
        print("\n" + "="*80)
        print("✓ ALL TEMPLATES SUCCESSFULLY CREATED!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ Some templates are missing")
        print("="*80)

    return all_good


if __name__ == '__main__':
    print("="*80)
    print("TEMPLATE BUILDING TEST")
    print("="*80)
    print("\nThis test will:")
    print("  1. Preprocess 4 subjects from each cohort (p30, p60, p90)")
    print("  2. Build templates for each cohort (4 subjects each)")
    print("  3. Validate outputs")
    print("\nTest subjects:")
    for cohort, subjects in TEST_SUBJECTS.items():
        print(f"  {cohort}: {', '.join(subjects)}")
    print("\nNote: This uses a small number of subjects for testing.")
    print("For production, use 25% of best-quality subjects per cohort (~10-20 subjects).")
    print("="*80 + "\n")

    # Step 1: Preprocess subjects
    preprocess_test_subjects()

    # Step 2: Build templates
    build_test_templates()

    # Step 3: Validate
    success = validate_outputs()

    if success:
        print("\n✅ Template building test PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Template building test FAILED!")
        sys.exit(1)
