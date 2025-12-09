#!/usr/bin/env python3
"""
Test complete functional preprocessing workflow with registration.

This script tests the full functional fMRI preprocessing pipeline including:
- Motion correction
- Brain extraction
- Smoothing
- Temporal filtering
- ICA denoising
- Registration to T2w anatomical
- Registration to SIGMA atlas
- Comprehensive QC
"""

import sys
from pathlib import Path

# Add neurofaune to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.preprocess.workflows.func_preprocess import run_functional_preprocessing


def main():
    print("="*80)
    print("TESTING COMPLETE FUNCTIONAL PREPROCESSING WORKFLOW WITH REGISTRATION")
    print("="*80)

    # Configuration
    config_file = Path('/home/edm9fd/sandbox/neurofaune/configs/default.yaml')
    config = load_config(config_file)

    # Test data paths
    subject = 'sub-Rat49'
    session = 'ses-p90'

    # Find functional scan
    bids_dir = Path('/mnt/arborea/bpa-rat/raw/bids')
    func_dir = bids_dir / subject / session / 'func'

    # Find a suitable run
    bold_files = sorted(func_dir.glob(f'{subject}_{session}_run-*_bold.nii.gz'))
    if not bold_files:
        print(f"ERROR: No BOLD files found in {func_dir}")
        return 1

    bold_file = bold_files[0]  # Use first run
    print(f"\nTest data:")
    print(f"  Subject: {subject}")
    print(f"  Session: {session}")
    print(f"  BOLD: {bold_file}")

    # Get T2w anatomical reference
    anat_dir = bids_dir / subject / session / 'anat'
    t2w_files = sorted(anat_dir.glob(f'{subject}_{session}_*T2w.nii.gz'))

    # Filter out localizers
    t2w_files = [f for f in t2w_files if 'Localizer' not in f.name]

    if not t2w_files:
        print(f"ERROR: No T2w anatomical files found in {anat_dir}")
        return 1

    t2w_file = t2w_files[0]  # Use first anatomical scan
    print(f"  T2w: {t2w_file}")

    # Output directory
    output_dir = Path('/mnt/arborea/bpa-rat/test/func_test_complete')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {output_dir}")

    # Create transform registry
    print("\nCreating transform registry...")
    registry = create_transform_registry(
        config=config,
        subject=subject,
        session=session,
        study_root=output_dir
    )

    # Check if T2w → SIGMA transform exists
    if not registry.has_transform('T2w', 'SIGMA'):
        print("\nWARNING: T2w → SIGMA transform not found in registry!")
        print("The workflow will register BOLD → T2w, but skip SIGMA registration.")
        print("To enable SIGMA registration, run anatomical preprocessing first:")
        print(f"  python scripts/batch_preprocess.py --subject {subject} --session {session} --modality anat")
        print("\nProceeding with BOLD → T2w registration only...")

    # Run workflow
    print("\n" + "="*80)
    print("STARTING FUNCTIONAL PREPROCESSING")
    print("="*80)

    try:
        results = run_functional_preprocessing(
            config=config,
            subject=subject,
            session=session,
            bold_file=bold_file,
            output_dir=output_dir,
            transform_registry=registry,
            t2w_file=t2w_file,  # Enable registration
            n_discard=5  # Discard first 5 volumes
        )

        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)

        print("\nOutput files:")
        print(f"  Preprocessed BOLD: {results['bold_preproc']}")
        print(f"  Brain mask: {results['brain_mask']}")
        print(f"  Confounds: {results['confounds']}")
        print(f"  Metadata: {results['metadata']}")

        if 'registration' in results:
            print(f"\nRegistration outputs:")
            print(f"  BOLD in T2w space: {results['registration']['bold_in_t2w']}")
            if 'normalization' in results:
                print(f"  BOLD in SIGMA space: {results['normalization']['bold_in_sigma']}")

        if 'ica_denoising' in results:
            print(f"\nICA denoising:")
            ica_results = results['ica_denoising']
            print(f"  Denoised BOLD: {ica_results['denoised_bold']}")
            print(f"  Signal components: {ica_results['classification']['summary']['n_signal']}")
            print(f"  Noise components: {ica_results['classification']['summary']['n_noise']}")

        print(f"\nQC reports:")
        for qc_type, qc_file in results['qc_reports'].items():
            print(f"  {qc_type}: {qc_file}")

        return 0

    except Exception as e:
        print(f"\n{'='*80}")
        print("TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
