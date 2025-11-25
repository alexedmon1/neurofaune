#!/usr/bin/env python3
"""
Test script for MSME T2 mapping and MWF calculation workflow.

Usage:
    python test_msme_workflow.py <subject_dir>

Example:
    python test_msme_workflow.py /mnt/arborea/bpa-rat/raw/bids/sub-Rat207/ses-p60
"""

import sys
import numpy as np
from pathlib import Path
from neurofaune.config import load_config
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.preprocess.workflows.msme_preprocess import run_msme_preprocessing


def find_msme_file(subject_dir: Path):
    """Find MSME file in subject directory."""
    msme_dir = subject_dir / 'msme'

    if not msme_dir.exists():
        # Try alternate locations
        msme_dir = subject_dir / 'anat'

    if not msme_dir.exists():
        raise FileNotFoundError(f"MSME directory not found in {subject_dir}")

    # Find MSME file
    msme_files = (
        list(msme_dir.glob('*_msme.nii.gz')) +
        list(msme_dir.glob('*_msme.nii')) +
        list(msme_dir.glob('*MSME*.nii.gz')) +
        list(msme_dir.glob('*MSME*.nii'))
    )

    if not msme_files:
        raise FileNotFoundError(f"No MSME file found in {msme_dir}")

    msme_file = msme_files[0]

    return msme_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_msme_workflow.py <subject_dir>")
        print("\nExample:")
        print("  python test_msme_workflow.py /mnt/arborea/bpa-rat/raw/bids/sub-Rat207/ses-p60")
        print("\nOptional: specify number of echoes and TE values")
        print("  python test_msme_workflow.py <subject_dir> --echoes 32")
        sys.exit(1)

    subject_dir = Path(sys.argv[1])

    if not subject_dir.exists():
        print(f"Error: Subject directory not found: {subject_dir}")
        sys.exit(1)

    # Parse optional arguments
    n_echoes = 32  # Default
    te_start = 10  # Default
    te_step = 10   # Default

    if '--echoes' in sys.argv:
        idx = sys.argv.index('--echoes')
        n_echoes = int(sys.argv[idx + 1])

    # Extract subject and session from path
    session = subject_dir.name  # e.g., 'ses-p60'
    subject = subject_dir.parent.name  # e.g., 'sub-Rat207'

    print("="*80)
    print("MSME T2 Mapping Workflow Test")
    print("="*80)
    print(f"Subject: {subject}")
    print(f"Session: {session}")
    print(f"Directory: {subject_dir}")
    print("="*80)

    # Find MSME file
    try:
        msme_file = find_msme_file(subject_dir)
        print(f"\n✓ Found MSME file: {msme_file.name}")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

    # Load configuration
    config_file = Path('configs/default.yaml')
    if not config_file.exists():
        print(f"\n✗ Error: Config file not found: {config_file}")
        sys.exit(1)

    print(f"✓ Loading config: {config_file}")
    config = load_config(config_file)

    # Set study root
    study_root = subject_dir.parent.parent.parent
    output_dir = study_root

    print(f"✓ Study root: {study_root}")

    # Create transform registry
    cohort = session.split('-')[1][:3]  # Extract 'p60' from 'ses-p60'
    registry = create_transform_registry(config, subject, cohort=cohort)
    print(f"✓ Transform registry created for cohort: {cohort}")

    # Create TE values
    te_values = np.arange(te_start, te_start + n_echoes * te_step, te_step)
    print(f"\n✓ TE values: {n_echoes} echoes from {te_values[0]:.0f} to {te_values[-1]:.0f} ms")

    # Run preprocessing
    print("\n" + "="*80)
    print("Starting MSME preprocessing...")
    print("="*80 + "\n")

    try:
        results = run_msme_preprocessing(
            config=config,
            subject=subject,
            session=session,
            msme_file=msme_file,
            output_dir=output_dir,
            transform_registry=registry,
            te_values=te_values
        )

        print("\n" + "="*80)
        print("✓ MSME Preprocessing Complete!")
        print("="*80)
        print("\nOutput files:")
        print(f"  - MWF map: {results['mwf']}")
        print(f"  - IWF map: {results['iwf']}")
        print(f"  - CSF fraction: {results['csf']}")
        print(f"  - T2 map: {results['t2']}")
        print(f"  - Brain mask: {results['brain_mask']}")

        print("\nQC Report:")
        print(f"  - MSME QC: {results['qc_results']['html_report']}")

        # Print MWF summary
        import nibabel as nib
        mwf_img = nib.load(results['mwf'])
        mask_img = nib.load(results['brain_mask'])

        mwf_data = mwf_img.get_fdata()
        mask_data = mask_img.get_fdata() > 0

        mwf_masked = mwf_data[mask_data]
        print("\nMWF Summary:")
        print(f"  Mean: {np.mean(mwf_masked):.3f}")
        print(f"  Std: {np.std(mwf_masked):.3f}")
        print(f"  Range: [{np.min(mwf_masked):.3f}, {np.max(mwf_masked):.3f}]")

        print("\n✓ Test completed successfully!")
        return 0

    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
