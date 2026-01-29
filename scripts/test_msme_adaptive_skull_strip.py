#!/usr/bin/env python3
"""
Test adaptive slice-wise skull stripping for MSME data.

Tests the new adaptive per-slice BET approach on MSME data,
which has only 5 thick coronal slices (160x160x5 at 2.0x2.0x8.0mm).
"""

from pathlib import Path
import nibabel as nib
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.preprocess.workflows.msme_preprocess import run_msme_preprocessing
from neurofaune.utils.transforms import create_transform_registry


def find_msme_file(bids_root: Path, subject: str, session: str) -> Path:
    """Find MSME file for a subject/session."""
    msme_dir = bids_root / subject / session / 'msme'
    if not msme_dir.exists():
        raise FileNotFoundError(f"MSME directory not found: {msme_dir}")

    msme_files = list(msme_dir.glob('*MSME*.nii.gz'))
    if not msme_files:
        raise FileNotFoundError(f"No MSME files found in {msme_dir}")

    return msme_files[0]


def find_t2w_file(derivatives_root: Path, subject: str, session: str) -> Path:
    """Find preprocessed T2w file for a subject/session."""
    anat_dir = derivatives_root / subject / session / 'anat'
    if not anat_dir.exists():
        raise FileNotFoundError(f"Anatomical directory not found: {anat_dir}")

    t2w_files = list(anat_dir.glob('*_T2w_brain.nii.gz'))
    if not t2w_files:
        # Try alternative naming
        t2w_files = list(anat_dir.glob('*T2w*.nii.gz'))

    if not t2w_files:
        raise FileNotFoundError(f"No T2w files found in {anat_dir}")

    return t2w_files[0]


def main():
    # Configuration
    study_root = Path('/mnt/arborea/bpa-rat')
    config_file = Path(__file__).parent.parent / 'configs' / 'bpa_rat_example.yaml'

    # Test subject (previously tested with Atropos approach)
    subject = 'sub-Rat110'
    session = 'ses-p90'

    print("="*80)
    print("MSME Adaptive Skull Stripping Test")
    print("="*80)
    print(f"Subject: {subject}")
    print(f"Session: {session}")
    print(f"Config: {config_file}")

    # Load config
    if config_file.exists():
        config = load_config(config_file)
    else:
        print(f"Config not found, using default")
        config = load_config(Path(__file__).parent.parent / 'configs' / 'default.yaml')

    # Find MSME file
    bids_root = study_root / 'raw' / 'bids'
    try:
        msme_file = find_msme_file(bids_root, subject, session)
        print(f"\nMSME file: {msme_file}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Searching for MSME files...")
        for msme in bids_root.rglob('*MSME*.nii.gz'):
            print(f"  Found: {msme}")
        return 1

    # Find T2w file for registration
    derivatives_root = study_root / 'derivatives'
    try:
        t2w_file = find_t2w_file(derivatives_root, subject, session)
        print(f"T2w file: {t2w_file}")
    except FileNotFoundError as e:
        print(f"\nWarning: {e}")
        t2w_file = None
        print("Will skip registration step")

    # Show MSME geometry
    msme_img = nib.load(msme_file)
    print(f"\nMSME geometry:")
    print(f"  Shape: {msme_img.shape}")
    print(f"  Voxel size: {msme_img.header.get_zooms()}")

    # Create transform registry
    registry = create_transform_registry(config, subject=subject, cohort='p90')

    # Work directory for this test
    work_dir = study_root / 'work' / subject / session / 'msme_adaptive_test'
    work_dir.mkdir(parents=True, exist_ok=True)

    # Run MSME preprocessing with adaptive skull stripping
    print("\n" + "="*80)
    print("Running MSME preprocessing with adaptive skull stripping...")
    print("="*80)

    results = run_msme_preprocessing(
        config=config,
        subject=subject,
        session=session,
        msme_file=msme_file,
        output_dir=study_root,
        transform_registry=registry,
        work_dir=work_dir,
        t2w_file=t2w_file,
        run_registration=t2w_file is not None
    )

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
    print("\nOutputs:")
    for key, value in results.items():
        if key != 'qc_results':
            print(f"  {key}: {value}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
