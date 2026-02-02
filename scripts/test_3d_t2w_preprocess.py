#!/usr/bin/env python3
"""
Test preprocessing on a single 3D T2w subject.

Usage:
    uv run python scripts/test_3d_t2w_preprocess.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing


def main():
    # Configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'bpa_rat_example.yaml'
    bids_dir = Path('/mnt/arborea/bpa-rat/raw/bids')
    output_dir = Path('/mnt/arborea/bpa-rat')

    # Test subject with 3D T2w
    subject = 'sub-Rat8'
    session = 'ses-p60'

    print("="*60)
    print(f"Testing 3D T2w preprocessing: {subject} {session}")
    print("="*60)

    # Load config
    config = load_config(config_path)

    # Check input file
    subject_dir = bids_dir / subject
    anat_dir = subject_dir / session / 'anat'
    t2w_files = list(anat_dir.glob('*T2w.nii.gz'))

    print(f"\nInput T2w files found: {len(t2w_files)}")
    for f in t2w_files:
        import nibabel as nib
        img = nib.load(f)
        print(f"  {f.name}: shape={img.shape}, voxel={img.header.get_zooms()[:3]}")

    # Create transform registry
    cohort = session.replace('ses-', '')
    registry = create_transform_registry(config, subject, cohort=cohort)

    print(f"\nRunning preprocessing...")
    print("-"*60)

    try:
        results = run_anatomical_preprocessing(
            config=config,
            subject=subject,
            session=session,
            output_dir=output_dir,
            transform_registry=registry,
            subject_dir=subject_dir
        )

        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"\nOutputs:")
        for key, value in results.items():
            if isinstance(value, Path) and value.exists():
                print(f"  {key}: {value}")
                if value.suffix == '.gz':
                    img = nib.load(value)
                    print(f"       shape={img.shape}, voxel={img.header.get_zooms()[:3]}")

    except Exception as e:
        print("\n" + "="*60)
        print("FAILED!")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
