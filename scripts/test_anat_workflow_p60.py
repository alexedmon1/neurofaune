#!/usr/bin/env python3
"""
Test anatomical preprocessing workflow on P60 subject (Rat207).

This script tests:
1. Automatic T2w scan selection from multiple acquisitions
2. Voxel size scaling (if needed)
3. Bias field correction
4. Skull stripping with age-specific parameters
5. ANTs registration to SIGMA atlas
6. Transform registry integration
"""

from pathlib import Path
import sys

# Add neurofaune to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing


def main():
    print("="*80)
    print("Testing Anatomical Preprocessing Workflow - P60 Subject")
    print("="*80)
    print()

    # Configuration
    config_file = Path('/home/edm9fd/sandbox/neurofaune/configs/default.yaml')
    subject = 'sub-Rat207'
    session = 'ses-p60'
    subject_dir = Path('/mnt/arborea/bpa-rat/raw/bids/sub-Rat207')
    output_dir = Path('/mnt/arborea/bpa-rat/test')

    print(f"Subject: {subject}")
    print(f"Session: {session}")
    print(f"Subject directory: {subject_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load configuration
    print("Loading configuration...")
    config = load_config(config_file)
    print(f"  Atlas: {config['atlas']['name']} at {config['atlas']['base_path']}")
    print()

    # Create transform registry
    print("Creating transform registry...")
    cohort = session.split('-')[1]  # Extract 'p60' from 'ses-p60'
    registry = create_transform_registry(config, subject, cohort=cohort)
    print(f"  Registry location: {registry.subject_dir}")
    print()

    # Run preprocessing workflow with automatic scan selection
    print("Running anatomical preprocessing workflow...")
    print("  Using automatic T2w scan selection")
    print()

    try:
        results = run_anatomical_preprocessing(
            config=config,
            subject=subject,
            session=session,
            subject_dir=subject_dir,  # Use automatic selection
            output_dir=output_dir,
            transform_registry=registry,
            modality='anat',
            prefer_orientation='axial'
        )

        print()
        print("="*80)
        print("Workflow completed successfully!")
        print("="*80)
        print()
        print("Output files:")
        print(f"  Brain: {results['brain']}")
        print(f"  Mask: {results['mask']}")
        print(f"  Warped to SIGMA: {results['warped']}")
        print()
        print("Transform files:")
        print(f"  Composite: {results['composite_transform']}")
        print(f"  Inverse: {results['inverse_composite_transform']}")
        print()
        print(f"Voxel scaling applied: {results['was_scaled']}")
        if results['was_scaled']:
            print(f"  Scale factor: {results['scale_factor']}x")
        print()

    except Exception as e:
        print()
        print("="*80)
        print("ERROR: Workflow failed!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
