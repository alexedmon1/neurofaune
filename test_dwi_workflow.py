#!/usr/bin/env python3
"""
Test script for DWI/DTI preprocessing workflow.

Usage:
    python test_dwi_workflow.py <subject_dir>

Example:
    python test_dwi_workflow.py /mnt/arborea/bpa-rat/raw/bids/sub-Rat207/ses-p60
"""

import sys
from pathlib import Path
from neurofaune.config import load_config
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.preprocess.workflows.dwi_preprocess import run_dwi_preprocessing


def find_dwi_files(subject_dir: Path):
    """Find DWI, bval, and bvec files in subject directory."""
    dwi_dir = subject_dir / 'dwi'

    if not dwi_dir.exists():
        raise FileNotFoundError(f"DWI directory not found: {dwi_dir}")

    # Find DWI file
    dwi_files = list(dwi_dir.glob('*_dwi.nii.gz')) + list(dwi_dir.glob('*_dwi.nii'))
    if not dwi_files:
        raise FileNotFoundError(f"No DWI file found in {dwi_dir}")
    dwi_file = dwi_files[0]

    # Find bval file
    bval_files = list(dwi_dir.glob('*.bval'))
    if not bval_files:
        raise FileNotFoundError(f"No bval file found in {dwi_dir}")
    bval_file = bval_files[0]

    # Find bvec file
    bvec_files = list(dwi_dir.glob('*.bvec'))
    if not bvec_files:
        raise FileNotFoundError(f"No bvec file found in {dwi_dir}")
    bvec_file = bvec_files[0]

    return dwi_file, bval_file, bvec_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_dwi_workflow.py <subject_dir>")
        print("\nExample:")
        print("  python test_dwi_workflow.py /mnt/arborea/bpa-rat/raw/bids/sub-Rat207/ses-p60")
        sys.exit(1)

    subject_dir = Path(sys.argv[1])

    if not subject_dir.exists():
        print(f"Error: Subject directory not found: {subject_dir}")
        sys.exit(1)

    # Extract subject and session from path
    session = subject_dir.name  # e.g., 'ses-p60'
    subject = subject_dir.parent.name  # e.g., 'sub-Rat207'

    print("="*80)
    print("DWI/DTI Workflow Test")
    print("="*80)
    print(f"Subject: {subject}")
    print(f"Session: {session}")
    print(f"Directory: {subject_dir}")
    print("="*80)

    # Find DWI files
    try:
        dwi_file, bval_file, bvec_file = find_dwi_files(subject_dir)
        print(f"\n✓ Found DWI files:")
        print(f"  DWI: {dwi_file.name}")
        print(f"  BVAL: {bval_file.name}")
        print(f"  BVEC: {bvec_file.name}")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

    # Load configuration
    config_file = Path('configs/default.yaml')
    if not config_file.exists():
        print(f"\n✗ Error: Config file not found: {config_file}")
        sys.exit(1)

    print(f"\n✓ Loading config: {config_file}")
    config = load_config(config_file)

    # Set study root (parent of raw/bids)
    study_root = subject_dir.parent.parent.parent  # Go up from ses-p60 -> sub-Rat -> bids -> raw
    output_dir = study_root

    print(f"✓ Study root: {study_root}")

    # Create transform registry
    cohort = session.split('-')[1][:3]  # Extract 'p60' from 'ses-p60'
    registry = create_transform_registry(config, subject, cohort=cohort)
    print(f"✓ Transform registry created for cohort: {cohort}")

    # Check for GPU
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        use_gpu = result.returncode == 0
        print(f"\n✓ GPU available: {use_gpu}")
    except:
        use_gpu = False
        print("\n⚠ GPU not available, using CPU")

    # Run preprocessing
    print("\n" + "="*80)
    print("Starting DWI preprocessing...")
    print("="*80 + "\n")

    try:
        results = run_dwi_preprocessing(
            config=config,
            subject=subject,
            session=session,
            dwi_file=dwi_file,
            bval_file=bval_file,
            bvec_file=bvec_file,
            output_dir=output_dir,
            transform_registry=registry,
            use_gpu=use_gpu
        )

        print("\n" + "="*80)
        print("✓ DWI Preprocessing Complete!")
        print("="*80)
        print("\nOutput files:")
        print(f"  - Preprocessed DWI: {results['dwi_preproc']}")
        print(f"  - Brain mask: {results['dwi_mask']}")
        print(f"  - FA map: {results['fa']}")
        print(f"  - MD map: {results['md']}")
        print(f"  - AD map: {results['ad']}")
        print(f"  - RD map: {results['rd']}")

        print("\nQC Reports:")
        if 'eddy_qc' in results['qc_results']:
            print(f"  - Eddy/Motion QC: {results['qc_results']['eddy_qc']['html_report']}")
        print(f"  - DTI Metrics QC: {results['qc_results']['dti_qc']['html_report']}")

        print("\n✓ Test completed successfully!")
        return 0

    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
