#!/usr/bin/env python3
"""
004_batch_preprocess_dwi.py

Batch DTI preprocessing to generate FA maps for template building.

This script:
1. Finds all DWI files in BIDS directory
2. Skips files with identity affine (header issues)
3. Runs DTI preprocessing (5D→4D, eddy, dtifit)
4. Outputs FA, MD, AD, RD maps

Prerequisites:
- DWI files must have correct voxel sizes in headers
- GPU recommended for eddy (set --no-gpu if unavailable)

Usage:
    python 004_batch_preprocess_dwi.py /path/to/bpa-rat --max-subjects 10
    python 004_batch_preprocess_dwi.py /path/to/bpa-rat --subjects sub-Rat1 sub-Rat102

Output:
    {derivatives}/{subject}/{session}/dwi/
        - {subject}_{session}_desc-preproc_dwi.nii.gz
        - {subject}_{session}_FA.nii.gz
        - {subject}_{session}_MD.nii.gz
        - etc.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import nibabel as nib
import numpy as np


def find_dwi_files(bids_root: Path, subjects: Optional[List[str]] = None) -> List[Tuple[str, str, Path]]:
    """
    Find all DWI files in BIDS directory.

    Returns list of (subject, session, dwi_path) tuples.
    """
    dwi_files = []

    pattern = 'sub-*/ses-*/dwi/*_dwi.nii.gz'
    for dwi_path in sorted(bids_root.glob(pattern)):
        parts = dwi_path.parts
        subject = [p for p in parts if p.startswith('sub-')][0]
        session = [p for p in parts if p.startswith('ses-')][0]

        if subjects and subject not in subjects:
            continue

        dwi_files.append((subject, session, dwi_path))

    return dwi_files


def check_dwi_geometry(dwi_path: Path) -> dict:
    """
    Check DWI geometry for issues.

    Returns dict with:
    - valid: bool
    - issues: list of issues found
    - voxel_size: tuple
    - shape: tuple
    """
    img = nib.load(dwi_path)
    affine = img.affine
    shape = img.shape

    result = {
        'valid': True,
        'issues': [],
        'voxel_size': tuple(round(v, 3) for v in img.header.get_zooms()[:3]),
        'shape': shape[:3] if len(shape) >= 3 else shape,
        'ndim': len(shape)
    }

    # Check for identity affine
    if np.allclose(affine, np.eye(4)):
        result['valid'] = False
        result['issues'].append('Identity affine (incorrect voxel sizes)')

    # Check for reasonable voxel sizes
    zooms = img.header.get_zooms()[:3]
    if any(z < 0.1 for z in zooms):
        result['issues'].append(f'Very small voxel size: {zooms}')
    if any(z > 20 for z in zooms):
        result['issues'].append(f'Very large voxel size: {zooms}')

    # Check for expected dimensions (5D for Bruker multi-average)
    if len(shape) == 5:
        result['is_5d'] = True
        result['n_directions'] = shape[3]
        result['n_averages'] = shape[4]
    elif len(shape) == 4:
        result['is_5d'] = False
        result['n_volumes'] = shape[3]
    else:
        result['issues'].append(f'Unexpected dimensions: {len(shape)}D')

    return result


def get_bval_bvec_paths(dwi_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Get bval and bvec file paths for a DWI file."""
    base = dwi_path.with_suffix('').with_suffix('')  # Remove .nii.gz
    bval = base.with_suffix('.bval')
    bvec = base.with_suffix('.bvec')

    return (
        bval if bval.exists() else None,
        bvec if bvec.exists() else None
    )


def preprocess_single_subject(
    study_root: Path,
    subject: str,
    session: str,
    dwi_path: Path,
    use_gpu: bool = True,
    force: bool = False
) -> dict:
    """
    Run DTI preprocessing for a single subject.

    Returns dict with status and output paths.
    """
    from neurofaune.config import load_config
    from neurofaune.utils.transforms import TransformRegistry
    from neurofaune.preprocess.workflows.dwi_preprocess import run_dwi_preprocessing

    derivatives_dir = study_root / 'derivatives' / subject / session / 'dwi'
    fa_output = derivatives_dir / f'{subject}_{session}_FA.nii.gz'

    # Check if already processed
    if fa_output.exists() and not force:
        return {
            'status': 'skipped',
            'reason': 'FA already exists',
            'fa_path': fa_output
        }

    # Get bval/bvec
    bval_path, bvec_path = get_bval_bvec_paths(dwi_path)
    if not bval_path or not bvec_path:
        return {
            'status': 'error',
            'reason': f'Missing bval/bvec files'
        }

    # Create minimal config
    config = {
        'study': {'name': 'BPA-Rat', 'code': 'BPARat'},
        'paths': {
            'study_root': str(study_root),
            'derivatives': str(study_root / 'derivatives'),
            'transforms': str(study_root / 'transforms'),
            'qc': str(study_root / 'qc'),
        },
        'diffusion': {
            'eddy': {
                'use_cuda': use_gpu,
                'repol': True,
            }
        },
        'execution': {
            'n_procs': 4
        }
    }

    # Create transform registry
    cohort = session.replace('ses-', '')
    registry = TransformRegistry(
        transforms_dir=study_root / 'transforms' / subject,
        subject=subject,
        cohort=cohort
    )

    try:
        results = run_dwi_preprocessing(
            config=config,
            subject=subject,
            session=session,
            dwi_file=dwi_path,
            bval_file=bval_path,
            bvec_file=bvec_path,
            output_dir=study_root,
            transform_registry=registry,
            use_gpu=use_gpu
        )

        return {
            'status': 'success',
            'fa_path': results.get('fa'),
            'md_path': results.get('md'),
        }

    except Exception as e:
        return {
            'status': 'error',
            'reason': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Batch DTI preprocessing'
    )
    parser.add_argument('study_root', type=Path, help='Path to study root')
    parser.add_argument('--subjects', nargs='+', help='Specific subjects to process')
    parser.add_argument('--max-subjects', type=int, help='Maximum number of subjects')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--force', action='store_true', help='Reprocess even if output exists')
    parser.add_argument('--dry-run', action='store_true', help='Only check files, do not process')
    args = parser.parse_args()

    bids_root = args.study_root / 'raw' / 'bids'

    if not bids_root.exists():
        print(f"ERROR: BIDS root not found: {bids_root}")
        sys.exit(1)

    # Normalize subject IDs
    subjects = None
    if args.subjects:
        subjects = [s if s.startswith('sub-') else f'sub-{s}' for s in args.subjects]

    # Find DWI files
    print("=" * 70)
    print("Batch DTI Preprocessing")
    print("=" * 70)

    dwi_files = find_dwi_files(bids_root, subjects)
    print(f"Found {len(dwi_files)} DWI files")

    if args.max_subjects:
        dwi_files = dwi_files[:args.max_subjects]
        print(f"Processing first {args.max_subjects}")

    # Check geometry and filter
    valid_files = []
    invalid_files = []

    print("\nChecking file geometry...")
    for subject, session, dwi_path in dwi_files:
        geom = check_dwi_geometry(dwi_path)

        if geom['valid']:
            valid_files.append((subject, session, dwi_path, geom))
        else:
            invalid_files.append((subject, session, dwi_path, geom))

    print(f"\n  Valid files: {len(valid_files)}")
    print(f"  Invalid files: {len(invalid_files)}")

    if invalid_files:
        print("\n  Invalid files (skipped):")
        for subj, sess, path, geom in invalid_files[:5]:
            print(f"    {subj}/{sess}: {', '.join(geom['issues'])}")
        if len(invalid_files) > 5:
            print(f"    ... and {len(invalid_files) - 5} more")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for subj, sess, path, geom in valid_files:
            print(f"  {subj}/{sess}: {geom['shape']}, {geom['voxel_size']} mm")
            if geom.get('is_5d'):
                print(f"    5D: {geom['n_directions']} directions × {geom['n_averages']} averages")
        return

    # Process valid files
    print("\n" + "=" * 70)
    print("Processing...")
    print("=" * 70)

    results = {'success': 0, 'skipped': 0, 'error': 0}

    for i, (subject, session, dwi_path, geom) in enumerate(valid_files):
        print(f"\n[{i+1}/{len(valid_files)}] {subject} / {session}")
        print(f"  Shape: {geom['shape']}, Voxels: {geom['voxel_size']} mm")

        result = preprocess_single_subject(
            study_root=args.study_root,
            subject=subject,
            session=session,
            dwi_path=dwi_path,
            use_gpu=not args.no_gpu,
            force=args.force
        )

        results[result['status']] += 1

        if result['status'] == 'success':
            print(f"  ✓ FA: {result['fa_path']}")
        elif result['status'] == 'skipped':
            print(f"  - Skipped: {result['reason']}")
        else:
            print(f"  ✗ Error: {result['reason']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Success: {results['success']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Errors:  {results['error']}")


if __name__ == '__main__':
    main()
