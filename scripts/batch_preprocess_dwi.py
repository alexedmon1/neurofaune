#!/usr/bin/env python3
"""
Batch DTI/DWI preprocessing script.

Processes all DWI scans in the BPA-Rat BIDS dataset with:
- Header validation (checks for identity affine issues)
- Optional automatic header fixing using Bruker source data
- GPU-accelerated eddy correction
- DTI fitting (FA, MD, AD, RD)
- Comprehensive QC
- FA to cohort template registration
- SIGMA atlas warping (FA, MD, AD, RD → SIGMA space)

Usage:
    # Dry run to see what would be processed
    python batch_preprocess_dwi.py --dry-run

    # Process all subjects
    python batch_preprocess_dwi.py

    # Process specific subjects
    python batch_preprocess_dwi.py --subjects sub-Rat1 sub-Rat102

    # Fix headers before processing (requires Bruker data)
    python batch_preprocess_dwi.py --fix-headers --bruker-root /mnt/arborea/bruker

    # Force reprocessing
    python batch_preprocess_dwi.py --force

    # Skip FA to T2w registration
    python batch_preprocess_dwi.py --skip-registration
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import traceback

import nibabel as nib
import numpy as np

# Add neurofaune to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.preprocess.workflows.dwi_preprocess import run_dwi_preprocessing
from neurofaune.utils.transforms import TransformRegistry
from neurofaune.utils.select_anatomical import is_3d_only_subject


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


def fix_headers_if_needed(
    bids_root: Path,
    bruker_root: Path,
    dwi_files: List[Tuple[str, str, Path]]
) -> int:
    """
    Fix headers for DWI files with identity affine.

    Returns number of files fixed.
    """
    from neurofaune.utils.fix_bruker_voxel_sizes import (
        find_bruker_scan_dir,
        parse_bruker_method,
        update_nifti_header,
        update_json_sidecar
    )
    import re

    fixed_count = 0

    for subject, session, dwi_path in dwi_files:
        img = nib.load(dwi_path)
        if not np.allclose(img.affine, np.eye(4)):
            continue  # Header is OK

        # Try to fix
        json_file = dwi_path.with_suffix('').with_suffix('.json')
        if not json_file.exists():
            print(f"  ⚠ No JSON sidecar for {dwi_path.name}, cannot fix")
            continue

        with open(json_file, 'r') as f:
            metadata = json.load(f)

        scan_name_full = metadata.get('ScanName', '')
        match = re.search(r'\(E(\d+)\)', scan_name_full)
        if not match:
            print(f"  ⚠ Cannot parse scan number from {scan_name_full}")
            continue

        scan_name = match.group(1)
        subject_id = subject.replace('sub-', '')
        session_id = session.replace('ses-', '')

        bruker_scan_dir = find_bruker_scan_dir(
            bruker_root, subject_id, session_id, scan_name
        )

        if not bruker_scan_dir:
            print(f"  ⚠ No Bruker data for {subject}/{session} scan {scan_name}")
            continue

        method_file = bruker_scan_dir / 'method'
        try:
            params = parse_bruker_method(method_file)
            if 'voxel_size' in params:
                update_nifti_header(dwi_path, params['voxel_size'])
                update_json_sidecar(json_file, params['voxel_size'])
                fixed_count += 1
                print(f"  ✓ Fixed {subject}/{session}")
        except Exception as e:
            print(f"  ✗ Error fixing {subject}/{session}: {e}")

    return fixed_count


def find_template_file(study_root: Path, session: str) -> Optional[Path]:
    """
    Find cohort template for FA→Template registration.

    Looks for the age-matched template at:
        {study_root}/templates/anat/{cohort}/tpl-BPARat_{cohort}_T2w.nii.gz

    Parameters
    ----------
    study_root : Path
        Study root directory
    session : str
        Session ID (e.g. 'ses-p60')

    Returns
    -------
    Path or None
        Path to cohort template if it exists
    """
    cohort = session.replace('ses-', '')
    template_path = study_root / 'templates' / 'anat' / cohort / f'tpl-BPARat_{cohort}_T2w.nii.gz'
    return template_path if template_path.exists() else None


def preprocess_single_subject(
    study_root: Path,
    subject: str,
    session: str,
    dwi_path: Path,
    config_path: Path,
    use_gpu: bool = True,
    force: bool = False,
    skip_registration: bool = False
) -> dict:
    """
    Run DTI preprocessing for a single subject.

    Returns dict with status and output paths.
    """
    derivatives_dir = study_root / 'derivatives' / subject / session / 'dwi'
    fa_output = derivatives_dir / f'{subject}_{session}_FA.nii.gz'

    # Check if already processed
    if fa_output.exists() and not force:
        return {
            'status': 'skipped',
            'reason': 'FA already exists',
            'fa_path': str(fa_output)
        }

    # Get bval/bvec
    bval_path, bvec_path = get_bval_bvec_paths(dwi_path)
    if not bval_path or not bvec_path:
        return {
            'status': 'error',
            'reason': 'Missing bval/bvec files'
        }

    # Load config
    config = load_config(config_path)

    # Create transform registry
    cohort = session.replace('ses-', '')
    registry = TransformRegistry(
        transforms_dir=study_root / 'transforms' / subject,
        subject=subject,
        cohort=cohort
    )

    # Find cohort template for FA→Template registration (+ SIGMA warping)
    template_file = None
    run_registration = not skip_registration
    if run_registration:
        template_file = find_template_file(study_root, session)
        if template_file is None:
            print(f"  No cohort template found for {subject}/{session}, registration will be skipped")

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
            use_gpu=use_gpu,
            template_file=template_file,
            run_registration=run_registration
        )

        result = {
            'status': 'success',
            'fa_path': str(results.get('fa', '')),
            'md_path': str(results.get('md', '')),
            'qc_metrics': results.get('qc_metrics', {})
        }

        if 'registration' in results:
            result['registration'] = str(results['registration'].get('affine_transform', ''))

        if results.get('sigma_fa'):
            result['sigma_fa'] = str(results['sigma_fa'])

        return result

    except Exception as e:
        return {
            'status': 'error',
            'reason': str(e),
            'traceback': traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(
        description='Batch DTI/DWI preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--bids-root',
        type=Path,
        default=Path('/mnt/arborea/bpa-rat/raw/bids'),
        help='BIDS root directory'
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('/mnt/arborea/bpa-rat'),
        help='Output root directory (study root)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('/home/edm9fd/sandbox/neurofaune/configs/default.yaml'),
        help='Configuration file'
    )
    parser.add_argument(
        '--subjects',
        nargs='+',
        help='Specific subjects to process (e.g., sub-Rat1 sub-Rat102)'
    )
    parser.add_argument(
        '--max-subjects',
        type=int,
        help='Maximum number of subjects to process'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration for eddy'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if output exists'
    )
    parser.add_argument(
        '--fix-headers',
        action='store_true',
        help='Attempt to fix broken headers using Bruker source data'
    )
    parser.add_argument(
        '--bruker-root',
        type=Path,
        default=Path('/mnt/arborea/bruker'),
        help='Bruker data root (required if --fix-headers)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only check files, do not process'
    )
    parser.add_argument(
        '--skip-registration',
        action='store_true',
        help='Skip FA to template registration and SIGMA warping'
    )
    parser.add_argument(
        '--exclude-3d',
        action='store_true',
        help='Exclude subjects that only have 3D T2w scans (no 2D available)'
    )

    args = parser.parse_args()

    # Normalize subject IDs
    subjects = None
    if args.subjects:
        subjects = [s if s.startswith('sub-') else f'sub-{s}' for s in args.subjects]

    # Find DWI files
    print("=" * 70)
    print("Batch DTI/DWI Preprocessing")
    print("=" * 70)
    print(f"BIDS root: {args.bids_root}")
    print(f"Output root: {args.output_root}")
    print(f"Config: {args.config}")
    print(f"GPU: {'disabled' if args.no_gpu else 'enabled'}")
    print()

    dwi_files = find_dwi_files(args.bids_root, subjects)
    print(f"Found {len(dwi_files)} DWI files")

    # Exclude 3D-only subjects if requested
    if args.exclude_3d:
        before = len(dwi_files)
        dwi_files = [
            (subj, sess, path) for subj, sess, path in dwi_files
            if not is_3d_only_subject(args.bids_root / subj, sess)
        ]
        n_excluded = before - len(dwi_files)
        if n_excluded > 0:
            print(f"Excluding {n_excluded} DWI files from 3D-only subjects (--exclude-3d)")

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

    # Attempt to fix headers if requested
    if invalid_files and args.fix_headers:
        print(f"\nAttempting to fix {len(invalid_files)} files with broken headers...")
        fixed = fix_headers_if_needed(
            args.bids_root,
            args.bruker_root,
            [(s, sess, p) for s, sess, p, _ in invalid_files]
        )
        print(f"Fixed {fixed} files")

        # Re-check geometry
        if fixed > 0:
            print("\nRe-checking geometry...")
            valid_files = []
            invalid_files = []
            for subject, session, dwi_path in dwi_files:
                geom = check_dwi_geometry(dwi_path)
                if geom['valid']:
                    valid_files.append((subject, session, dwi_path, geom))
                else:
                    invalid_files.append((subject, session, dwi_path, geom))
            print(f"  Valid files: {len(valid_files)}")
            print(f"  Invalid files: {len(invalid_files)}")

    if invalid_files:
        print("\n  Invalid files (will be skipped):")
        for subj, sess, path, geom in invalid_files[:10]:
            print(f"    {subj}/{sess}: {', '.join(geom['issues'])}")
        if len(invalid_files) > 10:
            print(f"    ... and {len(invalid_files) - 10} more")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for subj, sess, path, geom in valid_files[:20]:
            print(f"  {subj}/{sess}: {geom['shape']}, {geom['voxel_size']} mm")
        if len(valid_files) > 20:
            print(f"  ... and {len(valid_files) - 20} more")
        return 0

    if not valid_files:
        print("\nNo valid files to process!")
        return 1

    # Process valid files
    print("\n" + "=" * 70)
    print(f"Processing {len(valid_files)} files...")
    print("=" * 70)

    # Create log directory
    log_dir = Path('/tmp/dwi_batch_preprocessing')
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = log_dir / f'batch_results_{timestamp}.json'

    results = {'success': 0, 'skipped': 0, 'error': 0, 'details': []}

    for i, (subject, session, dwi_path, geom) in enumerate(valid_files):
        print(f"\n[{i+1}/{len(valid_files)}] {subject} / {session}")
        print(f"  Shape: {geom['shape']}, Voxels: {geom['voxel_size']} mm")

        result = preprocess_single_subject(
            study_root=args.output_root,
            subject=subject,
            session=session,
            dwi_path=dwi_path,
            config_path=args.config,
            use_gpu=not args.no_gpu,
            force=args.force,
            skip_registration=args.skip_registration
        )

        result['subject'] = subject
        result['session'] = session
        results['details'].append(result)
        results[result['status']] += 1

        if result['status'] == 'success':
            print(f"  ✓ FA: {result['fa_path']}")
            if result.get('registration'):
                print(f"  ✓ Registration: {result['registration']}")
            if result.get('sigma_fa'):
                print(f"  ✓ SIGMA FA: {result['sigma_fa']}")
        elif result['status'] == 'skipped':
            print(f"  → Skipped: {result['reason']}")
        else:
            print(f"  ✗ Error: {result['reason']}")

        # Save intermediate results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Success: {results['success']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Errors:  {results['error']}")
    print(f"\nResults saved to: {results_file}")

    if results['error'] > 0:
        print("\nFailed subjects:")
        for r in results['details']:
            if r['status'] == 'error':
                print(f"  {r['subject']}/{r['session']}: {r['reason']}")

    return 0 if results['error'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
