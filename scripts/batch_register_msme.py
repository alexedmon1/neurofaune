#!/usr/bin/env python3
"""
Batch MSME registration to SIGMA space via direct MSME→Template pipeline.

This script registers all preprocessed MSME data through the chain:
    Subject MSME (first echo) → Cohort Template (rigid) → SIGMA (pre-computed SyN)

For each subject with completed MSME preprocessing:
1. Prepare skull-stripped first echo as registration reference
2. MSME → Template registration (rigid, NCC Z-init via register_msme_to_template)
3. Warp MWF, IWF, CSFF, T2 maps to SIGMA space

Prerequisites:
- MSME preprocessing complete (MWF/IWF/CSFF/T2 + brain mask exist)
- Cohort templates built with SIGMA registration

Usage:
    # Register all subjects
    uv run python scripts/batch_register_msme.py \
        --study-root /mnt/arborea/bpa-rat \
        --n-cores 4

    # Dry run to see what would be processed
    uv run python scripts/batch_register_msme.py \
        --study-root /mnt/arborea/bpa-rat \
        --dry-run

    # Process specific subjects
    uv run python scripts/batch_register_msme.py \
        --study-root /mnt/arborea/bpa-rat \
        --subjects sub-Rat110 sub-Rat111

    # Skip subjects that already have SIGMA-space outputs
    uv run python scripts/batch_register_msme.py \
        --study-root /mnt/arborea/bpa-rat \
        --skip-existing
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.preprocess.workflows.msme_preprocess import register_msme_to_template


@dataclass
class SubjectRegistrationInfo:
    """Information about a subject's MSME registration status."""
    subject: str
    session: str
    cohort: str
    mwf_path: Path
    brain_mask_path: Path
    has_msme_to_template: bool = False
    has_sigma_outputs: bool = False
    error: Optional[str] = None


def discover_subjects(
    study_root: Path,
    subjects_filter: Optional[List[str]] = None
) -> List[SubjectRegistrationInfo]:
    """
    Discover subjects with MSME preprocessing complete.

    Returns list of SubjectRegistrationInfo with current registration status.
    """
    derivatives_root = study_root / 'derivatives'
    transforms_root = study_root / 'transforms'

    subjects = []

    # Find all MWF files (primary MSME output)
    mwf_files = sorted(derivatives_root.glob('sub-*/ses-*/msme/*_MWF.nii.gz'))

    for mwf_path in mwf_files:
        # Skip space-SIGMA files
        if 'space-' in mwf_path.name:
            continue

        # Parse subject/session from path
        parts = mwf_path.parts
        subject = [p for p in parts if p.startswith('sub-')][0]
        session = [p for p in parts if p.startswith('ses-')][0]
        cohort = session.replace('ses-', '')

        # Apply filter if specified
        if subjects_filter and subject not in subjects_filter:
            continue

        # Skip unknown cohort (no template)
        if cohort == 'unknown':
            continue

        # Check brain mask exists
        msme_dir = derivatives_root / subject / session / 'msme'
        brain_mask = msme_dir / f'{subject}_{session}_desc-brain_mask.nii.gz'

        info = SubjectRegistrationInfo(
            subject=subject,
            session=session,
            cohort=cohort,
            mwf_path=mwf_path,
            brain_mask_path=brain_mask
        )

        if not brain_mask.exists():
            info.error = f"Brain mask not found: {brain_mask}"

        # Check existing MSME→Template transform
        msme_to_template = transforms_root / subject / session / 'MSME_to_template_0GenericAffine.mat'
        info.has_msme_to_template = msme_to_template.exists()

        # Check for SIGMA-space outputs
        sigma_mwf = msme_dir / f'{subject}_{session}_space-SIGMA_MWF.nii.gz'
        info.has_sigma_outputs = sigma_mwf.exists()

        subjects.append(info)

    return subjects


def prepare_msme_ref(
    study_root: Path,
    info: SubjectRegistrationInfo
) -> Path:
    """
    Prepare skull-stripped first echo as MSME registration reference.

    Checks work dir for cached version first. If not found, extracts
    first echo from raw MSME and applies existing brain mask.

    Returns
    -------
    Path
        Path to skull-stripped first echo NIfTI
    """
    work_dir = study_root / 'work' / info.subject / info.session / 'msme_batch' / 'msme_registration'
    work_dir.mkdir(parents=True, exist_ok=True)

    msme_ref = work_dir / f'{info.subject}_{info.session}_msme_echo1.nii.gz'

    # Check if already prepared (from prior preprocessing run)
    if msme_ref.exists():
        print(f"    Using cached first echo ref: {msme_ref.name}")
        return msme_ref

    # Find raw MSME file
    bids_root = study_root / 'raw' / 'bids'
    msme_files = list(bids_root.glob(
        f'{info.subject}/{info.session}/msme/*MSME*.nii.gz'
    ))
    if not msme_files:
        raise FileNotFoundError(
            f"Raw MSME not found for {info.subject}/{info.session}"
        )

    print(f"    Extracting first echo from {msme_files[0].name}...")
    img = nib.load(msme_files[0])
    data = img.get_fdata()

    # MSME shape: (X, Y, echoes, slices) — extract first echo
    first_echo = data[:, :, 0, :]  # Shape: (X, Y, slices)

    # Build spatial affine: in-plane from header, slice thickness = 8mm
    in_plane = img.header.get_zooms()[:2]
    echo1_affine = np.diag([float(in_plane[0]), float(in_plane[1]), 8.0, 1.0])

    # Save raw first echo
    msme_ref_raw = work_dir / f'{info.subject}_{info.session}_msme_echo1_raw.nii.gz'
    nib.save(
        nib.Nifti1Image(first_echo.astype(np.float32), echo1_affine),
        msme_ref_raw
    )

    # Apply brain mask
    print(f"    Applying brain mask...")
    mask_data = nib.load(info.brain_mask_path).get_fdata() > 0
    brain_data = first_echo.astype(np.float32) * mask_data
    nib.save(nib.Nifti1Image(brain_data, echo1_affine), msme_ref)

    print(f"    First echo ref: shape={first_echo.shape}, "
          f"voxels=({in_plane[0]}, {in_plane[1]}, 8.0)mm")

    return msme_ref


def apply_transforms_to_sigma(
    input_image: Path,
    reference_image: Path,
    output_image: Path,
    transforms: List[str],
    interpolation: str = 'Linear'
) -> Path:
    """Apply transform chain to warp image to SIGMA space."""
    output_image.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(input_image),
        '-r', str(reference_image),
        '-o', str(output_image),
        '-n', interpolation,
    ]

    for t in transforms:
        cmd.extend(['-t', t])

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"antsApplyTransforms failed: {result.stderr}")

    return output_image


def register_subject(
    info: SubjectRegistrationInfo,
    study_root: Path,
    n_cores: int = 4,
    force: bool = False,
    z_range: Optional[tuple] = None
) -> Dict:
    """
    Run full registration chain for a single subject.

    Pipeline: MSME → Template (rigid) → SIGMA (pre-computed SyN)
    """
    derivatives_root = study_root / 'derivatives'
    transforms_root = study_root / 'transforms'
    templates_root = study_root / 'templates'

    result = {
        'subject': info.subject,
        'session': info.session,
        'status': 'success',
        'steps': {},
        'outputs': {}
    }

    # Check prerequisites
    if info.error:
        result['status'] = 'skipped'
        result['error'] = info.error
        return result

    # Check template exists
    template_path = templates_root / 'anat' / info.cohort / f'tpl-BPARat_{info.cohort}_T2w.nii.gz'
    if not template_path.exists():
        result['status'] = 'skipped'
        result['error'] = f"Template not found: {template_path}"
        return result

    # Check SIGMA reference exists
    sigma_template = study_root / 'atlas' / 'SIGMA_study_space' / 'SIGMA_InVivo_Brain_Template.nii.gz'
    if not sigma_template.exists():
        result['status'] = 'skipped'
        result['error'] = "SIGMA template not found"
        return result

    # Check template-to-SIGMA transforms exist
    tpl_to_sigma_affine = templates_root / 'anat' / info.cohort / 'transforms' / 'tpl-to-SIGMA_0GenericAffine.mat'
    tpl_to_sigma_warp = templates_root / 'anat' / info.cohort / 'transforms' / 'tpl-to-SIGMA_1Warp.nii.gz'

    if not tpl_to_sigma_affine.exists():
        result['status'] = 'skipped'
        result['error'] = f"Template-to-SIGMA transform not found: {tpl_to_sigma_affine}"
        return result

    try:
        # =================================================================
        # Step 1: Prepare MSME registration reference (skull-stripped echo1)
        # =================================================================
        print(f"    Step 1: Preparing MSME registration reference...")
        msme_ref = prepare_msme_ref(study_root, info)
        result['steps']['prepare_ref'] = 'completed'

        # =================================================================
        # Step 2: MSME → Template registration (rigid, NCC Z-init)
        # =================================================================
        msme_to_tpl_affine = transforms_root / info.subject / info.session / 'MSME_to_template_0GenericAffine.mat'

        if not info.has_msme_to_template or force:
            print(f"    Step 2: MSME → Template registration (rigid)...")

            work_dir = study_root / 'work' / info.subject / info.session / 'msme_batch' / 'msme_registration'
            work_dir.mkdir(parents=True, exist_ok=True)

            register_msme_to_template(
                msme_ref_file=msme_ref,
                template_file=template_path,
                output_dir=study_root,
                subject=info.subject,
                session=info.session,
                work_dir=work_dir,
                n_cores=n_cores,
                z_range=z_range
            )
            result['steps']['msme_to_template'] = 'computed'
        else:
            print(f"    Step 2: MSME → Template (exists)")
            result['steps']['msme_to_template'] = 'existing'

        if not msme_to_tpl_affine.exists():
            raise RuntimeError(f"MSME→Template transform not found after registration: {msme_to_tpl_affine}")

        # =================================================================
        # Step 3: Warp MSME maps to SIGMA space
        # =================================================================
        print(f"    Step 3: Warping MSME maps to SIGMA space...")

        # Build transform chain: MSME → Template → SIGMA
        # ANTs applies transforms in reverse order, so list them last-to-first
        transform_chain = [
            str(tpl_to_sigma_warp) if tpl_to_sigma_warp.exists() else None,
            str(tpl_to_sigma_affine),
            str(msme_to_tpl_affine),
        ]
        transform_chain = [t for t in transform_chain if t is not None]

        # Warp each metric map
        maps = ['MWF', 'IWF', 'CSFF', 'T2']
        msme_dir = derivatives_root / info.subject / info.session / 'msme'

        for map_name in maps:
            input_path = msme_dir / f'{info.subject}_{info.session}_{map_name}.nii.gz'
            output_path = msme_dir / f'{info.subject}_{info.session}_space-SIGMA_{map_name}.nii.gz'

            if not input_path.exists():
                print(f"      {map_name}: not found, skipping")
                continue

            if output_path.exists() and not force:
                print(f"      {map_name}: exists")
                result['outputs'][f'{map_name}_sigma'] = str(output_path)
                continue

            apply_transforms_to_sigma(
                input_image=input_path,
                reference_image=sigma_template,
                output_image=output_path,
                transforms=transform_chain,
                interpolation='Linear'
            )
            print(f"      {map_name}: done")
            result['outputs'][f'{map_name}_sigma'] = str(output_path)

        result['steps']['warp_to_sigma'] = 'completed'

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Batch register MSME to SIGMA space (direct MSME→Template pipeline)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--study-root', type=Path, required=True,
                        help='Path to study root directory')
    parser.add_argument('--subjects', nargs='+',
                        help='Specific subjects to process (default: all)')
    parser.add_argument('--n-cores', type=int, default=4,
                        help='Number of CPU cores for ANTs (default: 4)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip subjects with existing SIGMA outputs')
    parser.add_argument('--force', action='store_true',
                        help='Recompute all transforms even if they exist')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without running')
    parser.add_argument('--z-range', type=int, nargs=2, metavar=('MIN', 'MAX'),
                        help='Constrain NCC Z search to slice range (default: 14 28)')
    parser.add_argument('--output-json', type=Path,
                        help='Save results to JSON file')

    args = parser.parse_args()

    print("=" * 70)
    print("Batch MSME Registration to SIGMA Space (Direct MSME→Template Pipeline)")
    print("=" * 70)
    print(f"Study root: {args.study_root}")
    print(f"Cores: {args.n_cores}")
    print(f"Pipeline: MSME (rigid, NCC Z-init) → Template → SIGMA (pre-computed SyN)")
    print()

    # Discover subjects
    print("Discovering subjects with MSME preprocessing...")
    subjects = discover_subjects(args.study_root, args.subjects)
    print(f"Found {len(subjects)} subjects with MWF maps")

    # Filter based on options
    if args.skip_existing:
        subjects = [s for s in subjects if not s.has_sigma_outputs]
        print(f"After filtering existing: {len(subjects)} subjects")

    # Count status
    ready = [s for s in subjects if s.error is None]
    with_errors = [s for s in subjects if s.error is not None]

    print(f"\nStatus:")
    print(f"  Ready to process: {len(ready)}")
    print(f"  Already have MSME→Template: {sum(1 for s in subjects if s.has_msme_to_template)}")
    print(f"  Already have SIGMA outputs: {sum(1 for s in subjects if s.has_sigma_outputs)}")
    if with_errors:
        print(f"  With errors: {len(with_errors)}")
        for s in with_errors[:5]:
            print(f"    {s.subject}/{s.session}: {s.error}")
        if len(with_errors) > 5:
            print(f"    ... and {len(with_errors) - 5} more")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for s in ready[:20]:
            status = []
            if not s.has_msme_to_template:
                status.append("MSME→Tpl")
            status.append("→SIGMA")
            print(f"  {s.subject}/{s.session} ({s.cohort}): {' + '.join(status)}")
        if len(ready) > 20:
            print(f"  ... and {len(ready) - 20} more")
        return 0

    if not ready:
        print("\nNo subjects ready to process.")
        return 0

    # Process subjects
    print("\n" + "=" * 70)
    print("Processing subjects")
    print("=" * 70)

    results = []
    start_time = datetime.now()

    for i, info in enumerate(ready):
        print(f"\n[{i+1}/{len(ready)}] {info.subject} / {info.session} ({info.cohort})")

        z_range = tuple(args.z_range) if args.z_range else None
        result = register_subject(
            info=info,
            study_root=args.study_root,
            n_cores=args.n_cores,
            force=args.force,
            z_range=z_range
        )
        results.append(result)

        if result['status'] == 'success':
            print(f"    Complete")
        else:
            print(f"    {result['status']}: {result.get('error', 'unknown')}")

    # Summary
    elapsed = datetime.now() - start_time
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    skipped = sum(1 for r in results if r['status'] == 'skipped')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total processed: {len(results)}")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Elapsed time: {elapsed}")

    if failed > 0:
        print("\nFailed subjects:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  {r['subject']}/{r['session']}: {r.get('error', 'unknown')}")

    # Save results
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'study_root': str(args.study_root),
                'pipeline': 'MSME→Template→SIGMA (direct)',
                'total_subjects': len(subjects),
                'processed': len(results),
                'success': success,
                'failed': failed,
                'results': results
            }, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
