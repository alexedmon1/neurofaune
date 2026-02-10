#!/usr/bin/env python3
"""
Batch DTI registration to SIGMA space via direct FA→Template pipeline.

This script registers all preprocessed DTI data through the simplified chain:
    Subject FA → Cohort Template → SIGMA (2 stages, 3 transforms)

This replaces the old 3-stage pipeline (FA→T2w→Template→SIGMA) which used
5 transforms and produced worse atlas overlap, especially for 3D T2w subjects.

For each subject with completed DTI preprocessing:
1. FA → Template registration (affine to age-matched template)
2. Warp FA, MD, AD, RD to SIGMA space using FA→Template + Template→SIGMA

Prerequisites:
- DTI preprocessing complete (FA/MD/AD/RD exist)
- Cohort templates built with SIGMA registration

Usage:
    # Register all subjects
    uv run python scripts/batch_register_dwi.py \
        --study-root /mnt/arborea/bpa-rat \
        --n-cores 4

    # Dry run to see what would be processed
    uv run python scripts/batch_register_dwi.py \
        --study-root /mnt/arborea/bpa-rat \
        --dry-run

    # Process specific subjects
    uv run python scripts/batch_register_dwi.py \
        --study-root /mnt/arborea/bpa-rat \
        --subjects sub-Rat1 sub-Rat2

    # Skip subjects that already have SIGMA-space outputs
    uv run python scripts/batch_register_dwi.py \
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


@dataclass
class SubjectRegistrationInfo:
    """Information about a subject's registration status."""
    subject: str
    session: str
    cohort: str
    fa_path: Path
    has_fa_to_template: bool = False
    has_sigma_outputs: bool = False
    error: Optional[str] = None


def discover_subjects(
    study_root: Path,
    subjects_filter: Optional[List[str]] = None
) -> List[SubjectRegistrationInfo]:
    """
    Discover subjects with DTI preprocessing complete.

    Returns list of SubjectRegistrationInfo with current registration status.
    """
    derivatives_root = study_root / 'derivatives'
    transforms_root = study_root / 'transforms'

    subjects = []

    # Find all FA files
    fa_files = sorted(derivatives_root.glob('sub-*/ses-*/dwi/*_FA.nii.gz'))

    for fa_path in fa_files:
        # Skip space-SIGMA FA files
        if 'space-' in fa_path.name:
            continue

        # Parse subject/session from path
        parts = fa_path.parts
        subject = [p for p in parts if p.startswith('sub-')][0]
        session = [p for p in parts if p.startswith('ses-')][0]
        cohort = session.replace('ses-', '')

        # Apply filter if specified
        if subjects_filter and subject not in subjects_filter:
            continue

        # Skip unknown cohort (no template)
        if cohort == 'unknown':
            continue

        info = SubjectRegistrationInfo(
            subject=subject,
            session=session,
            cohort=cohort,
            fa_path=fa_path
        )

        # Check existing FA→Template transform
        fa_to_template = transforms_root / subject / session / 'FA_to_template_0GenericAffine.mat'
        info.has_fa_to_template = fa_to_template.exists()

        # Check for SIGMA-space outputs
        sigma_fa = derivatives_root / subject / session / 'dwi' / f'{subject}_{session}_space-SIGMA_FA.nii.gz'
        info.has_sigma_outputs = sigma_fa.exists()

        subjects.append(info)

    return subjects


def run_ants_registration(
    moving_image: Path,
    fixed_image: Path,
    output_prefix: Path,
    transform_type: str = 'Affine',
    n_cores: int = 4
) -> Dict[str, Path]:
    """Run ANTs registration."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    transform_flag = {'SyN': 's', 'Affine': 'a', 'Rigid': 'r'}[transform_type]

    cmd = [
        'antsRegistrationSyN.sh',
        '-d', '3',
        '-f', str(fixed_image),
        '-m', str(moving_image),
        '-o', str(output_prefix),
        '-n', str(n_cores),
        '-t', transform_flag
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ANTs registration failed: {result.stdout}")

    outputs = {
        'affine': Path(str(output_prefix) + '0GenericAffine.mat'),
        'warped': Path(str(output_prefix) + 'Warped.nii.gz'),
    }

    if transform_type == 'SyN':
        outputs['warp'] = Path(str(output_prefix) + '1Warp.nii.gz')
        outputs['inverse_warp'] = Path(str(output_prefix) + '1InverseWarp.nii.gz')

    return outputs


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
    force: bool = False
) -> Dict:
    """
    Run full registration chain for a single subject.

    Pipeline: FA → Template (affine) → SIGMA (pre-computed SyN)
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
        # Try alternate location
        sigma_template = Path('/mnt/arborea/atlases/SIGMA_scaled/SIGMA_Rat_Brain_Atlases/SIGMA_Anatomical_Atlas/InVivo_Atlas/SIGMA_InVivo_Brain_Template.nii')

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
        # =====================================================================
        # Step 1: FA → Template registration (affine)
        # =====================================================================
        fa_to_tpl_prefix = transforms_root / info.subject / info.session / 'FA_to_template_'
        fa_to_tpl_affine = Path(str(fa_to_tpl_prefix) + '0GenericAffine.mat')

        if not info.has_fa_to_template or force:
            print(f"    Step 1: FA → Template registration (affine)...")

            run_ants_registration(
                moving_image=info.fa_path,
                fixed_image=template_path,
                output_prefix=fa_to_tpl_prefix,
                transform_type='Affine',
                n_cores=n_cores
            )
            result['steps']['fa_to_template'] = 'computed'
        else:
            print(f"    Step 1: FA → Template (exists)")
            result['steps']['fa_to_template'] = 'existing'

        # =====================================================================
        # Step 2: Warp DTI metrics to SIGMA space
        # =====================================================================
        print(f"    Step 2: Warping DTI metrics to SIGMA space...")

        # Build transform chain: FA → Template → SIGMA
        # ANTs applies transforms in reverse order, so list them last-to-first
        transform_chain = [
            str(tpl_to_sigma_warp) if tpl_to_sigma_warp.exists() else None,
            str(tpl_to_sigma_affine),
            str(fa_to_tpl_affine),
        ]
        # Remove None entries
        transform_chain = [t for t in transform_chain if t is not None]

        # Warp each metric
        metrics = ['FA', 'MD', 'AD', 'RD']
        dwi_dir = derivatives_root / info.subject / info.session / 'dwi'

        for metric in metrics:
            input_path = dwi_dir / f'{info.subject}_{info.session}_{metric}.nii.gz'
            output_path = dwi_dir / f'{info.subject}_{info.session}_space-SIGMA_{metric}.nii.gz'

            if not input_path.exists():
                print(f"      {metric}: not found, skipping")
                continue

            if output_path.exists() and not force:
                print(f"      {metric}: exists")
                result['outputs'][f'{metric}_sigma'] = str(output_path)
                continue

            apply_transforms_to_sigma(
                input_image=input_path,
                reference_image=sigma_template,
                output_image=output_path,
                transforms=transform_chain,
                interpolation='Linear'
            )
            print(f"      {metric}: done")
            result['outputs'][f'{metric}_sigma'] = str(output_path)

        result['steps']['warp_to_sigma'] = 'completed'

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Batch register DTI to SIGMA space (direct FA→Template pipeline)',
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
    parser.add_argument('--output-json', type=Path,
                        help='Save results to JSON file')

    args = parser.parse_args()

    print("=" * 70)
    print("Batch DTI Registration to SIGMA Space (Direct FA→Template Pipeline)")
    print("=" * 70)
    print(f"Study root: {args.study_root}")
    print(f"Cores: {args.n_cores}")
    print(f"Pipeline: FA → Template (affine) → SIGMA (pre-computed SyN)")
    print()

    # Discover subjects
    print("Discovering subjects with DTI preprocessing...")
    subjects = discover_subjects(args.study_root, args.subjects)
    print(f"Found {len(subjects)} subjects with FA maps")

    # Filter based on options
    if args.skip_existing:
        subjects = [s for s in subjects if not s.has_sigma_outputs]
        print(f"After filtering existing: {len(subjects)} subjects")

    # Count status
    ready = [s for s in subjects if s.error is None]

    print(f"\nStatus:")
    print(f"  Ready to process: {len(ready)}")
    print(f"  Already have FA→Template: {sum(1 for s in subjects if s.has_fa_to_template)}")
    print(f"  Already have SIGMA outputs: {sum(1 for s in subjects if s.has_sigma_outputs)}")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for s in ready[:20]:
            status = []
            if not s.has_fa_to_template:
                status.append("FA→Tpl")
            status.append("→SIGMA")
            print(f"  {s.subject}/{s.session}: {' + '.join(status)}")
        if len(ready) > 20:
            print(f"  ... and {len(ready) - 20} more")
        return 0

    # Process subjects
    print("\n" + "=" * 70)
    print("Processing subjects")
    print("=" * 70)

    results = []
    start_time = datetime.now()

    for i, info in enumerate(ready):
        print(f"\n[{i+1}/{len(ready)}] {info.subject} / {info.session}")

        result = register_subject(
            info=info,
            study_root=args.study_root,
            n_cores=args.n_cores,
            force=args.force
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

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total processed: {len(results)}")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")
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
                'pipeline': 'FA→Template→SIGMA (direct)',
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
