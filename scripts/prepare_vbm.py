#!/usr/bin/env python3
"""
VBM Data Preparation Pipeline

Processes all subjects' T2w-derived tissue probability maps (GM, WM) for
voxel-based morphometry:
  1. Warp tissue maps from native space to SIGMA via T2w -> template -> SIGMA
  2. Compose the full warp field and compute Jacobian determinant
  3. Modulate warped tissue maps by Jacobian (volume preservation)
  4. Smooth modulated maps
  5. Stack into 4D volumes for FSL randomise

Prerequisites:
    - Anatomical preprocessing complete (tissue segmentation via Atropos)
    - T2w -> template registration done (transforms in transforms/ dir)
    - Template -> SIGMA registration done (transforms in templates/anat/{cohort}/)
    - SIGMA brain mask available in atlas/SIGMA_study_space/

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/prepare_vbm.py \
        --study-root /mnt/arborea/bpa-rat \
        --tissues GM WM \
        --fwhm 3.0 \
        --n-workers 4
"""

import argparse
import json
import logging
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from neurofaune.analysis.vbm.prepare_vbm import (
    compose_warp_field,
    compute_jacobian,
    modulate_tissue,
    smooth_volume,
    warp_tissue_to_sigma,
)


@dataclass
class VBMSubject:
    """Tracks per-subject VBM processing state."""
    subject: str
    session: str
    cohort: str
    tissue_files: Dict[str, Path] = field(default_factory=dict)
    t2w_to_tpl_affine: Optional[Path] = None
    t2w_to_tpl_warp: Optional[Path] = None
    tpl_to_sigma_affine: Optional[Path] = None
    tpl_to_sigma_warp: Optional[Path] = None
    included: bool = False
    exclusion_reason: Optional[str] = None


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"vbm_prepare_{timestamp}.log"

    logger = logging.getLogger("neurofaune.vbm")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def discover_vbm_subjects(
    study_root: Path,
    tissues: List[str],
    cohorts: List[str] = None,
) -> List[VBMSubject]:
    """
    Discover subjects with tissue maps and valid transform chains.

    Parameters
    ----------
    study_root : Path
        Study root directory.
    tissues : list of str
        Tissue types to require (e.g. ['GM', 'WM']).
    cohorts : list of str
        Cohorts to include (default: ['p30', 'p60', 'p90']).

    Returns
    -------
    list of VBMSubject
        All discovered subjects with validation status.
    """
    if cohorts is None:
        cohorts = ['p30', 'p60', 'p90']

    derivatives_dir = study_root / 'derivatives'
    transforms_dir = study_root / 'transforms'
    templates_dir = study_root / 'templates' / 'anat'

    subjects = []

    for sub_dir in sorted(derivatives_dir.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith('sub-'):
            continue
        subject = sub_dir.name

        for ses_dir in sorted(sub_dir.iterdir()):
            if not ses_dir.is_dir() or not ses_dir.name.startswith('ses-'):
                continue
            session = ses_dir.name
            cohort = session.replace('ses-', '')

            if cohort not in cohorts:
                continue

            sd = VBMSubject(subject=subject, session=session, cohort=cohort)

            # Check tissue probability maps
            anat_dir = ses_dir / 'anat'
            if not anat_dir.is_dir():
                sd.exclusion_reason = 'no anat directory'
                subjects.append(sd)
                continue

            all_tissues_found = True
            for tissue in tissues:
                tissue_file = (
                    anat_dir / f'{subject}_{session}_label-{tissue}_probseg.nii.gz'
                )
                if tissue_file.exists():
                    sd.tissue_files[tissue] = tissue_file
                else:
                    all_tissues_found = False
                    sd.exclusion_reason = f'missing {tissue} probseg'
                    break

            if not all_tissues_found:
                subjects.append(sd)
                continue

            # Check T2w -> template transforms
            xfm_dir = transforms_dir / subject / session
            t2w_affine = xfm_dir / f'{subject}_{session}_T2w_to_template_0GenericAffine.mat'
            t2w_warp = xfm_dir / f'{subject}_{session}_T2w_to_template_1Warp.nii.gz'

            if not t2w_affine.exists() or not t2w_warp.exists():
                sd.exclusion_reason = 'missing T2w-to-template transforms'
                subjects.append(sd)
                continue

            sd.t2w_to_tpl_affine = t2w_affine
            sd.t2w_to_tpl_warp = t2w_warp

            # Check template -> SIGMA transforms
            tpl_xfm_dir = templates_dir / cohort / 'transforms'
            tpl_sigma_affine = tpl_xfm_dir / 'tpl-to-SIGMA_0GenericAffine.mat'
            tpl_sigma_warp = tpl_xfm_dir / 'tpl-to-SIGMA_1Warp.nii.gz'

            if not tpl_sigma_affine.exists() or not tpl_sigma_warp.exists():
                sd.exclusion_reason = f'missing tpl-to-SIGMA transforms for {cohort}'
                subjects.append(sd)
                continue

            sd.tpl_to_sigma_affine = tpl_sigma_affine
            sd.tpl_to_sigma_warp = tpl_sigma_warp

            sd.included = True
            subjects.append(sd)

    return subjects


def process_single_subject(
    sd: VBMSubject,
    study_root: Path,
    sigma_reference: Path,
    tissues: List[str],
    fwhm_mm: float,
) -> Dict:
    """
    Process a single subject: warp, Jacobian, modulate, smooth.

    Returns a dict with status and output paths.
    """
    subject_key = f'{sd.subject}_{sd.session}'
    anat_dir = study_root / 'derivatives' / sd.subject / sd.session / 'anat'

    # Transform chain: ANTs applies in reverse order, so list as
    # [tpl_to_sigma_warp, tpl_to_sigma_affine, t2w_to_tpl_warp, t2w_to_tpl_affine]
    transforms = [
        sd.tpl_to_sigma_warp,
        sd.tpl_to_sigma_affine,
        sd.t2w_to_tpl_warp,
        sd.t2w_to_tpl_affine,
    ]

    result = {'subject_key': subject_key, 'success': True, 'outputs': {}}

    try:
        # Compose the full warp field for Jacobian computation
        composed_warp = anat_dir / f'{subject_key}_space-SIGMA_desc-composedWarp.nii.gz'
        if not composed_warp.exists():
            compose_warp_field(
                reference=sigma_reference,
                transforms=transforms,
                output_path=composed_warp,
            )

        # Compute Jacobian determinant
        jacobian_file = anat_dir / f'{subject_key}_space-SIGMA_desc-jacobian.nii.gz'
        if not jacobian_file.exists():
            compute_jacobian(
                warp_field=composed_warp,
                output_path=jacobian_file,
            )

        # Process each tissue type
        for tissue in tissues:
            # Warp tissue to SIGMA
            warped = anat_dir / f'{subject_key}_space-SIGMA_label-{tissue}_probseg.nii.gz'
            if not warped.exists():
                warp_tissue_to_sigma(
                    tissue_map=sd.tissue_files[tissue],
                    reference=sigma_reference,
                    transforms=transforms,
                    output_path=warped,
                )

            # Modulate
            modulated = anat_dir / f'{subject_key}_space-SIGMA_label-{tissue}_modulated.nii.gz'
            if not modulated.exists():
                modulate_tissue(
                    tissue_sigma=warped,
                    jacobian=jacobian_file,
                    output_path=modulated,
                )

            # Smooth
            smoothed = anat_dir / f'{subject_key}_space-SIGMA_label-{tissue}_modulated_smooth.nii.gz'
            if not smoothed.exists():
                smooth_volume(
                    input_path=modulated,
                    output_path=smoothed,
                    fwhm_mm=fwhm_mm,
                )

            result['outputs'][tissue] = str(smoothed)

        # Clean up composed warp (large file, only needed for Jacobian)
        if composed_warp.exists():
            composed_warp.unlink()

    except Exception as e:
        result['success'] = False
        result['error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Prepare VBM data: warp tissue maps to SIGMA, modulate, smooth, stack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output per subject (in derivatives/sub-*/ses-*/anat/):
    *_space-SIGMA_label-{GM,WM}_probseg.nii.gz     Warped tissue
    *_space-SIGMA_desc-jacobian.nii.gz               Jacobian determinant
    *_space-SIGMA_label-{GM,WM}_modulated.nii.gz     Tissue x Jacobian
    *_space-SIGMA_label-{GM,WM}_modulated_smooth.nii.gz  Smoothed modulated

Output group (in analysis/vbm/stats/):
    all_{GM,WM}.nii.gz     4D stacked volumes (all subjects)
    analysis_mask.nii.gz   SIGMA brain mask
    subject_list.txt       Subject order matching 4D volumes
        """
    )

    parser.add_argument('--study-root', type=Path, required=True,
                        help='Study root directory')
    parser.add_argument('--tissues', nargs='+', default=['GM', 'WM'],
                        help='Tissue types to process (default: GM WM)')
    parser.add_argument('--fwhm', type=float, default=3.0,
                        help='Smoothing FWHM in mm (default: 3.0)')
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--cohorts', nargs='+', default=['p30', 'p60', 'p90'],
                        help='Cohorts to include (default: p30 p60 p90)')

    args = parser.parse_args()
    study_root = args.study_root

    # Output directory
    vbm_dir = study_root / 'analysis' / 'vbm'
    stats_dir = vbm_dir / 'stats'
    stats_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(vbm_dir)

    logger.info("=" * 80)
    logger.info("VBM DATA PREPARATION")
    logger.info("=" * 80)
    logger.info(f"Study root: {study_root}")
    logger.info(f"Tissues: {args.tissues}")
    logger.info(f"Smoothing FWHM: {args.fwhm} mm")
    logger.info(f"Workers: {args.n_workers}")
    logger.info(f"Cohorts: {args.cohorts}")

    # SIGMA reference
    sigma_reference = (
        study_root / 'atlas' / 'SIGMA_study_space'
        / 'SIGMA_InVivo_Brain_Template.nii.gz'
    )
    sigma_mask = (
        study_root / 'atlas' / 'SIGMA_study_space'
        / 'SIGMA_InVivo_Brain_Mask.nii.gz'
    )

    if not sigma_reference.exists():
        logger.error(f"SIGMA reference not found: {sigma_reference}")
        sys.exit(1)
    if not sigma_mask.exists():
        logger.error(f"SIGMA brain mask not found: {sigma_mask}")
        sys.exit(1)

    # ── Phase 1: Discover subjects ──
    logger.info("\n[Phase 1] Discovering subjects...")
    subjects_data = discover_vbm_subjects(
        study_root, args.tissues, cohorts=args.cohorts
    )

    included = [s for s in subjects_data if s.included]
    excluded = [s for s in subjects_data if not s.included]

    logger.info(f"  Total discovered: {len(subjects_data)}")
    logger.info(f"  Included: {len(included)}")
    logger.info(f"  Excluded: {len(excluded)}")

    if excluded:
        reasons = {}
        for s in excluded:
            reasons.setdefault(s.exclusion_reason, []).append(
                f'{s.subject}_{s.session}'
            )
        for reason, subs in reasons.items():
            logger.info(f"    {reason}: {len(subs)}")

    if not included:
        logger.error("No subjects to process!")
        sys.exit(1)

    # Cohort distribution
    cohort_counts = {}
    for s in included:
        cohort_counts[s.cohort] = cohort_counts.get(s.cohort, 0) + 1
    logger.info(f"  Cohort distribution: {cohort_counts}")

    # ── Phase 2: Process subjects (warp, Jacobian, modulate, smooth) ──
    logger.info(f"\n[Phase 2] Processing {len(included)} subjects...")

    results = []
    if args.n_workers <= 1:
        for i, sd in enumerate(included):
            logger.info(f"  [{i+1}/{len(included)}] {sd.subject}_{sd.session}")
            r = process_single_subject(
                sd, study_root, sigma_reference, args.tissues, args.fwhm
            )
            results.append(r)
            if not r['success']:
                logger.warning(f"    FAILED: {r.get('error', 'unknown')}")
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            for sd in included:
                future = pool.submit(
                    process_single_subject,
                    sd, study_root, sigma_reference, args.tissues, args.fwhm,
                )
                futures[future] = f'{sd.subject}_{sd.session}'

            for i, future in enumerate(as_completed(futures)):
                subject_key = futures[future]
                r = future.result()
                results.append(r)
                status = "OK" if r['success'] else f"FAILED: {r.get('error', '')}"
                logger.info(f"  [{i+1}/{len(included)}] {subject_key}: {status}")

    n_ok = sum(1 for r in results if r['success'])
    n_fail = sum(1 for r in results if not r['success'])
    logger.info(f"\n  Processed: {n_ok} OK, {n_fail} failed")

    # Filter to successful subjects only
    successful_keys = {r['subject_key'] for r in results if r['success']}
    included = [s for s in included if f'{s.subject}_{s.session}' in successful_keys]

    if not included:
        logger.error("No subjects survived processing!")
        sys.exit(1)

    # ── Phase 3: Create analysis mask ──
    logger.info("\n[Phase 3] Creating analysis mask...")
    analysis_mask_file = stats_dir / 'analysis_mask.nii.gz'
    shutil.copy2(sigma_mask, analysis_mask_file)
    logger.info(f"  Copied SIGMA brain mask -> {analysis_mask_file.name}")

    mask_img = nib.load(analysis_mask_file)
    mask_data = mask_img.get_fdata() > 0

    # ── Phase 4: Stack into 4D volumes ──
    logger.info(f"\n[Phase 4] Stacking {len(included)} subjects into 4D volumes...")

    # Write subject list
    subject_list_file = vbm_dir / 'subject_list.txt'
    with open(subject_list_file, 'w') as f:
        for sd in included:
            f.write(f'{sd.subject}_{sd.session}\n')
    logger.info(f"  Wrote subject_list.txt ({len(included)} subjects)")

    # Get reference geometry from the SIGMA mask
    ref_affine = mask_img.affine
    ref_shape = mask_img.shape[:3]

    for tissue in args.tissues:
        logger.info(f"  Stacking {tissue}...")

        # Collect smoothed modulated maps in subject_list order
        data_4d = np.zeros((*ref_shape, len(included)), dtype=np.float32)

        for i, sd in enumerate(included):
            anat_dir = (
                study_root / 'derivatives' / sd.subject / sd.session / 'anat'
            )
            smooth_file = (
                anat_dir
                / f'{sd.subject}_{sd.session}_space-SIGMA_label-{tissue}_modulated_smooth.nii.gz'
            )
            if not smooth_file.exists():
                logger.warning(f"    Missing: {smooth_file.name}")
                continue
            data_4d[..., i] = nib.load(smooth_file).get_fdata(dtype=np.float32)

        # Apply analysis mask
        data_4d *= mask_data[:, :, :, np.newaxis]

        out_4d = stats_dir / f'all_{tissue}.nii.gz'
        nib.save(nib.Nifti1Image(data_4d, ref_affine), out_4d)
        logger.info(
            f"    -> {out_4d.name}: shape {data_4d.shape}, "
            f"range [{data_4d.min():.4f}, {data_4d.max():.4f}]"
        )

        del data_4d

    # Also copy subject_list.txt into stats/ for convenience
    shutil.copy2(subject_list_file, stats_dir / 'subject_list.txt')

    # ── Phase 4.5: Downsample to match smoothing resolution ──
    voxel_size = mask_img.header.get_zooms()[0]
    if args.fwhm > voxel_size:
        target_res = args.fwhm
        logger.info(
            f"\n[Phase 4.5] Downsampling from {voxel_size:.1f}mm to "
            f"{target_res:.1f}mm (matching FWHM)..."
        )

        # Back up full-resolution files
        for f in [analysis_mask_file] + [
            stats_dir / f'all_{t}.nii.gz' for t in args.tissues
        ]:
            backup = f.with_name(
                f.name.replace('.nii.gz', f'_{voxel_size:.1f}mm.nii.gz')
            )
            if not backup.exists():
                shutil.copy2(f, backup)
                logger.info(f"  Backed up: {f.name} -> {backup.name}")

        # Downsample mask (nearest neighbour)
        subprocess.run([
            'flirt',
            '-in', str(analysis_mask_file),
            '-ref', str(analysis_mask_file),
            '-out', str(analysis_mask_file),
            '-applyisoxfm', str(target_res),
            '-interp', 'nearestneighbour',
        ], check=True)

        # Downsample 4D tissue volumes (trilinear — already smoothed)
        for tissue in args.tissues:
            tissue_file = stats_dir / f'all_{tissue}.nii.gz'
            subprocess.run([
                'flirt',
                '-in', str(tissue_file),
                '-ref', str(analysis_mask_file),
                '-out', str(tissue_file),
                '-applyisoxfm', str(target_res),
                '-interp', 'trilinear',
            ], check=True)

        # Report new dimensions
        new_mask = nib.load(analysis_mask_file)
        new_voxels = int(np.count_nonzero(new_mask.get_fdata()))
        logger.info(
            f"  Downsampled: {ref_shape} -> {new_mask.shape}, "
            f"{np.count_nonzero(mask_data):,} -> {new_voxels:,} in-mask voxels"
        )
    else:
        logger.info(
            f"\n[Phase 4.5] Skipping downsample (voxel size {voxel_size:.1f}mm "
            f">= FWHM {args.fwhm:.1f}mm)"
        )

    # ── Phase 5: Write VBM config ──
    logger.info("\n[Phase 5] Writing VBM config...")
    vbm_config = {
        'date_created': datetime.now().isoformat(),
        'study_root': str(study_root),
        'tissues': args.tissues,
        'fwhm_mm': args.fwhm,
        'n_subjects': len(included),
        'cohort_counts': cohort_counts,
        'n_failed': n_fail,
        'sigma_reference': str(sigma_reference),
    }
    with open(vbm_dir / 'vbm_config.json', 'w') as f:
        json.dump(vbm_config, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VBM PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Subjects: {len(included)}")
    logger.info(f"  Tissues: {args.tissues}")
    logger.info(f"  Output: {vbm_dir}")
    logger.info(f"  4D volumes: {stats_dir / 'all_*.nii.gz'}")
    logger.info(f"  Analysis mask: {analysis_mask_file}")
    logger.info(f"\nNext: run prepare_vbm_designs.py to create design matrices")


if __name__ == '__main__':
    main()
