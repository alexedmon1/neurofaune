#!/usr/bin/env python3
"""
Template-Space TBSS Data Preparation

Prepares DTI data for group-level TBSS analysis in per-cohort template
space instead of SIGMA space.  This avoids the systematic z-alignment
errors introduced by the template→SIGMA warp while keeping the accurate
FA→template affine alignment (Dice ~0.94).

Workflow:
    1. Warp SIGMA atlas assets (WM prob, parcellation, brain mask) to
       template space using the inverse of template→SIGMA transforms
    2. Build tissue-informed WM analysis mask in template space
    3. Discover subjects with FA_to_template transforms
    4. Warp native DTI metrics to template space
    5. Build per-voxel coverage mask (intersect WM with coverage ≥ threshold)
    6. Stack masked 4D volumes for randomise
    7. Write manifests and config for downstream analysis

Usage:
    uv run python -m neurofaune.analysis.tbss.prepare_template_tbss \\
        --study-root /mnt/arborea/bpa-rat \\
        --output-dir /mnt/arborea/bpa-rat/analysis/tbss_template/p90 \\
        --cohort p90

    uv run python scripts/run_template_tbss_prepare.py \\
        --study-root /mnt/arborea/bpa-rat \\
        --output-dir /mnt/arborea/bpa-rat/analysis/tbss_template/p90 \\
        --cohort p90 --metrics FA MD AD RD
"""

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from neurofaune.analysis.tbss.prepare_tbss import (
    SubjectData,
    _find_metric_file,
    create_wm_mask,
    generate_manifest,
    setup_logging,
)

DTI_METRICS = ['FA', 'MD', 'AD', 'RD']
MSME_METRICS = ['MWF', 'IWF', 'CSFF', 'T2']

# Per-modality defaults: (transform_prefix, derivatives_subdir, coverage_metric)
MODALITY_DEFAULTS = {
    'dti':  ('FA',   'dwi',  'FA'),
    'msme': ('MSME', 'msme', 'MWF'),
}


def warp_atlas_to_template(
    study_root: Path,
    cohort: str,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Warp SIGMA atlas assets to per-cohort template space.

    Uses the *inverse* of the template→SIGMA transforms so that atlas
    images land in template space without touching individual subjects.

    Transform order for antsApplyTransforms (inverse mapping):
        -t [tpl-to-SIGMA_0GenericAffine.mat, 1]   # invert affine
        -t  tpl-to-SIGMA_1InverseWarp.nii.gz       # inverse warp

    Args:
        study_root: Study root directory
        cohort: Age cohort (p30, p60, p90)
        output_dir: Where to write template-space atlas files

    Returns:
        Dict with keys 'wm_probability', 'parcellation', 'brain_mask'
        mapping to output paths.
    """
    logger = logging.getLogger("neurofaune.tbss")

    atlas_dir = output_dir / 'atlas'
    atlas_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    sigma_dir = study_root / 'atlas' / 'SIGMA_study_space'
    tpl_transforms = study_root / 'templates' / 'anat' / cohort / 'transforms'
    reference = study_root / 'templates' / 'anat' / cohort / f'tpl-BPARat_{cohort}_T2w.nii.gz'

    affine = tpl_transforms / 'tpl-to-SIGMA_0GenericAffine.mat'
    inv_warp = tpl_transforms / 'tpl-to-SIGMA_1InverseWarp.nii.gz'

    for required in [reference, affine, inv_warp]:
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    # Assets to warp: (source filename, output filename, interpolation)
    assets = [
        ('SIGMA_InVivo_WM.nii.gz', 'wm_probability.nii.gz', 'Linear'),
        ('SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz', 'parcellation.nii.gz', 'NearestNeighbor'),
        ('SIGMA_InVivo_Brain_Mask.nii.gz', 'brain_mask.nii.gz', 'NearestNeighbor'),
    ]

    result_paths = {}
    key_map = {
        'wm_probability.nii.gz': 'wm_probability',
        'parcellation.nii.gz': 'parcellation',
        'brain_mask.nii.gz': 'brain_mask',
    }

    for src_name, out_name, interp in assets:
        src = sigma_dir / src_name
        out = atlas_dir / out_name

        if out.exists():
            logger.info(f"  Atlas cached: {out_name}")
            result_paths[key_map[out_name]] = out
            continue

        if not src.exists():
            raise FileNotFoundError(f"SIGMA atlas file not found: {src}")

        cmd = [
            'antsApplyTransforms',
            '-d', '3',
            '-i', str(src),
            '-r', str(reference),
            '-o', str(out),
            '-n', interp,
            '-t', f'[{affine}, 1]',
            '-t', str(inv_warp),
        ]

        logger.info(f"  Warping {src_name} → template space ({interp})")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"antsApplyTransforms failed for {src_name}: {proc.stderr}"
            )

        result_paths[key_map[out_name]] = out

    return result_paths


def discover_template_subjects(
    study_root: Path,
    cohort: str,
    metrics: List[str],
    exclude_file: Optional[Path] = None,
    subject_list: Optional[List[str]] = None,
    transform_prefix: str = 'FA',
    derivatives_subdir: str = 'dwi',
) -> List[SubjectData]:
    """
    Find subjects with modality-to-template transforms and native metrics.

    Args:
        study_root: Study root directory
        cohort: Age cohort (p30, p60, p90)
        metrics: Metrics to require (e.g. FA/MD/AD/RD or MWF/IWF/CSFF/T2)
        exclude_file: Optional exclusion list
        subject_list: Optional explicit subject list
        transform_prefix: Transform name prefix ('FA' or 'MSME')
        derivatives_subdir: Subdirectory under derivatives ('dwi' or 'msme')

    Returns:
        List of SubjectData with validation status
    """
    logger = logging.getLogger("neurofaune.tbss")

    transforms_dir = study_root / 'transforms'
    derivatives_dir = study_root / 'derivatives'
    session = f'ses-{cohort}'

    # Load exclusion list
    excluded_ids = set()
    if exclude_file and exclude_file.exists():
        with open(exclude_file) as f:
            excluded_ids = {line.strip() for line in f if line.strip()}
        logger.info(f"Loaded {len(excluded_ids)} excluded subjects")

    # Determine subject list
    if subject_list:
        subjects = []
        for entry in subject_list:
            entry = entry.strip()
            if '_ses-' in entry:
                subj = entry.split('_ses-')[0]
            else:
                subj = entry
            subjects.append(subj)
        logger.info(f"Using provided subject list: {len(subjects)} entries")
    else:
        # Discover from transforms directory
        subjects = []
        if not transforms_dir.exists():
            logger.error(f"Transforms directory not found: {transforms_dir}")
            return []

        for subj_dir in sorted(transforms_dir.iterdir()):
            if not subj_dir.is_dir() or not subj_dir.name.startswith('sub-'):
                continue
            ses_dir = subj_dir / session
            if ses_dir.is_dir():
                subjects.append(subj_dir.name)

        logger.info(f"Discovered {len(subjects)} subjects with {session} transforms")

    # Validate each subject
    subjects_data = []
    for subject in subjects:
        subject_session = f"{subject}_{session}"

        # Check exclusion
        if subject_session in excluded_ids or subject in excluded_ids:
            subjects_data.append(SubjectData(
                subject=subject, session=session, cohort=cohort,
                included=False, exclusion_reason='In exclusion list'
            ))
            continue

        sd = SubjectData(subject=subject, session=session, cohort=cohort)

        # Check modality_to_template transform
        subj_transforms = transforms_dir / subject / session
        mod_to_tpl = subj_transforms / f'{transform_prefix}_to_template_0GenericAffine.mat'
        if not mod_to_tpl.exists():
            sd.exclusion_reason = f'Missing {transform_prefix}_to_template transform'
            subjects_data.append(sd)
            continue
        sd.fa_to_t2w_affine = mod_to_tpl  # reusing field for the affine

        # Check native metrics
        dwi_dir = derivatives_dir / subject / session / derivatives_subdir
        missing_metrics = []
        for metric in metrics:
            metric_file = _find_metric_file(dwi_dir, subject, session, metric)
            if metric_file:
                sd.metric_files[metric] = metric_file
            else:
                missing_metrics.append(metric)

        if missing_metrics:
            sd.exclusion_reason = f"Missing native metrics: {', '.join(missing_metrics)}"
            subjects_data.append(sd)
            continue

        sd.included = True
        subjects_data.append(sd)

    included = sum(1 for s in subjects_data if s.included)
    excluded = sum(1 for s in subjects_data if not s.included)
    logger.info(f"Validation: {included} included, {excluded} excluded")

    return subjects_data


def warp_metric_to_template(
    metric_file: Path,
    transform: Path,
    reference: Path,
    output_file: Path,
) -> Path:
    """
    Warp one DTI metric to template space via the FA_to_template affine.

    Args:
        metric_file: Native-space metric NIfTI
        transform: FA_to_template_0GenericAffine.mat
        reference: Template image for geometry
        output_file: Output path

    Returns:
        Path to warped file
    """
    logger = logging.getLogger("neurofaune.tbss")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(metric_file),
        '-r', str(reference),
        '-o', str(output_file),
        '-n', 'Linear',
        '-t', str(transform),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"antsApplyTransforms failed for {metric_file.name}: {proc.stderr}"
        )

    return output_file


def build_coverage_mask(
    warped_fa_files: List[Path],
    wm_mask: Path,
    output_dir: Path,
    min_coverage: float = 0.75,
) -> Tuple[Path, Path]:
    """
    Intersect WM mask with per-voxel subject coverage.

    DTI is a partial-brain slab (11 slices), so some template voxels
    have zero coverage.  This mask keeps only WM voxels where at least
    ``min_coverage`` fraction of subjects have nonzero FA.

    Args:
        warped_fa_files: Warped FA files (one per subject)
        wm_mask: Interior WM mask from create_wm_mask()
        output_dir: Where to save outputs
        min_coverage: Minimum fraction of subjects with nonzero FA

    Returns:
        (analysis_mask_path, coverage_count_path)
    """
    logger = logging.getLogger("neurofaune.tbss")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_img = nib.load(warped_fa_files[0])
    shape = first_img.shape[:3]
    affine = first_img.affine

    # Count per-voxel nonzero coverage
    coverage = np.zeros(shape, dtype=np.int32)
    for fa_file in warped_fa_files:
        data = nib.load(fa_file).get_fdata()
        coverage += (data > 0).astype(np.int32)

    coverage_file = output_dir / 'coverage_count.nii.gz'
    nib.save(nib.Nifti1Image(coverage, affine), coverage_file)

    # Load WM mask
    wm_data = (nib.load(wm_mask).get_fdata() > 0)

    # Require coverage >= threshold
    min_count = int(np.ceil(min_coverage * len(warped_fa_files)))
    coverage_ok = (coverage >= min_count)

    analysis_mask = (wm_data & coverage_ok).astype(np.uint8)
    analysis_mask_file = output_dir / 'analysis_mask.nii.gz'
    nib.save(nib.Nifti1Image(analysis_mask, affine), analysis_mask_file)

    n_wm = int(wm_data.sum())
    n_analysis = int(analysis_mask.sum())
    n_removed = n_wm - n_analysis
    logger.info(
        f"Coverage mask: {n_analysis} voxels retained from {n_wm} WM voxels "
        f"({n_removed} removed, min_count={min_count}/{len(warped_fa_files)})"
    )

    return analysis_mask_file, coverage_file


def prepare_template_tbss_data(
    study_root: Path,
    output_dir: Path,
    cohort: str,
    metrics: List[str] = None,
    min_coverage: float = 0.75,
    wm_prob_threshold: float = 0.3,
    erosion_voxels: int = 2,
    exclude_file: Optional[Path] = None,
    subject_list: Optional[List[str]] = None,
    modality: str = 'dti',
) -> Dict:
    """
    Main 7-phase workflow for template-space TBSS preparation.

    Args:
        study_root: Study root directory
        output_dir: Output directory for this cohort's TBSS data
        cohort: Age cohort (p30, p60, p90)
        metrics: Metrics to prepare (default depends on modality)
        min_coverage: Minimum fraction of subjects with nonzero coverage per voxel
        wm_prob_threshold: WM probability threshold for mask
        erosion_voxels: Erosion from brain boundary in voxels
        exclude_file: Path to exclusion list
        subject_list: Optional explicit subject list
        modality: 'dti' or 'msme'

    Returns:
        Dict with preparation results
    """
    transform_prefix, derivatives_subdir, coverage_metric = MODALITY_DEFAULTS[modality]

    if metrics is None:
        metrics = DTI_METRICS if modality == 'dti' else MSME_METRICS

    study_root = Path(study_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    session = f'ses-{cohort}'
    reference = study_root / 'templates' / 'anat' / cohort / f'tpl-BPARat_{cohort}_T2w.nii.gz'

    logger.info("=" * 80)
    logger.info(f"TEMPLATE-SPACE TBSS PREPARATION — {cohort} ({modality.upper()})")
    logger.info("=" * 80)
    logger.info(f"Study root: {study_root}")
    logger.info(f"Cohort: {cohort}")
    logger.info(f"Modality: {modality} (transform={transform_prefix}, dir={derivatives_subdir})")
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Min coverage: {min_coverage}")
    logger.info(f"WM prob threshold: {wm_prob_threshold}")
    logger.info(f"Erosion voxels: {erosion_voxels}")

    # ── Phase 1: Warp atlas assets to template space ────────────────────
    logger.info("\n[Phase 1] Warping atlas assets to template space...")
    atlas_paths = warp_atlas_to_template(study_root, cohort, output_dir)
    logger.info(f"  WM probability: {atlas_paths['wm_probability']}")
    logger.info(f"  Parcellation:   {atlas_paths['parcellation']}")
    logger.info(f"  Brain mask:     {atlas_paths['brain_mask']}")

    # ── Phase 2: Build WM analysis mask ─────────────────────────────────
    logger.info("\n[Phase 2] Building WM analysis mask in template space...")
    wm_mask_dir = output_dir / 'wm_mask'
    # Pass a dummy config — the overrides bypass the SIGMA lookup
    dummy_config = {'paths': {'study_root': str(study_root)}}
    interior_mask, exterior_mask = create_wm_mask(
        config=dummy_config,
        output_dir=wm_mask_dir,
        reference_img=reference,
        wm_prob_threshold=wm_prob_threshold,
        erosion_voxels=erosion_voxels,
        wm_prob_file=atlas_paths['wm_probability'],
        brain_mask_file=atlas_paths['brain_mask'],
    )

    # ── Phase 3: Discover subjects ──────────────────────────────────────
    logger.info("\n[Phase 3] Discovering subjects...")
    subjects_data = discover_template_subjects(
        study_root=study_root,
        cohort=cohort,
        metrics=metrics,
        exclude_file=exclude_file,
        subject_list=subject_list,
        transform_prefix=transform_prefix,
        derivatives_subdir=derivatives_subdir,
    )

    included = [s for s in subjects_data if s.included]
    if not included:
        logger.error("No valid subjects found!")
        return {"success": False, "error": "No valid subjects found"}

    logger.info(f"Found {len(included)} valid subjects for {cohort}")

    # ── Phase 4: Warp metrics to template space ───────────────────────
    logger.info(f"\n[Phase 4] Warping {modality.upper()} metrics to template space...")
    transforms_dir = study_root / 'transforms'

    for metric in metrics:
        metric_dir = output_dir / metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        for i, sd in enumerate(included):
            out_file = metric_dir / f'{sd.subject}_{sd.session}_{metric}_template.nii.gz'

            if out_file.exists():
                logger.info(
                    f"  [{i+1}/{len(included)}] {sd.subject} {metric} (cached)"
                )
                continue

            # For DTI FA, reuse existing FA_to_template_Warped.nii.gz
            # (MSME Warped file is raw echo, not a derived metric — skip)
            if modality == 'dti' and metric == coverage_metric:
                prewarped = (
                    transforms_dir / sd.subject / session
                    / f'{transform_prefix}_to_template_Warped.nii.gz'
                )
                if prewarped.exists():
                    import shutil
                    shutil.copy2(prewarped, out_file)
                    logger.info(
                        f"  [{i+1}/{len(included)}] {sd.subject} {metric} "
                        f"(copied prewarped)"
                    )
                    continue

            # Warp native metric via modality_to_template affine
            transform = (
                transforms_dir / sd.subject / session
                / f'{transform_prefix}_to_template_0GenericAffine.mat'
            )
            logger.info(
                f"  [{i+1}/{len(included)}] {sd.subject} {metric}"
            )
            try:
                warp_metric_to_template(
                    metric_file=sd.metric_files[metric],
                    transform=transform,
                    reference=reference,
                    output_file=out_file,
                )
            except Exception as e:
                logger.error(f"  Failed: {e}")
                sd.included = False
                sd.exclusion_reason = f"Warp failed: {e}"

    # Recompute included after potential failures
    included = [s for s in subjects_data if s.included]
    if not included:
        logger.error("No subjects remaining after Phase 4!")
        return {"success": False, "error": "All subjects failed warping"}

    # ── Phase 5: Build coverage mask ────────────────────────────────────
    logger.info("\n[Phase 5] Building coverage mask...")
    cov_dir = output_dir / coverage_metric
    warped_cov_files = [
        cov_dir / f'{sd.subject}_{sd.session}_{coverage_metric}_template.nii.gz'
        for sd in included
    ]
    missing_cov = [f for f in warped_cov_files if not f.exists()]
    if missing_cov:
        logger.error(f"Missing {coverage_metric} files for {len(missing_cov)} subjects")
        return {"success": False, "error": f"Missing {coverage_metric}: {missing_cov[:5]}"}

    stats_dir = output_dir / 'stats'
    stats_dir.mkdir(parents=True, exist_ok=True)

    analysis_mask_file, coverage_file = build_coverage_mask(
        warped_fa_files=warped_cov_files,
        wm_mask=interior_mask,
        output_dir=stats_dir,
        min_coverage=min_coverage,
    )

    # ── Phase 6: Stack 4D volumes ───────────────────────────────────────
    logger.info("\n[Phase 6] Stacking 4D volumes...")
    first_img = nib.load(warped_cov_files[0])
    shape_3d = first_img.shape[:3]
    affine = first_img.affine
    analysis_mask_data = (nib.load(analysis_mask_file).get_fdata() > 0)

    for metric in metrics:
        metric_dir = output_dir / metric
        metric_files = [
            metric_dir / f'{sd.subject}_{sd.session}_{metric}_template.nii.gz'
            for sd in included
        ]

        stack = np.zeros((*shape_3d, len(metric_files)), dtype=np.float32)
        for i, f in enumerate(metric_files):
            stack[..., i] = nib.load(f).get_fdata()

        # Apply analysis mask
        mask_4d = analysis_mask_data[:, :, :, np.newaxis]
        stack_masked = stack * mask_4d

        out_4d = stats_dir / f'all_{metric}.nii.gz'
        nib.save(nib.Nifti1Image(stack_masked, affine), out_4d)
        logger.info(f"  {metric}: {stack_masked.shape}")

    # Mean of coverage metric (FA for DTI, MWF for MSME)
    cov_stack = nib.load(stats_dir / f'all_{coverage_metric}.nii.gz').get_fdata()
    mean_cov = np.mean(cov_stack, axis=-1).astype(np.float32)
    mean_fa_file = stats_dir / 'mean_FA.nii.gz'  # keep name for FSL compat
    nib.save(nib.Nifti1Image(mean_cov, affine), mean_fa_file)
    logger.info(f"  mean {coverage_metric} saved as mean_FA.nii.gz")

    # ── Phase 7: Write manifests ────────────────────────────────────────
    logger.info("\n[Phase 7] Writing manifests...")
    manifest = generate_manifest(subjects_data, output_dir, metrics)

    # Write tbss_config.json with atlas paths for downstream scripts
    tbss_config = {
        'space': 'template',
        'cohort': cohort,
        'template': str(reference),
        'atlas': {k: str(v) for k, v in atlas_paths.items()},
        'analysis_mask': str(analysis_mask_file),
        'mean_FA': str(mean_fa_file),
        'date_prepared': datetime.now().isoformat(),
        'n_subjects': len(included),
        'min_coverage': min_coverage,
        'wm_prob_threshold': wm_prob_threshold,
        'erosion_voxels': erosion_voxels,
    }
    with open(output_dir / 'tbss_config.json', 'w') as f:
        json.dump(tbss_config, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("TEMPLATE-SPACE TBSS PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Cohort: {cohort}")
    logger.info(f"Subjects included: {manifest['subjects_included']}")
    logger.info(f"Subjects excluded: {manifest['subjects_excluded']}")
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Analysis mask: {int(analysis_mask_data.sum())} voxels")
    logger.info(f"Output: {output_dir}")
    logger.info("\nNext steps:")
    logger.info(f"  1. Generate designs:")
    logger.info(f"     uv run python scripts/prepare_tbss_designs.py \\")
    logger.info(f"         --study-tracker $TRACKER --tbss-dir {output_dir} \\")
    logger.info(f"         --output-dir {output_dir}/designs \\")
    logger.info(f"         --analyses per_pnd_{cohort} dose_response_{cohort}")
    logger.info(f"  2. Run randomise:")
    logger.info(f"     uv run python scripts/run_tbss_analysis.py \\")
    logger.info(f"         --tbss-dir {output_dir} --metrics {' '.join(metrics)} \\")
    logger.info(f"         --parcellation {atlas_paths['parcellation']}")

    return {
        "success": True,
        "manifest": manifest,
        "output_dir": str(output_dir),
        "analysis_mask": str(analysis_mask_file),
        "n_mask_voxels": int(analysis_mask_data.sum()),
        "n_subjects": len(included),
        "atlas_paths": {k: str(v) for k, v in atlas_paths.items()},
    }


def main():
    """Command-line interface for template-space TBSS preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare DTI data for TBSS in per-cohort template space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare p90 cohort
  uv run python -m neurofaune.analysis.tbss.prepare_template_tbss \\
      --study-root /mnt/arborea/bpa-rat \\
      --output-dir /mnt/arborea/bpa-rat/analysis/tbss_template/p90 \\
      --cohort p90

  # All metrics with custom coverage threshold
  uv run python -m neurofaune.analysis.tbss.prepare_template_tbss \\
      --study-root /mnt/arborea/bpa-rat \\
      --output-dir /mnt/arborea/bpa-rat/analysis/tbss_template/p60 \\
      --cohort p60 --metrics FA MD AD RD --min-coverage 0.80
        """
    )

    parser.add_argument('--study-root', type=Path, required=True,
                        help='Study root directory')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for this cohort')
    parser.add_argument('--cohort', type=str, required=True,
                        choices=['p30', 'p60', 'p90'],
                        help='Age cohort')
    parser.add_argument('--modality', type=str, default='dti',
                        choices=['dti', 'msme'],
                        help='Modality (default: dti)')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='Metrics to prepare (default: FA MD AD RD for dti, '
                             'MWF IWF CSFF T2 for msme)')
    parser.add_argument('--min-coverage', type=float, default=0.75,
                        help='Min fraction of subjects with nonzero voxel (default: 0.75)')
    parser.add_argument('--wm-prob-threshold', type=float, default=0.3,
                        help='WM probability threshold (default: 0.3)')
    parser.add_argument('--erosion-voxels', type=int, default=2,
                        help='Erosion from brain boundary in voxels (default: 2)')
    parser.add_argument('--exclude-file', type=Path,
                        help='Path to exclusion list')
    parser.add_argument('--subject-list', type=Path,
                        help='Explicit subject list file')

    args = parser.parse_args()

    # Load subject list if provided
    subjects = None
    if args.subject_list:
        if not args.subject_list.exists():
            parser.error(f"Subject list not found: {args.subject_list}")
        with open(args.subject_list) as f:
            subjects = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(subjects)} subjects from {args.subject_list}")

    result = prepare_template_tbss_data(
        study_root=args.study_root,
        output_dir=args.output_dir,
        cohort=args.cohort,
        metrics=args.metrics,
        min_coverage=args.min_coverage,
        wm_prob_threshold=args.wm_prob_threshold,
        erosion_voxels=args.erosion_voxels,
        exclude_file=args.exclude_file,
        subject_list=subjects,
        modality=args.modality,
    )

    exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
