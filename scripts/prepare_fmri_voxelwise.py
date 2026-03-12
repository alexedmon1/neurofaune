#!/usr/bin/env python3
"""
Prepare fMRI resting-state metric data for whole-brain voxel-wise analysis.

Discovers SIGMA-space fALFF and ReHo maps, merges with study tracker metadata,
applies exclusions, stacks into masked 4D volumes, and generates a coverage-
based analysis mask restricted to voxels with adequate subject coverage.

Unlike TBSS which uses a skeletonized WM mask, this analysis uses the full
SIGMA brain mask intersected with a coverage threshold — only voxels where
a minimum fraction of subjects have signal are included.

Usage:
    # Prepare ReHo with exclusions
    PYTHONUNBUFFERED=1 uv run python scripts/prepare_fmri_voxelwise.py \
        --study-root $STUDY_ROOT \
        --output-dir $STUDY_ROOT/analysis/reho \
        --metrics ReHo \
        --exclusion-csv $STUDY_ROOT/exclusions/func_exclusions.csv \
        --min-volumes 200

    # Prepare fALFF with exclusions
    PYTHONUNBUFFERED=1 uv run python scripts/prepare_fmri_voxelwise.py \
        --study-root $STUDY_ROOT \
        --output-dir $STUDY_ROOT/analysis/falff \
        --metrics fALFF \
        --exclusion-csv $STUDY_ROOT/exclusions/func_exclusions.csv \
        --min-volumes 200
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import pandas as pd

FMRI_METRICS = ['fALFF', 'ReHo']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def load_exclusions(exclusion_csv: Path) -> set:
    """
    Load exclusion CSV and return set of (subject, session) tuples.

    Expected CSV format: subject,session,reason,date_added
    """
    df = pd.read_csv(exclusion_csv)
    excl = set(zip(df['subject'], df['session']))
    logger.info(f"Loaded {len(excl)} exclusions from {exclusion_csv}")
    return excl


def get_n_volumes(derivatives_dir: Path, subject: str, session: str) -> int:
    """
    Get BOLD volume count from fALFF/preprocessing analysis JSON.

    Checks analysis JSON files for n_timepoints (the number of BOLD
    volumes used in spectral analysis), then falls back to the
    preprocessed 4D BOLD file header.
    """
    func_dir = derivatives_dir / subject / session / 'func'

    # Try fALFF analysis JSON (has n_timepoints in statistics)
    for json_name in [
        f'{subject}_{session}_desc-falff_analysis.json',
        f'{subject}_{session}_desc-reho_analysis.json',
    ]:
        analysis_json = func_dir / json_name
        if analysis_json.exists():
            try:
                with open(analysis_json) as f:
                    data = json.load(f)
                # n_timepoints nested under statistics.falff.parameters
                stats = data.get('statistics', {})
                for key in stats:
                    params = stats[key].get('parameters', {})
                    if 'n_timepoints' in params:
                        return int(params['n_timepoints'])
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    # Try preprocessed BOLD 4D file
    preproc_bold = func_dir / f'{subject}_{session}_desc-preproc_bold.nii.gz'
    if preproc_bold.exists():
        img = nib.load(preproc_bold)
        return img.shape[3] if len(img.shape) > 3 else 1

    return 0


def discover_fmri_subjects(
    derivatives_dir: Path,
    metrics: List[str],
    exclusions: set = None,
    min_volumes: int = 0,
) -> List[Dict]:
    """
    Discover subjects with complete fMRI SIGMA-space maps.

    Looks for files matching:
        {sub}_{ses}_space-SIGMA_desc-{metric}_bold.nii.gz

    Parameters
    ----------
    derivatives_dir : Path
        Path to derivatives directory.
    metrics : list[str]
        Metric names to require (e.g. ['fALFF', 'ReHo']).
    exclusions : set, optional
        Set of (subject, session) tuples to exclude.
    min_volumes : int
        Minimum BOLD volume count. Sessions with fewer volumes are excluded.

    Returns list of dicts with keys: subject, session, subject_key, metric_files.
    """
    subjects = []
    n_excluded_qc = 0
    n_excluded_volumes = 0

    for subject_dir in sorted(derivatives_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith('sub-'):
            continue
        subject = subject_dir.name

        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir() or not session_dir.name.startswith('ses-'):
                continue
            session = session_dir.name

            # Check exclusion list
            if exclusions and (subject, session) in exclusions:
                n_excluded_qc += 1
                continue

            func_dir = session_dir / 'func'
            if not func_dir.is_dir():
                continue

            # Check volume count
            if min_volumes > 0:
                n_vol = get_n_volumes(derivatives_dir, subject, session)
                if n_vol < min_volumes:
                    logger.info(
                        f"  Excluding {subject}/{session}: "
                        f"{n_vol} volumes < {min_volumes} minimum"
                    )
                    n_excluded_volumes += 1
                    continue

            # Check all metrics present
            metric_files = {}
            for metric in metrics:
                f = func_dir / f'{subject}_{session}_space-SIGMA_desc-{metric}_bold.nii.gz'
                if f.exists():
                    metric_files[metric] = f

            if len(metric_files) == len(metrics):
                subjects.append({
                    'subject': subject,
                    'session': session,
                    'subject_key': f'{subject}_{session}',
                    'metric_files': metric_files,
                })

    if n_excluded_qc > 0:
        logger.info(f"Excluded {n_excluded_qc} sessions by QC exclusion list")
    if n_excluded_volumes > 0:
        logger.info(f"Excluded {n_excluded_volumes} sessions by volume count (<{min_volumes})")

    return subjects


def merge_with_tracker(
    subjects: List[Dict],
    tracker_path: Path,
) -> pd.DataFrame:
    """
    Merge discovered fMRI subjects with study tracker to get dose/sex/PND.

    Returns DataFrame sorted alphabetically by subject_key.
    """
    tracker = pd.read_csv(tracker_path)
    valid = tracker[tracker['irc.ID'].notna()].copy()
    valid['bids_id'] = 'sub-' + valid['irc.ID']
    logger.info(f"Tracker rows with irc.ID: {len(valid)}")

    # Build subject dataframe
    rows = []
    for s in subjects:
        m = re.match(r'(sub-\w+)_(ses-\w+)', s['subject_key'])
        if m:
            rows.append({
                'subject_key': s['subject_key'],
                'bids_id': m.group(1),
                'session': m.group(2),
                'metric_files': s['metric_files'],
            })

    subj_df = pd.DataFrame(rows)
    subj_df['PND'] = subj_df['session'].str.replace('ses-', '').str.upper()

    # Merge on bids_id
    merged = subj_df.merge(
        valid[['bids_id', 'sex', 'dose.level']],
        on='bids_id',
        how='inner',
    )
    merged = merged.rename(columns={'dose.level': 'dose'})

    unmatched = len(subj_df) - len(merged)
    if unmatched > 0:
        missing_ids = set(subj_df['bids_id']) - set(valid['bids_id'])
        logger.warning(
            f"{unmatched} fMRI subjects not in tracker (dropped): "
            f"{sorted(missing_ids)}"
        )

    # Deterministic sort
    merged = merged.sort_values('subject_key').reset_index(drop=True)
    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Prepare fMRI resting-state data for whole-brain voxel-wise analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output structure:
  {output_dir}/
    config.json               # Provenance and analysis parameters
    subject_list.txt          # Deterministic subject order
    subject_manifest.json     # Per-subject metadata
    stats/
      analysis_mask.nii.gz    # SIGMA brain mask (whole-brain)
      all_fALFF.nii.gz        # 4D masked volumes
      all_ReHo.nii.gz
        """
    )

    parser.add_argument('--study-root', type=Path, required=True,
                        help='Path to study root (contains derivatives/, atlas/)')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for voxelwise fMRI data')
    parser.add_argument('--study-tracker', type=Path, default=None,
                        help='Path to study_tracker_combined CSV '
                             '(default: {study-root}/study_tracker_combined_250916.csv)')
    parser.add_argument('--metrics', nargs='+', default=FMRI_METRICS,
                        help=f'Metrics to prepare (default: {FMRI_METRICS})')
    parser.add_argument('--exclusion-csv', type=Path, default=None,
                        help='Path to exclusion CSV (columns: subject, session, reason)')
    parser.add_argument('--min-volumes', type=int, default=0,
                        help='Minimum BOLD volume count to include a session (0 = no check)')
    parser.add_argument('--coverage-threshold', type=float, default=0.75,
                        help='Fraction of subjects required at each voxel for coverage mask '
                             '(default: 0.75). Set to 0 to use whole-brain SIGMA mask.')

    args = parser.parse_args()

    # Derive paths
    derivatives_dir = args.study_root / 'derivatives'
    brain_mask_path = (
        args.study_root / 'atlas' / 'SIGMA_study_space'
        / 'SIGMA_InVivo_Brain_Mask.nii.gz'
    )
    tracker_path = args.study_tracker or (
        args.study_root / 'study_tracker_combined_250916.csv'
    )

    logger.info("=" * 70)
    logger.info("fMRI Whole-Brain Voxelwise Data Preparation")
    logger.info("=" * 70)
    logger.info(f"Study root: {args.study_root}")
    logger.info(f"Derivatives: {derivatives_dir}")
    logger.info(f"Brain mask: {brain_mask_path}")
    logger.info(f"Tracker: {tracker_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Metrics: {args.metrics}")

    # Validate inputs
    if not derivatives_dir.exists():
        logger.error(f"Derivatives dir not found: {derivatives_dir}")
        sys.exit(1)

    if not brain_mask_path.exists():
        logger.error(f"SIGMA brain mask not found: {brain_mask_path}")
        sys.exit(1)

    if not tracker_path.exists():
        logger.error(f"Study tracker not found: {tracker_path}")
        sys.exit(1)

    # Load exclusions if provided
    exclusions = None
    if args.exclusion_csv:
        if not args.exclusion_csv.exists():
            logger.error(f"Exclusion CSV not found: {args.exclusion_csv}")
            sys.exit(1)
        exclusions = load_exclusions(args.exclusion_csv)
        logger.info(f"Exclusion CSV: {args.exclusion_csv}")
    if args.min_volumes > 0:
        logger.info(f"Minimum volume count: {args.min_volumes}")

    # Phase 1: Discover fMRI subjects
    logger.info("\n[Phase 1] Discovering fMRI SIGMA-space maps...")
    subjects = discover_fmri_subjects(
        derivatives_dir, args.metrics,
        exclusions=exclusions,
        min_volumes=args.min_volumes,
    )
    logger.info(f"Found {len(subjects)} subjects with complete metrics")

    if not subjects:
        logger.error("No subjects found with complete fMRI SIGMA-space maps!")
        sys.exit(1)

    # Phase 2: Merge with tracker
    logger.info("\n[Phase 2] Merging with study tracker...")
    merged = merge_with_tracker(subjects, tracker_path)
    logger.info(f"Matched {len(merged)} subjects with tracker metadata")

    # Distribution summary
    logger.info("\nDistribution:")
    for pnd in sorted(merged['PND'].unique()):
        subset = merged[merged['PND'] == pnd]
        dose_dist = dict(subset['dose'].value_counts().sort_index())
        sex_dist = dict(subset['sex'].value_counts().sort_index())
        logger.info(f"  {pnd}: n={len(subset)}, dose={dose_dist}, sex={sex_dist}")

    # Phase 3: Set up output directories
    logger.info("\n[Phase 3] Setting up output...")
    output_dir = args.output_dir
    stats_dir = output_dir / 'stats'
    log_dir = output_dir / 'logs'
    stats_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load SIGMA brain mask as reference
    sigma_mask_img = nib.load(brain_mask_path)
    sigma_mask_data = sigma_mask_img.get_fdata() > 0
    ref_shape = sigma_mask_img.shape[:3]
    ref_affine = sigma_mask_img.affine
    n_sigma_voxels = int(sigma_mask_data.sum())
    logger.info(f"SIGMA brain mask: {n_sigma_voxels} voxels")

    # Save SIGMA mask for reference
    sigma_mask_dst = stats_dir / 'sigma_brain_mask.nii.gz'
    shutil.copy2(brain_mask_path, sigma_mask_dst)

    # Phase 4: Build subject list and manifest
    logger.info("\n[Phase 4] Writing subject list and manifest...")

    # Write subject_list.txt
    subject_list_file = output_dir / 'subject_list.txt'
    with open(subject_list_file, 'w') as f:
        for _, row in merged.iterrows():
            f.write(f"{row['subject_key']}\n")

    # Build metric_files lookup from discovered subjects (keyed by subject_key)
    metric_files_lookup = {
        s['subject_key']: s['metric_files'] for s in subjects
    }

    # Write subject_manifest.json
    manifest = {
        'analysis_type': 'voxelwise_whole_brain',
        'modality': 'func',
        'pipeline': 'neurofaune',
        'metrics': args.metrics,
        'date_prepared': datetime.now().isoformat(),
        'n_subjects': len(merged),
        'subjects': [
            {
                'subject_key': row['subject_key'],
                'bids_id': row['bids_id'],
                'session': row['session'],
                'PND': row['PND'],
                'sex': row['sex'],
                'dose': row['dose'],
            }
            for _, row in merged.iterrows()
        ],
    }
    with open(output_dir / 'subject_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Subject list: {len(merged)} subjects")

    # Phase 5: Stack 4D volumes
    logger.info("\n[Phase 5] Building 4D volumes...")

    n_subjects = len(merged)
    all_4d = {}  # metric -> 4D array

    for metric in args.metrics:
        logger.info(f"\n  {metric}:")
        volume_4d = np.zeros((*ref_shape, n_subjects), dtype=np.float32)

        for i, (_, row) in enumerate(merged.iterrows()):
            subject_key = row['subject_key']
            metric_file = metric_files_lookup[subject_key][metric]
            img = nib.load(metric_file)
            data = img.get_fdata()

            if data.shape[:3] != ref_shape:
                logger.error(
                    f"    Shape mismatch for {subject_key}: "
                    f"{data.shape[:3]} vs {ref_shape}"
                )
                sys.exit(1)

            volume_4d[..., i] = data

        all_4d[metric] = volume_4d

    # Phase 6: Build coverage mask
    logger.info("\n[Phase 6] Building coverage mask...")

    # Use the first metric to compute coverage (all metrics share the same FOV)
    first_metric = args.metrics[0]
    data_4d = all_4d[first_metric]

    # Count how many subjects have non-zero signal at each voxel
    coverage = np.sum(np.abs(data_4d) > 0.01, axis=3)

    # Save coverage count map
    coverage_img = nib.Nifti1Image(coverage.astype(np.float32), ref_affine)
    nib.save(coverage_img, stats_dir / 'coverage_count.nii.gz')

    if args.coverage_threshold > 0:
        min_subjects = int(n_subjects * args.coverage_threshold)
        coverage_mask = (coverage >= min_subjects) & sigma_mask_data
        n_coverage_voxels = int(coverage_mask.sum())

        logger.info(
            f"Coverage threshold: {args.coverage_threshold:.0%} "
            f"({min_subjects}/{n_subjects} subjects)"
        )
        logger.info(
            f"Coverage mask: {n_coverage_voxels:,} voxels "
            f"({100 * n_coverage_voxels / n_sigma_voxels:.1f}% of SIGMA mask)"
        )

        if n_coverage_voxels == 0:
            logger.error(
                "No voxels meet coverage threshold! "
                "Try lowering --coverage-threshold."
            )
            sys.exit(1)

        mask_data = coverage_mask
        mask_type = 'coverage_mask'
        n_mask_voxels = n_coverage_voxels
    else:
        mask_data = sigma_mask_data
        mask_type = 'SIGMA_brain_mask'
        n_mask_voxels = n_sigma_voxels
        logger.info("Using whole-brain SIGMA mask (no coverage threshold)")

    # Save analysis mask
    mask_dst = stats_dir / 'analysis_mask.nii.gz'
    nib.save(nib.Nifti1Image(mask_data.astype(np.uint8), ref_affine), mask_dst)

    # Phase 7: Apply mask and save 4D volumes
    logger.info("\n[Phase 7] Applying mask and saving 4D volumes...")

    mask_4d = mask_data[:, :, :, np.newaxis]
    for metric in args.metrics:
        volume_4d_masked = all_4d[metric] * mask_4d

        out_file = stats_dir / f'all_{metric}.nii.gz'
        nib.save(nib.Nifti1Image(volume_4d_masked, ref_affine), out_file)

        masked_vals = volume_4d_masked[mask_data]
        logger.info(f"  {metric}: shape={volume_4d_masked.shape}")
        logger.info(
            f"    Within-mask range: [{masked_vals.min():.4f}, {masked_vals.max():.4f}]"
        )
        logger.info(f"    Within-mask mean: {masked_vals.mean():.4f}")
        logger.info(f"    Saved: {out_file}")

    # Write config.json with provenance
    subject_list_hash = sha256_file(subject_list_file)
    config = {
        'analysis_type': 'voxelwise_whole_brain',
        'modality': 'func',
        'metrics': args.metrics,
        'n_subjects': len(merged),
        'subject_list_sha256': subject_list_hash,
        'mask_source': str(brain_mask_path),
        'mask_type': mask_type,
        'n_mask_voxels': n_mask_voxels,
        'n_sigma_mask_voxels': n_sigma_voxels,
        'coverage_threshold': args.coverage_threshold,
        'min_subjects_per_voxel': int(n_subjects * args.coverage_threshold)
            if args.coverage_threshold > 0 else 0,
        'exclusion_csv': str(args.exclusion_csv) if args.exclusion_csv else None,
        'min_volumes': args.min_volumes,
        'tfce_mode': '3D (-T)',
        'date_prepared': datetime.now().isoformat(),
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"\nSubject list SHA256: {subject_list_hash[:16]}...")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("fMRI VOXELWISE DATA PREPARATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Subjects: {len(merged)}")
    logger.info(f"Metrics: {args.metrics}")
    logger.info(f"Analysis mask: {n_mask_voxels:,} voxels ({mask_type})")
    if args.exclusion_csv:
        logger.info(f"Exclusion CSV: {args.exclusion_csv}")
    if args.min_volumes > 0:
        logger.info(f"Min volumes: {args.min_volumes}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Generate categorical designs:")
    logger.info(f"     uv run python scripts/prepare_tbss_designs.py \\")
    logger.info(f"         --study-tracker {tracker_path} \\")
    logger.info(f"         --tbss-dir {output_dir} \\")
    logger.info(f"         --output-dir {output_dir}/designs")
    logger.info(f"  2. Generate dose-response designs:")
    logger.info(f"     uv run python scripts/prepare_tbss_dose_response_designs.py \\")
    logger.info(f"         --study-tracker {tracker_path} \\")
    logger.info(f"         --tbss-dir {output_dir} \\")
    logger.info(f"         --output-dir {output_dir}/designs")
    logger.info(f"  3. Run voxelwise analysis:")
    logger.info(f"     uv run python scripts/run_voxelwise_fmri_analysis.py \\")
    logger.info(f"         --analysis-dir {output_dir} \\")
    logger.info(f"         --metrics {' '.join(args.metrics)}")
    logger.info(f"\n  NOTE: Use -T (3D TFCE), not --T2 (2D skeleton TFCE)")


if __name__ == '__main__':
    main()
