#!/usr/bin/env python3
"""
Prepare fMRI resting-state metric data for whole-brain voxel-wise analysis.

Discovers SIGMA-space fALFF and ReHo maps, merges with study tracker metadata,
stacks into masked 4D volumes using the SIGMA brain mask (whole-brain, not WM
skeleton), and writes subject list + config JSON.

Unlike TBSS which uses a skeletonized WM mask, this analysis uses the full
SIGMA brain mask because fALFF and ReHo are whole-brain metrics.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/prepare_fmri_voxelwise.py \
        --study-root /mnt/arborea/bpa-rat \
        --output-dir /mnt/arborea/bpa-rat/analysis/voxelwise_fmri \
        --metrics fALFF ReHo
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


def discover_fmri_subjects(
    derivatives_dir: Path,
    metrics: List[str],
) -> List[Dict]:
    """
    Discover subjects with complete fMRI SIGMA-space maps.

    Looks for files matching:
        {sub}_{ses}_space-SIGMA_desc-{metric}_bold.nii.gz

    Returns list of dicts with keys: subject, session, subject_key, metric_files.
    """
    subjects = []

    for subject_dir in sorted(derivatives_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith('sub-'):
            continue
        subject = subject_dir.name

        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir() or not session_dir.name.startswith('ses-'):
                continue
            session = session_dir.name
            func_dir = session_dir / 'func'
            if not func_dir.is_dir():
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

    # Phase 1: Discover fMRI subjects
    logger.info("\n[Phase 1] Discovering fMRI SIGMA-space maps...")
    subjects = discover_fmri_subjects(derivatives_dir, args.metrics)
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

    # Copy SIGMA brain mask as analysis mask
    mask_dst = stats_dir / 'analysis_mask.nii.gz'
    shutil.copy2(brain_mask_path, mask_dst)
    logger.info(f"Copied brain mask from {brain_mask_path}")

    mask_img = nib.load(mask_dst)
    mask_data = mask_img.get_fdata() > 0
    n_mask_voxels = int(mask_data.sum())
    logger.info(f"Analysis mask (whole-brain): {n_mask_voxels} voxels")

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
        'mask': 'SIGMA_InVivo_Brain_Mask',
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

    # Write config.json with provenance
    subject_list_hash = sha256_file(subject_list_file)
    config = {
        'analysis_type': 'voxelwise_whole_brain',
        'modality': 'func',
        'metrics': args.metrics,
        'n_subjects': len(merged),
        'subject_list_sha256': subject_list_hash,
        'mask_source': str(brain_mask_path),
        'mask_type': 'SIGMA_brain_mask',
        'n_mask_voxels': n_mask_voxels,
        'tfce_mode': '3D (-T)',
        'date_prepared': datetime.now().isoformat(),
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Subject list: {len(merged)} subjects")
    logger.info(f"Subject list SHA256: {subject_list_hash[:16]}...")

    # Phase 5: Stack 4D volumes
    logger.info("\n[Phase 5] Building 4D volumes...")

    # Get reference shape from mask
    ref_shape = mask_img.shape[:3]
    ref_affine = mask_img.affine

    for metric in args.metrics:
        logger.info(f"\n  {metric}:")
        n_subjects = len(merged)
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

        # Apply mask
        mask_4d = mask_data[:, :, :, np.newaxis]
        volume_4d_masked = volume_4d * mask_4d

        # Save
        out_file = stats_dir / f'all_{metric}.nii.gz'
        nib.save(nib.Nifti1Image(volume_4d_masked, ref_affine), out_file)

        # Summary stats within mask
        masked_vals = volume_4d_masked[mask_data]
        logger.info(f"    Shape: {volume_4d_masked.shape}")
        logger.info(
            f"    Within-mask range: [{masked_vals.min():.4f}, {masked_vals.max():.4f}]"
        )
        logger.info(f"    Within-mask mean: {masked_vals.mean():.4f}")
        logger.info(f"    Saved: {out_file}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("fMRI VOXELWISE DATA PREPARATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Subjects: {len(merged)}")
    logger.info(f"Metrics: {args.metrics}")
    logger.info(f"Mask voxels: {n_mask_voxels} (whole-brain)")
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
