#!/usr/bin/env python3
"""
Prepare MSME metric data for TBSS-style voxel-wise analysis.

Discovers MSME SIGMA-space maps (MWF, IWF, CSFF, T2), merges with
study tracker metadata, stacks into masked 4D volumes using the
DTI-derived WM analysis mask, and writes subject list + tbss_config.json.

The analysis mask from DTI TBSS is reused because:
  - It's derived from FA skeletonization / WM thresholding
  - MSME maps cover the same white matter regions
  - Using the same mask enables direct cross-modality comparison

Usage:
    uv run python scripts/prepare_msme_tbss.py \
        --derivatives-dir /mnt/arborea/bpa-rat/derivatives \
        --dti-tbss-dir /mnt/arborea/bpa-rat/analysis/tbss \
        --output-dir /mnt/arborea/bpa-rat/analysis/tbss_msme \
        --study-tracker /mnt/arborea/bpa-rat/study_tracker_combined_250916.csv
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

MSME_METRICS = ['MWF', 'IWF', 'CSFF', 'T2']

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


def discover_msme_subjects(
    derivatives_dir: Path,
    metrics: List[str],
) -> List[Dict]:
    """
    Discover subjects with complete MSME SIGMA-space maps.

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
            msme_dir = session_dir / 'msme'
            if not msme_dir.is_dir():
                continue

            # Check all metrics present
            metric_files = {}
            for metric in metrics:
                f = msme_dir / f'{subject}_{session}_space-SIGMA_{metric}.nii.gz'
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
    Merge discovered MSME subjects with study tracker to get dose/sex/PND.

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
            f"{unmatched} MSME subjects not in tracker (dropped): "
            f"{sorted(missing_ids)}"
        )

    # Deterministic sort
    merged = merged.sort_values('subject_key').reset_index(drop=True)
    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Prepare MSME metric data for TBSS-style voxel-wise analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output structure:
  {output_dir}/
    tbss_config.json          # Modality, metrics, subject hash
    subject_list.txt          # Deterministic subject order
    subject_manifest.json     # Per-subject metadata
    stats/
      analysis_mask.nii.gz    # Copied from DTI TBSS
      all_MWF.nii.gz          # 4D masked volumes
      all_IWF.nii.gz
      all_CSFF.nii.gz
      all_T2.nii.gz
        """
    )

    parser.add_argument('--derivatives-dir', type=Path, required=True,
                        help='Path to derivatives directory')
    parser.add_argument('--dti-tbss-dir', type=Path, required=True,
                        help='Path to DTI TBSS directory (for analysis mask)')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for MSME TBSS data')
    parser.add_argument('--study-tracker', type=Path, required=True,
                        help='Path to study_tracker_combined CSV')
    parser.add_argument('--metrics', nargs='+', default=MSME_METRICS,
                        help=f'Metrics to prepare (default: {MSME_METRICS})')

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("MSME TBSS Data Preparation")
    logger.info("=" * 70)
    logger.info(f"Derivatives: {args.derivatives_dir}")
    logger.info(f"DTI TBSS: {args.dti_tbss_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Metrics: {args.metrics}")

    # Validate inputs
    if not args.derivatives_dir.exists():
        logger.error(f"Derivatives dir not found: {args.derivatives_dir}")
        sys.exit(1)

    dti_mask = args.dti_tbss_dir / 'stats' / 'analysis_mask.nii.gz'
    if not dti_mask.exists():
        logger.error(f"DTI analysis mask not found: {dti_mask}")
        sys.exit(1)

    if not args.study_tracker.exists():
        logger.error(f"Study tracker not found: {args.study_tracker}")
        sys.exit(1)

    # Phase 1: Discover MSME subjects
    logger.info("\n[Phase 1] Discovering MSME SIGMA-space maps...")
    subjects = discover_msme_subjects(args.derivatives_dir, args.metrics)
    logger.info(f"Found {len(subjects)} subjects with complete metrics")

    if not subjects:
        logger.error("No subjects found with complete MSME SIGMA-space maps!")
        sys.exit(1)

    # Phase 2: Merge with tracker
    logger.info("\n[Phase 2] Merging with study tracker...")
    merged = merge_with_tracker(subjects, args.study_tracker)
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

    # Copy analysis mask from DTI TBSS
    mask_dst = stats_dir / 'analysis_mask.nii.gz'
    shutil.copy2(dti_mask, mask_dst)
    logger.info(f"Copied analysis mask from {dti_mask}")

    mask_img = nib.load(mask_dst)
    mask_data = mask_img.get_fdata() > 0
    n_mask_voxels = int(mask_data.sum())
    logger.info(f"Analysis mask: {n_mask_voxels} voxels")

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
        'analysis_type': 'TBSS',
        'modality': 'msme',
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

    # Write tbss_config.json with provenance
    subject_list_hash = sha256_file(subject_list_file)
    tbss_config = {
        'modality': 'msme',
        'metrics': args.metrics,
        'n_subjects': len(merged),
        'subject_list_sha256': subject_list_hash,
        'analysis_mask_source': str(dti_mask),
        'date_prepared': datetime.now().isoformat(),
    }
    with open(output_dir / 'tbss_config.json', 'w') as f:
        json.dump(tbss_config, f, indent=2)

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
    logger.info("MSME TBSS PREPARATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Subjects: {len(merged)}")
    logger.info(f"Metrics: {args.metrics}")
    logger.info(f"Mask voxels: {n_mask_voxels}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Generate designs:")
    logger.info(f"     uv run python scripts/prepare_tbss_designs.py \\")
    logger.info(f"         --study-tracker {args.study_tracker} \\")
    logger.info(f"         --tbss-dir {output_dir} \\")
    logger.info(f"         --output-dir {output_dir}/designs")
    logger.info(f"  2. Run analysis:")
    logger.info(f"     uv run python scripts/run_tbss_analysis.py \\")
    logger.info(f"         --tbss-dir {output_dir} \\")
    logger.info(f"         --metrics {' '.join(args.metrics)}")


if __name__ == '__main__':
    main()
