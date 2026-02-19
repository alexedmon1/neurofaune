#!/usr/bin/env python3
"""
Prepare FSL randomise design matrices for BPA-Rat TBSS analysis.

Dose is treated as a categorical factor (C=Control, L=Low, M=Medium, H=High)
to allow detection of non-monotonic dose-response patterns.

Creates 4 sets of design files:
  - per_pnd_p30: dose group comparisons within P30 (dose + sex)
  - per_pnd_p60: dose group comparisons within P60 (dose + sex)
  - per_pnd_p90: dose group comparisons within P90 (dose + sex)
  - pooled: all subjects with dose × PND interaction (dose + PND + sex + dose×PND)

Usage:
    uv run python scripts/prepare_tbss_designs.py \
        --study-tracker /mnt/arborea/bpa-rat/study_tracker_combined_250916.csv \
        --tbss-dir /mnt/arborea/bpa-rat/analysis/tbss \
        --output-dir /mnt/arborea/bpa-rat/analysis/tbss/designs
"""

import argparse
import gc
import hashlib
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np
import pandas as pd

# Add neuroaider to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'neuroaider'))
from neuroaider import DesignHelper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def write_provenance(
    design_dir: Path,
    tbss_dir: Path,
    n_subjects_in_design: int,
    design_type: str,
) -> None:
    """Write provenance.json linking design to its TBSS subject list."""
    subject_list_path = tbss_dir / 'subject_list.txt'
    h = hashlib.sha256()
    with open(subject_list_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    subject_list_hash = h.hexdigest()

    n_subjects_in_list = sum(
        1 for line in open(subject_list_path) if line.strip()
    )

    provenance = {
        'tbss_dir': str(tbss_dir),
        'subject_list_sha256': subject_list_hash,
        'n_subjects_in_list': n_subjects_in_list,
        'n_subjects_in_design': n_subjects_in_design,
        'design_type': design_type,
        'date_created': datetime.now().isoformat(),
    }
    with open(design_dir / 'provenance.json', 'w') as f:
        json.dump(provenance, f, indent=2)
    logger.info(f"  Wrote provenance.json (hash: {subject_list_hash[:16]}...)")


def pre_subset_4d_volumes(
    tbss_dir: Path,
    design_dirs: List[Path],
    metrics: Optional[List[str]] = None,
) -> None:
    """
    Pre-create per-analysis 4D subsets from the master volumes.

    Iterates sequentially (one metric at a time) to keep peak memory at
    ~2.6 GB instead of loading multiple volumes in parallel.

    Args:
        tbss_dir: TBSS root directory containing stats/ and subject_list.txt
        design_dirs: List of design directories, each with subject_order.txt
        metrics: Metric names to subset (read from tbss_config.json if None)
    """
    # Discover metrics from tbss_config.json
    if metrics is None:
        config_file = tbss_dir / 'tbss_config.json'
        if not config_file.exists():
            logger.warning(f"No tbss_config.json found at {tbss_dir}, skipping subset")
            return
        with open(config_file) as f:
            metrics = json.load(f)['metrics']

    stats_dir = tbss_dir / 'stats'
    randomise_dir = tbss_dir / 'randomise'

    # Load master subject list
    subject_list_file = tbss_dir / 'subject_list.txt'
    with open(subject_list_file) as f:
        master_subjects = [line.strip() for line in f if line.strip()]

    logger.info(f"\n[Pre-subset] Creating 4D subsets for {len(design_dirs)} analyses × {len(metrics)} metrics")
    logger.info(f"  Master subject list: {len(master_subjects)} subjects")

    master_index = {subj: i for i, subj in enumerate(master_subjects)}

    for metric in metrics:
        master_4d_path = stats_dir / f'all_{metric}.nii.gz'
        if not master_4d_path.exists():
            logger.warning(f"  Master 4D not found: {master_4d_path}, skipping {metric}")
            continue

        # Load master volume once per metric
        logger.info(f"\n  Loading master {metric} ({master_4d_path.name})...")
        master_img = nib.load(master_4d_path)
        master_data = master_img.get_fdata()

        if master_data.shape[3] != len(master_subjects):
            logger.error(
                f"  {metric}: 4D has {master_data.shape[3]} volumes but "
                f"subject list has {len(master_subjects)} — skipping"
            )
            del master_data
            gc.collect()
            continue

        for design_dir in design_dirs:
            analysis_name = design_dir.name
            subject_order_file = design_dir / 'subject_order.txt'
            if not subject_order_file.exists():
                logger.warning(f"  No subject_order.txt in {design_dir}, skipping")
                continue

            with open(subject_order_file) as f:
                design_subjects = [line.strip() for line in f if line.strip()]

            subset_dir = randomise_dir / analysis_name / 'data'
            subset_file = subset_dir / f'all_{metric}.nii.gz'

            # Skip if already exists with correct shape
            if subset_file.exists():
                existing_shape = nib.load(subset_file).shape
                if len(existing_shape) == 4 and existing_shape[3] == len(design_subjects):
                    logger.info(f"  {analysis_name}/{metric}: exists ({existing_shape[3]} vols), skipping")
                    continue
                else:
                    logger.info(f"  {analysis_name}/{metric}: wrong shape {existing_shape}, re-creating")

            # Build index mapping and subset
            indices = []
            missing = []
            for subj in design_subjects:
                if subj in master_index:
                    indices.append(master_index[subj])
                else:
                    missing.append(subj)

            if missing:
                logger.error(
                    f"  {analysis_name}/{metric}: {len(missing)} subjects not in master — skipping"
                )
                continue

            subset_data = master_data[:, :, :, indices].astype(np.float32)
            subset_dir.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(subset_data, master_img.affine, master_img.header), subset_file)
            logger.info(
                f"  {analysis_name}/{metric}: {master_data.shape[3]} -> {subset_data.shape[3]} volumes"
            )
            del subset_data

        # Free master volume before loading next metric
        del master_data
        gc.collect()

    logger.info("\n[Pre-subset] Done")


def load_and_merge_data(
    study_tracker_path: Path,
    tbss_dir: Path,
) -> pd.DataFrame:
    """
    Load study tracker and merge with TBSS subject list.

    The study tracker has one row per rat (at terminal PND), but many rats
    were scanned longitudinally at multiple timepoints. We match on rat ID
    and take dose/sex from the tracker, PND from the TBSS session label.

    Returns:
        DataFrame with columns: subject_key, bids_id, session, PND, sex, dose
    """
    # Load study tracker
    tracker = pd.read_csv(study_tracker_path)
    logger.info(f"Loaded tracker: {len(tracker)} rows")

    # Filter to rows with valid irc.ID
    valid = tracker[tracker['irc.ID'].notna()].copy()
    valid['bids_id'] = 'sub-' + valid['irc.ID']
    logger.info(f"Tracker rows with irc.ID: {len(valid)}")

    # Load TBSS subject list
    subject_list_path = tbss_dir / 'subject_list.txt'
    if not subject_list_path.exists():
        raise FileNotFoundError(f"TBSS subject list not found: {subject_list_path}")

    with open(subject_list_path) as f:
        tbss_subjects = [line.strip() for line in f if line.strip()]
    logger.info(f"TBSS subjects: {len(tbss_subjects)}")

    # Parse TBSS subjects into rat ID + session
    tbss_parsed = []
    for s in tbss_subjects:
        m = re.match(r'(sub-\w+)_(ses-\w+)', s)
        if m:
            tbss_parsed.append({
                'subject_key': s,
                'bids_id': m.group(1),
                'session': m.group(2),
            })
    tbss_df = pd.DataFrame(tbss_parsed)
    tbss_df['PND'] = tbss_df['session'].str.replace('ses-', '').str.upper()

    # Merge on bids_id to get dose/sex (rat-level, not session-level)
    merged = tbss_df.merge(
        valid[['bids_id', 'sex', 'dose.level']],
        on='bids_id',
        how='inner'
    )
    merged = merged.rename(columns={'dose.level': 'dose'})

    unmatched = len(tbss_df) - len(merged)
    if unmatched > 0:
        missing_ids = set(tbss_df['bids_id']) - set(valid['bids_id'])
        logger.warning(
            f"{unmatched} TBSS subjects not found in tracker "
            f"(dropped): {sorted(missing_ids)}"
        )

    logger.info(f"Final merged dataset: {len(merged)} subjects")
    return merged


def create_per_pnd_design(
    data: pd.DataFrame,
    pnd: str,
    output_dir: Path,
    tbss_dir: Path = None,
) -> None:
    """Create design files for a single PND timepoint."""
    subset = data[data['PND'] == pnd].copy().reset_index(drop=True)
    n = len(subset)
    if n == 0:
        logger.warning(f"No subjects for {pnd}, skipping")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Per-PND design: {pnd} (n={n})")
    logger.info(f"{'='*60}")

    # Prepare DataFrame for DesignHelper
    design_df = pd.DataFrame({
        'participant_id': subset['subject_key'].values,
        'dose': subset['dose'].values,
        'sex': subset['sex'].values,
    })

    helper = DesignHelper(design_df, subject_column='participant_id')
    helper.add_categorical('dose', coding='dummy', reference='C')
    helper.add_categorical('sex', coding='effect', reference='F')

    helper.build_design_matrix()
    cols = helper.design_column_names
    n_cols = len(cols)

    # Each dose level vs Control (bidirectional)
    # With dummy coding, each dose coefficient IS the difference from Control
    for dose_level in ['H', 'L', 'M']:
        col_name = f'dose_{dose_level}'
        idx = cols.index(col_name)
        pos_vec = [0.0] * n_cols
        pos_vec[idx] = 1.0
        neg_vec = [0.0] * n_cols
        neg_vec[idx] = -1.0
        helper.add_contrast(f'{dose_level}_gt_C', vector=pos_vec)
        helper.add_contrast(f'C_gt_{dose_level}', vector=neg_vec)

    helper.build_design_matrix()
    logger.info(f"Columns: {helper.design_column_names}")
    logger.info(helper.summary())

    # Save
    design_dir = output_dir / f'per_pnd_{pnd.lower()}'
    design_dir.mkdir(parents=True, exist_ok=True)

    helper.save(
        design_mat_file=design_dir / 'design.mat',
        design_con_file=design_dir / 'design.con',
        summary_file=design_dir / 'design_summary.json',
    )
    helper.write_description(design_dir / 'design_description.txt')

    # Save subject order for verification
    subset[['subject_key']].to_csv(
        design_dir / 'subject_order.txt', index=False, header=False
    )

    # Write provenance
    if tbss_dir is not None:
        write_provenance(design_dir, tbss_dir, n, f'per_pnd_{pnd.lower()}')

    logger.info(f"Saved to {design_dir}")


def create_pooled_design(
    data: pd.DataFrame,
    output_dir: Path,
    tbss_dir: Path = None,
) -> None:
    """Create pooled design with dose × PND interaction."""
    n = len(data)
    logger.info(f"\n{'='*60}")
    logger.info(f"Pooled design (n={n})")
    logger.info(f"{'='*60}")

    # Prepare DataFrame for DesignHelper
    design_df = pd.DataFrame({
        'participant_id': data['subject_key'].values,
        'dose': data['dose'].values,
        'PND': data['PND'].values,
        'sex': data['sex'].values,
    })

    helper = DesignHelper(design_df, subject_column='participant_id')
    helper.add_categorical('dose', coding='dummy', reference='C')
    helper.add_categorical('PND', coding='effect', reference='P30')
    helper.add_categorical('sex', coding='effect', reference='F')
    helper.add_interaction('dose', 'PND')

    helper.build_design_matrix()
    cols = helper.design_column_names
    n_cols = len(cols)

    # Main effect contrasts: each dose level vs Control (bidirectional)
    for dose_level in ['H', 'L', 'M']:
        col_name = f'dose_{dose_level}'
        idx = cols.index(col_name)
        pos_vec = [0.0] * n_cols
        pos_vec[idx] = 1.0
        neg_vec = [0.0] * n_cols
        neg_vec[idx] = -1.0
        helper.add_contrast(f'{dose_level}_gt_C', vector=pos_vec)
        helper.add_contrast(f'C_gt_{dose_level}', vector=neg_vec)

    # Interaction contrasts: does the dose-vs-control difference change across PND?
    for dose_level in ['H', 'L', 'M']:
        for pnd in ['P60', 'P90']:
            col_name = f'dose_{dose_level}×PND_{pnd}'
            idx = cols.index(col_name)
            pos_vec = [0.0] * n_cols
            pos_vec[idx] = 1.0
            neg_vec = [0.0] * n_cols
            neg_vec[idx] = -1.0
            helper.add_contrast(f'{dose_level}_x_{pnd}_pos', vector=pos_vec)
            helper.add_contrast(f'{dose_level}_x_{pnd}_neg', vector=neg_vec)
    logger.info(f"Columns: {helper.design_column_names}")
    logger.info(helper.summary())

    # Check matrix rank
    mat = helper.design_matrix
    rank = np.linalg.matrix_rank(mat)
    n_cols = mat.shape[1]
    if rank < n_cols:
        logger.warning(f"Design matrix is rank-deficient! rank={rank}, cols={n_cols}")
    else:
        logger.info(f"Design matrix is full rank ({rank}/{n_cols})")

    # Save
    design_dir = output_dir / 'pooled'
    design_dir.mkdir(parents=True, exist_ok=True)

    helper.save(
        design_mat_file=design_dir / 'design.mat',
        design_con_file=design_dir / 'design.con',
        summary_file=design_dir / 'design_summary.json',
    )
    helper.write_description(design_dir / 'design_description.txt')

    # Save subject order for verification
    data[['subject_key']].to_csv(
        design_dir / 'subject_order.txt', index=False, header=False
    )

    # Write provenance
    if tbss_dir is not None:
        write_provenance(design_dir, tbss_dir, n, 'pooled')

    logger.info(f"Saved to {design_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare FSL randomise design matrices for BPA-Rat TBSS'
    )
    parser.add_argument(
        '--study-tracker', type=Path, required=True,
        help='Path to study_tracker_combined CSV'
    )
    parser.add_argument(
        '--tbss-dir', type=Path, required=True,
        help='Path to TBSS directory (must contain subject_list.txt)'
    )
    parser.add_argument(
        '--output-dir', type=Path, required=True,
        help='Output directory for design files'
    )
    parser.add_argument(
        '--skip-subset', action='store_true',
        help='Skip pre-creating 4D subsets for each analysis'
    )
    args = parser.parse_args()

    # Load and merge data
    data = load_and_merge_data(args.study_tracker, args.tbss_dir)

    # Print distribution summary
    logger.info("\nData distribution:")
    for pnd in ['P30', 'P60', 'P90']:
        subset = data[data['PND'] == pnd]
        dose_dist = dict(subset['dose'].value_counts().sort_index())
        sex_dist = dict(subset['sex'].value_counts().sort_index())
        logger.info(f"  {pnd}: n={len(subset)}, dose={dose_dist}, sex={sex_dist}")

    # Create per-PND designs
    for pnd in ['P30', 'P60', 'P90']:
        create_per_pnd_design(data, pnd, args.output_dir, tbss_dir=args.tbss_dir)

    # Create pooled design
    create_pooled_design(data, args.output_dir, tbss_dir=args.tbss_dir)

    logger.info(f"\nAll designs saved to {args.output_dir}")

    # Pre-create 4D subsets
    if not args.skip_subset:
        design_dirs = sorted(args.output_dir.glob('per_pnd_*')) + \
                      [args.output_dir / 'pooled']
        design_dirs = [d for d in design_dirs if d.is_dir()]
        pre_subset_4d_volumes(args.tbss_dir, design_dirs)
    else:
        logger.info("\nSkipping 4D subset creation (--skip-subset)")


if __name__ == '__main__':
    main()
