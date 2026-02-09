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
import logging
import re
import sys
from pathlib import Path

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
    helper.add_categorical('dose', coding='effect', reference='C')
    helper.add_categorical('sex', coding='effect', reference='F')

    helper.build_design_matrix()
    cols = helper.design_column_names
    n_cols = len(cols)

    # Each dose level vs Control (bidirectional)
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

    # Save subject order for verification
    subset[['subject_key']].to_csv(
        design_dir / 'subject_order.txt', index=False, header=False
    )

    logger.info(f"Saved to {design_dir}")


def create_pooled_design(
    data: pd.DataFrame,
    output_dir: Path,
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
    helper.add_categorical('dose', coding='effect', reference='C')
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

    # Save subject order for verification
    data[['subject_key']].to_csv(
        design_dir / 'subject_order.txt', index=False, header=False
    )

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
        create_per_pnd_design(data, pnd, args.output_dir)

    # Create pooled design
    create_pooled_design(data, args.output_dir)

    logger.info(f"\nAll designs saved to {args.output_dir}")


if __name__ == '__main__':
    main()
