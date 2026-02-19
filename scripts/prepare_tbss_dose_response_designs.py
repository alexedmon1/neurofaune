#!/usr/bin/env python3
"""
Prepare FSL randomise design matrices for BPA-Rat TBSS dose-response analysis.

Dose is treated as an ordinal numeric variable (C=0, L=1, M=2, H=3) to test
for monotonic linear dose-response trends. This has more statistical power than
categorical (dummy-coded) designs when the true effect is monotonic (1 df for
dose trend vs 3 df for categorical dose).

Creates 4 sets of design files:
  - dose_response_p30: linear dose trend within P30 (dose + sex)
  - dose_response_p60: linear dose trend within P60 (dose + sex)
  - dose_response_p90: linear dose trend within P90 (dose + sex)
  - dose_response_pooled: dose trend + PND + sex + dose×PND interaction

Usage:
    uv run python scripts/prepare_tbss_dose_response_designs.py \
        --study-tracker /mnt/arborea/bpa-rat/study_tracker_combined_250916.csv \
        --tbss-dir /mnt/arborea/bpa-rat/analysis/tbss \
        --output-dir /mnt/arborea/bpa-rat/analysis/tbss/designs
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Reuse shared functions from the categorical design script
from prepare_tbss_designs import load_and_merge_data, write_provenance, pre_subset_4d_volumes

# Add neuroaider to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'neuroaider'))
from neuroaider import DesignHelper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ordinal dose mapping
DOSE_MAP = {'C': 0, 'L': 1, 'M': 2, 'H': 3}


def create_per_pnd_dose_design(data, pnd, output_dir, tbss_dir=None):
    """Create dose-response design for a single PND timepoint.

    Design columns: [Intercept, dose (mean-centered), sex_M (effect)]
    Contrasts: dose_pos (+1 on dose), dose_neg (-1 on dose)
    """
    subset = data[data['PND'] == pnd].copy().reset_index(drop=True)
    n = len(subset)
    if n == 0:
        logger.warning(f"No subjects for {pnd}, skipping")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Dose-response design: {pnd} (n={n})")
    logger.info(f"{'='*60}")

    # Map dose to ordinal numeric
    subset['dose_numeric'] = subset['dose'].map(DOSE_MAP)

    # Log distribution
    for dose_label in ['C', 'L', 'M', 'H']:
        count = (subset['dose'] == dose_label).sum()
        logger.info(f"  {dose_label} (={DOSE_MAP[dose_label]}): n={count}")

    import pandas as pd
    design_df = pd.DataFrame({
        'participant_id': subset['subject_key'].values,
        'dose_numeric': subset['dose_numeric'].values,
        'sex': subset['sex'].values,
    })

    helper = DesignHelper(design_df, subject_column='participant_id')
    helper.add_covariate('dose_numeric', mean_center=True)
    helper.add_categorical('sex', coding='effect', reference='F')

    helper.build_design_matrix()
    logger.info(f"Columns: {helper.design_column_names}")

    # Contrasts: positive and negative dose trend
    helper.add_contrast('dose_pos', covariate='dose_numeric', direction='+')
    helper.add_contrast('dose_neg', covariate='dose_numeric', direction='-')

    logger.info(helper.summary())

    # Save
    design_dir = output_dir / f'dose_response_{pnd.lower()}'
    design_dir.mkdir(parents=True, exist_ok=True)

    helper.save(
        design_mat_file=design_dir / 'design.mat',
        design_con_file=design_dir / 'design.con',
        summary_file=design_dir / 'design_summary.json',
    )
    helper.write_description(design_dir / 'design_description.txt')

    subset[['subject_key']].to_csv(
        design_dir / 'subject_order.txt', index=False, header=False
    )

    # Write provenance
    if tbss_dir is not None:
        write_provenance(design_dir, tbss_dir, n, f'dose_response_{pnd.lower()}')

    # Rank check
    mat = helper.design_matrix
    rank = np.linalg.matrix_rank(mat)
    n_cols = mat.shape[1]
    if rank < n_cols:
        logger.warning(f"Design matrix is rank-deficient! rank={rank}, cols={n_cols}")
    else:
        logger.info(f"Design matrix is full rank ({rank}/{n_cols})")

    logger.info(f"Saved to {design_dir}")


def create_pooled_dose_design(data, output_dir, tbss_dir=None):
    """Create pooled dose-response design with dose × PND interaction.

    Design columns: [Intercept, dose_numeric (centered), PND_P60, PND_P90,
                     sex_M, dose_numeric×PND_P60, dose_numeric×PND_P90]
    Contrasts:
      - dose_pos / dose_neg: main linear dose trend
      - dose_x_P60_pos / dose_x_P60_neg: dose slope differs at P60 vs P30?
      - dose_x_P90_pos / dose_x_P90_neg: dose slope differs at P90 vs P30?
    """
    n = len(data)
    logger.info(f"\n{'='*60}")
    logger.info(f"Pooled dose-response design (n={n})")
    logger.info(f"{'='*60}")

    data = data.copy()
    data['dose_numeric'] = data['dose'].map(DOSE_MAP)

    # Log distribution
    for pnd in ['P30', 'P60', 'P90']:
        subset = data[data['PND'] == pnd]
        dose_dist = {d: (subset['dose'] == d).sum() for d in ['C', 'L', 'M', 'H']}
        logger.info(f"  {pnd}: n={len(subset)}, dose={dose_dist}")

    import pandas as pd
    design_df = pd.DataFrame({
        'participant_id': data['subject_key'].values,
        'dose_numeric': data['dose_numeric'].values,
        'PND': data['PND'].values,
        'sex': data['sex'].values,
    })

    helper = DesignHelper(design_df, subject_column='participant_id')
    helper.add_covariate('dose_numeric', mean_center=True)
    helper.add_categorical('PND', coding='effect', reference='P30')
    helper.add_categorical('sex', coding='effect', reference='F')
    helper.add_interaction('dose_numeric', 'PND')

    helper.build_design_matrix()
    logger.info(f"Columns: {helper.design_column_names}")

    # Main dose trend
    helper.add_contrast('dose_pos', covariate='dose_numeric', direction='+')
    helper.add_contrast('dose_neg', covariate='dose_numeric', direction='-')

    # Dose × PND interactions
    helper.add_contrast(
        'dose_x_P60_pos', interaction='dose_numeric×PND', level='P60', direction='+'
    )
    helper.add_contrast(
        'dose_x_P60_neg', interaction='dose_numeric×PND', level='P60', direction='-'
    )
    helper.add_contrast(
        'dose_x_P90_pos', interaction='dose_numeric×PND', level='P90', direction='+'
    )
    helper.add_contrast(
        'dose_x_P90_neg', interaction='dose_numeric×PND', level='P90', direction='-'
    )

    logger.info(helper.summary())

    # Rank check
    mat = helper.design_matrix
    rank = np.linalg.matrix_rank(mat)
    n_cols = mat.shape[1]
    if rank < n_cols:
        logger.warning(f"Design matrix is rank-deficient! rank={rank}, cols={n_cols}")
    else:
        logger.info(f"Design matrix is full rank ({rank}/{n_cols})")

    # Save
    design_dir = output_dir / 'dose_response_pooled'
    design_dir.mkdir(parents=True, exist_ok=True)

    helper.save(
        design_mat_file=design_dir / 'design.mat',
        design_con_file=design_dir / 'design.con',
        summary_file=design_dir / 'design_summary.json',
    )
    helper.write_description(design_dir / 'design_description.txt')

    data[['subject_key']].to_csv(
        design_dir / 'subject_order.txt', index=False, header=False
    )

    # Write provenance
    if tbss_dir is not None:
        write_provenance(design_dir, tbss_dir, n, 'dose_response_pooled')

    logger.info(f"Saved to {design_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare FSL randomise designs for ordinal dose-response TBSS analysis'
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

    # Load and merge data (same function as categorical design script)
    data = load_and_merge_data(args.study_tracker, args.tbss_dir)

    # Print distribution summary
    logger.info("\nData distribution:")
    for pnd in ['P30', 'P60', 'P90']:
        subset = data[data['PND'] == pnd]
        dose_dist = dict(subset['dose'].value_counts().sort_index())
        sex_dist = dict(subset['sex'].value_counts().sort_index())
        logger.info(f"  {pnd}: n={len(subset)}, dose={dose_dist}, sex={sex_dist}")

    # Create per-PND dose-response designs
    for pnd in ['P30', 'P60', 'P90']:
        create_per_pnd_dose_design(data, pnd, args.output_dir, tbss_dir=args.tbss_dir)

    # Create pooled dose-response design
    create_pooled_dose_design(data, args.output_dir, tbss_dir=args.tbss_dir)

    logger.info(f"\nAll dose-response designs saved to {args.output_dir}")

    # Pre-create 4D subsets
    if not args.skip_subset:
        design_dirs = sorted(args.output_dir.glob('dose_response_*'))
        design_dirs = [d for d in design_dirs if d.is_dir()]
        pre_subset_4d_volumes(args.tbss_dir, design_dirs)
    else:
        logger.info("\nSkipping 4D subset creation (--skip-subset)")


if __name__ == '__main__':
    main()
