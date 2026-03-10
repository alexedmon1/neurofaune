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
  - dose_response_pooled: dose trend + PND + sex + dose\u00d7PND interaction

For non-dose targets (e.g. --target auc --target-csv auc_lookup.csv):
  - {target}_response_p30/p60/p90: per-PND continuous target designs
  - {target}_response_pooled: pooled with target \u00d7 PND interaction

Usage:
    uv run python scripts/prepare_tbss_dose_response_designs.py \\
        --study-tracker $STUDY_ROOT/study_tracker_combined_250916.csv \\
        --tbss-dir $STUDY_ROOT/analysis/tbss/dwi \\
        --output-dir $STUDY_ROOT/analysis/tbss/dwi/designs

    # With a custom target column:
    uv run python scripts/prepare_tbss_dose_response_designs.py \\
        --study-tracker ... --tbss-dir ... --output-dir ... \\
        --target auc --target-csv auc_lookup.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


def _load_target_lookup(csv_path: Path, column_name: str) -> pd.DataFrame:
    """Load a target lookup CSV (subject, session, <column_name>).

    Parameters
    ----------
    csv_path : Path
        CSV with at least subject, session, and *column_name* columns.
    column_name : str
        Name of the target column to use.
    """
    df = pd.read_csv(csv_path)
    if not {'subject', 'session'}.issubset(df.columns):
        raise ValueError(
            f"Target CSV must have 'subject' and 'session' columns. "
            f"Found: {list(df.columns)}"
        )
    if column_name not in df.columns:
        raise ValueError(
            f"Target column {column_name!r} not found in CSV. "
            f"Available: {[c for c in df.columns if c not in ('subject', 'session')]}"
        )
    logger.info("Loaded target lookup (%s): %d rows from %s", column_name, len(df), csv_path)
    return df


def _merge_target(data: pd.DataFrame, lookup_df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Merge target values into subject data.

    Creates a column named *column_name* by matching subject+session keys.
    Rows with missing values are dropped.
    """
    lookup_df = lookup_df.copy()
    lookup_df['_merge_key'] = lookup_df['subject'] + '_' + lookup_df['session']
    value_lookup = dict(zip(lookup_df['_merge_key'], lookup_df[column_name]))

    data = data.copy()
    data[column_name] = data['subject_key'].map(value_lookup)

    n_missing = data[column_name].isna().sum()
    if n_missing > 0:
        logger.warning("Dropping %d subjects with no %s match", n_missing, column_name)
        data = data.dropna(subset=[column_name]).reset_index(drop=True)

    return data


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
    """Create pooled dose-response design with dose \u00d7 PND interaction.

    Design columns: [Intercept, dose_numeric (centered), PND_P60, PND_P90,
                     sex_M, dose_numeric\u00d7PND_P60, dose_numeric\u00d7PND_P90]
    Contrasts:
      - dose_pos / dose_neg: main linear dose trend
      - dose_x_P60_pos / dose_x_P60_neg: dose slope differs at P60 vs P30?
      - dose_x_P90_pos / dose_x_P90_neg: dose slope differs at P90 vs P30?
    """
    n = len(data)
    n_pnds = data['PND'].nunique()
    if n_pnds < 2:
        logger.info(f"\nSkipping pooled dose-response design: only {n_pnds} PND level(s) present")
        return

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

    # Dose \u00d7 PND interactions
    helper.add_contrast(
        'dose_x_P60_pos', interaction='dose_numeric\u00d7PND', level='P60', direction='+'
    )
    helper.add_contrast(
        'dose_x_P60_neg', interaction='dose_numeric\u00d7PND', level='P60', direction='-'
    )
    helper.add_contrast(
        'dose_x_P90_pos', interaction='dose_numeric\u00d7PND', level='P90', direction='+'
    )
    helper.add_contrast(
        'dose_x_P90_neg', interaction='dose_numeric\u00d7PND', level='P90', direction='-'
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


def create_per_pnd_target_design(data, pnd, target_name, output_dir, tbss_dir=None):
    """Create continuous-target response design for a single PND timepoint.

    Design columns: [Intercept, <target> (mean-centered), sex_M (effect)]
    Contrasts: <target>_pos, <target>_neg
    """
    subset = data[data['PND'] == pnd].copy().reset_index(drop=True)
    n = len(subset)
    if n == 0:
        logger.warning(f"No subjects for {pnd}, skipping {target_name} design")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"{target_name}-response design: {pnd} (n={n})")
    logger.info(f"{'='*60}")

    logger.info(f"  {target_name} range: [{subset[target_name].min():.2f}, {subset[target_name].max():.2f}]")
    for dose_label in ['C', 'L', 'M', 'H']:
        vals = subset.loc[subset['dose'] == dose_label, target_name]
        if len(vals) > 0:
            logger.info(f"  {dose_label}: n={len(vals)}, {target_name} mean={vals.mean():.2f}")

    design_df = pd.DataFrame({
        'participant_id': subset['subject_key'].values,
        target_name: subset[target_name].values,
        'sex': subset['sex'].values,
    })

    helper = DesignHelper(design_df, subject_column='participant_id')
    helper.add_covariate(target_name, mean_center=True)
    helper.add_categorical('sex', coding='effect', reference='F')

    helper.build_design_matrix()
    logger.info(f"Columns: {helper.design_column_names}")

    helper.add_contrast(f'{target_name}_pos', covariate=target_name, direction='+')
    helper.add_contrast(f'{target_name}_neg', covariate=target_name, direction='-')

    logger.info(helper.summary())

    design_dir = output_dir / f'{target_name}_response_{pnd.lower()}'
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

    if tbss_dir is not None:
        write_provenance(design_dir, tbss_dir, n, f'{target_name}_response_{pnd.lower()}')

    # Rank check
    mat = helper.design_matrix
    rank = np.linalg.matrix_rank(mat)
    n_cols = mat.shape[1]
    if rank < n_cols:
        logger.warning(f"Design matrix is rank-deficient! rank={rank}, cols={n_cols}")
    else:
        logger.info(f"Design matrix is full rank ({rank}/{n_cols})")

    logger.info(f"Saved to {design_dir}")


def create_pooled_target_design(data, target_name, output_dir, tbss_dir=None):
    """Create pooled continuous-target response design with target \u00d7 PND interaction.

    Design columns: [Intercept, <target> (centered), PND_P60, PND_P90,
                     sex_M, <target>\u00d7PND_P60, <target>\u00d7PND_P90]
    """
    n = len(data)
    n_pnds = data['PND'].nunique()
    if n_pnds < 2:
        logger.info(f"\nSkipping pooled {target_name}-response design: only {n_pnds} PND level(s) present")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Pooled {target_name}-response design (n={n})")
    logger.info(f"{'='*60}")

    data = data.copy()

    for pnd in ['P30', 'P60', 'P90']:
        subset = data[data['PND'] == pnd]
        logger.info(
            f"  {pnd}: n={len(subset)}, {target_name} range=[{subset[target_name].min():.2f}, "
            f"{subset[target_name].max():.2f}], mean={subset[target_name].mean():.2f}"
        )

    design_df = pd.DataFrame({
        'participant_id': data['subject_key'].values,
        target_name: data[target_name].values,
        'PND': data['PND'].values,
        'sex': data['sex'].values,
    })

    helper = DesignHelper(design_df, subject_column='participant_id')
    helper.add_covariate(target_name, mean_center=True)
    helper.add_categorical('PND', coding='effect', reference='P30')
    helper.add_categorical('sex', coding='effect', reference='F')
    helper.add_interaction(target_name, 'PND')

    helper.build_design_matrix()
    logger.info(f"Columns: {helper.design_column_names}")

    interaction_name = f'{target_name}\u00d7PND'
    helper.add_contrast(f'{target_name}_pos', covariate=target_name, direction='+')
    helper.add_contrast(f'{target_name}_neg', covariate=target_name, direction='-')
    helper.add_contrast(f'{target_name}_x_P60_pos', interaction=interaction_name, level='P60', direction='+')
    helper.add_contrast(f'{target_name}_x_P60_neg', interaction=interaction_name, level='P60', direction='-')
    helper.add_contrast(f'{target_name}_x_P90_pos', interaction=interaction_name, level='P90', direction='+')
    helper.add_contrast(f'{target_name}_x_P90_neg', interaction=interaction_name, level='P90', direction='-')

    logger.info(helper.summary())

    # Rank check
    mat = helper.design_matrix
    rank = np.linalg.matrix_rank(mat)
    n_cols = mat.shape[1]
    if rank < n_cols:
        logger.warning(f"Design matrix is rank-deficient! rank={rank}, cols={n_cols}")
    else:
        logger.info(f"Design matrix is full rank ({rank}/{n_cols})")

    design_dir = output_dir / f'{target_name}_response_pooled'
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

    if tbss_dir is not None:
        write_provenance(design_dir, tbss_dir, n, f'{target_name}_response_pooled')

    logger.info(f"Saved to {design_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare FSL randomise designs for dose-response TBSS analysis'
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
        '--target', type=str, default='dose',
        help="Target variable: 'dose' (ordinal C=0..H=3) or any column name from --target-csv",
    )
    parser.add_argument(
        '--target-csv', type=Path, default=None,
        help='Path to target lookup CSV with subject, session, <target> columns (required for non-dose targets)',
    )
    parser.add_argument(
        '--auc-csv', type=Path, default=None,
        help='Deprecated alias for --target-csv (backward compatibility)',
    )
    parser.add_argument(
        '--skip-subset', action='store_true',
        help='Skip pre-creating 4D subsets for each analysis'
    )
    args = parser.parse_args()

    # Handle deprecated --auc-csv alias
    if args.auc_csv is not None and args.target_csv is None:
        args.target_csv = args.auc_csv
        logger.warning("--auc-csv is deprecated, use --target-csv instead")

    if args.target != 'dose' and args.target_csv is None:
        parser.error(f"--target-csv is required when --target is '{args.target}'")

    # Load and merge data (same function as categorical design script)
    data = load_and_merge_data(args.study_tracker, args.tbss_dir)

    # Print distribution summary
    logger.info("\nData distribution:")
    for pnd in ['P30', 'P60', 'P90']:
        subset = data[data['PND'] == pnd]
        dose_dist = dict(subset['dose'].value_counts().sort_index())
        sex_dist = dict(subset['sex'].value_counts().sort_index())
        logger.info(f"  {pnd}: n={len(subset)}, dose={dose_dist}, sex={sex_dist}")

    if args.target == 'dose':
        # Create per-PND dose-response designs
        for pnd in ['P30', 'P60', 'P90']:
            create_per_pnd_dose_design(data, pnd, args.output_dir, tbss_dir=args.tbss_dir)

        # Create pooled dose-response design
        create_pooled_dose_design(data, args.output_dir, tbss_dir=args.tbss_dir)

        design_pattern = 'dose_response_*'

    else:
        # Generic target: merge from lookup CSV
        target_name = args.target
        target_df = _load_target_lookup(args.target_csv, target_name)
        data_target = _merge_target(data, target_df, target_name)

        logger.info(f"\n{target_name} distribution after merge:")
        for pnd in ['P30', 'P60', 'P90']:
            subset = data_target[data_target['PND'] == pnd]
            if len(subset) > 0:
                logger.info(
                    f"  {pnd}: n={len(subset)}, {target_name} range=[{subset[target_name].min():.2f}, "
                    f"{subset[target_name].max():.2f}]"
                )

        # Create per-PND target designs
        for pnd in ['P30', 'P60', 'P90']:
            create_per_pnd_target_design(data_target, pnd, target_name, args.output_dir, tbss_dir=args.tbss_dir)

        # Create pooled target design
        create_pooled_target_design(data_target, target_name, args.output_dir, tbss_dir=args.tbss_dir)

        design_pattern = f'{target_name}_response_*'

    logger.info(f"\nAll {args.target}-response designs saved to {args.output_dir}")

    # Pre-create 4D subsets
    if not args.skip_subset:
        design_dirs = sorted(args.output_dir.glob(design_pattern))
        design_dirs = [d for d in design_dirs if d.is_dir()]
        pre_subset_4d_volumes(args.tbss_dir, design_dirs)
    else:
        logger.info("\nSkipping 4D subset creation (--skip-subset)")


if __name__ == '__main__':
    main()
