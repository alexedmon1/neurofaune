#!/usr/bin/env python3
"""
Prepare design matrices for MVPA analysis of SIGMA-space DTI metrics.

Creates both categorical (dose group classification) and dose-response
(ordinal regression) designs for whole-brain decoding. Discovers
SIGMA-space NIfTIs from the derivatives tree and builds NeuroAider
design matrices per-cohort and pooled.

Design sets created:
  - per_pnd_p30, per_pnd_p60, per_pnd_p90: within-cohort dose (+ sex)
  - pooled: all subjects with dose + PND + sex + dose x PND
  - dose_response_p30/p60/p90: ordinal dose trend within cohort
  - dose_response_pooled: ordinal dose + PND interaction

Usage:
    uv run python scripts/prepare_mvpa_designs.py \
        --study-tracker /mnt/arborea/bpa-rat/study_tracker_combined_250916.csv \
        --derivatives-root /mnt/arborea/bpa-rat/derivatives \
        --output-dir /mnt/arborea/bpa-rat/analysis/mvpa/designs \
        --metrics FA MD AD RD
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'neuroaider'))

from neuroaider import DesignHelper
from neurofaune.analysis.mvpa.data_loader import discover_sigma_images

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DOSE_MAP = {'C': 0, 'L': 1, 'M': 2, 'H': 3}


def write_provenance(
    design_dir: Path,
    subject_list_path: Path,
    n_subjects: int,
    design_type: str,
) -> None:
    """Write provenance.json linking design to its subject list."""
    h = hashlib.sha256()
    with open(subject_list_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    subject_list_hash = h.hexdigest()

    provenance = {
        'subject_list_path': str(subject_list_path),
        'subject_list_sha256': subject_list_hash,
        'n_subjects': n_subjects,
        'design_type': design_type,
        'date_created': datetime.now().isoformat(),
    }
    with open(design_dir / 'provenance.json', 'w') as f:
        json.dump(provenance, f, indent=2)
    logger.info(f"  Wrote provenance.json (hash: {subject_list_hash[:16]}...)")


def discover_and_merge(
    study_tracker_path: Path,
    derivatives_root: Path,
    metrics: list,
    exclusion_csv: Path = None,
) -> pd.DataFrame:
    """Discover SIGMA-space images and merge with study tracker metadata.

    Args:
        study_tracker_path: Path to study tracker CSV.
        derivatives_root: Path to derivatives directory.
        metrics: List of DTI metrics to check.
        exclusion_csv: Optional CSV with subject,session columns to exclude.

    Returns:
        DataFrame with columns: subject_key, subject, session, PND, sex, dose.
    """
    # Discover subjects that have all requested metrics in SIGMA space
    all_subjects = {}
    for metric in metrics:
        images = discover_sigma_images(derivatives_root, metric)
        for info in images:
            key = f"{info['subject']}_{info['session']}"
            if key not in all_subjects:
                all_subjects[key] = {
                    'subject': info['subject'],
                    'session': info['session'],
                    'cohort': info['cohort'],
                    'metrics': set(),
                }
            all_subjects[key]['metrics'].add(metric)

    # Filter to subjects with all metrics
    complete = {
        k: v for k, v in all_subjects.items()
        if v['metrics'] == set(metrics)
    }
    logger.info(
        "Discovered %d subjects with all %d metrics in SIGMA space",
        len(complete), len(metrics),
    )

    # Load exclusion list
    excluded_ids = set()
    if exclusion_csv and exclusion_csv.exists():
        excl_df = pd.read_csv(exclusion_csv)
        for _, row in excl_df.iterrows():
            excluded_ids.add(f"{row['subject']}_{row['session']}")
        logger.info("Loaded %d exclusions from %s", len(excluded_ids), exclusion_csv)

    # Load study tracker for dose/sex metadata
    tracker = pd.read_csv(study_tracker_path)
    valid = tracker[tracker['irc.ID'].notna()].copy()
    valid['bids_id'] = 'sub-' + valid['irc.ID']
    tracker_lookup = {}
    for _, row in valid.iterrows():
        tracker_lookup[row['bids_id']] = {
            'sex': row['sex'],
            'dose': row['dose.level'],
        }

    # Build merged dataset
    rows = []
    for key, info in sorted(complete.items()):
        if key in excluded_ids:
            continue
        subject = info['subject']
        session = info['session']
        cohort = info['cohort']

        meta = tracker_lookup.get(subject)
        if meta is None:
            logger.warning("Subject %s not found in tracker, skipping", subject)
            continue

        rows.append({
            'subject_key': key,
            'subject': subject,
            'session': session,
            'PND': cohort.upper(),
            'sex': meta['sex'],
            'dose': meta['dose'],
        })

    df = pd.DataFrame(rows)
    logger.info("Final merged dataset: %d subjects", len(df))
    return df


def create_per_pnd_categorical(data, pnd, output_dir, subject_list_path):
    """Create categorical design for a single PND."""
    subset = data[data['PND'] == pnd].copy().reset_index(drop=True)
    n = len(subset)
    if n == 0:
        logger.warning("No subjects for %s, skipping categorical", pnd)
        return

    logger.info("\n%s\nCategorical design: %s (n=%d)\n%s", "=" * 60, pnd, n, "=" * 60)

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

    for dose_level in ['H', 'L', 'M']:
        col_name = f'dose_{dose_level}'
        idx = cols.index(col_name)
        pos_vec = [0.0] * n_cols
        pos_vec[idx] = 1.0
        neg_vec = [0.0] * n_cols
        neg_vec[idx] = -1.0
        helper.add_contrast(f'{dose_level}_gt_C', vector=pos_vec)
        helper.add_contrast(f'C_gt_{dose_level}', vector=neg_vec)

    logger.info(helper.summary())

    design_dir = output_dir / f'per_pnd_{pnd.lower()}'
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
    write_provenance(design_dir, subject_list_path, n, f'per_pnd_{pnd.lower()}')
    logger.info("Saved categorical design to %s", design_dir)


def create_pooled_categorical(data, output_dir, subject_list_path):
    """Create pooled categorical design with dose x PND interaction."""
    n = len(data)
    logger.info("\n%s\nPooled categorical design (n=%d)\n%s", "=" * 60, n, "=" * 60)

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

    for dose_level in ['H', 'L', 'M']:
        col_name = f'dose_{dose_level}'
        idx = cols.index(col_name)
        pos_vec = [0.0] * n_cols
        pos_vec[idx] = 1.0
        neg_vec = [0.0] * n_cols
        neg_vec[idx] = -1.0
        helper.add_contrast(f'{dose_level}_gt_C', vector=pos_vec)
        helper.add_contrast(f'C_gt_{dose_level}', vector=neg_vec)

    for dose_level in ['H', 'L', 'M']:
        for pnd in ['P60', 'P90']:
            col_name = f'dose_{dose_level}\u00d7PND_{pnd}'
            idx = cols.index(col_name)
            pos_vec = [0.0] * n_cols
            pos_vec[idx] = 1.0
            neg_vec = [0.0] * n_cols
            neg_vec[idx] = -1.0
            helper.add_contrast(f'{dose_level}_x_{pnd}_pos', vector=pos_vec)
            helper.add_contrast(f'{dose_level}_x_{pnd}_neg', vector=neg_vec)

    logger.info(helper.summary())

    design_dir = output_dir / 'pooled'
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
    write_provenance(design_dir, subject_list_path, n, 'pooled')
    logger.info("Saved pooled categorical design to %s", design_dir)


def create_per_pnd_dose_response(data, pnd, output_dir, subject_list_path):
    """Create dose-response design for a single PND."""
    subset = data[data['PND'] == pnd].copy().reset_index(drop=True)
    n = len(subset)
    if n == 0:
        logger.warning("No subjects for %s, skipping dose-response", pnd)
        return

    logger.info(
        "\n%s\nDose-response design: %s (n=%d)\n%s", "=" * 60, pnd, n, "=" * 60
    )

    subset['dose_numeric'] = subset['dose'].map(DOSE_MAP)

    design_df = pd.DataFrame({
        'participant_id': subset['subject_key'].values,
        'dose_numeric': subset['dose_numeric'].values,
        'sex': subset['sex'].values,
    })

    helper = DesignHelper(design_df, subject_column='participant_id')
    helper.add_covariate('dose_numeric', mean_center=True)
    helper.add_categorical('sex', coding='effect', reference='F')
    helper.build_design_matrix()

    helper.add_contrast('dose_pos', covariate='dose_numeric', direction='+')
    helper.add_contrast('dose_neg', covariate='dose_numeric', direction='-')

    logger.info(helper.summary())

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
    write_provenance(design_dir, subject_list_path, n, f'dose_response_{pnd.lower()}')
    logger.info("Saved dose-response design to %s", design_dir)


def create_pooled_dose_response(data, output_dir, subject_list_path):
    """Create pooled dose-response design with dose x PND interaction."""
    n = len(data)
    logger.info(
        "\n%s\nPooled dose-response design (n=%d)\n%s", "=" * 60, n, "=" * 60
    )

    data = data.copy()
    data['dose_numeric'] = data['dose'].map(DOSE_MAP)

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

    helper.add_contrast('dose_pos', covariate='dose_numeric', direction='+')
    helper.add_contrast('dose_neg', covariate='dose_numeric', direction='-')

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
    write_provenance(design_dir, subject_list_path, n, 'dose_response_pooled')
    logger.info("Saved pooled dose-response design to %s", design_dir)


def _load_auc_lookup(auc_csv_path: Path) -> pd.DataFrame:
    """Load AUC lookup CSV (subject, session, auc)."""
    auc_df = pd.read_csv(auc_csv_path)
    if not {'subject', 'session', 'auc'}.issubset(auc_df.columns):
        raise ValueError(
            f"AUC CSV must have columns: subject, session, auc. "
            f"Found: {list(auc_df.columns)}"
        )
    logger.info("Loaded AUC lookup: %d rows from %s", len(auc_df), auc_csv_path)
    return auc_df


def _merge_auc(data: pd.DataFrame, auc_df: pd.DataFrame) -> pd.DataFrame:
    """Merge AUC values into MVPA subject data."""
    auc_df = auc_df.copy()
    auc_df['_merge_key'] = auc_df['subject'] + '_' + auc_df['session']
    auc_lookup = dict(zip(auc_df['_merge_key'], auc_df['auc']))

    data = data.copy()
    data['auc'] = data['subject_key'].map(auc_lookup)

    n_missing = data['auc'].isna().sum()
    if n_missing > 0:
        logger.warning("Dropping %d subjects with no AUC match", n_missing)
        data = data.dropna(subset=['auc']).reset_index(drop=True)

    return data


def create_per_pnd_auc_response(data, pnd, output_dir, subject_list_path):
    """Create AUC-response design for a single PND."""
    subset = data[data['PND'] == pnd].copy().reset_index(drop=True)
    n = len(subset)
    if n == 0:
        logger.warning("No subjects for %s, skipping AUC-response", pnd)
        return

    logger.info(
        "\n%s\nAUC-response design: %s (n=%d)\n%s", "=" * 60, pnd, n, "=" * 60
    )

    design_df = pd.DataFrame({
        'participant_id': subset['subject_key'].values,
        'auc': subset['auc'].values,
        'sex': subset['sex'].values,
    })

    helper = DesignHelper(design_df, subject_column='participant_id')
    helper.add_covariate('auc', mean_center=True)
    helper.add_categorical('sex', coding='effect', reference='F')
    helper.build_design_matrix()

    helper.add_contrast('auc_pos', covariate='auc', direction='+')
    helper.add_contrast('auc_neg', covariate='auc', direction='-')

    logger.info(helper.summary())

    design_dir = output_dir / f'auc_response_{pnd.lower()}'
    design_dir.mkdir(parents=True, exist_ok=True)

    helper.save(
        design_mat_file=design_dir / 'design.mat',
        design_con_file=design_dir / 'design.con',
        summary_file=design_dir / 'design_summary.json',
    )
    helper.write_description(design_dir / 'design_description.txt')

    # Save target values for load_design() to use directly
    import json
    target_vals = dict(zip(subset['subject_key'].values, subset['auc'].values))
    with open(design_dir / 'target_values.json', 'w') as f:
        json.dump(target_vals, f, indent=2)

    subset[['subject_key']].to_csv(
        design_dir / 'subject_order.txt', index=False, header=False
    )
    write_provenance(design_dir, subject_list_path, n, f'auc_response_{pnd.lower()}')
    logger.info("Saved AUC-response design to %s", design_dir)


def create_pooled_auc_response(data, output_dir, subject_list_path):
    """Create pooled AUC-response design with auc x PND interaction."""
    n = len(data)
    logger.info(
        "\n%s\nPooled AUC-response design (n=%d)\n%s", "=" * 60, n, "=" * 60
    )

    data = data.copy()

    design_df = pd.DataFrame({
        'participant_id': data['subject_key'].values,
        'auc': data['auc'].values,
        'PND': data['PND'].values,
        'sex': data['sex'].values,
    })

    helper = DesignHelper(design_df, subject_column='participant_id')
    helper.add_covariate('auc', mean_center=True)
    helper.add_categorical('PND', coding='effect', reference='P30')
    helper.add_categorical('sex', coding='effect', reference='F')
    helper.add_interaction('auc', 'PND')
    helper.build_design_matrix()

    helper.add_contrast('auc_pos', covariate='auc', direction='+')
    helper.add_contrast('auc_neg', covariate='auc', direction='-')

    helper.add_contrast(
        'auc_x_P60_pos', interaction='auc\u00d7PND', level='P60', direction='+'
    )
    helper.add_contrast(
        'auc_x_P60_neg', interaction='auc\u00d7PND', level='P60', direction='-'
    )
    helper.add_contrast(
        'auc_x_P90_pos', interaction='auc\u00d7PND', level='P90', direction='+'
    )
    helper.add_contrast(
        'auc_x_P90_neg', interaction='auc\u00d7PND', level='P90', direction='-'
    )

    logger.info(helper.summary())

    design_dir = output_dir / 'auc_response_pooled'
    design_dir.mkdir(parents=True, exist_ok=True)

    helper.save(
        design_mat_file=design_dir / 'design.mat',
        design_con_file=design_dir / 'design.con',
        summary_file=design_dir / 'design_summary.json',
    )
    helper.write_description(design_dir / 'design_description.txt')

    # Save target values for load_design()
    import json
    target_vals = dict(zip(data['subject_key'].values, data['auc'].values))
    with open(design_dir / 'target_values.json', 'w') as f:
        json.dump(target_vals, f, indent=2)

    data[['subject_key']].to_csv(
        design_dir / 'subject_order.txt', index=False, header=False
    )
    write_provenance(design_dir, subject_list_path, n, 'auc_response_pooled')
    logger.info("Saved pooled AUC-response design to %s", design_dir)


def write_design_description(args, output_path):
    """Write a human-readable description of the MVPA design preparation."""
    lines = [
        "ANALYSIS DESCRIPTION",
        "====================",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Analysis: MVPA (Multi-Voxel Pattern Analysis) Design Preparation",
        "",
        "DATA SOURCE",
        "-----------",
        f"Derivatives root: {args.derivatives_root}",
        f"Study tracker: {args.study_tracker}",
        f"Exclusion list: {args.exclusion_csv or 'None'}",
        f"Metrics checked: {', '.join(args.metrics)}",
        "",
        "DESIGN TYPES",
        "------------",
        "1. Categorical (per-PND + pooled):",
        "   - Dose as factor: C (reference), L, M, H",
        "   - Dummy coding, sex as nuisance (effect coding)",
        "   - Pooled includes dose x PND interaction",
        "",
        "2. Dose-response (per-PND + pooled):",
        "   - Dose as ordinal: C=0, L=1, M=2, H=3 (mean-centered)",
        "   - Sex as nuisance (effect coding)",
        "   - Pooled includes dose_numeric x PND interaction",
        "",
        "CONTRASTS",
        "---------",
        "Categorical: H>C, C>H, L>C, C>L, M>C, C>M",
        "  (pooled adds dose x PND interaction contrasts)",
        "Dose-response: dose_pos, dose_neg",
        "  (pooled adds dose x P60 and dose x P90 interactions)",
        "",
        "SUBJECT SELECTION",
        "-----------------",
        "Subjects must have all requested metrics in SIGMA space:",
        f"  Pattern: sub-*/ses-*/dwi/*_space-SIGMA_{{metric}}.nii.gz",
        "  ses-unknown sessions excluded",
    ]
    output_path.write_text("\n".join(lines))
    logger.info("Saved design description: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare design matrices for MVPA analysis of SIGMA-space DTI metrics'
    )
    parser.add_argument(
        '--study-tracker', type=Path, required=True,
        help='Path to study_tracker_combined CSV'
    )
    parser.add_argument(
        '--derivatives-root', type=Path, required=True,
        help='Path to derivatives directory with SIGMA-space NIfTIs'
    )
    parser.add_argument(
        '--output-dir', type=Path, required=True,
        help='Output directory for design files'
    )
    parser.add_argument(
        '--metrics', nargs='+', default=['FA', 'MD', 'AD', 'RD'],
        help='DTI metrics to check for subject inclusion (default: FA MD AD RD)'
    )
    parser.add_argument(
        '--exclusion-csv', type=Path, default=None,
        help='CSV of sessions to exclude (must have subject, session columns)'
    )
    parser.add_argument(
        '--target', choices=['dose', 'auc'], default='dose',
        help="Target variable: 'dose' (categorical + ordinal) or 'auc' (continuous AUC)",
    )
    parser.add_argument(
        '--auc-csv', type=Path, default=None,
        help='Path to AUC lookup CSV (required when --target auc)',
    )
    parser.add_argument(
        '--cohorts', nargs='+', default=['P30', 'P60', 'P90'],
        help='Cohorts to create designs for (default: P30 P60 P90)'
    )
    args = parser.parse_args()

    if args.target == 'auc' and args.auc_csv is None:
        parser.error("--auc-csv is required when --target is 'auc'")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover and merge data
    data = discover_and_merge(
        args.study_tracker, args.derivatives_root, args.metrics, args.exclusion_csv,
    )

    if len(data) == 0:
        logger.error("No subjects found. Check paths and metrics.")
        sys.exit(1)

    # Write master subject list
    subject_list_path = args.output_dir / 'subject_list.txt'
    data[['subject_key']].to_csv(subject_list_path, index=False, header=False)
    logger.info("Wrote master subject list: %d subjects", len(data))

    # Print distribution
    logger.info("\nData distribution:")
    for pnd in args.cohorts:
        subset = data[data['PND'] == pnd]
        dose_dist = dict(subset['dose'].value_counts().sort_index())
        sex_dist = dict(subset['sex'].value_counts().sort_index())
        logger.info("  %s: n=%d, dose=%s, sex=%s", pnd, len(subset), dose_dist, sex_dist)

    if args.target == 'dose':
        # Create categorical designs (per-PND + pooled)
        for pnd in args.cohorts:
            create_per_pnd_categorical(data, pnd, args.output_dir, subject_list_path)
        create_pooled_categorical(data, args.output_dir, subject_list_path)

        # Create dose-response designs (per-PND + pooled)
        for pnd in args.cohorts:
            create_per_pnd_dose_response(data, pnd, args.output_dir, subject_list_path)
        create_pooled_dose_response(data, args.output_dir, subject_list_path)

    elif args.target == 'auc':
        # Merge AUC data
        auc_df = _load_auc_lookup(args.auc_csv)
        data_auc = _merge_auc(data, auc_df)

        # Create AUC-response designs (per-PND + pooled)
        for pnd in args.cohorts:
            create_per_pnd_auc_response(data_auc, pnd, args.output_dir, subject_list_path)
        create_pooled_auc_response(data_auc, args.output_dir, subject_list_path)

    # Write analysis description
    write_design_description(args, args.output_dir / 'design_description.txt')

    logger.info("\nAll MVPA designs saved to %s", args.output_dir)


if __name__ == '__main__':
    main()
