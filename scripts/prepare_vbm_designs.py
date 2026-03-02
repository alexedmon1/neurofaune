#!/usr/bin/env python3
"""
Prepare FSL randomise design matrices for BPA-Rat VBM analysis.

Creates per-PND categorical and dose-response designs using the same
statistical models as TBSS, but matched to the VBM subject list.

Design sets:
  Categorical (per_pnd_{p30,p60,p90}):
    Dummy-coded dose (ref=C), effect-coded sex (ref=F)
    Contrasts: H_gt_C, C_gt_H, L_gt_C, C_gt_L, M_gt_C, C_gt_M

  Dose-response (dose_response_{p30,p60,p90}):
    Ordinal dose (C=0, L=1, M=2, H=3, mean-centered), effect-coded sex
    Contrasts: dose_pos, dose_neg

Usage:
    uv run python scripts/prepare_vbm_designs.py \
        --vbm-dir /mnt/arborea/bpa-rat/analysis/vbm \
        --study-tracker /mnt/arborea/bpa-rat/study_tracker_combined_250916.csv
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

# Ordinal dose mapping for dose-response designs
DOSE_MAP = {'C': 0, 'L': 1, 'M': 2, 'H': 3}


def write_provenance(
    design_dir: Path,
    vbm_dir: Path,
    n_subjects_in_design: int,
    design_type: str,
) -> None:
    """Write provenance.json linking design to its VBM subject list."""
    subject_list_path = vbm_dir / 'subject_list.txt'
    h = hashlib.sha256()
    with open(subject_list_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    subject_list_hash = h.hexdigest()

    n_subjects_in_list = sum(
        1 for line in open(subject_list_path) if line.strip()
    )

    provenance = {
        'vbm_dir': str(vbm_dir),
        'subject_list_sha256': subject_list_hash,
        'n_subjects_in_list': n_subjects_in_list,
        'n_subjects_in_design': n_subjects_in_design,
        'design_type': design_type,
        'date_created': datetime.now().isoformat(),
    }
    with open(design_dir / 'provenance.json', 'w') as f:
        json.dump(provenance, f, indent=2)
    logger.info(f"  Wrote provenance.json (hash: {subject_list_hash[:16]}...)")


def load_and_merge_data(
    study_tracker_path: Path,
    vbm_dir: Path,
) -> pd.DataFrame:
    """
    Load study tracker and merge with VBM subject list.

    Returns DataFrame with columns: subject_key, bids_id, session, PND, sex, dose
    """
    # Load study tracker
    tracker = pd.read_csv(study_tracker_path)
    logger.info(f"Loaded tracker: {len(tracker)} rows")

    valid = tracker[tracker['irc.ID'].notna()].copy()
    valid['bids_id'] = 'sub-' + valid['irc.ID']
    logger.info(f"Tracker rows with irc.ID: {len(valid)}")

    # Load VBM subject list
    subject_list_path = vbm_dir / 'subject_list.txt'
    if not subject_list_path.exists():
        raise FileNotFoundError(f"VBM subject list not found: {subject_list_path}")

    with open(subject_list_path) as f:
        vbm_subjects = [line.strip() for line in f if line.strip()]
    logger.info(f"VBM subjects: {len(vbm_subjects)}")

    # Parse subjects into rat ID + session
    parsed = []
    for s in vbm_subjects:
        m = re.match(r'(sub-\w+)_(ses-\w+)', s)
        if m:
            parsed.append({
                'subject_key': s,
                'bids_id': m.group(1),
                'session': m.group(2),
            })
    vbm_df = pd.DataFrame(parsed)
    vbm_df['PND'] = vbm_df['session'].str.replace('ses-', '').str.upper()

    # Merge on bids_id
    merged = vbm_df.merge(
        valid[['bids_id', 'sex', 'dose.level']],
        on='bids_id',
        how='inner'
    )
    merged = merged.rename(columns={'dose.level': 'dose'})

    unmatched = len(vbm_df) - len(merged)
    if unmatched > 0:
        missing_ids = set(vbm_df['bids_id']) - set(valid['bids_id'])
        logger.warning(
            f"{unmatched} VBM subjects not found in tracker "
            f"(dropped): {sorted(missing_ids)}"
        )

    logger.info(f"Final merged dataset: {len(merged)} subjects")
    return merged


def pre_subset_4d_volumes(
    vbm_dir: Path,
    design_dirs: List[Path],
    tissues: Optional[List[str]] = None,
) -> None:
    """
    Pre-create per-analysis 4D subsets from the master VBM volumes.

    Processes one tissue at a time to control peak memory.
    """
    if tissues is None:
        config_file = vbm_dir / 'vbm_config.json'
        if not config_file.exists():
            logger.warning(f"No vbm_config.json found, skipping subset")
            return
        with open(config_file) as f:
            tissues = json.load(f)['tissues']

    stats_dir = vbm_dir / 'stats'
    randomise_dir = vbm_dir / 'randomise'

    # Load master subject list
    subject_list_file = vbm_dir / 'subject_list.txt'
    with open(subject_list_file) as f:
        master_subjects = [line.strip() for line in f if line.strip()]

    logger.info(
        f"\n[Pre-subset] Creating 4D subsets for "
        f"{len(design_dirs)} analyses x {len(tissues)} tissues"
    )
    logger.info(f"  Master subject list: {len(master_subjects)} subjects")

    master_index = {subj: i for i, subj in enumerate(master_subjects)}

    for tissue in tissues:
        master_4d_path = stats_dir / f'all_{tissue}.nii.gz'
        if not master_4d_path.exists():
            logger.warning(f"  Master 4D not found: {master_4d_path}, skipping {tissue}")
            continue

        logger.info(f"\n  Loading master {tissue} ({master_4d_path.name})...")
        master_img = nib.load(master_4d_path)
        master_data = master_img.get_fdata()

        if master_data.shape[3] != len(master_subjects):
            logger.error(
                f"  {tissue}: 4D has {master_data.shape[3]} volumes but "
                f"subject list has {len(master_subjects)} - skipping"
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
            subset_file = subset_dir / f'all_{tissue}.nii.gz'

            # Skip if already exists with correct shape
            if subset_file.exists():
                existing_shape = nib.load(subset_file).shape
                if len(existing_shape) == 4 and existing_shape[3] == len(design_subjects):
                    logger.info(
                        f"  {analysis_name}/{tissue}: exists "
                        f"({existing_shape[3]} vols), skipping"
                    )
                    continue

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
                    f"  {analysis_name}/{tissue}: "
                    f"{len(missing)} subjects not in master - skipping"
                )
                continue

            subset_data = master_data[:, :, :, indices].astype(np.float32)
            subset_dir.mkdir(parents=True, exist_ok=True)
            nib.save(
                nib.Nifti1Image(subset_data, master_img.affine, master_img.header),
                subset_file,
            )
            logger.info(
                f"  {analysis_name}/{tissue}: "
                f"{master_data.shape[3]} -> {subset_data.shape[3]} volumes"
            )
            del subset_data

        del master_data
        gc.collect()

    logger.info("\n[Pre-subset] Done")


def create_per_pnd_design(
    data: pd.DataFrame,
    pnd: str,
    output_dir: Path,
    vbm_dir: Path,
) -> None:
    """Create categorical design files for a single PND timepoint."""
    subset = data[data['PND'] == pnd].copy().reset_index(drop=True)
    n = len(subset)
    if n == 0:
        logger.warning(f"No subjects for {pnd}, skipping")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Per-PND design: {pnd} (n={n})")
    logger.info(f"{'='*60}")

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

    write_provenance(design_dir, vbm_dir, n, f'per_pnd_{pnd.lower()}')
    logger.info(f"Saved to {design_dir}")


def create_per_pnd_dose_response_design(
    data: pd.DataFrame,
    pnd: str,
    output_dir: Path,
    vbm_dir: Path,
) -> None:
    """Create dose-response design for a single PND timepoint."""
    subset = data[data['PND'] == pnd].copy().reset_index(drop=True)
    n = len(subset)
    if n == 0:
        logger.warning(f"No subjects for {pnd}, skipping")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Dose-response design: {pnd} (n={n})")
    logger.info(f"{'='*60}")

    subset['dose_numeric'] = subset['dose'].map(DOSE_MAP)

    for dose_label in ['C', 'L', 'M', 'H']:
        count = (subset['dose'] == dose_label).sum()
        logger.info(f"  {dose_label} (={DOSE_MAP[dose_label]}): n={count}")

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

    helper.add_contrast('dose_pos', covariate='dose_numeric', direction='+')
    helper.add_contrast('dose_neg', covariate='dose_numeric', direction='-')

    logger.info(helper.summary())

    # Rank check
    mat = helper.design_matrix
    rank = np.linalg.matrix_rank(mat)
    n_cols = mat.shape[1]
    if rank < n_cols:
        logger.warning(f"Design matrix is rank-deficient! rank={rank}, cols={n_cols}")
    else:
        logger.info(f"Design matrix is full rank ({rank}/{n_cols})")

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

    write_provenance(design_dir, vbm_dir, n, f'dose_response_{pnd.lower()}')
    logger.info(f"Saved to {design_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare FSL randomise design matrices for BPA-Rat VBM analysis'
    )
    parser.add_argument(
        '--vbm-dir', type=Path, required=True,
        help='VBM directory (must contain subject_list.txt from prepare_vbm.py)'
    )
    parser.add_argument(
        '--study-tracker', type=Path, required=True,
        help='Path to study_tracker_combined CSV'
    )
    parser.add_argument(
        '--skip-subset', action='store_true',
        help='Skip pre-creating 4D subsets for each analysis'
    )
    args = parser.parse_args()

    vbm_dir = args.vbm_dir
    designs_dir = vbm_dir / 'designs'

    # Load and merge data
    data = load_and_merge_data(args.study_tracker, vbm_dir)

    # Print distribution summary
    logger.info("\nData distribution:")
    for pnd in ['P30', 'P60', 'P90']:
        subset = data[data['PND'] == pnd]
        dose_dist = dict(subset['dose'].value_counts().sort_index())
        sex_dist = dict(subset['sex'].value_counts().sort_index())
        logger.info(f"  {pnd}: n={len(subset)}, dose={dose_dist}, sex={sex_dist}")

    # Create per-PND categorical designs
    for pnd in ['P30', 'P60', 'P90']:
        create_per_pnd_design(data, pnd, designs_dir, vbm_dir)

    # Create per-PND dose-response designs
    for pnd in ['P30', 'P60', 'P90']:
        create_per_pnd_dose_response_design(data, pnd, designs_dir, vbm_dir)

    logger.info(f"\nAll designs saved to {designs_dir}")

    # Pre-create 4D subsets
    if not args.skip_subset:
        design_dirs = sorted(designs_dir.glob('per_pnd_*')) + \
                      sorted(designs_dir.glob('dose_response_*'))
        design_dirs = [d for d in design_dirs if d.is_dir()]
        pre_subset_4d_volumes(vbm_dir, design_dirs)
    else:
        logger.info("\nSkipping 4D subset creation (--skip-subset)")


if __name__ == '__main__':
    main()
