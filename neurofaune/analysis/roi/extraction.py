"""
Core ROI extraction logic for SIGMA atlas parcellation.

Extracts mean metric values (FA, MD, T2, etc.) within atlas-defined
regions of interest from images warped to SIGMA space.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_parcellation(
    parcellation_path: Path,
    labels_csv_path: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load SIGMA parcellation NIfTI and labels CSV.

    Normalizes territory names ("Olfactive Bulb" → "Olfactory Bulb")
    and sanitizes ROI names (dots → underscores) for use as column names.

    Parameters
    ----------
    parcellation_path : Path
        Path to parcellation NIfTI (e.g. SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz)
    labels_csv_path : Path
        Path to labels CSV with columns: Labels, Hemisphere, Matter,
        Territories, System, Region of interest

    Returns
    -------
    parcellation_data : ndarray
        3D integer array of label IDs
    labels_df : DataFrame
        Labels table with sanitized ROI names and a 'roi_name' column
    """
    parc_img = nib.load(str(parcellation_path))
    parcellation_data = np.asarray(parc_img.dataobj, dtype=np.int32)

    labels_df = pd.read_csv(labels_csv_path, encoding='utf-8-sig')

    # Normalize territory naming inconsistency
    labels_df['Territories'] = labels_df['Territories'].str.replace(
        'Olfactive Bulb', 'Olfactory Bulb', regex=False
    )

    # Sanitize ROI names: dots → underscores for valid column/variable names
    labels_df['roi_name'] = labels_df['Region of interest'].str.replace(
        '.', '_', regex=False
    )

    # Warn about labels in CSV but absent from parcellation volume
    csv_labels = set(labels_df['Labels'].values)
    vol_labels = set(np.unique(parcellation_data)) - {0}
    missing = csv_labels - vol_labels
    if missing:
        logger.warning(
            f"{len(missing)} labels in CSV but not in parcellation volume: "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
        )

    logger.info(
        f"Loaded parcellation: {parcellation_data.shape}, "
        f"{len(vol_labels)} volume labels, {len(labels_df)} CSV entries"
    )

    return parcellation_data, labels_df


def extract_roi_means(
    metric_img: np.ndarray,
    parcellation_data: np.ndarray,
    labels_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute mean metric value within each labeled ROI.

    Parameters
    ----------
    metric_img : ndarray
        3D metric volume (FA, MD, T2, etc.) in SIGMA space
    parcellation_data : ndarray
        3D integer parcellation array (same shape as metric_img)
    labels_df : DataFrame
        Labels table with 'Labels' and 'roi_name' columns

    Returns
    -------
    dict
        Mapping of ROI name → mean metric value. NaN for labels
        present in CSV but absent from parcellation.
    """
    roi_means = {}
    for _, row in labels_df.iterrows():
        label_id = row['Labels']
        roi_name = row['roi_name']
        mask = parcellation_data == label_id
        n_voxels = mask.sum()
        if n_voxels == 0:
            roi_means[roi_name] = np.nan
        else:
            roi_means[roi_name] = float(np.nanmean(metric_img[mask]))

    return roi_means


def compute_territory_means(
    roi_means: dict[str, float],
    labels_df: pd.DataFrame,
    parcellation_data: np.ndarray,
) -> dict[str, float]:
    """
    Aggregate region means into territory-level means, weighted by voxel count.

    Parameters
    ----------
    roi_means : dict
        ROI name → mean metric value (from extract_roi_means)
    labels_df : DataFrame
        Labels table with 'Labels', 'roi_name', and 'Territories' columns
    parcellation_data : ndarray
        3D parcellation array for computing voxel counts

    Returns
    -------
    dict
        Mapping of "territory_{name}" → volume-weighted mean
    """
    territory_means = {}

    for territory, group in labels_df.groupby('Territories'):
        weighted_sum = 0.0
        total_voxels = 0

        for _, row in group.iterrows():
            label_id = row['Labels']
            roi_name = row['roi_name']
            mean_val = roi_means.get(roi_name, np.nan)
            if np.isnan(mean_val):
                continue

            n_voxels = int((parcellation_data == label_id).sum())
            if n_voxels == 0:
                continue

            weighted_sum += mean_val * n_voxels
            total_voxels += n_voxels

        col_name = f"territory_{territory.replace(' ', '_')}"
        if total_voxels > 0:
            territory_means[col_name] = weighted_sum / total_voxels
        else:
            territory_means[col_name] = np.nan

    return territory_means


def discover_sigma_metrics(
    derivatives_dir: Path,
    modality: str,
    metrics: list[str],
) -> list[dict]:
    """
    Find all SIGMA-space metric images in derivatives.

    Parameters
    ----------
    derivatives_dir : Path
        Path to derivatives directory containing sub-*/ses-*/ folders
    modality : str
        Subdirectory name: 'dwi', 'msme', 'func'
    metrics : list[str]
        Metric suffixes to search for (e.g. ['FA', 'MD', 'AD', 'RD'])

    Returns
    -------
    list[dict]
        Each dict has keys: subject, session, metric, path
    """
    found = []

    for metric in metrics:
        pattern = f"sub-*/ses-*/{modality}/sub-*_ses-*_space-SIGMA_{metric}.nii.gz"
        for path in sorted(derivatives_dir.glob(pattern)):
            match = re.match(
                r'(sub-\w+)_(ses-\w+)_space-SIGMA_(\w+)\.nii\.gz',
                path.name,
            )
            if match:
                found.append({
                    'subject': match.group(1),
                    'session': match.group(2),
                    'metric': match.group(3),
                    'path': path,
                })

    logger.info(
        f"Discovered {len(found)} SIGMA-space {modality} images "
        f"for metrics {metrics}"
    )
    return found


def extract_all_subjects(
    derivatives_dir: Path,
    parcellation_path: Path,
    labels_csv_path: Path,
    modality: str,
    metrics: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Extract ROI means for all subjects, one DataFrame per metric.

    Parameters
    ----------
    derivatives_dir : Path
        Path to derivatives directory
    parcellation_path : Path
        Path to SIGMA parcellation NIfTI
    labels_csv_path : Path
        Path to SIGMA labels CSV
    modality : str
        Modality subdirectory ('dwi', 'msme', 'func')
    metrics : list[str]
        Metric names to extract (e.g. ['FA', 'MD'])

    Returns
    -------
    dict[str, DataFrame]
        Mapping of metric name → wide DataFrame with columns:
        subject, session + ROI columns + territory columns
    """
    parcellation_data, labels_df = load_parcellation(
        parcellation_path, labels_csv_path
    )

    file_list = discover_sigma_metrics(derivatives_dir, modality, metrics)
    if not file_list:
        logger.warning("No SIGMA-space metric images found")
        return {}

    # Group by metric
    by_metric: dict[str, list[dict]] = {}
    for entry in file_list:
        by_metric.setdefault(entry['metric'], []).append(entry)

    result = {}

    for metric, entries in by_metric.items():
        rows = []
        for i, entry in enumerate(entries):
            sub = entry['subject']
            ses = entry['session']
            logger.info(
                f"[{metric}] {i+1}/{len(entries)}: {sub}_{ses}"
            )

            img = nib.load(str(entry['path']))
            img_data = np.asarray(img.dataobj, dtype=np.float32)

            roi_means = extract_roi_means(img_data, parcellation_data, labels_df)
            territory_means = compute_territory_means(
                roi_means, labels_df, parcellation_data
            )

            row = {'subject': sub, 'session': ses}
            row.update(roi_means)
            row.update(territory_means)
            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(
            f"[{metric}] Extracted {len(df)} subjects × "
            f"{len(df.columns) - 2} ROI/territory columns"
        )
        result[metric] = df

    return result


def to_long_format(
    wide_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    metric_name: str,
) -> pd.DataFrame:
    """
    Melt wide ROI DataFrame into tidy long format.

    Parameters
    ----------
    wide_df : DataFrame
        Wide DataFrame from extract_all_subjects (subject, session, ROI cols, territory cols)
    labels_df : DataFrame
        Labels table for annotating ROI metadata
    metric_name : str
        Name of the metric (e.g. 'FA', 'MD')

    Returns
    -------
    DataFrame
        Long format with columns: subject, session, cohort, roi, hemisphere,
        matter, territory, system, level, metric, value
    """
    id_cols = ['subject', 'session']
    # Add phenotype columns if already merged
    for col in ['dose', 'sex']:
        if col in wide_df.columns:
            id_cols.append(col)

    value_cols = [c for c in wide_df.columns if c not in id_cols]

    long = wide_df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name='roi',
        value_name='value',
    )

    long['metric'] = metric_name

    # Derive cohort from session
    long['cohort'] = long['session'].str.replace('ses-', '', regex=False)

    # Build lookup from sanitized roi_name to metadata
    roi_meta = labels_df.set_index('roi_name')[
        ['Hemisphere', 'Matter', 'Territories', 'System']
    ].to_dict('index')

    # Annotate region-level rows
    long['hemisphere'] = long['roi'].map(
        lambda r: roi_meta[r]['Hemisphere'] if r in roi_meta else np.nan
    )
    long['matter'] = long['roi'].map(
        lambda r: roi_meta[r]['Matter'] if r in roi_meta else np.nan
    )
    long['territory'] = long['roi'].map(
        lambda r: roi_meta[r]['Territories'] if r in roi_meta else np.nan
    )
    long['system'] = long['roi'].map(
        lambda r: roi_meta[r]['System'] if r in roi_meta else np.nan
    )

    # Mark level: territory-aggregated vs individual region
    long['level'] = long['roi'].apply(
        lambda r: 'territory' if r.startswith('territory_') else 'region'
    )

    # For territory rows, fill territory column from the name
    territory_mask = long['level'] == 'territory'
    long.loc[territory_mask, 'territory'] = (
        long.loc[territory_mask, 'roi']
        .str.replace('territory_', '', regex=False)
        .str.replace('_', ' ')
    )

    return long


def merge_phenotype(
    df: pd.DataFrame,
    study_tracker_path: Path,
) -> pd.DataFrame:
    """
    Merge dose and sex from study tracker onto ROI DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame with a 'subject' column (e.g. 'sub-Rat84')
    study_tracker_path : Path
        Path to study tracker CSV (must have 'irc.ID', 'dose.level', 'sex')

    Returns
    -------
    DataFrame
        Input DataFrame with 'dose' and 'sex' columns added
    """
    tracker = pd.read_csv(study_tracker_path, encoding='utf-8-sig')
    valid = tracker[tracker['irc.ID'].notna()].copy()
    valid['subject'] = 'sub-' + valid['irc.ID']

    pheno = valid[['subject', 'dose.level', 'sex']].rename(
        columns={'dose.level': 'dose'}
    )

    merged = df.merge(pheno, on='subject', how='left')

    n_missing = merged['dose'].isna().sum()
    if n_missing > 0:
        missing_subs = sorted(
            merged.loc[merged['dose'].isna(), 'subject'].unique()
        )
        logger.warning(
            f"{n_missing} rows missing phenotype data. "
            f"Subjects: {missing_subs[:10]}"
        )

    logger.info(
        f"Merged phenotype: {len(merged)} rows, "
        f"{merged['dose'].notna().sum()} with dose/sex"
    )

    return merged
