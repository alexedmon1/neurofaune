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

# Entity parsers for SIGMA-space derivative filenames.
#
# [\w-], not \w, in the metric group: BIDS entity VALUES legitimately contain
# hyphens, and the multi-shell metrics are named model-DKI_MK / model-NODDI_ODI.
# Under a \w-only group the glob found those files and the regex then dropped
# every one of them with no error -- 364 of 572 images on the cuprizone study,
# i.e. the whole of DKI and NODDI, invisible to the analysis stage.
SIGMA_MAP_RE = re.compile(
    r'(sub-[\w-]+?)_(ses-[\w-]+?)_space-SIGMA_([\w-]+)\.nii\.gz'
)
SIGMA_FUNC_RE = re.compile(
    r'(sub-[\w-]+?)_(ses-[\w-]+?)_space-SIGMA_desc-([\w-]+)_bold\.nii\.gz'
)


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
    coverage_mask: Optional[np.ndarray] = None,
    min_coverage: float = 0.0,
    return_coverage: bool = False,
):
    """
    Compute mean metric value within each labeled ROI, over COVERED voxels only.

    An acquisition slab rarely spans the whole atlas: a 27-slice DWI or an
    11-slice MSME covers part of it, and every atlas voxel outside the slab is
    zero in the warped image. Averaging those zeros in makes the ROI value a
    function of how much of the ROI the slab reached rather than of the tissue.
    Measured on this data (rat MSME T2, 234 ROIs): ``corr(coverage, ROI mean)``
    was **0.932** including zeros and **0.03** excluding them, with the median
    ROI reading 51.1 ms instead of 60.0 ms. The bias is silent and it is
    strongest exactly where coverage is worst, so it survives as plausible
    numbers rather than as missing data.

    Voxels are therefore restricted to `coverage_mask` when one is given, and
    otherwise to finite non-zero voxels.

    Parameters
    ----------
    metric_img : ndarray
        3D metric volume (FA, MD, T2, etc.) in SIGMA space
    parcellation_data : ndarray
        3D integer parcellation array (same shape as metric_img)
    labels_df : DataFrame
        Labels table with 'Labels' and 'roi_name' columns
    coverage_mask : ndarray, optional
        3D boolean mask of voxels the acquisition actually reached (e.g. the
        session brain mask warped to SIGMA). **Prefer this.** The nonzero
        fallback cannot distinguish "outside the slab" from a genuine zero, and
        some metrics do take exact zeros in-slab — measured at 0.07% of in-slab
        voxels for MWF, where NNLS returns no short-T2 component.
    min_coverage : float
        Return NaN for any ROI whose covered fraction is below this (0-1).
        Default 0.0 keeps every ROI; the coverage is reported regardless, so
        the caller can threshold later instead.
    return_coverage : bool
        If True, return ``(roi_means, roi_coverage)`` instead of just means.

    Returns
    -------
    dict, or (dict, dict) when return_coverage
        Mapping of ROI name → mean over covered voxels (NaN when the label is
        absent from the parcellation, nothing is covered, or coverage is below
        `min_coverage`), and optionally ROI name → covered fraction (0-1).
    """
    if coverage_mask is None:
        covered_all = np.isfinite(metric_img) & (metric_img != 0)
    else:
        covered_all = np.asarray(coverage_mask, dtype=bool) & np.isfinite(metric_img)

    roi_means = {}
    roi_coverage = {}
    for _, row in labels_df.iterrows():
        label_id = row['Labels']
        roi_name = row['roi_name']
        mask = parcellation_data == label_id
        n_voxels = int(mask.sum())
        if n_voxels == 0:
            roi_means[roi_name] = np.nan
            roi_coverage[roi_name] = np.nan
            continue

        covered = mask & covered_all
        n_covered = int(covered.sum())
        frac = n_covered / n_voxels
        roi_coverage[roi_name] = frac

        if n_covered == 0 or frac < min_coverage:
            roi_means[roi_name] = np.nan
        else:
            roi_means[roi_name] = float(np.nanmean(metric_img[covered]))

    if return_coverage:
        return roi_means, roi_coverage
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
        Metric suffixes to search for (e.g. ['FA', 'MD', 'AD', 'RD']
        for DWI/MSME, or ['fALFF', 'ReHo'] for func)

    Returns
    -------
    list[dict]
        Each dict has keys: subject, session, metric, path
    """
    found = []

    for metric in metrics:
        if modality == "func":
            # Functional files: sub-*_ses-*_space-SIGMA_desc-{metric}_bold.nii.gz
            pattern = (
                f"sub-*/ses-*/{modality}/"
                f"sub-*_ses-*_space-SIGMA_desc-{metric}_bold.nii.gz"
            )
            fname_re = SIGMA_FUNC_RE
        else:
            # DWI/MSME files: sub-*_ses-*_space-SIGMA_{metric}.nii.gz
            pattern = (
                f"sub-*/ses-*/{modality}/"
                f"sub-*_ses-*_space-SIGMA_{metric}.nii.gz"
            )
            fname_re = SIGMA_MAP_RE

        for path in sorted(derivatives_dir.glob(pattern)):
            match = fname_re.match(path.name)
            if match:
                found.append({
                    'subject': match.group(1),
                    'session': match.group(2),
                    'metric': match.group(3),
                    'path': path,
                })
            else:
                # The glob matched but the parse did not. Never drop this on
                # the floor: it means files exist that no one will analyse.
                logger.warning(
                    f"Could not parse entities from {path.name} -- skipped. "
                    f"This file will be invisible to the analysis stage."
                )

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
    exclusions: Optional[set] = None,
    min_volumes: int = 0,
    min_coverage: float = 0.0,
    use_coverage_mask: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Extract ROI means for all subjects, one DataFrame per metric.

    ROI means are computed over covered voxels only — see `extract_roi_means`
    for why averaging in out-of-slab zeros makes an ROI value track coverage
    rather than tissue.

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
    exclusions : set, optional
        Set of (subject, session) tuples to exclude.
    min_volumes : int
        Minimum BOLD volume count for func sessions (0 = no check).
    min_coverage : float
        ROIs covered below this fraction (0-1) are returned as NaN. Default 0.0
        keeps everything and lets the caller threshold using the reported
        coverage. Partial-coverage ROIs are real for slab acquisitions: on this
        study's 11-slice MSME, 86 of 234 atlas ROIs fall below 50%.
    use_coverage_mask : bool
        Prefer the session's own brain mask warped to SIGMA
        (``{sub}_{ses}_space-SIGMA_desc-brain_mask.nii.gz``) as the coverage
        mask, falling back to finite-nonzero when it is absent. Only the mask
        distinguishes "outside the slab" from a genuine zero.

    Returns
    -------
    dict[str, DataFrame]
        Mapping of metric name → wide DataFrame with columns:
        subject, session + ROI columns + territory columns
    """
    import json as _json

    parcellation_data, labels_df = load_parcellation(
        parcellation_path, labels_csv_path
    )

    file_list = discover_sigma_metrics(derivatives_dir, modality, metrics)
    if not file_list:
        logger.warning("No SIGMA-space metric images found")
        return {}

    # Apply exclusions
    if exclusions:
        n_before = len(file_list)
        file_list = [
            e for e in file_list
            if (e['subject'], e['session']) not in exclusions
        ]
        n_excluded = n_before - len(file_list)
        if n_excluded > 0:
            logger.info(f"Excluded {n_excluded} sessions by exclusion list")

    # Apply volume count filter for func modality
    if min_volumes > 0 and modality == 'func':
        filtered = []
        n_vol_excluded = 0
        for entry in file_list:
            sub, ses = entry['subject'], entry['session']
            func_dir = derivatives_dir / sub / ses / 'func'
            n_vol = 0
            # Check analysis JSONs for n_timepoints
            for json_name in [
                f'{sub}_{ses}_desc-falff_analysis.json',
                f'{sub}_{ses}_desc-reho_analysis.json',
            ]:
                analysis_json = func_dir / json_name
                if analysis_json.exists():
                    try:
                        with open(analysis_json) as f:
                            data = _json.load(f)
                        stats = data.get('statistics', {})
                        for key in stats:
                            params = stats[key].get('parameters', {})
                            if 'n_timepoints' in params:
                                n_vol = int(params['n_timepoints'])
                                break
                    except (ValueError, KeyError, TypeError):
                        pass
                if n_vol > 0:
                    break
            # Fall back to preprocessed BOLD header
            if n_vol == 0:
                import nibabel as _nib
                preproc = func_dir / f'{sub}_{ses}_desc-preproc_bold.nii.gz'
                if preproc.exists():
                    img = _nib.load(preproc)
                    n_vol = img.shape[3] if len(img.shape) > 3 else 1
            if n_vol >= min_volumes:
                filtered.append(entry)
            else:
                logger.info(
                    f"Excluding {sub}/{ses}: {n_vol} volumes < {min_volumes}"
                )
                n_vol_excluded += 1
        if n_vol_excluded > 0:
            logger.info(
                f"Excluded {n_vol_excluded} sessions by volume count "
                f"(<{min_volumes})"
            )
        file_list = filtered

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

            coverage_mask = None
            if use_coverage_mask:
                mask_path = (
                    entry['path'].parent
                    / f'{sub}_{ses}_space-SIGMA_desc-brain_mask.nii.gz'
                )
                if mask_path.exists():
                    coverage_mask = np.asarray(
                        nib.load(str(mask_path)).dataobj
                    ) > 0
                else:
                    logger.warning(
                        f"[{metric}] {sub}_{ses}: no SIGMA brain mask "
                        f"({mask_path.name}); falling back to finite-nonzero "
                        f"coverage, which cannot tell an out-of-slab voxel "
                        f"from a genuine zero."
                    )

            roi_means, roi_cov = extract_roi_means(
                img_data, parcellation_data, labels_df,
                coverage_mask=coverage_mask,
                min_coverage=min_coverage,
                return_coverage=True,
            )
            territory_means = compute_territory_means(
                roi_means, labels_df, parcellation_data
            )

            cov_vals = [c for c in roi_cov.values() if np.isfinite(c)]
            if cov_vals:
                logger.info(
                    f"[{metric}] {sub}_{ses}: coverage median "
                    f"{float(np.median(cov_vals)):.2f}, "
                    f"{sum(c < 0.5 for c in cov_vals)} ROIs <50%"
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
