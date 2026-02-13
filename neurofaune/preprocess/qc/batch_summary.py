"""
Batch QC Summary Report Generator

Aggregates QC metrics across all subjects in a batch and generates:
- Summary HTML dashboard with outlier flagging
- Metrics CSV for all subjects
- Thumbnail gallery of key QC images
- Distribution plots by cohort
- Skull strip omnibus reports (scrollable per-modality galleries)

QC Directory Structure:
    {study_root}/qc/
    ├── subjects/                       # Per-subject QC
    │   ├── sub-Rat1/
    │   │   └── ses-p60/
    │   │       ├── anat/
    │   │       ├── dwi/
    │   │       └── func/
    │   └── sub-Rat2/
    └── reports/                        # Module-wide omnibus reports
        ├── skull_strip_anat.html       # Scrollable mosaic galleries
        ├── skull_strip_dwi.html
        ├── thumbnails/                 # Copied images for reports
        ├── anat_batch_summary/         # Batch summaries by modality
        ├── dwi_batch_summary/
        └── templates/{cohort}/         # Template QC
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

# Use non-interactive backend for headless operation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# QC Path Utilities
# =============================================================================

def get_subject_qc_dir(
    study_root: Path,
    subject: str,
    session: str,
    modality: str
) -> Path:
    """
    Get the QC output directory for a subject/session/modality.

    Uses structure: qc/subjects/{subject}/{session}/{modality}/

    Parameters
    ----------
    study_root : Path
        Study root directory
    subject : str
        Subject identifier (e.g., 'sub-Rat1')
    session : str
        Session identifier (e.g., 'ses-p60')
    modality : str
        Modality ('anat', 'dwi', 'func', 'msme')

    Returns
    -------
    Path
        QC directory path (creates if doesn't exist)
    """
    qc_dir = Path(study_root) / 'qc' / 'subjects' / subject / session / modality
    qc_dir.mkdir(parents=True, exist_ok=True)
    return qc_dir


def get_batch_summary_dir(
    study_root: Path,
    modality: str
) -> Path:
    """
    Get the batch summary output directory for a modality.

    Uses structure: qc/reports/{modality}_batch_summary/

    Parameters
    ----------
    study_root : Path
        Study root directory
    modality : str
        Modality ('anat', 'dwi', 'func', 'msme')

    Returns
    -------
    Path
        Batch summary directory path (creates if doesn't exist)
    """
    summary_dir = Path(study_root) / 'qc' / 'reports' / f'{modality}_batch_summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir


def get_reports_dir(study_root: Path) -> Path:
    """
    Get the QC reports directory for module-wide omnibus reports.

    Uses structure: qc/reports/

    Parameters
    ----------
    study_root : Path
        Study root directory

    Returns
    -------
    Path
        Reports directory path (creates if doesn't exist)
    """
    reports_dir = Path(study_root) / 'qc' / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


# =============================================================================
# Slice-Level QC (for TBSS and partial-coverage modalities)
# =============================================================================

def compute_slice_metrics(
    fa_file: Path,
    mask_file: Optional[Path] = None
) -> pd.DataFrame:
    """
    Compute per-slice QC metrics for a DTI FA map.

    Parameters
    ----------
    fa_file : Path
        Path to FA NIfTI file
    mask_file : Path, optional
        Path to brain mask. If None, uses FA > 0 as mask.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per slice, columns: slice_idx, fa_mean, fa_std,
        fa_median, coverage, outlier_pct, snr_estimate
    """
    import nibabel as nib

    fa_img = nib.load(fa_file)
    fa_data = fa_img.get_fdata()

    if mask_file and Path(mask_file).exists():
        mask_data = nib.load(mask_file).get_fdata() > 0
    else:
        mask_data = fa_data > 0

    n_slices = fa_data.shape[2]
    records = []

    for s in range(n_slices):
        fa_slice = fa_data[:, :, s]
        mask_slice = mask_data[:, :, s]

        if mask_slice.sum() == 0:
            records.append({
                'slice_idx': s,
                'fa_mean': np.nan,
                'fa_std': np.nan,
                'fa_median': np.nan,
                'coverage': 0.0,
                'outlier_pct': 0.0,
                'snr_estimate': np.nan,
                'n_voxels': 0,
            })
            continue

        fa_masked = fa_slice[mask_slice]

        # Basic statistics
        fa_mean = float(np.mean(fa_masked))
        fa_std = float(np.std(fa_masked))
        fa_median = float(np.median(fa_masked))

        # Coverage: fraction of mask voxels with signal
        coverage = float(np.sum(fa_masked > 0) / mask_slice.sum())

        # Outlier percentage: FA > 1.0 or FA < 0 (invalid values)
        outlier_pct = float(np.sum((fa_masked > 1.0) | (fa_masked < 0)) / len(fa_masked) * 100)

        # SNR estimate
        snr = fa_mean / (fa_std + 1e-10)

        records.append({
            'slice_idx': s,
            'fa_mean': fa_mean,
            'fa_std': fa_std,
            'fa_median': fa_median,
            'coverage': coverage,
            'outlier_pct': outlier_pct,
            'snr_estimate': snr,
            'n_voxels': int(mask_slice.sum()),
        })

    return pd.DataFrame(records)


def flag_bad_slices(
    slice_metrics: pd.DataFrame,
    fa_mean_min: float = 0.15,
    fa_mean_max: float = 0.7,
    coverage_min: float = 0.5,
    outlier_max: float = 5.0,
    snr_min: float = 1.0
) -> Tuple[List[int], List[int], pd.DataFrame]:
    """
    Flag bad slices based on QC thresholds.

    Parameters
    ----------
    slice_metrics : pd.DataFrame
        Output from compute_slice_metrics()
    fa_mean_min : float
        Minimum acceptable mean FA (default: 0.15)
    fa_mean_max : float
        Maximum acceptable mean FA (default: 0.7)
    coverage_min : float
        Minimum coverage fraction (default: 0.5)
    outlier_max : float
        Maximum outlier percentage (default: 5.0%)
    snr_min : float
        Minimum SNR estimate (default: 1.0)

    Returns
    -------
    good_slices : list
        List of good slice indices
    bad_slices : list
        List of bad slice indices
    flags_df : DataFrame
        DataFrame with flag reasons per slice
    """
    flags = []

    for _, row in slice_metrics.iterrows():
        slice_idx = int(row['slice_idx'])
        reasons = []

        if pd.isna(row['fa_mean']) or row['n_voxels'] == 0:
            reasons.append('empty')
        else:
            if row['fa_mean'] < fa_mean_min:
                reasons.append(f'fa_low ({row["fa_mean"]:.3f} < {fa_mean_min})')
            if row['fa_mean'] > fa_mean_max:
                reasons.append(f'fa_high ({row["fa_mean"]:.3f} > {fa_mean_max})')
            if row['coverage'] < coverage_min:
                reasons.append(f'low_coverage ({row["coverage"]:.2f} < {coverage_min})')
            if row['outlier_pct'] > outlier_max:
                reasons.append(f'outliers ({row["outlier_pct"]:.1f}% > {outlier_max}%)')
            if row['snr_estimate'] < snr_min:
                reasons.append(f'low_snr ({row["snr_estimate"]:.2f} < {snr_min})')

        flags.append({
            'slice_idx': slice_idx,
            'is_bad': len(reasons) > 0,
            'n_flags': len(reasons),
            'reasons': '; '.join(reasons) if reasons else ''
        })

    flags_df = pd.DataFrame(flags)

    good_slices = flags_df[~flags_df['is_bad']]['slice_idx'].tolist()
    bad_slices = flags_df[flags_df['is_bad']]['slice_idx'].tolist()

    return good_slices, bad_slices, flags_df


def generate_slice_qc_summary(
    derivatives_dir: Path,
    output_dir: Path,
    subjects: Optional[List[str]] = None,
    fa_pattern: str = '*_FA.nii.gz',
    mask_pattern: str = '*_desc-brain_mask.nii.gz'
) -> Dict[str, Path]:
    """
    Generate slice-level QC summary for all DTI subjects.

    Creates:
    - slice_metrics.csv: Per-subject, per-slice metrics
    - slice_exclusions.json: Subjects with bad slices and slice indices
    - slice_heatmap.png: Visual summary of slice quality across subjects

    Parameters
    ----------
    derivatives_dir : Path
        Derivatives directory containing DTI outputs
    output_dir : Path
        Output directory for slice QC results
    subjects : list, optional
        Specific subjects to include
    fa_pattern : str
        Glob pattern for FA files
    mask_pattern : str
        Glob pattern for mask files

    Returns
    -------
    dict
        Paths to generated files
    """
    import nibabel as nib

    derivatives_dir = Path(derivatives_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Computing slice-level metrics...")

    all_slice_metrics = []
    slice_exclusions = {}

    # Find all FA files
    fa_files = sorted(derivatives_dir.glob(f'**/dwi/{fa_pattern}'))

    for fa_file in fa_files:
        # Extract subject/session from path
        parts = fa_file.parts
        try:
            subj_idx = next(i for i, p in enumerate(parts) if p.startswith('sub-'))
            subject = parts[subj_idx]
            session = parts[subj_idx + 1] if parts[subj_idx + 1].startswith('ses-') else 'ses-unknown'
        except (StopIteration, IndexError):
            continue

        # Filter if specific subjects requested
        if subjects and subject not in subjects:
            continue

        # Find corresponding mask
        mask_file = fa_file.parent / mask_pattern.replace('*', f'{subject}_{session}')
        if not mask_file.exists():
            mask_files = list(fa_file.parent.glob(mask_pattern))
            mask_file = mask_files[0] if mask_files else None

        # Compute slice metrics
        try:
            slice_df = compute_slice_metrics(fa_file, mask_file)
            slice_df['subject'] = subject
            slice_df['session'] = session
            all_slice_metrics.append(slice_df)

            # Flag bad slices
            good, bad, flags = flag_bad_slices(slice_df)

            if bad:
                slice_exclusions[f'{subject}_{session}'] = {
                    'subject': subject,
                    'session': session,
                    'bad_slices': bad,
                    'good_slices': good,
                    'n_bad': len(bad),
                    'n_total': len(slice_df),
                    'flags': flags[flags['is_bad']][['slice_idx', 'reasons']].to_dict('records')
                }
        except Exception as e:
            print(f"    Warning: Could not process {fa_file}: {e}")

    if not all_slice_metrics:
        print("  No slice metrics computed.")
        return {}

    # Combine all metrics
    combined_df = pd.concat(all_slice_metrics, ignore_index=True)

    # Reorder columns
    cols = ['subject', 'session', 'slice_idx', 'fa_mean', 'fa_std', 'fa_median',
            'coverage', 'outlier_pct', 'snr_estimate', 'n_voxels']
    combined_df = combined_df[[c for c in cols if c in combined_df.columns]]

    # Save slice metrics
    metrics_path = output_dir / 'slice_metrics.csv'
    combined_df.to_csv(metrics_path, index=False)
    print(f"    Slice metrics: {len(combined_df)} rows -> {metrics_path}")

    # Save slice exclusions
    exclusions_path = output_dir / 'slice_exclusions.json'

    # Add summary
    summary = {
        'total_subjects': combined_df['subject'].nunique(),
        'subjects_with_bad_slices': len(slice_exclusions),
        'total_bad_slices': sum(e['n_bad'] for e in slice_exclusions.values()),
        'thresholds': {
            'fa_mean_min': 0.15,
            'fa_mean_max': 0.7,
            'coverage_min': 0.5,
            'outlier_max': 5.0,
            'snr_min': 1.0
        }
    }

    with open(exclusions_path, 'w') as f:
        json.dump({
            'summary': summary,
            'exclusions': slice_exclusions
        }, f, indent=2)
    print(f"    Slice exclusions: {len(slice_exclusions)} subjects with bad slices -> {exclusions_path}")

    # Generate simple text list of bad slices
    bad_slices_path = output_dir / 'bad_slices.tsv'
    bad_rows = []
    for subj_sess, info in slice_exclusions.items():
        for slice_idx in info['bad_slices']:
            bad_rows.append({
                'subject': info['subject'],
                'session': info['session'],
                'slice_idx': slice_idx
            })

    if bad_rows:
        bad_df = pd.DataFrame(bad_rows)
        bad_df.to_csv(bad_slices_path, sep='\t', index=False)
        print(f"    Bad slices list: {len(bad_rows)} entries -> {bad_slices_path}")

    # Generate heatmap visualization
    try:
        heatmap_path = _generate_slice_heatmap(combined_df, output_dir)
    except Exception as e:
        print(f"    Warning: Could not generate heatmap: {e}")
        heatmap_path = None

    return {
        'slice_metrics_csv': metrics_path,
        'slice_exclusions_json': exclusions_path,
        'bad_slices_tsv': bad_slices_path if bad_rows else None,
        'heatmap_png': heatmap_path,
    }


def _generate_slice_heatmap(
    slice_df: pd.DataFrame,
    output_dir: Path
) -> Path:
    """Generate heatmap showing FA quality across subjects and slices."""
    # Pivot to create subject x slice matrix
    pivot = slice_df.pivot_table(
        index=['subject', 'session'],
        columns='slice_idx',
        values='fa_mean',
        aggfunc='first'
    )

    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.3)))

    # Create heatmap
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0.1, vmax=0.6)

    # Labels
    ax.set_xlabel('Slice Index')
    ax.set_ylabel('Subject')
    ax.set_title('FA Mean by Slice (Red=Low, Green=Good)')

    # Y-axis labels (subject_session)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([f'{s[0]}_{s[1]}' for s in pivot.index], fontsize=6)

    # X-axis labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)

    # Colorbar
    plt.colorbar(im, ax=ax, label='Mean FA')

    plt.tight_layout()

    heatmap_path = output_dir / 'slice_quality_heatmap.png'
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()

    return heatmap_path


@dataclass
class BatchQCConfig:
    """Configuration for batch QC summary generation.

    NOTE on voxel scaling: Rodent MRI has sub-mm voxels which are scaled 10x
    for FSL/ANTs compatibility. Motion parameters from eddy are in the scaled
    space, so FD values are 10x larger than actual motion. Thresholds below
    are in SCALED units (multiply by voxel_scale for actual mm).

    Example: With voxel_scale=10, mean_fd threshold of 20.0 means 2.0mm actual.
    """

    # Outlier detection thresholds (z-score)
    outlier_z_threshold: float = 2.5

    # Voxel scaling factor (rodent MRI is scaled 10x for FSL/ANTs)
    voxel_scale: float = 10.0

    # DWI-specific thresholds (in SCALED space - divide by voxel_scale for actual mm)
    dwi_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'mean_fd': (0.0, 20.0),       # 20mm scaled = 2mm actual, warn if > 2mm actual motion
        'max_fd': (0.0, 50.0),        # 50mm scaled = 5mm actual, warn if > 5mm actual motion
        'fa_mean': (0.25, 0.55),      # typical range (not affected by scaling)
        'md_mean': (0.0004, 0.0012),  # mm²/s (not affected by scaling)
        'mean_snr': (1.0, None),      # warn if < 1.0
    })

    # Functional-specific thresholds (in SCALED space)
    # Note: Motion params from MCFLIRT are in scaled mm due to 10x voxel scaling
    func_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'motion_mean_fd': (0.0, 5.0),        # 5mm scaled = 0.5mm actual, stricter for fMRI
        'motion_max_fd': (0.0, 20.0),        # 20mm scaled = 2mm actual
        'motion_pct_bad_volumes': (0.0, 20.0),  # %, warn if > 20%
        'motion_mean_dvars': (0.0, None),    # DVARS (no upper bound, use z-score)
    })

    # Anatomical thresholds (not affected by voxel scaling)
    # Note: brain_to_total_ratio can be > 1 due to how background is handled
    # These thresholds are based on observed distributions
    anat_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'skull_stripping_brain_to_total_ratio': (0.8, 2.5),  # Ratio varies widely
        'skull_stripping_snr_estimate': (0.8, None),  # SNR should be > 0.8 (low due to rodent brain contrast)
        'segmentation_gm_volume_fraction': (0.1, 0.8),  # GM should be 10-80% of brain
        'segmentation_wm_volume_fraction': (0.05, 0.85),  # WM can vary significantly
    })

    # Thumbnail settings
    thumbnail_size: Tuple[int, int] = (200, 200)
    thumbnail_quality: int = 85


def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Flatten a nested dictionary.

    Example:
        {'skull_stripping': {'snr': 2.5, 'brain_volume': 1000}}
        becomes
        {'skull_stripping_snr': 2.5, 'skull_stripping_brain_volume': 1000}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def collect_qc_metrics(
    qc_dir: Path,
    modality: str,
    subjects: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Collect QC metrics from all subjects for a given modality.

    Parameters
    ----------
    qc_dir : Path
        Root QC directory (e.g., /study/qc/)
    modality : str
        Modality to collect ('dwi', 'anat', 'func', 'msme')
    subjects : list, optional
        Specific subjects to include. If None, discovers all.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per subject/session, columns are metrics
    """
    qc_dir = Path(qc_dir)
    records = []

    # Look under qc/subjects/ first (new structure), fall back to flat qc/sub-*
    subjects_dir = qc_dir / 'subjects'
    if subjects_dir.exists():
        subject_dirs = sorted(subjects_dir.glob('sub-*'))
    else:
        subject_dirs = sorted(qc_dir.glob('sub-*'))

    for subj_dir in subject_dirs:
        subject = subj_dir.name

        # Filter if specific subjects requested
        if subjects and subject not in subjects:
            continue

        # Find session directories
        session_dirs = sorted(subj_dir.glob('ses-*'))

        for sess_dir in session_dirs:
            session = sess_dir.name
            mod_dir = sess_dir / modality

            if not mod_dir.exists():
                continue

            # Collect metrics from JSON files
            record = {
                'subject': subject,
                'session': session,
                'cohort': _extract_cohort(session),
            }

            # Load all *_metrics.json and *_qc.json files
            for json_file in mod_dir.glob('*metrics*.json'):
                try:
                    with open(json_file) as f:
                        metrics = json.load(f)
                    # Flatten nested dictionaries (e.g., anat has skull_stripping, segmentation)
                    flat_metrics = _flatten_dict(metrics)
                    record.update(flat_metrics)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not read {json_file}: {e}")

            # Also check for basic QC JSON
            basic_qc = mod_dir / f'{subject}_{session}_{modality}_basic_qc.json'
            if basic_qc.exists():
                try:
                    with open(basic_qc) as f:
                        record.update(json.load(f))
                except (json.JSONDecodeError, IOError):
                    pass

            if len(record) > 3:  # Has more than just subject/session/cohort
                records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Ensure subject/session/cohort are first columns
    cols = ['subject', 'session', 'cohort']
    other_cols = [c for c in df.columns if c not in cols]
    df = df[cols + sorted(other_cols)]

    return df


def _extract_cohort(session: str) -> str:
    """Extract cohort from session name (e.g., ses-p60 -> p60)."""
    if session.startswith('ses-'):
        return session[4:]
    return session


def detect_outliers(
    df: pd.DataFrame,
    config: BatchQCConfig,
    modality: str
) -> pd.DataFrame:
    """
    Detect outliers based on z-scores and absolute thresholds.

    Returns DataFrame with outlier flags and reasons.
    """
    if df.empty:
        return pd.DataFrame()

    # Get modality-specific thresholds
    thresholds = getattr(config, f'{modality}_thresholds', {})

    outlier_records = []

    for idx, row in df.iterrows():
        flags = []

        # Check absolute thresholds
        for metric, (low, high) in thresholds.items():
            if metric not in df.columns:
                continue
            val = row.get(metric)
            if pd.isna(val):
                continue
            if low is not None and val < low:
                flags.append(f"{metric}={val:.3f} (< {low})")
            if high is not None and val > high:
                flags.append(f"{metric}={val:.3f} (> {high})")

        # Check z-scores for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['subject', 'session']:
                continue
            val = row.get(col)
            if pd.isna(val):
                continue
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std > 0:
                z = abs((val - col_mean) / col_std)
                if z > config.outlier_z_threshold:
                    flags.append(f"{col}: z={z:.1f}")

        outlier_records.append({
            'subject': row['subject'],
            'session': row['session'],
            'is_outlier': len(flags) > 0,
            'n_flags': len(flags),
            'flags': '; '.join(flags) if flags else ''
        })

    return pd.DataFrame(outlier_records)


def generate_distribution_plots(
    df: pd.DataFrame,
    output_dir: Path,
    modality: str,
    key_metrics: Optional[List[str]] = None
) -> List[Path]:
    """
    Generate distribution plots for key metrics.

    Returns list of generated figure paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return []

    # Default key metrics per modality
    if key_metrics is None:
        key_metrics = _get_key_metrics(modality)

    # Filter to metrics that exist in the data
    available_metrics = [m for m in key_metrics if m in df.columns]

    if not available_metrics:
        return []

    generated = []

    # 1. Distribution histograms
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        data = df[metric].dropna()

        ax.hist(data, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.3f}')
        ax.axvline(data.median(), color='green', linestyle=':', label=f'Median: {data.median():.3f}')
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')
        ax.set_title(f'{metric} Distribution (n={len(data)})')
        ax.legend(fontsize=8)

    # Hide unused axes
    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    hist_path = output_dir / f'{modality}_distributions.png'
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    generated.append(hist_path)

    # 2. Box plots by cohort (if cohort column exists)
    if 'cohort' in df.columns and df['cohort'].nunique() > 1:
        fig, axes = plt.subplots(1, min(4, len(available_metrics)),
                                  figsize=(4*min(4, len(available_metrics)), 4))
        if len(available_metrics) == 1:
            axes = [axes]

        cohorts = sorted(df['cohort'].dropna().unique())

        for i, metric in enumerate(available_metrics[:4]):
            ax = axes[i]
            data_by_cohort = [df[df['cohort'] == c][metric].dropna() for c in cohorts]

            bp = ax.boxplot(data_by_cohort, labels=cohorts, patch_artist=True)
            colors = plt.cm.Set2(np.linspace(0, 1, len(cohorts)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            ax.set_xlabel('Cohort')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} by Cohort')

        plt.tight_layout()
        box_path = output_dir / f'{modality}_by_cohort.png'
        plt.savefig(box_path, dpi=150, bbox_inches='tight')
        plt.close()
        generated.append(box_path)

    return generated


def _get_key_metrics(modality: str) -> List[str]:
    """Get default key metrics for a modality."""
    metrics = {
        'dwi': ['fa_mean', 'md_mean', 'mean_fd', 'max_fd', 'mean_snr'],
        'anat': [
            'skull_stripping_brain_to_total_ratio',
            'skull_stripping_snr_estimate',
            'segmentation_gm_volume_fraction',
            'segmentation_wm_volume_fraction'
        ],
        'func': [
            'motion_mean_fd',
            'motion_max_fd',
            'motion_pct_bad_volumes',
            'motion_mean_dvars'
        ],
        'msme': ['t2_mean', 'mwf_mean', 'iwf_mean'],
    }
    return metrics.get(modality, [])


def generate_thumbnail_gallery(
    qc_dir: Path,
    output_dir: Path,
    modality: str,
    df: pd.DataFrame,
    config: BatchQCConfig
) -> Path:
    """
    Generate HTML thumbnail gallery of key QC images.

    Returns path to generated HTML file.
    """
    qc_dir = Path(qc_dir)
    output_dir = Path(output_dir)
    thumb_dir = output_dir / 'thumbnails'
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # Key images per modality
    image_patterns = {
        'dwi': ['*FA_montage.png', '*motion_params.png'],
        'anat': ['*mask_overlay*.png', '*segmentation*.png'],
        'func': ['*motion_params.png', '*fd_dvars.png'],
        'msme': ['*t2_montage.png', '*mwf_montage.png'],
    }

    patterns = image_patterns.get(modality, ['*.png'])

    # Collect images and create thumbnails
    gallery_items = []

    # Determine base path (new structure: qc/subjects/, legacy: qc/sub/, old flat: qc/)
    subjects_dir = qc_dir / 'subjects'
    sub_dir = qc_dir / 'sub'
    if subjects_dir.exists():
        base_dir = subjects_dir
    elif sub_dir.exists():
        base_dir = sub_dir
    else:
        base_dir = qc_dir

    for _, row in df.iterrows():
        subject = row['subject']
        session = row['session']

        subj_qc_dir = base_dir / subject / session / modality / 'figures'
        if not subj_qc_dir.exists():
            subj_qc_dir = base_dir / subject / session / modality

        for pattern in patterns:
            for img_path in subj_qc_dir.glob(pattern):
                # Copy to thumbnails (could resize here for true thumbnails)
                thumb_name = f'{subject}_{session}_{img_path.name}'
                thumb_path = thumb_dir / thumb_name

                try:
                    shutil.copy(img_path, thumb_path)
                    gallery_items.append({
                        'subject': subject,
                        'session': session,
                        'image': thumb_name,
                        'original': str(img_path),
                        'type': pattern.replace('*', '').replace('.png', '')
                    })
                except IOError:
                    pass

    # Generate HTML gallery
    html_path = output_dir / 'thumbnail_gallery.html'
    _write_gallery_html(html_path, gallery_items, modality)

    return html_path


def _write_gallery_html(output_path: Path, items: List[Dict], modality: str):
    """Write HTML thumbnail gallery."""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{modality.upper()} QC Thumbnail Gallery</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .filters {{ margin: 20px 0; padding: 10px; background: white; border-radius: 5px; }}
        .gallery {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .gallery-item {{
            background: white;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .gallery-item img {{
            max-width: 250px;
            max-height: 200px;
            cursor: pointer;
            border: 1px solid #ddd;
        }}
        .gallery-item img:hover {{ border-color: #007bff; }}
        .gallery-item .label {{
            font-size: 11px;
            color: #666;
            margin-top: 5px;
            word-break: break-all;
        }}
        .modal {{
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
        }}
        .modal img {{
            max-width: 95%;
            max-height: 95%;
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
        }}
        .modal-close {{
            position: absolute;
            top: 20px; right: 30px;
            color: white;
            font-size: 30px;
            cursor: pointer;
        }}
        input[type="text"] {{ padding: 8px; width: 200px; }}
    </style>
</head>
<body>
    <h1>{modality.upper()} QC Thumbnail Gallery</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <p>Total images: {len(items)}</p>

    <div class="filters">
        <label>Filter by subject: <input type="text" id="filter" onkeyup="filterGallery()" placeholder="e.g., Rat102"></label>
    </div>

    <div class="gallery">
"""

    for item in items:
        html += f"""        <div class="gallery-item" data-subject="{item['subject']}">
            <img src="thumbnails/{item['image']}" onclick="openModal(this.src)" alt="{item['subject']}">
            <div class="label">{item['subject']}<br>{item['session']}<br>{item['type']}</div>
        </div>
"""

    html += """    </div>

    <div class="modal" id="modal" onclick="closeModal()">
        <span class="modal-close">&times;</span>
        <img id="modal-img" src="">
    </div>

    <script>
        function openModal(src) {
            document.getElementById('modal-img').src = src;
            document.getElementById('modal').style.display = 'block';
        }
        function closeModal() {
            document.getElementById('modal').style.display = 'none';
        }
        function filterGallery() {
            const filter = document.getElementById('filter').value.toLowerCase();
            const items = document.querySelectorAll('.gallery-item');
            items.forEach(item => {
                const subject = item.dataset.subject.toLowerCase();
                item.style.display = subject.includes(filter) ? 'block' : 'none';
            });
        }
        document.addEventListener('keydown', e => { if(e.key === 'Escape') closeModal(); });
    </script>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)


def generate_summary_html(
    df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    output_dir: Path,
    modality: str,
    figure_paths: List[Path]
) -> Path:
    """
    Generate main HTML summary dashboard.

    Returns path to generated HTML file.
    """
    output_dir = Path(output_dir)
    html_path = output_dir / 'summary.html'

    n_subjects = len(df)
    n_outliers = outliers_df['is_outlier'].sum() if not outliers_df.empty else 0
    cohort_counts = df['cohort'].value_counts().to_dict() if 'cohort' in df.columns else {}

    # Get key metrics for this modality
    key_metrics = _get_key_metrics(modality)
    available_metrics = [m for m in key_metrics if m in df.columns]

    # Compute summary statistics
    stats_html = ""
    if available_metrics:
        stats_data = []
        for metric in available_metrics:
            col = df[metric].dropna()
            if len(col) > 0:
                stats_data.append({
                    'Metric': metric,
                    'Mean': f'{col.mean():.4f}',
                    'Std': f'{col.std():.4f}',
                    'Min': f'{col.min():.4f}',
                    'Max': f'{col.max():.4f}',
                    'Median': f'{col.median():.4f}',
                })
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            stats_html = stats_df.to_html(index=False, classes='stats-table')

    # Create outlier summary
    outlier_html = ""
    if not outliers_df.empty and n_outliers > 0:
        flagged = outliers_df[outliers_df['is_outlier']]
        outlier_html = flagged[['subject', 'session', 'n_flags', 'flags']].to_html(
            index=False, classes='outlier-table'
        )

    # Full metrics table (sortable)
    metrics_table = df.to_html(index=False, classes='metrics-table', table_id='metricsTable')

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{modality.upper()} Batch QC Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1, h2 {{ color: #333; }}
        .summary-box {{
            display: inline-block;
            background: white;
            padding: 20px;
            margin: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-box .number {{ font-size: 36px; font-weight: bold; color: #007bff; }}
        .summary-box .label {{ color: #666; }}
        .summary-box.warning .number {{ color: #dc3545; }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #007bff; color: white; cursor: pointer; }}
        th:hover {{ background: #0056b3; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #e9ecef; }}
        .outlier-table tr {{ background: #fff3cd; }}
        .figure {{ max-width: 100%; margin: 10px 0; }}
        .nav {{ margin: 20px 0; }}
        .nav a {{
            display: inline-block;
            padding: 10px 20px;
            background: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin-right: 10px;
        }}
        .nav a:hover {{ background: #0056b3; }}
        .cohort-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            margin: 2px;
        }}
        .cohort-p30 {{ background: #d4edda; color: #155724; }}
        .cohort-p60 {{ background: #cce5ff; color: #004085; }}
        .cohort-p90 {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <h1>{modality.upper()} Batch QC Summary</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

    <div class="nav">
        <a href="thumbnail_gallery.html">View Thumbnail Gallery</a>
        <a href="metrics.csv" download>Download Metrics CSV</a>
    </div>

    <div>
        <div class="summary-box">
            <div class="number">{n_subjects}</div>
            <div class="label">Total Subjects</div>
        </div>
        <div class="summary-box {'warning' if n_outliers > 0 else ''}">
            <div class="number">{n_outliers}</div>
            <div class="label">Outliers Flagged</div>
        </div>
"""

    # Add cohort counts
    for cohort, count in sorted(cohort_counts.items()):
        html += f"""        <div class="summary-box">
            <div class="number">{count}</div>
            <div class="label">Cohort {cohort}</div>
        </div>
"""

    html += """    </div>

    <div class="section">
        <h2>Summary Statistics</h2>
"""
    html += stats_html if stats_html else "<p>No metrics available.</p>"
    html += """    </div>
"""

    # Outliers section
    if n_outliers > 0:
        html += f"""    <div class="section">
        <h2>Flagged Outliers ({n_outliers})</h2>
        <p>Subjects flagged based on z-score > 2.5 or absolute thresholds:</p>
        {outlier_html}
    </div>
"""

    # Distribution figures
    if figure_paths:
        html += """    <div class="section">
        <h2>Metric Distributions</h2>
"""
        for fig_path in figure_paths:
            html += f'        <img class="figure" src="figures/{fig_path.name}" alt="{fig_path.stem}">\n'
        html += """    </div>
"""

    # Full metrics table
    html += f"""    <div class="section">
        <h2>All Metrics</h2>
        <p>Click column headers to sort. <a href="metrics.csv" download>Download CSV</a></p>
        {metrics_table}
    </div>

    <script>
        // Simple table sorting
        document.querySelectorAll('#metricsTable th').forEach((th, idx) => {{
            th.addEventListener('click', () => {{
                const table = document.getElementById('metricsTable');
                const rows = Array.from(table.querySelectorAll('tr')).slice(1);
                const asc = th.dataset.sort !== 'asc';
                th.dataset.sort = asc ? 'asc' : 'desc';

                rows.sort((a, b) => {{
                    const aVal = a.cells[idx].textContent;
                    const bVal = b.cells[idx].textContent;
                    const aNum = parseFloat(aVal);
                    const bNum = parseFloat(bVal);
                    if (!isNaN(aNum) && !isNaN(bNum)) {{
                        return asc ? aNum - bNum : bNum - aNum;
                    }}
                    return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                }});

                rows.forEach(row => table.querySelector('tbody').appendChild(row));
            }});
        }});
    </script>
</body>
</html>
"""

    with open(html_path, 'w') as f:
        f.write(html)

    return html_path


def generate_exclusion_lists(
    df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    output_dir: Path,
    modality: str,
    config: BatchQCConfig
) -> Dict[str, Path]:
    """
    Generate usable exclusion lists for downstream analysis.

    Creates multiple output formats:
    - exclude_subjects.txt: Simple list of subjects to exclude (one per line)
    - include_subjects.txt: Simple list of subjects to include (one per line)
    - exclusions.tsv: Tab-separated with subject, session, reason, metrics
    - exclusions_by_reason.json: Grouped by exclusion reason for targeted filtering

    Parameters
    ----------
    df : pd.DataFrame
        Full metrics DataFrame
    outliers_df : pd.DataFrame
        Outlier detection results
    output_dir : Path
        Output directory
    modality : str
        Modality being processed
    config : BatchQCConfig
        Configuration with thresholds

    Returns
    -------
    dict
        Paths to generated exclusion files
    """
    output_dir = Path(output_dir)
    results = {}

    if outliers_df.empty:
        return results

    # Merge metrics with outlier flags for detailed output
    merged = outliers_df.merge(
        df[['subject', 'session', 'cohort'] + _get_key_metrics(modality)],
        on=['subject', 'session'],
        how='left'
    )

    # 1. Simple exclude list (subject_session format)
    exclude_list = []
    include_list = []

    for _, row in outliers_df.iterrows():
        entry = f"{row['subject']}_{row['session']}"
        if row['is_outlier']:
            exclude_list.append(entry)
        else:
            include_list.append(entry)

    exclude_path = output_dir / 'exclude_subjects.txt'
    with open(exclude_path, 'w') as f:
        f.write('\n'.join(sorted(exclude_list)))
    results['exclude_txt'] = exclude_path
    print(f"    Exclude list: {len(exclude_list)} subjects -> {exclude_path}")

    include_path = output_dir / 'include_subjects.txt'
    with open(include_path, 'w') as f:
        f.write('\n'.join(sorted(include_list)))
    results['include_txt'] = include_path
    print(f"    Include list: {len(include_list)} subjects -> {include_path}")

    # 2. Detailed TSV with reasons and key metrics
    flagged = merged[merged['is_outlier']].copy()
    if not flagged.empty:
        # Reorder columns for readability
        cols = ['subject', 'session', 'cohort', 'n_flags', 'flags']
        metric_cols = [c for c in flagged.columns if c not in cols + ['is_outlier']]
        flagged = flagged[cols + metric_cols]

        tsv_path = output_dir / 'exclusions.tsv'
        flagged.to_csv(tsv_path, sep='\t', index=False)
        results['exclusions_tsv'] = tsv_path

    # 3. Group by exclusion reason for targeted filtering
    exclusions_by_reason = {
        'high_motion': [],
        'extreme_diffusion': [],
        'low_snr': [],
        'other': []
    }

    for _, row in outliers_df[outliers_df['is_outlier']].iterrows():
        entry = {
            'subject': row['subject'],
            'session': row['session'],
            'flags': row['flags']
        }

        flags = row['flags'].lower()

        # Categorize by primary issue
        if 'fd' in flags or 'motion' in flags or 'rotation' in flags or 'translation' in flags:
            exclusions_by_reason['high_motion'].append(entry)
        elif any(m in flags for m in ['fa_', 'md_', 'ad_', 'rd_']):
            exclusions_by_reason['extreme_diffusion'].append(entry)
        elif 'snr' in flags:
            exclusions_by_reason['low_snr'].append(entry)
        else:
            exclusions_by_reason['other'].append(entry)

    # Add summary counts
    summary = {
        'total_subjects': len(df),
        'excluded': len(exclude_list),
        'included': len(include_list),
        'exclusion_rate': f"{len(exclude_list) / len(df) * 100:.1f}%",
        'by_reason': {k: len(v) for k, v in exclusions_by_reason.items()},
        'thresholds_used': {
            'z_score': config.outlier_z_threshold,
            **getattr(config, f'{modality}_thresholds', {})
        }
    }

    by_reason_path = output_dir / 'exclusions_by_reason.json'
    with open(by_reason_path, 'w') as f:
        json.dump({
            'summary': summary,
            'by_reason': exclusions_by_reason
        }, f, indent=2)
    results['by_reason_json'] = by_reason_path

    # 4. Create cohort-specific exclusion lists (useful for group analysis)
    if 'cohort' in df.columns:
        cohorts_dir = output_dir / 'by_cohort'
        cohorts_dir.mkdir(exist_ok=True)

        for cohort in df['cohort'].dropna().unique():
            cohort_exclude = [
                f"{row['subject']}_{row['session']}"
                for _, row in outliers_df.iterrows()
                if row['is_outlier'] and
                df[(df['subject'] == row['subject']) & (df['session'] == row['session'])]['cohort'].iloc[0] == cohort
            ]

            cohort_include = [
                f"{row['subject']}_{row['session']}"
                for _, row in outliers_df.iterrows()
                if not row['is_outlier'] and
                df[(df['subject'] == row['subject']) & (df['session'] == row['session'])]['cohort'].iloc[0] == cohort
            ]

            if cohort_exclude or cohort_include:
                with open(cohorts_dir / f'exclude_{cohort}.txt', 'w') as f:
                    f.write('\n'.join(sorted(cohort_exclude)))
                with open(cohorts_dir / f'include_{cohort}.txt', 'w') as f:
                    f.write('\n'.join(sorted(cohort_include)))

        results['by_cohort_dir'] = cohorts_dir

    return results


def generate_batch_qc_summary(
    qc_dir: Path,
    modality: str,
    output_dir: Optional[Path] = None,
    subjects: Optional[List[str]] = None,
    config: Optional[BatchQCConfig] = None
) -> Dict[str, Path]:
    """
    Generate complete batch QC summary for a modality.

    Parameters
    ----------
    qc_dir : Path
        Root QC directory containing per-subject QC outputs
    modality : str
        Modality to summarize ('dwi', 'anat', 'func', 'msme')
    output_dir : Path, optional
        Output directory. Defaults to qc_dir/reports/{modality}_batch_summary/
    subjects : list, optional
        Specific subjects to include. If None, includes all.
    config : BatchQCConfig, optional
        Configuration for outlier detection and thresholds

    Returns
    -------
    dict
        Paths to generated outputs:
        - 'summary_html': Main dashboard
        - 'metrics_csv': All metrics CSV
        - 'gallery_html': Thumbnail gallery
        - 'outliers_json': Outlier flags
        - 'figures': List of distribution plots
    """
    qc_dir = Path(qc_dir)

    if output_dir is None:
        output_dir = qc_dir / 'reports' / f'{modality}_batch_summary'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = BatchQCConfig()

    print(f"Generating {modality.upper()} batch QC summary...")
    print(f"  QC directory: {qc_dir}")
    print(f"  Output: {output_dir}")

    # 1. Collect metrics
    print("  Collecting metrics...")
    df = collect_qc_metrics(qc_dir, modality, subjects)

    if df.empty:
        print(f"  Warning: No QC metrics found for {modality}")
        return {}

    print(f"  Found {len(df)} subjects with QC data")

    # 2. Save metrics CSV
    csv_path = output_dir / 'metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved metrics to {csv_path}")

    # 3. Detect outliers
    print("  Detecting outliers...")
    outliers_df = detect_outliers(df, config, modality)
    n_outliers = outliers_df['is_outlier'].sum() if not outliers_df.empty else 0
    print(f"  Found {n_outliers} outliers")

    # Save outliers JSON
    outliers_path = output_dir / 'outliers.json'
    if not outliers_df.empty:
        outliers_df.to_json(outliers_path, orient='records', indent=2)

    # 4. Generate exclusion lists
    print("  Generating exclusion lists...")
    exclusion_results = generate_exclusion_lists(df, outliers_df, output_dir, modality, config)

    # 5. Generate distribution plots
    print("  Generating distribution plots...")
    figures_dir = output_dir / 'figures'
    figure_paths = generate_distribution_plots(df, figures_dir, modality)

    # 6. Generate thumbnail gallery
    print("  Generating thumbnail gallery...")
    gallery_path = generate_thumbnail_gallery(qc_dir, output_dir, modality, df, config)

    # 7. Generate summary HTML
    print("  Generating summary dashboard...")
    summary_path = generate_summary_html(df, outliers_df, output_dir, modality, figure_paths)

    print(f"\nBatch QC summary complete!")
    print(f"  Summary: {summary_path}")
    print(f"  Gallery: {gallery_path}")
    print(f"  Metrics: {csv_path}")
    if exclusion_results.get('exclude_txt'):
        print(f"  Exclusions: {exclusion_results['exclude_txt']}")

    return {
        'summary_html': summary_path,
        'metrics_csv': csv_path,
        'gallery_html': gallery_path,
        'outliers_json': outliers_path,
        'figures': figure_paths,
        **exclusion_results,  # Include all exclusion list paths
    }


# =============================================================================
# Skull Strip Omnibus Reports
# =============================================================================

def generate_skull_strip_omnibus(
    study_root: Path,
    modality: str,
) -> Optional[Path]:
    """
    Generate a scrollable HTML omnibus report for skull strip QC across all subjects.

    Scans qc/subjects/sub-*/ses-*/{modality}/ for skull strip metrics and mosaic
    images, copies thumbnails to qc/reports/thumbnails/, and writes a single HTML
    page with a filterable card grid.

    Parameters
    ----------
    study_root : Path
        Study root directory
    modality : str
        Modality ('anat', 'dwi', 'func', 'msme')

    Returns
    -------
    Path or None
        Path to generated HTML report, or None if no data found
    """
    study_root = Path(study_root)
    reports_dir = get_reports_dir(study_root)
    thumb_dir = reports_dir / 'thumbnails'
    thumb_dir.mkdir(parents=True, exist_ok=True)

    subjects_qc = study_root / 'qc' / 'subjects'
    if not subjects_qc.exists():
        print(f"  No qc/subjects/ directory found at {subjects_qc}")
        return None

    # Collect skull strip data from all sessions
    cards = []

    for subj_dir in sorted(subjects_qc.glob('sub-*')):
        subject = subj_dir.name
        for sess_dir in sorted(subj_dir.glob('ses-*')):
            session = sess_dir.name
            cohort = _extract_cohort(session)
            mod_dir = sess_dir / modality

            if not mod_dir.exists():
                continue

            # Load metrics
            metrics_files = list(mod_dir.glob('*_skull_strip_metrics.json'))
            if not metrics_files:
                continue

            try:
                with open(metrics_files[0]) as f:
                    metrics = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            # Find mosaic image
            mosaic_files = list(mod_dir.glob('figures/*skull_strip_mosaic*.png'))
            if not mosaic_files:
                mosaic_files = list(mod_dir.glob('*skull_strip_mosaic*.png'))
            if not mosaic_files:
                continue

            # Copy mosaic to thumbnails
            mosaic_src = mosaic_files[0]
            thumb_name = f'{subject}_{session}_{modality}_skull_strip_mosaic.png'
            thumb_path = thumb_dir / thumb_name
            try:
                shutil.copy(mosaic_src, thumb_path)
            except IOError:
                continue

            # Find per-subject HTML report for linking
            report_files = list(mod_dir.glob('*skull_strip_qc.html'))
            report_rel = None
            if report_files:
                report_rel = str(report_files[0].relative_to(study_root / 'qc'))

            # Determine quality status
            ratio = metrics.get('brain_to_total_ratio', 0)
            snr = metrics.get('snr_estimate', 0)
            quality = _classify_skull_strip_quality(ratio, snr, modality)

            cards.append({
                'subject': subject,
                'session': session,
                'cohort': cohort,
                'ratio': ratio,
                'snr': snr,
                'quality': quality,
                'thumb': f'thumbnails/{thumb_name}',
                'report_link': f'../{report_rel}' if report_rel else None,
            })

    if not cards:
        print(f"  No skull strip data found for {modality}")
        return None

    # Compute summary stats
    n_total = len(cards)
    n_good = sum(1 for c in cards if c['quality'] == 'good')
    n_warning = sum(1 for c in cards if c['quality'] == 'warning')
    n_poor = sum(1 for c in cards if c['quality'] == 'poor')
    cohort_counts = {}
    for c in cards:
        cohort_counts[c['cohort']] = cohort_counts.get(c['cohort'], 0) + 1

    # Generate HTML
    output_path = reports_dir / f'skull_strip_{modality}.html'
    _write_omnibus_html(output_path, cards, modality, n_total, n_good, n_warning, n_poor, cohort_counts)

    print(f"  Generated {output_path.name}: {n_total} sessions ({n_good} good, {n_warning} warning, {n_poor} poor)")
    return output_path


def _classify_skull_strip_quality(
    ratio: float,
    snr: float,
    modality: str,
) -> str:
    """Classify skull strip quality as good/warning/poor.

    Note: DWI/func/MSME are partial-coverage acquisitions (9-11 slices out of
    41 T2w slices), so brain_to_total_ratio is inherently low (~0.1-0.2).
    Thresholds are adjusted accordingly.
    """
    if modality == 'anat':
        if ratio < 0.5 or ratio > 3.0 or snr < 0.5:
            return 'poor'
        if ratio < 0.8 or ratio > 2.5 or snr < 0.8:
            return 'warning'
    else:
        # DWI/func/MSME — partial-coverage, ratio is naturally ~0.08-0.20
        if ratio < 0.02 or ratio > 1.0 or snr < 0.5:
            return 'poor'
        if ratio < 0.05 or ratio > 0.5 or snr < 1.0:
            return 'warning'
    return 'good'


def _write_omnibus_html(
    output_path: Path,
    cards: List[Dict],
    modality: str,
    n_total: int,
    n_good: int,
    n_warning: int,
    n_poor: int,
    cohort_counts: Dict[str, int],
):
    """Write skull strip omnibus HTML report."""

    # Build cohort options for filter dropdown
    cohorts = sorted(cohort_counts.keys())
    cohort_options = '\n'.join(
        f'            <option value="{c}">{c} ({cohort_counts[c]})</option>'
        for c in cohorts
    )

    # Build card HTML
    cards_html = []
    for card in cards:
        border_color = {'good': '#27ae60', 'warning': '#f39c12', 'poor': '#e74c3c'}[card['quality']]
        badge_class = card['quality']

        link_open = f'<a href="{card["report_link"]}" target="_blank">' if card['report_link'] else ''
        link_close = '</a>' if card['report_link'] else ''

        cards_html.append(f"""        <div class="card {badge_class}" data-subject="{card['subject']}"
             data-session="{card['session']}" data-cohort="{card['cohort']}"
             data-quality="{card['quality']}" style="border-left: 5px solid {border_color};">
            <img src="{card['thumb']}" onclick="openModal(this.src)" alt="{card['subject']}">
            <div class="card-info">
                <div class="card-label">{link_open}{card['subject']}{link_close}<br>
                    <span class="session">{card['session']}</span>
                    <span class="badge {badge_class}">{card['quality']}</span>
                </div>
                <div class="card-metrics">
                    Ratio: {card['ratio']:.2f} | SNR: {card['snr']:.2f}
                </div>
            </div>
        </div>
""")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Skull Strip QC — {modality.upper()} Omnibus</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 0; padding: 20px; background: #f0f2f5; color: #333; }}
        h1 {{ margin: 0 0 5px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px 25px; border-radius: 8px;
                   margin-bottom: 20px; }}
        .header p {{ margin: 5px 0; opacity: 0.85; }}

        .summary {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }}
        .summary-box {{ background: white; padding: 15px 20px; border-radius: 8px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; min-width: 100px; }}
        .summary-box .number {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
        .summary-box .label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
        .summary-box.good .number {{ color: #27ae60; }}
        .summary-box.warning .number {{ color: #f39c12; }}
        .summary-box.poor .number {{ color: #e74c3c; }}

        .filters {{ background: white; padding: 15px 20px; border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px;
                    display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }}
        .filters label {{ font-size: 13px; font-weight: 600; color: #555; }}
        .filters select, .filters input {{ padding: 6px 10px; border: 1px solid #ddd;
                                           border-radius: 4px; font-size: 13px; }}
        .filters input {{ width: 180px; }}

        .grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
        .card {{ background: white; border-radius: 8px; overflow: hidden;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); width: calc(25% - 9px);
                 transition: transform 0.15s, box-shadow 0.15s; }}
        .card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
        .card img {{ width: 100%; display: block; cursor: pointer; }}
        .card-info {{ padding: 8px 10px; }}
        .card-label {{ font-size: 12px; font-weight: 600; }}
        .card-label a {{ color: #2980b9; text-decoration: none; }}
        .card-label a:hover {{ text-decoration: underline; }}
        .session {{ color: #888; font-weight: normal; }}
        .card-metrics {{ font-size: 11px; color: #666; margin-top: 4px; }}
        .badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px;
                  font-size: 10px; font-weight: 600; text-transform: uppercase;
                  margin-left: 4px; }}
        .badge.good {{ background: #d4edda; color: #155724; }}
        .badge.warning {{ background: #fff3cd; color: #856404; }}
        .badge.poor {{ background: #f8d7da; color: #721c24; }}

        .modal {{ display: none; position: fixed; top: 0; left: 0;
                  width: 100%; height: 100%; background: rgba(0,0,0,0.92);
                  z-index: 1000; }}
        .modal img {{ max-width: 95%; max-height: 95%; position: absolute;
                      top: 50%; left: 50%; transform: translate(-50%, -50%); }}
        .modal-close {{ position: absolute; top: 15px; right: 25px; color: white;
                        font-size: 32px; cursor: pointer; }}
        .modal-nav {{ position: absolute; top: 50%; transform: translateY(-50%);
                      color: white; font-size: 48px; cursor: pointer; padding: 20px;
                      user-select: none; }}
        .modal-nav.prev {{ left: 10px; }}
        .modal-nav.next {{ right: 10px; }}
        .modal-nav:hover {{ color: #ccc; }}

        .count-display {{ font-size: 13px; color: #888; margin-bottom: 10px; }}

        @media (max-width: 1200px) {{ .card {{ width: calc(33.33% - 8px); }} }}
        @media (max-width: 900px)  {{ .card {{ width: calc(50% - 6px); }} }}
        @media (max-width: 600px)  {{ .card {{ width: 100%; }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Skull Strip QC — {modality.upper()}</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | {n_total} sessions</p>
    </div>

    <div class="summary">
        <div class="summary-box">
            <div class="number">{n_total}</div>
            <div class="label">Total</div>
        </div>
        <div class="summary-box good">
            <div class="number">{n_good}</div>
            <div class="label">Good</div>
        </div>
        <div class="summary-box warning">
            <div class="number">{n_warning}</div>
            <div class="label">Warning</div>
        </div>
        <div class="summary-box poor">
            <div class="number">{n_poor}</div>
            <div class="label">Poor</div>
        </div>
"""

    for cohort in cohorts:
        html += f"""        <div class="summary-box">
            <div class="number">{cohort_counts[cohort]}</div>
            <div class="label">{cohort}</div>
        </div>
"""

    html += f"""    </div>

    <div class="filters">
        <label>Cohort:
            <select id="filterCohort" onchange="applyFilters()">
                <option value="">All</option>
{cohort_options}
            </select>
        </label>
        <label>Quality:
            <select id="filterQuality" onchange="applyFilters()">
                <option value="">All</option>
                <option value="good">Good ({n_good})</option>
                <option value="warning">Warning ({n_warning})</option>
                <option value="poor">Poor ({n_poor})</option>
            </select>
        </label>
        <label>Search:
            <input type="text" id="filterSearch" onkeyup="applyFilters()" placeholder="e.g. Rat102">
        </label>
        <label>Sort:
            <select id="sortBy" onchange="applyFilters()">
                <option value="subject">Subject</option>
                <option value="ratio-asc">Ratio (low first)</option>
                <option value="ratio-desc">Ratio (high first)</option>
                <option value="snr-asc">SNR (low first)</option>
                <option value="snr-desc">SNR (high first)</option>
            </select>
        </label>
    </div>

    <div class="count-display" id="countDisplay">Showing {n_total} of {n_total} sessions</div>

    <div class="grid" id="grid">
{''.join(cards_html)}    </div>

    <div class="modal" id="modal">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <span class="modal-nav prev" onclick="navModal(-1)">&#8249;</span>
        <span class="modal-nav next" onclick="navModal(1)">&#8250;</span>
        <img id="modal-img" src="">
    </div>

    <script>
        let visibleImages = [];
        let modalIdx = -1;

        function applyFilters() {{
            const cohort = document.getElementById('filterCohort').value;
            const quality = document.getElementById('filterQuality').value;
            const search = document.getElementById('filterSearch').value.toLowerCase();
            const sortBy = document.getElementById('sortBy').value;

            const cards = Array.from(document.querySelectorAll('.card'));
            let visible = 0;

            cards.forEach(card => {{
                const matchCohort = !cohort || card.dataset.cohort === cohort;
                const matchQuality = !quality || card.dataset.quality === quality;
                const matchSearch = !search ||
                    card.dataset.subject.toLowerCase().includes(search) ||
                    card.dataset.session.toLowerCase().includes(search);

                const show = matchCohort && matchQuality && matchSearch;
                card.style.display = show ? '' : 'none';
                if (show) visible++;
            }});

            // Sort visible cards
            const grid = document.getElementById('grid');
            const sorted = cards.slice().sort((a, b) => {{
                if (a.style.display === 'none' && b.style.display !== 'none') return 1;
                if (a.style.display !== 'none' && b.style.display === 'none') return -1;
                if (sortBy === 'subject') {{
                    return a.dataset.subject.localeCompare(b.dataset.subject) ||
                           a.dataset.session.localeCompare(b.dataset.session);
                }}
                const aMetrics = a.querySelector('.card-metrics').textContent;
                const bMetrics = b.querySelector('.card-metrics').textContent;
                let aVal, bVal;
                if (sortBy.startsWith('ratio')) {{
                    aVal = parseFloat(aMetrics.match(/Ratio:\\s*([\\d.]+)/)?.[1] || 0);
                    bVal = parseFloat(bMetrics.match(/Ratio:\\s*([\\d.]+)/)?.[1] || 0);
                }} else {{
                    aVal = parseFloat(aMetrics.match(/SNR:\\s*([\\d.]+)/)?.[1] || 0);
                    bVal = parseFloat(bMetrics.match(/SNR:\\s*([\\d.]+)/)?.[1] || 0);
                }}
                return sortBy.endsWith('asc') ? aVal - bVal : bVal - aVal;
            }});
            sorted.forEach(c => grid.appendChild(c));

            document.getElementById('countDisplay').textContent =
                `Showing ${{visible}} of {n_total} sessions`;

            // Update visible images list for modal navigation
            visibleImages = sorted
                .filter(c => c.style.display !== 'none')
                .map(c => c.querySelector('img').src);
        }}

        function openModal(src) {{
            visibleImages = Array.from(document.querySelectorAll('.card'))
                .filter(c => c.style.display !== 'none')
                .map(c => c.querySelector('img').src);
            modalIdx = visibleImages.indexOf(src);
            document.getElementById('modal-img').src = src;
            document.getElementById('modal').style.display = 'block';
        }}

        function closeModal() {{
            document.getElementById('modal').style.display = 'none';
            modalIdx = -1;
        }}

        function navModal(dir) {{
            if (visibleImages.length === 0) return;
            modalIdx = (modalIdx + dir + visibleImages.length) % visibleImages.length;
            document.getElementById('modal-img').src = visibleImages[modalIdx];
        }}

        document.addEventListener('keydown', e => {{
            if (e.key === 'Escape') closeModal();
            if (e.key === 'ArrowLeft') navModal(-1);
            if (e.key === 'ArrowRight') navModal(1);
        }});
    </script>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

