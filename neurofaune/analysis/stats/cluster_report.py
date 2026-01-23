#!/usr/bin/env python3
"""
Cluster Extraction and Reporting for FSL Randomise Results

Extracts significant clusters from randomise corrected p-value maps and
generates reports with SIGMA atlas labels for anatomical localization.

Adapted from neurovrai for rodent TBSS with SIGMA parcellation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd


class ClusterReportError(Exception):
    """Raised when cluster extraction fails"""
    pass


def extract_clusters(
    corrp_file: Path,
    stat_file: Path,
    threshold: float = 0.95,
    min_cluster_size: int = 10
) -> pd.DataFrame:
    """
    Extract significant clusters from corrected p-value map.

    Uses connected-component labeling to identify clusters of significant
    voxels, then extracts peak statistics from the stat map.

    Args:
        corrp_file: Corrected p-value map (1-p values from randomise)
        stat_file: T-statistic or F-statistic map
        threshold: Threshold for significance (default: 0.95 = p<0.05)
        min_cluster_size: Minimum cluster size in voxels

    Returns:
        DataFrame with cluster information (index, size, peak coords, peak stat)
    """
    logger = logging.getLogger("neurofaune.tbss")

    if not corrp_file.exists():
        raise ClusterReportError(f"Corrected p-value file not found: {corrp_file}")
    if not stat_file.exists():
        raise ClusterReportError(f"Stat file not found: {stat_file}")

    corrp_img = nib.load(corrp_file)
    corrp_data = corrp_img.get_fdata()
    affine = corrp_img.affine

    stat_img = nib.load(stat_file)
    stat_data = stat_img.get_fdata()

    sig_mask = corrp_data >= threshold
    n_sig_voxels = int(np.sum(sig_mask))

    if n_sig_voxels == 0:
        logger.info(f"No significant voxels at threshold {threshold}")
        return pd.DataFrame()

    logger.info(f"Found {n_sig_voxels} significant voxels")

    # Connected component labeling
    from scipy import ndimage
    labeled, n_clusters = ndimage.label(sig_mask)

    clusters = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labeled == cluster_id
        cluster_size = int(np.sum(cluster_mask))

        if cluster_size < min_cluster_size:
            continue

        # Peak statistic location
        cluster_stats = stat_data * cluster_mask
        peak_idx = np.unravel_index(np.argmax(cluster_stats), cluster_stats.shape)
        peak_stat = float(stat_data[peak_idx])

        # Peak corrected p-value
        peak_corrp = float(corrp_data[peak_idx])

        # Convert voxel to mm coordinates
        peak_mm = _voxel_to_mm(peak_idx, affine)

        # Center of gravity
        cog_voxel = ndimage.center_of_mass(cluster_mask)
        cog_mm = _voxel_to_mm(cog_voxel, affine)

        # Mean statistic in cluster
        mean_stat = float(np.mean(stat_data[cluster_mask]))

        clusters.append({
            'cluster_id': cluster_id,
            'size_voxels': cluster_size,
            'peak_stat': peak_stat,
            'peak_corrp': peak_corrp,
            'peak_x_mm': peak_mm[0],
            'peak_y_mm': peak_mm[1],
            'peak_z_mm': peak_mm[2],
            'cog_x_mm': cog_mm[0],
            'cog_y_mm': cog_mm[1],
            'cog_z_mm': cog_mm[2],
            'mean_stat': mean_stat,
            'peak_x_vox': int(peak_idx[0]),
            'peak_y_vox': int(peak_idx[1]),
            'peak_z_vox': int(peak_idx[2]),
        })

    df = pd.DataFrame(clusters)
    if not df.empty:
        df = df.sort_values('size_voxels', ascending=False).reset_index(drop=True)

    logger.info(f"Extracted {len(df)} clusters (min size={min_cluster_size})")
    return df


def _voxel_to_mm(voxel_coords, affine: np.ndarray) -> Tuple[float, float, float]:
    """Convert voxel indices to mm coordinates using affine."""
    coords = np.array([*voxel_coords, 1.0])
    mm = affine @ coords
    return (float(mm[0]), float(mm[1]), float(mm[2]))


def add_sigma_labels(
    df: pd.DataFrame,
    sigma_parcellation: Path,
    label_names: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Add anatomical labels from SIGMA parcellation atlas.

    For each cluster peak, looks up the corresponding SIGMA atlas region.

    Args:
        df: DataFrame with cluster coordinates (peak_x_vox, peak_y_vox, peak_z_vox)
        sigma_parcellation: Path to SIGMA parcellation NIfTI
        label_names: Optional mapping of label index to region name.
                     If None, uses integer labels.

    Returns:
        DataFrame with added 'region' and 'region_id' columns
    """
    if df.empty:
        return df

    if not sigma_parcellation.exists():
        logging.getLogger("neurofaune.tbss").warning(
            f"SIGMA parcellation not found: {sigma_parcellation}"
        )
        df['region'] = 'Unknown'
        df['region_id'] = 0
        return df

    parc_img = nib.load(sigma_parcellation)
    parc_data = parc_img.get_fdata().astype(int)

    regions = []
    region_ids = []

    for _, row in df.iterrows():
        x, y, z = int(row['peak_x_vox']), int(row['peak_y_vox']), int(row['peak_z_vox'])

        # Bounds check
        if (0 <= x < parc_data.shape[0] and
                0 <= y < parc_data.shape[1] and
                0 <= z < parc_data.shape[2]):
            label_id = int(parc_data[x, y, z])
        else:
            label_id = 0

        region_ids.append(label_id)

        if label_names and label_id in label_names:
            regions.append(label_names[label_id])
        elif label_id == 0:
            regions.append('Outside atlas')
        else:
            regions.append(f'SIGMA region {label_id}')

    df['region_id'] = region_ids
    df['region'] = regions
    return df


def generate_cluster_report(
    stat_file: Path,
    corrp_file: Path,
    contrast_name: str,
    output_dir: Path,
    sigma_parcellation: Optional[Path] = None,
    label_names: Optional[Dict[int, str]] = None,
    threshold: float = 0.95,
    min_cluster_size: int = 10
) -> Dict:
    """
    Generate comprehensive cluster report for a single contrast.

    Args:
        stat_file: T-statistic or F-statistic map
        corrp_file: Corrected p-value map
        contrast_name: Name of contrast for report
        output_dir: Output directory for reports
        sigma_parcellation: Path to SIGMA parcellation NIfTI
        label_names: Optional label index to name mapping
        threshold: Significance threshold (default: 0.95 = p<0.05)
        min_cluster_size: Minimum cluster size

    Returns:
        Dictionary with report paths and summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("neurofaune.tbss")
    logger.info(f"Generating cluster report: {contrast_name}")

    df = extract_clusters(
        corrp_file=corrp_file,
        stat_file=stat_file,
        threshold=threshold,
        min_cluster_size=min_cluster_size
    )

    if df.empty:
        return {
            'contrast_name': contrast_name,
            'n_clusters': 0,
            'total_voxels': 0,
            'significant': False
        }

    # Add SIGMA labels
    if sigma_parcellation is not None:
        df = add_sigma_labels(df, sigma_parcellation, label_names)

    # Save CSV
    csv_file = output_dir / f"{contrast_name}_clusters.csv"
    df.to_csv(csv_file, index=False)

    # Save HTML
    html_file = output_dir / f"{contrast_name}_clusters.html"
    _generate_html_report(df, contrast_name, html_file, threshold)

    total_voxels = int(df['size_voxels'].sum())

    logger.info(f"  {len(df)} clusters, {total_voxels} total voxels")
    logger.info(f"  CSV: {csv_file}")
    logger.info(f"  HTML: {html_file}")

    return {
        'contrast_name': contrast_name,
        'n_clusters': len(df),
        'total_voxels': total_voxels,
        'significant': True,
        'csv_file': str(csv_file),
        'html_file': str(html_file),
        'clusters': df.to_dict('records')
    }


def generate_reports_for_all_contrasts(
    randomise_output_dir: Path,
    output_dir: Path,
    contrast_names: Optional[List[str]] = None,
    sigma_parcellation: Optional[Path] = None,
    label_names: Optional[Dict[int, str]] = None,
    threshold: float = 0.95,
    min_cluster_size: int = 10
) -> Dict:
    """
    Generate cluster reports for all contrasts in randomise output.

    Args:
        randomise_output_dir: Directory containing randomise outputs
        output_dir: Directory for cluster reports
        contrast_names: Optional list of contrast names (order matches contrasts)
        sigma_parcellation: Path to SIGMA parcellation NIfTI
        label_names: Optional label index to name mapping
        threshold: Significance threshold
        min_cluster_size: Minimum cluster size

    Returns:
        Dictionary summarizing all reports
    """
    randomise_output_dir = Path(randomise_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corrp_files = sorted(randomise_output_dir.glob('*_tfce_corrp_tstat*.nii.gz'))

    if not corrp_files:
        # Try alternate naming pattern
        corrp_files = sorted(randomise_output_dir.glob('*_corrp_*.nii.gz'))

    reports = []

    for i, corrp_file in enumerate(corrp_files):
        # Find corresponding stat file
        stat_name = corrp_file.name.replace('_tfce_corrp_', '_')
        stat_file = randomise_output_dir / stat_name

        if not stat_file.exists():
            # Try without tfce prefix
            stat_name = corrp_file.name.replace('_corrp_', '_').replace('_tfce', '')
            stat_file = randomise_output_dir / stat_name

        if not stat_file.exists():
            logging.getLogger("neurofaune.tbss").warning(
                f"Stat file not found for {corrp_file.name}"
            )
            continue

        contrast_name = (
            contrast_names[i] if contrast_names and i < len(contrast_names)
            else f"contrast_{i+1}"
        )

        report = generate_cluster_report(
            stat_file=stat_file,
            corrp_file=corrp_file,
            contrast_name=contrast_name,
            output_dir=output_dir,
            sigma_parcellation=sigma_parcellation,
            label_names=label_names,
            threshold=threshold,
            min_cluster_size=min_cluster_size
        )

        reports.append(report)

    return {
        'reports': reports,
        'output_dir': str(output_dir),
        'n_significant': sum(1 for r in reports if r['significant'])
    }


def _generate_html_report(
    df: pd.DataFrame,
    contrast_name: str,
    output_file: Path,
    threshold: float
):
    """Generate HTML report for clusters."""
    total_voxels = int(df['size_voxels'].sum()) if 'size_voxels' in df.columns else 0

    # Select display columns
    display_cols = ['cluster_id', 'size_voxels', 'peak_stat', 'peak_corrp',
                    'peak_x_mm', 'peak_y_mm', 'peak_z_mm']
    if 'region' in df.columns:
        display_cols.append('region')

    display_df = df[display_cols].copy()
    display_df.columns = [
        'Cluster', 'Voxels', 'Peak T', 'Peak 1-p',
        'X (mm)', 'Y (mm)', 'Z (mm)'
    ] + (['Region'] if 'region' in df.columns else [])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cluster Report: {contrast_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #2E7D32; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #e8f5e9; padding: 15px; border-left: 4px solid #2E7D32; margin-bottom: 20px; }}
        .footer {{ margin-top: 20px; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>TBSS Cluster Report: {contrast_name}</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Threshold:</strong> {threshold} (p &lt; {1-threshold:.3f})</p>
        <p><strong>Number of clusters:</strong> {len(df)}</p>
        <p><strong>Total significant voxels:</strong> {total_voxels}</p>
        <p><strong>Atlas:</strong> SIGMA Rat Brain Atlas</p>
    </div>

    <h2>Clusters</h2>
    {display_df.to_html(index=False, border=0, float_format='%.3f')}

    <div class="footer">
        <p>Generated by neurofaune TBSS analysis pipeline</p>
        <p>Coordinates in SIGMA study-space (mm)</p>
    </div>
</body>
</html>"""

    with open(output_file, 'w') as f:
        f.write(html)
