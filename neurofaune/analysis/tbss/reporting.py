#!/usr/bin/env python3
"""
TBSS Summary Report Generation

Generates comprehensive HTML reports summarizing the full TBSS analysis:
- Subject inclusion/exclusion
- Skeleton parameters and coverage
- Statistical results across all metrics and contrasts
- Significant clusters with SIGMA atlas labels
- Slice QC summary (if applicable)

Usage:
    from neurofaune.analysis.tbss.reporting import generate_tbss_report

    report_path = generate_tbss_report(
        analysis_name='dose_response',
        tbss_dir=Path('/study/analysis/tbss'),
        randomise_dir=Path('/study/analysis/tbss/randomise/dose_response'),
        output_file=Path('/study/analysis/tbss/reports/tbss_summary.html'),
        metrics=['FA', 'MD', 'AD', 'RD']
    )
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np


def load_analysis_summary(randomise_dir: Path) -> Optional[Dict]:
    """
    Load analysis summary JSON from randomise output directory.

    Args:
        randomise_dir: Directory containing randomise results

    Returns:
        Analysis summary dict or None if not found
    """
    summary_file = randomise_dir / 'analysis_summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return None


def load_subject_manifest(tbss_dir: Path) -> Optional[Dict]:
    """
    Load subject manifest from TBSS preparation directory.

    Args:
        tbss_dir: TBSS output directory

    Returns:
        Manifest dict or None if not found
    """
    manifest_file = tbss_dir / 'subject_manifest.json'
    if manifest_file.exists():
        with open(manifest_file) as f:
            return json.load(f)
    return None


def load_slice_qc_summary(tbss_dir: Path) -> Optional[Dict]:
    """
    Load slice QC validity report if available.

    Args:
        tbss_dir: TBSS output directory

    Returns:
        Slice QC report dict or None if not available
    """
    report_file = tbss_dir / 'slice_qc' / 'validity_report.json'
    if report_file.exists():
        with open(report_file) as f:
            return json.load(f)
    return None


def get_skeleton_stats(tbss_dir: Path) -> Dict:
    """
    Compute skeleton coverage statistics.

    Args:
        tbss_dir: TBSS output directory

    Returns:
        Dict with skeleton voxel counts and coverage info
    """
    stats_dir = tbss_dir / 'stats'
    stats = {}

    skeleton_mask = stats_dir / 'mean_FA_skeleton_mask.nii.gz'
    if skeleton_mask.exists():
        img = nib.load(skeleton_mask)
        data = img.get_fdata() > 0
        stats['skeleton_voxels'] = int(np.sum(data))
        voxel_dims = img.header.get_zooms()[:3]
        voxel_vol = float(np.prod(voxel_dims))
        stats['skeleton_volume_mm3'] = stats['skeleton_voxels'] * voxel_vol

    mean_fa = stats_dir / 'mean_FA.nii.gz'
    if mean_fa.exists():
        img = nib.load(mean_fa)
        data = img.get_fdata()
        brain_mask = data > 0
        stats['brain_voxels'] = int(np.sum(brain_mask))
        stats['mean_fa_value'] = float(np.mean(data[brain_mask])) if np.any(brain_mask) else 0.0

    return stats


def load_cluster_reports(randomise_dir: Path, metrics: List[str]) -> Dict[str, List[Dict]]:
    """
    Load cluster report CSVs for all metrics.

    Args:
        randomise_dir: Directory containing randomise results
        metrics: List of metrics to check

    Returns:
        Dict mapping metric -> list of cluster report dicts
    """
    import pandas as pd

    all_clusters = {}

    for metric in metrics:
        reports_dir = randomise_dir / f'cluster_reports_{metric}'
        if not reports_dir.exists():
            continue

        metric_clusters = []
        for csv_file in sorted(reports_dir.glob('*_clusters.csv')):
            try:
                df = pd.read_csv(csv_file)
                contrast_name = csv_file.stem.replace('_clusters', '')
                metric_clusters.append({
                    'contrast_name': contrast_name,
                    'n_clusters': len(df),
                    'total_voxels': int(df['size_voxels'].sum()) if not df.empty else 0,
                    'clusters': df.to_dict('records') if not df.empty else []
                })
            except Exception:
                continue

        if metric_clusters:
            all_clusters[metric] = metric_clusters

    return all_clusters


def generate_tbss_report(
    analysis_name: str,
    tbss_dir: Path,
    randomise_dir: Path,
    output_file: Path,
    metrics: List[str] = None,
    config: Optional[Dict] = None
) -> Path:
    """
    Generate comprehensive HTML summary report for TBSS analysis.

    Aggregates information from all pipeline stages into a single
    navigable HTML report.

    Args:
        analysis_name: Name of the analysis run
        tbss_dir: TBSS data preparation directory
        randomise_dir: Directory containing randomise results
        output_file: Output path for HTML report
        metrics: Metrics to include (default: ['FA', 'MD', 'AD', 'RD'])
        config: Optional config dict

    Returns:
        Path to generated HTML report
    """
    logger = logging.getLogger("neurofaune.tbss")

    if metrics is None:
        metrics = ['FA', 'MD', 'AD', 'RD']

    tbss_dir = Path(tbss_dir)
    randomise_dir = Path(randomise_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Gather data from all sources
    manifest = load_subject_manifest(tbss_dir)
    analysis_summary = load_analysis_summary(randomise_dir)
    slice_qc = load_slice_qc_summary(tbss_dir)
    skeleton_stats = get_skeleton_stats(tbss_dir)
    cluster_reports = load_cluster_reports(randomise_dir, metrics)

    # Build HTML
    html = _build_html_report(
        analysis_name=analysis_name,
        manifest=manifest,
        analysis_summary=analysis_summary,
        slice_qc=slice_qc,
        skeleton_stats=skeleton_stats,
        cluster_reports=cluster_reports,
        metrics=metrics
    )

    with open(output_file, 'w') as f:
        f.write(html)

    logger.info(f"TBSS summary report: {output_file}")
    return output_file


def _build_html_report(
    analysis_name: str,
    manifest: Optional[Dict],
    analysis_summary: Optional[Dict],
    slice_qc: Optional[Dict],
    skeleton_stats: Dict,
    cluster_reports: Dict[str, List[Dict]],
    metrics: List[str]
) -> str:
    """Build the complete HTML report string."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Subject section
    subjects_html = _build_subjects_section(manifest)

    # Skeleton section
    skeleton_html = _build_skeleton_section(skeleton_stats)

    # Analysis parameters section
    params_html = _build_params_section(analysis_summary)

    # Results section (per metric)
    results_html = _build_results_section(analysis_summary, cluster_reports, metrics)

    # Slice QC section
    slice_qc_html = _build_slice_qc_section(slice_qc)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>TBSS Report: {analysis_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a5276;
            border-bottom: 3px solid #2E7D32;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2E7D32;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #1a5276;
            margin-top: 20px;
        }}
        .nav {{
            background: #1a5276;
            padding: 10px 20px;
            margin: -30px -30px 30px -30px;
        }}
        .nav a {{
            color: white;
            text-decoration: none;
            margin-right: 20px;
            font-size: 0.9em;
        }}
        .nav a:hover {{
            text-decoration: underline;
        }}
        .summary-box {{
            background-color: #e8f5e9;
            padding: 15px 20px;
            border-left: 4px solid #2E7D32;
            margin: 15px 0;
        }}
        .warning-box {{
            background-color: #fff3e0;
            padding: 15px 20px;
            border-left: 4px solid #f57c00;
            margin: 15px 0;
        }}
        .info-box {{
            background-color: #e3f2fd;
            padding: 15px 20px;
            border-left: 4px solid #1976d2;
            margin: 15px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 0.9em;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #2E7D32;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .significant {{
            color: #2E7D32;
            font-weight: bold;
        }}
        .not-significant {{
            color: #666;
        }}
        .metric-section {{
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
        }}
        .metric-header {{
            font-size: 1.1em;
            font-weight: bold;
            color: #1a5276;
            margin-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }}
        .stat-card {{
            background: #f5f5f5;
            padding: 12px;
            border-radius: 4px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.4em;
            font-weight: bold;
            color: #1a5276;
        }}
        .stat-label {{
            font-size: 0.85em;
            color: #666;
            margin-top: 4px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
            color: #888;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="nav">
        <a href="#subjects">Subjects</a>
        <a href="#skeleton">Skeleton</a>
        <a href="#parameters">Parameters</a>
        <a href="#results">Results</a>
        {('<a href="#sliceqc">Slice QC</a>' if slice_qc else '')}
    </div>

    <h1>TBSS Analysis Report: {analysis_name}</h1>
    <p>Generated: {timestamp}</p>

    {subjects_html}
    {skeleton_html}
    {params_html}
    {results_html}
    {slice_qc_html}

    <div class="footer">
        <p>Generated by neurofaune TBSS analysis pipeline</p>
        <p>Atlas: SIGMA Rat Brain Atlas (study-space)</p>
        <p>Statistical inference: FSL randomise with TFCE</p>
    </div>
</div>
</body>
</html>"""

    return html


def _build_subjects_section(manifest: Optional[Dict]) -> str:
    """Build the subjects summary section."""
    if manifest is None:
        return '<h2 id="subjects">Subjects</h2><p>Subject manifest not available.</p>'

    n_included = manifest.get('subjects_included', 0)
    n_excluded = manifest.get('subjects_excluded', 0)
    n_total = n_included + n_excluded

    # Cohort breakdown
    cohort_counts = {}
    for subj in manifest.get('subjects', []):
        cohort = subj.get('cohort', 'unknown')
        cohort_counts[cohort] = cohort_counts.get(cohort, 0) + 1

    cohort_html = ""
    if cohort_counts:
        rows = "".join(
            f"<tr><td>{cohort}</td><td>{count}</td></tr>"
            for cohort, count in sorted(cohort_counts.items())
        )
        cohort_html = f"""
        <table>
            <tr><th>Cohort</th><th>N</th></tr>
            {rows}
        </table>"""

    # Exclusions
    exclusion_html = ""
    excluded = manifest.get('excluded_subjects', [])
    if excluded:
        rows = "".join(
            f"<tr><td>{e.get('subject', 'N/A')}</td><td>{e.get('reason', 'N/A')}</td></tr>"
            for e in excluded[:20]  # Limit display
        )
        exclusion_html = f"""
        <h3>Excluded Subjects</h3>
        <table>
            <tr><th>Subject</th><th>Reason</th></tr>
            {rows}
        </table>"""
        if len(excluded) > 20:
            exclusion_html += f"<p>... and {len(excluded) - 20} more</p>"

    return f"""
    <h2 id="subjects">Subjects</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{n_included}</div>
            <div class="stat-label">Included</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{n_excluded}</div>
            <div class="stat-label">Excluded</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{n_total}</div>
            <div class="stat-label">Total</div>
        </div>
    </div>
    {cohort_html}
    {exclusion_html}
    """


def _build_skeleton_section(skeleton_stats: Dict) -> str:
    """Build the skeleton statistics section."""
    if not skeleton_stats:
        return '<h2 id="skeleton">White Matter Skeleton</h2><p>Skeleton statistics not available.</p>'

    cards = []
    if 'skeleton_voxels' in skeleton_stats:
        cards.append(f"""
        <div class="stat-card">
            <div class="stat-value">{skeleton_stats['skeleton_voxels']:,}</div>
            <div class="stat-label">Skeleton Voxels</div>
        </div>""")
    if 'skeleton_volume_mm3' in skeleton_stats:
        cards.append(f"""
        <div class="stat-card">
            <div class="stat-value">{skeleton_stats['skeleton_volume_mm3']:.1f}</div>
            <div class="stat-label">Skeleton Volume (mm3)</div>
        </div>""")
    if 'brain_voxels' in skeleton_stats:
        cards.append(f"""
        <div class="stat-card">
            <div class="stat-value">{skeleton_stats['brain_voxels']:,}</div>
            <div class="stat-label">Brain Voxels</div>
        </div>""")
    if 'mean_fa_value' in skeleton_stats:
        cards.append(f"""
        <div class="stat-card">
            <div class="stat-value">{skeleton_stats['mean_fa_value']:.3f}</div>
            <div class="stat-label">Mean FA (brain)</div>
        </div>""")

    return f"""
    <h2 id="skeleton">White Matter Skeleton</h2>
    <div class="stats-grid">
        {''.join(cards)}
    </div>
    """


def _build_params_section(analysis_summary: Optional[Dict]) -> str:
    """Build the analysis parameters section."""
    if analysis_summary is None:
        return '<h2 id="parameters">Analysis Parameters</h2><p>Analysis summary not available.</p>'

    params = [
        ('Permutations', analysis_summary.get('n_permutations', 'N/A')),
        ('TFCE', 'Yes' if analysis_summary.get('tfce', True) else 'No'),
        ('Metrics', ', '.join(analysis_summary.get('metrics', []))),
        ('N Subjects', analysis_summary.get('n_subjects', 'N/A')),
    ]

    rows = "".join(f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>" for k, v in params)

    return f"""
    <h2 id="parameters">Analysis Parameters</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        {rows}
    </table>
    """


def _build_results_section(
    analysis_summary: Optional[Dict],
    cluster_reports: Dict[str, List[Dict]],
    metrics: List[str]
) -> str:
    """Build the results section with per-metric cluster tables."""

    sections = []

    for metric in metrics:
        metric_html = _build_metric_results(metric, analysis_summary, cluster_reports.get(metric))
        sections.append(metric_html)

    return f"""
    <h2 id="results">Statistical Results</h2>
    {''.join(sections)}
    """


def _build_metric_results(
    metric: str,
    analysis_summary: Optional[Dict],
    cluster_reports: Optional[List[Dict]]
) -> str:
    """Build results section for a single metric."""

    # Get contrast-level summary from analysis_summary
    contrast_summary = ""
    if analysis_summary and 'results' in analysis_summary:
        metric_results = analysis_summary['results'].get(metric, {})
        contrasts = metric_results.get('contrasts', [])

        if contrasts:
            rows = []
            for c in contrasts:
                status_class = 'significant' if c.get('significant') else 'not-significant'
                status_text = 'SIGNIFICANT' if c.get('significant') else 'n.s.'
                n_vox = c.get('n_significant_voxels', 0)
                rows.append(
                    f"<tr><td>{c.get('type', '')}{c.get('contrast_number', '')}</td>"
                    f"<td>{n_vox:,}</td>"
                    f"<td class=\"{status_class}\">{status_text}</td></tr>"
                )

            contrast_summary = f"""
            <table>
                <tr><th>Contrast</th><th>Significant Voxels</th><th>Status</th></tr>
                {''.join(rows)}
            </table>"""

    # Cluster details
    cluster_html = ""
    if cluster_reports:
        for report in cluster_reports:
            if not report.get('clusters'):
                continue

            contrast_name = report.get('contrast_name', 'unknown')
            clusters = report['clusters']

            rows = []
            for cl in clusters[:20]:  # Limit display
                region = cl.get('region', '')
                rows.append(
                    f"<tr>"
                    f"<td>{cl.get('cluster_id', '')}</td>"
                    f"<td>{cl.get('size_voxels', 0):,}</td>"
                    f"<td>{cl.get('peak_stat', 0):.2f}</td>"
                    f"<td>{cl.get('peak_x_mm', 0):.1f}, "
                    f"{cl.get('peak_y_mm', 0):.1f}, "
                    f"{cl.get('peak_z_mm', 0):.1f}</td>"
                    f"<td>{region}</td>"
                    f"</tr>"
                )

            cluster_html += f"""
            <h3>{contrast_name}</h3>
            <table>
                <tr><th>#</th><th>Voxels</th><th>Peak T</th><th>Peak (mm)</th><th>Region</th></tr>
                {''.join(rows)}
            </table>"""
            if len(clusters) > 20:
                cluster_html += f"<p>... and {len(clusters) - 20} more clusters</p>"

    # Determine significance status for the box
    has_significant = False
    if analysis_summary and 'results' in analysis_summary:
        metric_results = analysis_summary['results'].get(metric, {})
        has_significant = metric_results.get('n_significant_contrasts', 0) > 0

    box_class = 'summary-box' if has_significant else 'info-box'
    status_msg = 'Significant results found' if has_significant else 'No significant results'

    return f"""
    <div class="metric-section">
        <div class="metric-header">{metric}</div>
        <div class="{box_class}"><strong>{status_msg}</strong></div>
        {contrast_summary}
        {cluster_html}
    </div>
    """


def _build_slice_qc_section(slice_qc: Optional[Dict]) -> str:
    """Build the slice QC summary section."""
    if slice_qc is None:
        return ""

    imputed = slice_qc.get('imputed_files', {})
    n_metrics_imputed = len(imputed)

    return f"""
    <h2 id="sliceqc">Slice-Level QC</h2>
    <div class="info-box">
        <p>Slice-level validity masking was applied to handle partial-coverage DTI artifacts.</p>
        <p>Bad slices were imputed with group mean values.</p>
    </div>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{n_metrics_imputed}</div>
            <div class="stat-label">Metrics Imputed</div>
        </div>
    </div>
    <p><strong>Validity masks:</strong> {slice_qc.get('validity_masks_dir', 'N/A')}</p>
    <p><strong>Analysis mask:</strong> {slice_qc.get('analysis_mask', 'N/A')}</p>
    """
