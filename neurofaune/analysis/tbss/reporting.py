#!/usr/bin/env python3
"""
TBSS Summary Report Generation

Generates comprehensive HTML reports summarizing the full TBSS analysis:
- Subject inclusion/exclusion
- Skeleton parameters and coverage
- Every test, significant or not, with its whole-mask effect size (Cohen's d and
  CI), direction and extent -- from tests.csv written by run_tbss_stats
- Clusters with extent, peak, named SIGMA regions and effect -- from clusters.csv
- A Null results section: every test with no voxel surviving FWE correction
- Slice QC summary (if applicable)

The report renders the read-out tables (neurofaune.analysis.stats.readout); it
computes nothing itself, and never headlines a count of significant voxels.

Usage:
    from neurofaune.analysis.tbss.reporting import generate_tbss_report

    report_path = generate_tbss_report(
        analysis_name='dose_response',
        tbss_dir=Path('/study/analysis/tbss/dwi'),
        randomise_dir=Path('/study/analysis/tbss/dwi/randomise/dose_response'),
        output_file=Path('/study/analysis/tbss/dwi/reports/tbss_summary.html'),
        metrics=['FA', 'MD', 'AD', 'RD']
    )
"""

import html
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd


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


def load_slice_qc_summary(
    tbss_dir: Path,
    qc_dir: Optional[Path] = None,
    modality: Optional[str] = None,
) -> Optional[Dict]:
    """
    Load slice QC validity report if available.

    Checks both the centralized ``qc/tbss/{modality}/`` location and the
    legacy ``{tbss_dir}/slice_qc/`` location.

    Args:
        tbss_dir: TBSS output directory
        qc_dir: Optional study-level QC root
        modality: Modality name (e.g. 'dwi', 'msme')

    Returns:
        Slice QC report dict or None if not available
    """
    # Check centralized QC location first
    if qc_dir is not None and modality is not None:
        report_file = Path(qc_dir) / 'tbss' / modality / 'validity_report.json'
        if report_file.exists():
            with open(report_file) as f:
                return json.load(f)

    # Fall back to legacy location
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


def load_readout(randomise_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """tests.csv and clusters.csv written by run_tbss_stats (None when absent)."""
    out = []
    for name in ("tests.csv", "clusters.csv"):
        f = Path(randomise_dir) / name
        try:
            out.append(pd.read_csv(f) if f.exists() else None)
        except pd.errors.EmptyDataError:        # a run with no clusters writes an empty file
            out.append(pd.DataFrame())
    return out[0], out[1]


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
    tests, clusters = load_readout(randomise_dir)

    # Build HTML
    html = _build_html_report(
        analysis_name=analysis_name,
        manifest=manifest,
        analysis_summary=analysis_summary,
        slice_qc=slice_qc,
        skeleton_stats=skeleton_stats,
        tests=tests,
        clusters=clusters,
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
    tests: Optional[pd.DataFrame],
    clusters: Optional[pd.DataFrame],
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
    results_html = _build_results_section(tests, clusters, metrics)
    nulls_html = _build_null_section(tests)

    # Slice QC section
    slice_qc_html = _build_slice_qc_section(slice_qc)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>TBSS Report: {_e(analysis_name)}</title>
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
        <a href="#nulls">Null results</a>
        {('<a href="#sliceqc">Slice QC</a>' if slice_qc else '')}
    </div>

    <h1>TBSS Analysis Report: {_e(analysis_name)}</h1>
    <p>Generated: {timestamp}</p>

    {subjects_html}
    {skeleton_html}
    {params_html}
    {results_html}
    {nulls_html}
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
            f"<tr><td>{_e(cohort)}</td><td>{count}</td></tr>"
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
            f"<tr><td>{_e(e.get('subject', 'N/A'))}</td><td>{_e(e.get('reason', 'N/A'))}</td></tr>"
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
        ('Clusters', analysis_summary.get('cluster_definition', 'N/A')),
    ]

    rows = "".join(f"<tr><td><strong>{_e(k)}</strong></td><td>{_e(v)}</td></tr>" for k, v in params)

    return f"""
    <h2 id="parameters">Analysis Parameters</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        {rows}
    </table>
    """


def _e(x) -> str:
    """Text for HTML: escaped, and empty for a missing value. Every name and free-text
    value (contrasts, measures, regions, directions, subjects, analysis names) goes
    through this; only numbers formatted by _f are written without it."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return html.escape(str(x))


def _f(x, fmt: str = "{:+.2f}") -> str:
    return "&ndash;" if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


def _d_cell(r, prefix: str = "") -> str:
    return (f"{_f(r.get(prefix + 'd'))} [{_f(r.get(prefix + 'd_ci_low'))}, "
            f"{_f(r.get(prefix + 'd_ci_high'))}]")


def _means(r, prefix: str = "") -> str:
    if r.get("design") == "two-group":
        return f"{_f(r.get(prefix + 'mean_pos'), '{:.4g}')} vs {_f(r.get(prefix + 'mean_neg'), '{:.4g}')}"
    if r.get("design") == "one-sample":
        return _f(r.get(prefix + "mean"), "{:.4g}")
    return "&ndash;"


def _build_results_section(
    tests: Optional[pd.DataFrame],
    clusters: Optional[pd.DataFrame],
    metrics: List[str],
    max_rows: int = 20,
) -> str:
    """Every test per metric with its effect, then its clusters."""
    if tests is None or tests.empty:
        return ('<h2 id="results">Statistical Results</h2><div class="warning-box">'
                'tests.csv not found -- run_tbss_stats writes it; nothing is reported without it.</div>')
    sections = [_build_metric_results(m, tests[tests.metric == m],
                                      clusters[clusters.metric == m] if clusters is not None
                                      and not clusters.empty else None, max_rows)
                for m in metrics if (tests.metric == m).any()]
    return f"""
    <h2 id="results">Statistical Results</h2>
    <p>Every contrast is listed, whether or not any voxel survived correction. <em>Whole-mask d</em>
    is the effect of the contrast on each subject's mean over the whole analysis mask, with an exact
    95% CI; it is not selected on significance. <em>Raw d</em> is the same difference standardised by
    the raw (not covariate-adjusted) SD. Cluster effects are computed over voxels selected because
    they were significant, so they are inflated.</p>
    {''.join(sections)}
    """


def _build_metric_results(
    metric: str,
    tests: pd.DataFrame,
    clusters: Optional[pd.DataFrame],
    max_rows: int = 20,
) -> str:
    """One metric: a row per contrast, then the clusters (all of them, or a link to the CSV)."""
    rows = []
    for _, r in tests.iterrows():
        rows.append(
            f"<tr><td>{_e(r.contrast_name)}</td><td>{_e(r.tested_direction)}</td>"
            f"<td>{r.n} / {r.df}</td><td>{_d_cell(r, 'whole_')}</td><td>{_f(r.get('whole_d_raw'))}</td>"
            f"<td>{_means(r, 'whole_')}</td><td>{_e(r.get('whole_observed_direction', ''))}</td>"
            f"<td>{int(r.n_vox_fwe):,} ({100 * r.frac_mask_fwe:.1f}%)</td><td>{_f(r.min_p_fwe, '{:.3g}')}</td>"
            f"<td>{int(r.n_clusters)}</td></tr>")
    table = f"""
        <table>
            <tr><th>Contrast</th><th>Tests</th><th>n / df</th><th>Whole-mask d [95% CI]</th>
                <th>Raw d</th><th>Means</th><th>Observed</th><th>Voxels at FWE threshold</th>
                <th>Min FWE p</th><th>Clusters</th></tr>
            {''.join(rows)}
        </table>"""

    cluster_html = "<p>No clusters under the cluster definition.</p>"
    if clusters is not None and not clusters.empty:
        c = clusters.sort_values("n_voxels", ascending=False)
        shown = c.head(max_rows)
        crow = []
        for _, r in shown.iterrows():
            top = "; ".join(str(r.get("regions", "")).split("; ")[:3]) if "regions" in r else ""
            crow.append(
                f"<tr><td>{_e(r.contrast_name)}</td><td>{int(r.cluster)}</td><td>{int(r.n_voxels):,}</td>"
                f"<td>{r.mm3:.2f}</td><td>{r.peak_t:.2f}</td><td>{_e(r.peak_xyz_mm)}</td>"
                f"<td>{_e(r.get('peak_region', '') or '')}</td><td>{_e(top)}</td>"
                f"<td>{_f(r.min_p_fwe, '{:.3g}')}</td><td>{_d_cell(r)}</td></tr>")
        more = (f"<p>Showing the {len(shown)} largest of {len(c)} clusters; every cluster, with all "
                f"regions it covers, is in <a href=\"clusters.csv\">clusters.csv</a>.</p>"
                if len(c) > len(shown) else
                '<p>Full table, with every region each cluster covers: <a href="clusters.csv">clusters.csv</a>.</p>')
        cluster_html = f"""
        <table>
            <tr><th>Contrast</th><th>#</th><th>Voxels</th><th>mm&sup3;</th><th>Peak t</th>
                <th>Peak (mm)</th><th>Peak region</th><th>Largest regions (voxels)</th>
                <th>Min FWE p</th><th>Cluster d [95% CI] (selected)</th></tr>
            {''.join(crow)}
        </table>{more}"""

    return f"""
    <div class="metric-section">
        <div class="metric-header">{_e(metric)}</div>
        {table}
        <h3>{_e(metric)}: clusters</h3>
        {cluster_html}
    </div>
    """


def _build_null_section(tests: Optional[pd.DataFrame]) -> str:
    """Every test in which no voxel survived correction, with its effect size."""
    if tests is None or tests.empty:
        return ""
    nulls = tests[~tests.significant_fwe.astype(bool)]
    if nulls.empty:
        body = "<p>Every test had at least one voxel surviving FWE correction.</p>"
    else:
        rows = "".join(
            f"<tr><td>{_e(r.metric)}</td><td>{_e(r.contrast_name)}</td><td>{_e(r.tested_direction)}</td>"
            f"<td>{r.n} / {r.df}</td><td>{_d_cell(r, 'whole_')}</td><td>{_f(r.min_p_fwe, '{:.3g}')}</td></tr>"
            for _, r in nulls.iterrows())
        body = f"""
        <table>
            <tr><th>Metric</th><th>Contrast</th><th>Tests</th><th>n / df</th>
                <th>Whole-mask d [95% CI]</th><th>Min FWE p</th></tr>
            {rows}
        </table>"""
    return f"""
    <h2 id="nulls">Null results</h2>
    <p>{len(nulls)} of {len(tests)} tests had no voxel surviving FWE correction. They are results:
    each is listed with its whole-mask effect and interval.</p>
    {body}
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
    <p><strong>Validity masks:</strong> {_e(slice_qc.get('validity_masks_dir', 'N/A'))}</p>
    <p><strong>Analysis mask:</strong> {_e(slice_qc.get('analysis_mask', 'N/A'))}</p>
    """
