#!/usr/bin/env python3
"""
Generate functional registration QC: per-subject figures + group reports.

Produces:
  Per-subject (qc/subjects/{sub}/{ses}/func/):
    - figures/{sub}_{ses}_bold_to_template_registration.png (4-row checkerboard)
    - {sub}_{ses}_registration_qc_metrics.json (correlation, NMI)

  Group-level (qc/reports/):
    - func_registration_gallery.html   (sorted worst→best, filters)
    - func_preprocessing_dashboard.html (all metrics, sortable table)
    - func_preprocessing_summary.csv    (one row per run, qc_pass column)

Usage:
    uv run python scripts/generate_func_registration_qc.py /mnt/arborea/bpa-rat
    uv run python scripts/generate_func_registration_qc.py /mnt/arborea/bpa-rat --subjects sub-Rat1 sub-Rat102
    uv run python scripts/generate_func_registration_qc.py /mnt/arborea/bpa-rat --skip-figures
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.config import load_config, get_config_value
from neurofaune.preprocess.qc.batch_summary import (
    get_subject_qc_dir,
    get_reports_dir,
    collect_qc_metrics,
)


# ─── Defaults for config thresholds ────────────────────────────────────────
DEFAULT_THRESHOLDS = {
    'max_mean_fd': 0.05,
    'max_pct_bad_volumes': 50.0,
    'min_registration_correlation': 0.5,
    'min_registration_nmi': 1.2,
}


# ─── Per-subject registration QC ───────────────────────────────────────────

def _session_to_cohort(session: str) -> str:
    """ses-p60 → p60"""
    return session.replace('ses-', '')


def _find_template(study_root: Path, cohort: str) -> Path:
    """Locate the age-specific T2w template."""
    tpl = study_root / 'templates' / 'anat' / cohort / f'tpl-BPARat_{cohort}_T2w.nii.gz'
    if not tpl.exists():
        raise FileNotFoundError(f"Template not found: {tpl}")
    return tpl


def _detect_bold_coverage(warped_data: np.ndarray) -> dict:
    """Detect the slice range with actual BOLD signal on the template grid."""
    slice_means = np.array([
        warped_data[:, :, s].mean() for s in range(warped_data.shape[2])
    ])
    threshold = slice_means.max() * 0.01
    nonzero = np.where(slice_means > threshold)[0]
    if len(nonzero) == 0:
        return {'start_slice': 0, 'end_slice': 0, 'n_slices': 0}
    return {
        'start_slice': int(nonzero[0]),
        'end_slice': int(nonzero[-1]),
        'n_slices': int(nonzero[-1] - nonzero[0] + 1),
    }


def generate_per_subject_qc(
    study_root: Path,
    subject: str,
    session: str,
    skip_figures: bool = False,
) -> dict | None:
    """
    Generate registration QC for one subject/session.

    Returns dict with metrics or None if data missing.
    """
    import nibabel as nib

    transforms_dir = study_root / 'transforms' / subject / session
    warped_file = transforms_dir / 'BOLD_to_template_Warped.nii.gz'
    if not warped_file.exists():
        return None

    cohort = _session_to_cohort(session)
    try:
        template_file = _find_template(study_root, cohort)
    except FileNotFoundError:
        print(f"  Warning: template not found for {cohort}, skipping {subject}/{session}")
        return None

    # Compute metrics
    from neurofaune.preprocess.qc.func.registration_qc import calculate_registration_metrics
    metrics = calculate_registration_metrics(fixed=template_file, moving=warped_file)

    if 'error' in metrics:
        print(f"  Warning: metrics error for {subject}/{session}: {metrics['error']}")
        return None

    # Detect coverage
    warped_data = nib.load(warped_file).get_fdata()
    coverage = _detect_bold_coverage(warped_data)
    metrics['coverage'] = coverage

    # Output directory
    qc_dir = get_subject_qc_dir(study_root, subject, session, 'func')
    figures_dir = qc_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_file = qc_dir / f'{subject}_{session}_registration_qc_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Generate 4-row figure
    figure_file = figures_dir / f'{subject}_{session}_bold_to_template_registration.png'
    if not skip_figures:
        from neurofaune.templates.registration_qc import generate_registration_qc_figure

        # Use coverage-informed slice selection
        start = coverage['start_slice']
        end = coverage['end_slice']
        n_template_slices = warped_data.shape[2]
        if end > start:
            margin = max(1, (end - start) // 10)
            s0 = max(0, start - margin)
            s1 = min(n_template_slices - 1, end + margin)
            slice_indices = np.linspace(s0, s1, 9, dtype=int).tolist()
        else:
            slice_indices = None

        generate_registration_qc_figure(
            fixed_file=template_file,
            warped_file=warped_file,
            output_file=figure_file,
            title=f"BOLD→Template Registration — {subject} {session}",
            slice_indices=slice_indices,
        )

    return {
        'subject': subject,
        'session': session,
        'cohort': cohort,
        'correlation': metrics['correlation'],
        'nmi': metrics['normalized_mutual_information'],
        'n_voxels_overlap': metrics['n_voxels_overlap'],
        'coverage_start': coverage['start_slice'],
        'coverage_end': coverage['end_slice'],
        'coverage_n_slices': coverage['n_slices'],
        'figure': str(figure_file) if not skip_figures else None,
        'metrics_json': str(metrics_file),
    }


# ─── Collect all func QC metrics into unified rows ─────────────────────────

def _load_ica_summary(qc_dir: Path, subject: str, session: str) -> dict:
    """Extract summary stats from ica_classification.json."""
    ica_file = qc_dir / 'subjects' / subject / session / 'func' / f'{subject}_{session}_ica_classification.json'
    if not ica_file.exists():
        return {}
    try:
        with open(ica_file) as f:
            data = json.load(f)
        components = data.get('components', [])
        n_total = data.get('n_components', len(components))
        n_noise = sum(1 for c in components if c.get('label') == 'noise')
        # Estimate variance removed (noise_score-weighted approximation)
        pct_noise = (n_noise / n_total * 100) if n_total > 0 else 0.0
        return {
            'ica_n_components': n_total,
            'ica_n_noise': n_noise,
            'ica_pct_noise_components': round(pct_noise, 1),
        }
    except (json.JSONDecodeError, IOError):
        return {}


def build_comprehensive_df(
    study_root: Path,
    registration_results: list[dict],
) -> pd.DataFrame:
    """
    Build one-row-per-run DataFrame with motion + registration + ICA metrics.
    """
    qc_root = study_root / 'qc'

    # Collect existing func QC metrics (motion, confounds)
    existing_df = collect_qc_metrics(qc_root, 'func')

    # Build registration DataFrame
    reg_df = pd.DataFrame(registration_results)
    reg_df = reg_df.rename(columns={
        'correlation': 'registration_correlation',
        'nmi': 'registration_nmi',
        'n_voxels_overlap': 'registration_n_voxels',
    })

    # Merge on subject + session
    if not existing_df.empty:
        df = existing_df.merge(
            reg_df[['subject', 'session', 'cohort', 'registration_correlation',
                     'registration_nmi', 'registration_n_voxels',
                     'coverage_start', 'coverage_end', 'coverage_n_slices']],
            on=['subject', 'session'],
            how='outer',
            suffixes=('', '_reg'),
        )
        # Fill cohort from registration if missing
        if 'cohort_reg' in df.columns:
            df['cohort'] = df['cohort'].fillna(df['cohort_reg'])
            df.drop(columns=['cohort_reg'], inplace=True)
    else:
        df = reg_df.copy()

    # Add ICA summary
    ica_records = []
    for _, row in df.iterrows():
        ica = _load_ica_summary(qc_root, row['subject'], row['session'])
        ica['subject'] = row['subject']
        ica['session'] = row['session']
        ica_records.append(ica)

    if ica_records:
        ica_df = pd.DataFrame(ica_records)
        if not ica_df.drop(columns=['subject', 'session']).dropna(how='all').empty:
            df = df.merge(ica_df, on=['subject', 'session'], how='left')

    return df


# ─── QC pass/fail logic ────────────────────────────────────────────────────

def compute_qc_pass(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Add qc_pass boolean and qc_fail_reasons columns based on thresholds.
    """
    passes = []
    reasons_list = []

    for _, row in df.iterrows():
        fails = []

        # Motion thresholds
        max_mean_fd = thresholds.get('max_mean_fd', DEFAULT_THRESHOLDS['max_mean_fd'])
        val = row.get('motion_mean_fd')
        if pd.notna(val) and val > max_mean_fd:
            fails.append(f"mean_fd={val:.4f}>{max_mean_fd}")

        max_pct = thresholds.get('max_pct_bad_volumes', DEFAULT_THRESHOLDS['max_pct_bad_volumes'])
        val = row.get('motion_pct_bad_volumes')
        if pd.notna(val) and val > max_pct:
            fails.append(f"pct_bad_vol={val:.1f}%>{max_pct}%")

        # Registration thresholds
        min_corr = thresholds.get('min_registration_correlation',
                                  DEFAULT_THRESHOLDS['min_registration_correlation'])
        val = row.get('registration_correlation')
        if pd.notna(val) and val < min_corr:
            fails.append(f"reg_corr={val:.3f}<{min_corr}")

        min_nmi = thresholds.get('min_registration_nmi',
                                 DEFAULT_THRESHOLDS['min_registration_nmi'])
        val = row.get('registration_nmi')
        if pd.notna(val) and val < min_nmi:
            fails.append(f"reg_nmi={val:.3f}<{min_nmi}")

        passes.append(len(fails) == 0)
        reasons_list.append('; '.join(fails) if fails else '')

    df = df.copy()
    df['qc_pass'] = passes
    df['qc_fail_reasons'] = reasons_list
    return df


def _classify_registration_quality(corr: float, nmi: float) -> str:
    """Classify as good/fair/poor."""
    if corr > 0.7 and nmi > 1.5:
        return 'good'
    if corr > 0.5 and nmi > 1.2:
        return 'fair'
    return 'poor'


# ─── Gallery HTML ───────────────────────────────────────────────────────────

def _write_gallery_html(
    output_path: Path,
    cards: list[dict],
    n_total: int,
    n_good: int,
    n_fair: int,
    n_poor: int,
    cohort_counts: dict[str, int],
):
    """Write sorted registration gallery HTML."""
    cohorts = sorted(cohort_counts.keys())
    cohort_options = '\n'.join(
        f'            <option value="{c}">{c} ({cohort_counts[c]})</option>'
        for c in cohorts
    )

    cards_html = []
    for card in cards:
        border_color = {'good': '#27ae60', 'fair': '#f39c12', 'poor': '#e74c3c'}[card['quality']]

        # Link to per-subject figure
        link = f'href="{card["figure_rel"]}" target="_blank"' if card.get('figure_rel') else ''

        cards_html.append(f"""        <div class="card {card['quality']}" data-subject="{card['subject']}"
             data-session="{card['session']}" data-cohort="{card['cohort']}"
             data-quality="{card['quality']}" data-corr="{card['correlation']:.4f}"
             data-nmi="{card['nmi']:.4f}"
             style="border-left: 5px solid {border_color};">
            <img src="{card['thumb']}" onclick="openModal(this.src)" alt="{card['subject']}">
            <div class="card-info">
                <div class="card-label">
                    {'<a ' + link + '>' if link else ''}{card['subject']}{'</a>' if link else ''}<br>
                    <span class="session">{card['session']}</span>
                    <span class="badge {card['quality']}">{card['quality']}</span>
                </div>
                <div class="card-metrics">
                    Corr: {card['correlation']:.3f} | NMI: {card['nmi']:.3f}
                </div>
            </div>
        </div>
""")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Functional Registration QC Gallery</title>
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
        .summary-box.fair .number {{ color: #f39c12; }}
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
                  font-size: 10px; font-weight: 600; text-transform: uppercase; margin-left: 4px; }}
        .badge.good {{ background: #d4edda; color: #155724; }}
        .badge.fair {{ background: #fff3cd; color: #856404; }}
        .badge.poor {{ background: #f8d7da; color: #721c24; }}

        .modal {{ display: none; position: fixed; top: 0; left: 0;
                  width: 100%; height: 100%; background: rgba(0,0,0,0.92); z-index: 1000; }}
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
        <h1>Functional Registration QC Gallery</h1>
        <p>BOLD-to-Template registration sorted worst-to-best | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
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
        <div class="summary-box fair">
            <div class="number">{n_fair}</div>
            <div class="label">Fair</div>
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
                <option value="fair">Fair ({n_fair})</option>
                <option value="poor">Poor ({n_poor})</option>
            </select>
        </label>
        <label>Search:
            <input type="text" id="filterSearch" onkeyup="applyFilters()" placeholder="e.g. Rat102">
        </label>
        <label>Sort:
            <select id="sortBy" onchange="applyFilters()">
                <option value="corr-asc">Correlation (worst first)</option>
                <option value="corr-desc">Correlation (best first)</option>
                <option value="nmi-asc">NMI (worst first)</option>
                <option value="nmi-desc">NMI (best first)</option>
                <option value="subject">Subject</option>
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

            const grid = document.getElementById('grid');
            const sorted = cards.slice().sort((a, b) => {{
                if (a.style.display === 'none' && b.style.display !== 'none') return 1;
                if (a.style.display !== 'none' && b.style.display === 'none') return -1;
                if (sortBy === 'subject') {{
                    return a.dataset.subject.localeCompare(b.dataset.subject) ||
                           a.dataset.session.localeCompare(b.dataset.session);
                }}
                let aVal, bVal;
                if (sortBy.startsWith('corr')) {{
                    aVal = parseFloat(a.dataset.corr);
                    bVal = parseFloat(b.dataset.corr);
                }} else {{
                    aVal = parseFloat(a.dataset.nmi);
                    bVal = parseFloat(b.dataset.nmi);
                }}
                return sortBy.endsWith('asc') ? aVal - bVal : bVal - aVal;
            }});
            sorted.forEach(c => grid.appendChild(c));

            document.getElementById('countDisplay').textContent =
                `Showing ${{visible}} of {n_total} sessions`;

            visibleImages = sorted.filter(c => c.style.display !== 'none')
                                  .map(c => c.querySelector('img').src);
        }}

        function openModal(src) {{
            modalIdx = visibleImages.indexOf(src);
            document.getElementById('modal-img').src = src;
            document.getElementById('modal').style.display = 'block';
        }}
        function closeModal() {{ document.getElementById('modal').style.display = 'none'; }}
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

        // Initialize
        applyFilters();
    </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)


# ─── Dashboard HTML ─────────────────────────────────────────────────────────

def _cell_color(val, metric: str, thresholds: dict) -> str:
    """Return CSS color class for a cell value based on thresholds."""
    if pd.isna(val):
        return ''

    checks = {
        'motion_mean_fd': ('max', thresholds.get('max_mean_fd', DEFAULT_THRESHOLDS['max_mean_fd'])),
        'motion_pct_bad_volumes': ('max', thresholds.get('max_pct_bad_volumes', DEFAULT_THRESHOLDS['max_pct_bad_volumes'])),
        'registration_correlation': ('min', thresholds.get('min_registration_correlation', DEFAULT_THRESHOLDS['min_registration_correlation'])),
        'registration_nmi': ('min', thresholds.get('min_registration_nmi', DEFAULT_THRESHOLDS['min_registration_nmi'])),
    }

    if metric not in checks:
        return ''

    direction, thresh = checks[metric]
    if direction == 'max':
        if val > thresh:
            return 'cell-red'
        if val > thresh * 0.7:
            return 'cell-yellow'
        return 'cell-green'
    else:
        if val < thresh:
            return 'cell-red'
        if val < thresh * 1.3:
            return 'cell-yellow'
        return 'cell-green'


def _write_dashboard_html(
    output_path: Path,
    df: pd.DataFrame,
    thresholds: dict,
):
    """Write comprehensive func preprocessing dashboard HTML."""

    # Columns to display (in order)
    display_cols = [
        'subject', 'session', 'cohort',
        'motion_mean_fd', 'motion_max_fd', 'motion_pct_bad_volumes', 'motion_mean_dvars',
        'registration_correlation', 'registration_nmi',
        'coverage_n_slices',
        'ica_n_components', 'ica_n_noise', 'ica_pct_noise_components',
        'qc_pass', 'qc_fail_reasons',
    ]

    available_cols = [c for c in display_cols if c in df.columns]
    display_df = df[available_cols].copy()

    n_total = len(display_df)
    n_pass = int(display_df['qc_pass'].sum()) if 'qc_pass' in display_df.columns else n_total
    n_fail = n_total - n_pass

    # Build table rows
    rows_html = []
    for _, row in display_df.iterrows():
        cells = []
        for col in available_cols:
            val = row[col]
            css_class = _cell_color(val, col, thresholds) if col not in ('subject', 'session', 'cohort', 'qc_pass', 'qc_fail_reasons') else ''

            if col == 'qc_pass':
                css_class = 'cell-green' if val else 'cell-red'
                display_val = 'PASS' if val else 'FAIL'
            elif isinstance(val, float):
                display_val = f'{val:.4f}' if abs(val) < 1 else f'{val:.1f}'
            elif pd.isna(val):
                display_val = '—'
            else:
                display_val = str(val)

            cells.append(f'<td class="{css_class}">{display_val}</td>')

        row_class = 'row-fail' if 'qc_pass' in available_cols and not row.get('qc_pass', True) else ''
        rows_html.append(f'<tr class="{row_class}">{"".join(cells)}</tr>')

    # Header
    headers = ''.join(f'<th>{c}</th>' for c in available_cols)

    cohort_counts = df['cohort'].value_counts().to_dict() if 'cohort' in df.columns else {}

    rows_body = '\n'.join(rows_html)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Functional Preprocessing Dashboard</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 0; padding: 20px; background: #f0f2f5; color: #333; }}
        .header {{ background: #2c3e50; color: white; padding: 20px 25px; border-radius: 8px;
                   margin-bottom: 20px; }}
        .header h1 {{ margin: 0 0 5px; }}
        .header p {{ margin: 5px 0; opacity: 0.85; }}

        .summary {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }}
        .summary-box {{ background: white; padding: 15px 20px; border-radius: 8px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; min-width: 100px; }}
        .summary-box .number {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
        .summary-box .label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
        .summary-box.pass .number {{ color: #27ae60; }}
        .summary-box.fail .number {{ color: #e74c3c; }}

        .controls {{ background: white; padding: 15px 20px; border-radius: 8px;
                     box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px;
                     display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }}
        .controls label {{ font-size: 13px; font-weight: 600; color: #555; }}
        .controls select, .controls input {{ padding: 6px 10px; border: 1px solid #ddd;
                                              border-radius: 4px; font-size: 13px; }}
        .controls a {{ padding: 6px 14px; background: #007bff; color: white; text-decoration: none;
                       border-radius: 4px; font-size: 13px; }}
        .controls a:hover {{ background: #0056b3; }}

        .table-container {{ background: white; border-radius: 8px; padding: 15px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow-x: auto; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
        th {{ background: #2c3e50; color: white; padding: 8px 10px; text-align: left;
              cursor: pointer; white-space: nowrap; position: sticky; top: 0; }}
        th:hover {{ background: #34495e; }}
        td {{ padding: 6px 10px; border-bottom: 1px solid #eee; white-space: nowrap; }}
        tr:hover {{ background: #f5f5f5; }}
        .row-fail {{ background: #fff5f5; }}
        .row-fail:hover {{ background: #ffe8e8; }}

        .cell-green {{ background: #d4edda; }}
        .cell-yellow {{ background: #fff3cd; }}
        .cell-red {{ background: #f8d7da; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Functional Preprocessing Dashboard</h1>
        <p>All QC metrics per subject-session | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <div class="summary">
        <div class="summary-box">
            <div class="number">{n_total}</div>
            <div class="label">Total Runs</div>
        </div>
        <div class="summary-box pass">
            <div class="number">{n_pass}</div>
            <div class="label">QC Pass</div>
        </div>
        <div class="summary-box fail">
            <div class="number">{n_fail}</div>
            <div class="label">QC Fail</div>
        </div>
"""

    for cohort in sorted(cohort_counts.keys()):
        html += f"""        <div class="summary-box">
            <div class="number">{cohort_counts[cohort]}</div>
            <div class="label">{cohort}</div>
        </div>
"""

    html += f"""    </div>

    <div class="controls">
        <label>Show:
            <select id="showFilter" onchange="filterTable()">
                <option value="all">All ({n_total})</option>
                <option value="fail">Failures only ({n_fail})</option>
                <option value="pass">Pass only ({n_pass})</option>
            </select>
        </label>
        <label>Cohort:
            <select id="cohortFilter" onchange="filterTable()">
                <option value="">All</option>
"""
    for cohort in sorted(cohort_counts.keys()):
        html += f'                <option value="{cohort}">{cohort}</option>\n'

    html += f"""            </select>
        </label>
        <label>Search:
            <input type="text" id="searchFilter" onkeyup="filterTable()" placeholder="e.g. Rat102">
        </label>
        <a href="func_preprocessing_summary.csv" download>Download CSV</a>
        <a href="func_registration_gallery.html">Registration Gallery</a>
    </div>

    <div class="table-container">
        <table id="dashTable">
            <thead><tr>{headers}</tr></thead>
            <tbody>
{rows_body}
            </tbody>
        </table>
    </div>
"""

    # JS section uses plain string to avoid Python 3.10 f-string parser issues with #
    html += """
    <script>
        // Sortable columns
        document.querySelectorAll('#dashTable th').forEach((th, idx) => {
            th.addEventListener('click', () => {
                const tbody = document.querySelector('#dashTable tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const asc = th.dataset.sort !== 'asc';
                th.dataset.sort = asc ? 'asc' : 'desc';

                rows.sort((a, b) => {
                    const aVal = a.cells[idx].textContent.trim();
                    const bVal = b.cells[idx].textContent.trim();
                    const aNum = parseFloat(aVal);
                    const bNum = parseFloat(bVal);
                    if (!isNaN(aNum) && !isNaN(bNum)) {
                        return asc ? aNum - bNum : bNum - aNum;
                    }
                    return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                });
                rows.forEach(r => tbody.appendChild(r));
            });
        });

        function filterTable() {
            const show = document.getElementById('showFilter').value;
            const cohort = document.getElementById('cohortFilter').value;
            const search = document.getElementById('searchFilter').value.toLowerCase();

            const rows = document.querySelectorAll('#dashTable tbody tr');
            rows.forEach(row => {
                const cells = row.cells;
                const subject = cells[0].textContent.toLowerCase();
                const sess = cells[1].textContent.toLowerCase();
                const rowCohort = cells[2].textContent.trim();
                const isPass = row.classList.contains('row-fail') ? false : true;

                let visible = true;
                if (show === 'fail' && isPass) visible = false;
                if (show === 'pass' && !isPass) visible = false;
                if (cohort && rowCohort !== cohort) visible = false;
                if (search && !subject.includes(search) && !sess.includes(search)) visible = false;

                row.style.display = visible ? '' : 'none';
            });
        }
    </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate functional registration QC')
    parser.add_argument('study_root', type=Path, help='Study root directory')
    parser.add_argument('--subjects', nargs='+', help='Specific subjects to process')
    parser.add_argument('--skip-figures', action='store_true',
                        help='Skip generating per-subject registration figures')
    parser.add_argument('--config', type=Path, help='Config file (default: study_root/config.yaml)')
    args = parser.parse_args()

    study_root = args.study_root.resolve()
    config_path = args.config or study_root / 'config.yaml'

    # Load config
    config = load_config(config_path)
    func_thresholds = get_config_value(config, 'qc.thresholds.func', default={})
    # Merge with defaults
    thresholds = {**DEFAULT_THRESHOLDS, **func_thresholds}

    print(f"Study root: {study_root}")
    print(f"Thresholds: {thresholds}")

    # ── Discover subjects ──────────────────────────────────────────────
    transforms_dir = study_root / 'transforms'
    if args.subjects:
        subject_dirs = [transforms_dir / s for s in args.subjects if (transforms_dir / s).exists()]
    else:
        subject_dirs = sorted(d for d in transforms_dir.iterdir()
                              if d.is_dir() and d.name.startswith('sub-'))

    print(f"Processing {len(subject_dirs)} subjects...")

    # ── Per-subject registration QC ────────────────────────────────────
    registration_results = []
    seen = set()
    for subj_dir in subject_dirs:
        subject = subj_dir.name
        # Collect session dirs: prefer ses-* at top level, fall back to sub-X/ses-*
        session_dirs = sorted(d for d in subj_dir.iterdir() if d.name.startswith('ses-'))
        if not session_dirs:
            # Legacy nested: transforms/sub-X/sub-X/ses-Y/
            nested = subj_dir / subject
            if nested.is_dir():
                session_dirs = sorted(d for d in nested.iterdir() if d.name.startswith('ses-'))

        for sess_dir in session_dirs:
            session = sess_dir.name
            key = (subject, session)
            if key in seen:
                continue
            seen.add(key)

            result = generate_per_subject_qc(
                study_root, subject, session,
                skip_figures=args.skip_figures,
            )
            if result:
                registration_results.append(result)
                print(f"  {subject}/{session}: corr={result['correlation']:.3f}, nmi={result['nmi']:.3f}")

    print(f"\nRegistration QC computed for {len(registration_results)} runs")

    if not registration_results:
        print("No results to summarize. Exiting.")
        return

    # ── Build comprehensive DataFrame ──────────────────────────────────
    print("\nBuilding comprehensive metrics DataFrame...")
    df = build_comprehensive_df(study_root, registration_results)
    df = compute_qc_pass(df, thresholds)

    # ── Save CSV ───────────────────────────────────────────────────────
    reports_dir = get_reports_dir(study_root)
    csv_path = reports_dir / 'func_preprocessing_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV: {csv_path} ({len(df)} rows)")

    n_pass = int(df['qc_pass'].sum())
    n_fail = len(df) - n_pass
    print(f"  QC pass: {n_pass}, fail: {n_fail}")
    if n_fail > 0:
        print(f"  Failed subjects:")
        for _, row in df[~df['qc_pass']].iterrows():
            print(f"    {row['subject']}/{row['session']}: {row['qc_fail_reasons']}")

    # ── Gallery HTML ───────────────────────────────────────────────────
    print("\nGenerating registration gallery...")
    thumb_dir = reports_dir / 'thumbnails'
    thumb_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    cards = []
    for result in sorted(registration_results, key=lambda r: r['correlation']):
        subject = result['subject']
        session = result['session']
        cohort = result['cohort']
        quality = _classify_registration_quality(result['correlation'], result['nmi'])

        # Copy figure to thumbnails
        thumb_name = f'{subject}_{session}_bold_to_template_registration.png'
        figure_path = result.get('figure')
        if figure_path and Path(figure_path).exists():
            try:
                shutil.copy(figure_path, thumb_dir / thumb_name)
            except IOError:
                pass

        # Relative path from reports dir to per-subject figure
        figure_rel = None
        if figure_path:
            try:
                figure_rel = str(Path(figure_path).relative_to(study_root / 'qc'))
                figure_rel = f'../{figure_rel}'
            except ValueError:
                pass

        cards.append({
            'subject': subject,
            'session': session,
            'cohort': cohort,
            'correlation': result['correlation'],
            'nmi': result['nmi'],
            'quality': quality,
            'thumb': f'thumbnails/{thumb_name}',
            'figure_rel': figure_rel,
        })

    n_total = len(cards)
    n_good = sum(1 for c in cards if c['quality'] == 'good')
    n_fair = sum(1 for c in cards if c['quality'] == 'fair')
    n_poor = sum(1 for c in cards if c['quality'] == 'poor')
    cohort_counts = {}
    for c in cards:
        cohort_counts[c['cohort']] = cohort_counts.get(c['cohort'], 0) + 1

    gallery_path = reports_dir / 'func_registration_gallery.html'
    _write_gallery_html(gallery_path, cards, n_total, n_good, n_fair, n_poor, cohort_counts)
    print(f"Gallery: {gallery_path}")

    # ── Dashboard HTML ─────────────────────────────────────────────────
    print("Generating preprocessing dashboard...")
    dashboard_path = reports_dir / 'func_preprocessing_dashboard.html'
    _write_dashboard_html(dashboard_path, df, thresholds)
    print(f"Dashboard: {dashboard_path}")

    print(f"\nDone! Summary: {n_total} runs — {n_good} good, {n_fair} fair, {n_poor} poor registration")


if __name__ == '__main__':
    main()
