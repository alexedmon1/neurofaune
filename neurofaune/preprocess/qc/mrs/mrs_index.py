"""Study-level spectroscopy QC index.

Scans an MRS output root and renders one navigable ``index.html`` tying the
per-session material together:

* a **cohort summary** -- session counts, QC pass rate, and the spread of SNR,
  linewidth and voxel tissue composition,
* a **session table** with the metrics that decide whether a fit is usable,
  sortable by clicking a column and flagging anything outside threshold, and
* per-session links to the neurofaune QC page, the **fsl_mrs report**, the
  voxel-placement overlay and the CRLB chart.

Spectroscopy sits outside the ``qc/subjects/<sub>/<ses>/<mod>/`` tree the
preprocessing index walks -- it is self-contained under ``{study}/mrs`` because
it never enters the BIDS tree -- so it gets its own index rather than a
modality column there. The layout and styling deliberately match
:mod:`neurofaune.preprocess.qc.qc_index` so the two feel like one system.

Idempotent: it renders whatever is present and can be regenerated after any
run. All links are relative to the MRS root, so the tree stays portable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Metric, label, and the direction that counts as bad. Thresholds come from
# neurofaune.preprocess.qc.mrs.QC_THRESHOLDS so the index and the per-session
# reports cannot disagree.
_COLUMNS: List[Tuple[str, str, str]] = [
    ('snr', 'SNR', 'low'),
    ('fwhm_hz', 'Linewidth (Hz)', 'high'),
    ('voxel_coverage', 'Voxel coverage', 'low'),
    ('target_overlap', 'On target', 'low'),
]


def _rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _collect(mrs_root: Path) -> List[Dict[str, Any]]:
    """One record per session, from the QC metrics and metadata on disk."""
    sessions: List[Dict[str, Any]] = []
    qc_root = mrs_root / 'qc'
    if not qc_root.exists():
        return sessions

    for metrics_file in sorted(qc_root.glob('sub-*/ses-*/*_mrs-qc.json')):
        subject, session = metrics_file.parts[-3], metrics_file.parts[-2]
        try:
            metrics = json.loads(metrics_file.read_text())
        except json.JSONDecodeError:
            continue

        record: Dict[str, Any] = {'subject': subject, 'session': session, **metrics}

        meta_file = mrs_root / subject / session / f'{subject}_{session}_mrs.json'
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except json.JSONDecodeError:
                meta = {}
            fractions = meta.get('tissue_fractions', {})
            record.update({
                'fitter': meta.get('fitter', 'fsl_mrs'),
                'source': meta.get('source', ''),
                'n_averages': meta.get('n_averages'),
                'water_removed': meta.get('water_removed', False),
                'frac_GM': fractions.get('GM'),
                'frac_WM': fractions.get('WM'),
                'frac_CSF': fractions.get('CSF'),
                'fractions_measured': fractions.get('measured', False),
            })

        # Links, all relative to the MRS root so the tree can be moved.
        links: List[Tuple[str, str]] = []
        figures = metrics_file.parent / 'figures'
        for pattern, label in (
            (f'{subject}_{session}_mrs-qc.html', 'QC'),
            (f'{subject}_{session}_voxel-placement.png', 'voxel'),
            (f'{subject}_{session}_crlb.png', 'CRLB'),
        ):
            candidate = figures / pattern
            if candidate.exists():
                links.append((label, _rel(candidate, mrs_root)))
        for candidate, label in (
            (mrs_root / subject / session / 'fit' / 'report.html', 'fsl_mrs report'),
            (mrs_root / subject / session / 'preproc' / 'mergedReports.html', 'preproc'),
            (mrs_root / subject / session / 'fit' / 'lcmodel.ps', 'LCModel plot'),
        ):
            if candidate.exists():
                links.append((label, _rel(candidate, mrs_root)))
        record['links'] = links
        sessions.append(record)

    return sessions


def _summary(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cohort-level counts and metric spreads."""
    def values(key: str) -> List[float]:
        return [s[key] for s in sessions
                if isinstance(s.get(key), (int, float)) and s[key] == s[key]]

    def spread(key: str) -> Optional[Tuple[float, float, float]]:
        vals = sorted(values(key))
        if not vals:
            return None
        return (vals[len(vals) // 2], vals[0], vals[-1])

    return {
        'n_sessions': len(sessions),
        'n_subjects': len({s['subject'] for s in sessions}),
        'n_pass': sum(1 for s in sessions if s.get('overall_pass')),
        'n_measured_fractions': sum(1 for s in sessions if s.get('fractions_measured')),
        'n_water_removed': sum(1 for s in sessions if s.get('water_removed')),
        'snr': spread('snr'),
        'fwhm_hz': spread('fwhm_hz'),
        'target_overlap': spread('target_overlap'),
        'n_on_target': sum(1 for s in sessions
                           if s.get('placement_pass') is not False),
    }


def _cell(record: Dict[str, Any], key: str, direction: str,
          thresholds: Dict[str, float]) -> str:
    """One metric cell, flagged when it falls outside its threshold."""
    value = record.get(key)
    if not isinstance(value, (int, float)) or value != value:
        return '<td class="none">-</td>'

    limit = thresholds.get({'snr': 'min_snr', 'fwhm_hz': 'max_fwhm_hz',
                            'voxel_coverage': 'min_voxel_coverage',
                            'target_overlap': 'min_target_overlap'}[key])
    bad = limit is not None and (
        (value < limit) if direction == 'low' else (value > limit))
    shown = (f'{value * 100:.0f}%'
             if key in ('voxel_coverage', 'target_overlap') else f'{value:.1f}')
    css = ' class="flag"' if bad else ''
    return f'<td data-v="{value}"{css}>{shown}</td>'


def generate_mrs_index(
    mrs_root: Path,
    study_name: Optional[str] = None,
    output_file: Optional[Path] = None,
) -> Optional[Path]:
    """Render ``index.html`` for a study's spectroscopy output.

    Parameters
    ----------
    mrs_root : Path
        MRS output root, e.g. ``{study}/mrs``.
    study_name : str, optional
        Shown in the header.
    output_file : Path, optional
        Defaults to ``mrs_root/index.html``.

    Returns
    -------
    Path or None
        The written index, or None when there is no QC to index.
    """
    from neurofaune.preprocess.qc.mrs.mrs_qc import QC_THRESHOLDS

    mrs_root = Path(mrs_root)
    if not mrs_root.exists():
        return None
    sessions = _collect(mrs_root)
    if not sessions:
        return None

    out = Path(output_file) if output_file else mrs_root / 'index.html'
    stats = _summary(sessions)

    def spread_text(key: str, unit: str = '') -> str:
        s = stats.get(key)
        return '-' if not s else f'{s[0]:.1f}{unit} <span class="sub">({s[1]:.1f}-{s[2]:.1f})</span>'

    rows = []
    for record in sorted(sessions, key=lambda r: (r['subject'], r['session'])):
        links = ' '.join(f'<a href="{href}">{label}</a>' for label, href in record['links'])
        flags = []
        if not record.get('fractions_measured'):
            flags.append('assumed tissue fractions')
        if record.get('water_removed'):
            flags.append('HLSVD retry')
        if record.get('source') == 'bruker_averaged':
            flags.append('pre-averaged data')
        if record.get('placement_pass') is False:
            flags.append('voxel off target -- check the geometry')
        note = f'<span class="warn">{"; ".join(flags)}</span>' if flags else ''
        status = ('<span class="ok">PASS</span>' if record.get('overall_pass')
                  else '<span class="flag">REVIEW</span>')
        reliable = (f"{record.get('n_metabolites_reliable', '-')}"
                    f"/{record.get('n_metabolites', '-')}")
        gm, wm, csf = (record.get('frac_GM'), record.get('frac_WM'), record.get('frac_CSF'))
        tissue = ('-' if gm is None
                  else f'{gm:.2f} / {wm:.2f} / {csf:.2f}')
        rows.append(
            f'<tr><td class="sess">{record["subject"]} '
            f'<span class="ses">{record["session"]}</span></td>'
            f'<td>{status}</td>'
            + ''.join(_cell(record, key, direction, QC_THRESHOLDS)
                      for key, _, direction in _COLUMNS)
            + f'<td>{reliable}</td><td>{tissue}</td>'
            f'<td class="cell">{links}</td><td>{note}</td></tr>'
        )

    headers = ''.join(f'<th>{label}</th>' for _, label, _ in _COLUMNS)
    title = f'MRS QC: {study_name}' if study_name else 'MRS QC'
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: -apple-system, Arial, sans-serif; margin: 24px; color: #222; background:#fafafa; }}
 h1 {{ border-bottom: 3px solid #4CAF50; padding-bottom: 8px; }}
 h2 {{ color:#444; margin-top: 32px; }}
 .sub {{ color:#777; font-size: 0.9em; }}
 .cards {{ display:flex; flex-wrap:wrap; gap:14px; }}
 .card {{ background:#fff; border:1px solid #e0e0e0; border-radius:8px; padding:14px 16px; min-width:200px; box-shadow:0 1px 3px rgba(0,0,0,.05); }}
 .card h3 {{ margin:0 0 6px; font-size:.95em; color:#555; }}
 .card .big {{ font-size:1.6em; font-weight:600; }}
 table.matrix {{ border-collapse:collapse; width:100%; background:#fff; }}
 table.matrix th, table.matrix td {{ border:1px solid #e0e0e0; padding:6px 8px; text-align:left; font-size:.86em; }}
 table.matrix thead th {{ background:#f0f0f0; position:sticky; top:0; cursor:pointer; user-select:none; }}
 table.matrix thead th:hover {{ background:#e6e6e6; }}
 td.sess {{ white-space:nowrap; font-weight:bold; }} td.sess .ses {{ color:#888; font-weight:normal; }}
 td.cell a {{ color:#1565c0; text-decoration:none; margin-right:8px; }}
 td.cell a:hover {{ text-decoration:underline; }}
 td.none {{ color:#ccc; text-align:center; }}
 .flag {{ color:#c62828; font-weight:bold; }} .ok {{ color:#2e7d32; }} .warn {{ color:#e65100; }}
</style></head><body>
<h1>{title}</h1>
<p class="sub">Single-voxel spectroscopy. Thresholds: SNR &ge; {QC_THRESHOLDS['min_snr']:.0f},
 linewidth &le; {QC_THRESHOLDS['max_fwhm_hz']:.0f} Hz, voxel coverage &ge;
 {QC_THRESHOLDS['min_voxel_coverage'] * 100:.0f}%, CRLB &le;
 {QC_THRESHOLDS['max_crlb_percent']:.0f}%.</p>

<div class="cards">
 <div class="card"><h3>Sessions</h3><div class="big">{stats['n_sessions']}</div>
  <div class="sub">{stats['n_subjects']} subjects</div></div>
 <div class="card"><h3>Passing QC</h3><div class="big">{stats['n_pass']}/{stats['n_sessions']}</div></div>
 <div class="card"><h3>SNR</h3><div class="big">{spread_text('snr')}</div></div>
 <div class="card"><h3>Linewidth</h3><div class="big">{spread_text('fwhm_hz', ' Hz')}</div></div>
 <div class="card"><h3>Voxel on target</h3>
  <div class="big">{stats['n_on_target']}/{stats['n_sessions']}</div>
  <div class="sub">overlap with the intended structure</div></div>
 <div class="card"><h3>Measured tissue fractions</h3>
  <div class="big">{stats['n_measured_fractions']}/{stats['n_sessions']}</div>
  <div class="sub">rest use assumed values</div></div>
</div>

<h2>Sessions</h2>
<p class="sub">Click a column to sort. Always check the voxel overlay: placement
 comes from Bruker geometry, not from the image affine.</p>
<table class="matrix" id="sessions">
<thead><tr><th>Session</th><th>Status</th>{headers}
 <th>CRLB ok</th><th>GM / WM / CSF</th><th>Reports</th><th>Notes</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>

<script>
document.querySelectorAll('#sessions thead th').forEach(function (th, col) {{
  var asc = true;
  th.addEventListener('click', function () {{
    var body = document.querySelector('#sessions tbody');
    var rows = Array.prototype.slice.call(body.rows);
    rows.sort(function (a, b) {{
      var x = a.cells[col], y = b.cells[col];
      // Numeric columns carry the raw value in data-v; the rest sort as text.
      var vx = x.dataset.v !== undefined ? parseFloat(x.dataset.v) : null;
      var vy = y.dataset.v !== undefined ? parseFloat(y.dataset.v) : null;
      if (vx !== null && vy !== null) return asc ? vx - vy : vy - vx;
      return asc ? x.textContent.localeCompare(y.textContent)
                 : y.textContent.localeCompare(x.textContent);
    }});
    rows.forEach(function (r) {{ body.appendChild(r); }});
    asc = !asc;
  }});
}});
</script>
</body></html>
"""
    out.write_text(html)
    return out
