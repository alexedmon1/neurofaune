"""
Generate a self-contained index.html dashboard from the report registry.

Inline CSS/JS, no external dependencies. Visual language matches
neurofaune/analysis/tbss/reporting.py (#1a5276 primary, #2E7D32 accent).
"""

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .registry import load_registry
from .section_renderers import render_entry

logger = logging.getLogger(__name__)

# Display order for analysis type groups
_TYPE_ORDER = ["tbss", "roi_extraction", "covnet", "connectome", "classification", "regression", "mvpa", "batch_qc"]

_TYPE_LABELS = {
    "tbss": "TBSS (Tract-Based Spatial Statistics)",
    "roi_extraction": "ROI Extraction",
    "covnet": "Covariance Network",
    "connectome": "Functional Connectome",
    "classification": "Multivariate Classification",
    "regression": "Dose-Response Regression",
    "mvpa": "MVPA (Voxel-Level Decoding)",
    "batch_qc": "Batch QC",
}


def _summary_cards(entries: list) -> str:
    """Build the summary cards row at the top of the dashboard."""
    total = len(entries)
    type_counts = Counter(e.get("analysis_type", "other") for e in entries)
    status_counts = Counter(e.get("status", "completed") for e in entries)

    cards = [
        f'<div class="stat-card">'
        f'<div class="stat-value">{total}</div>'
        f'<div class="stat-label">Total Analyses</div>'
        f"</div>"
    ]

    for atype in _TYPE_ORDER:
        count = type_counts.get(atype, 0)
        if count:
            label = _TYPE_LABELS.get(atype, atype)
            cards.append(
                f'<div class="stat-card">'
                f'<div class="stat-value">{count}</div>'
                f'<div class="stat-label">{label}</div>'
                f"</div>"
            )

    # Unknown types
    for atype, count in sorted(type_counts.items()):
        if atype not in _TYPE_ORDER:
            cards.append(
                f'<div class="stat-card">'
                f'<div class="stat-value">{count}</div>'
                f'<div class="stat-label">{atype}</div>'
                f"</div>"
            )

    n_completed = status_counts.get("completed", 0)
    n_failed = status_counts.get("failed", 0)
    if n_failed:
        cards.append(
            f'<div class="stat-card" style="border-left:3px solid #c62828;">'
            f'<div class="stat-value" style="color:#c62828;">{n_failed}</div>'
            f'<div class="stat-label">Failed</div>'
            f"</div>"
        )

    return f'<div class="stats-grid">{"".join(cards)}</div>'


def _group_entries(entries: list) -> list:
    """Group entries by analysis_type in display order."""
    by_type: Dict[str, list] = {}
    for e in entries:
        atype = e.get("analysis_type", "other")
        by_type.setdefault(atype, []).append(e)

    groups = []
    for atype in _TYPE_ORDER:
        if atype in by_type:
            groups.append((atype, by_type.pop(atype)))
    # Remaining types
    for atype in sorted(by_type.keys()):
        groups.append((atype, by_type[atype]))

    return groups


def generate_index_html(
    analysis_root: Path,
    *,
    registry: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Build a self-contained index.html dashboard from the registry.

    Args:
        analysis_root: Absolute path to analysis root directory.
        registry: Pre-loaded registry dict (loads from disk if None).
        output_path: Override output path (defaults to analysis_root/index.html).

    Returns:
        Path to the generated HTML file.
    """
    analysis_root = Path(analysis_root)
    if registry is None:
        registry = load_registry(analysis_root)

    if output_path is None:
        output_path = analysis_root / "index.html"

    entries = list(registry.get("entries", {}).values())
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    study_name = registry.get("study_name", "") or "Analysis Dashboard"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_html = _summary_cards(entries)

    # Grouped sections
    sections_html = ""
    groups = _group_entries(entries)
    for atype, group_entries in groups:
        label = _TYPE_LABELS.get(atype, atype)
        entries_html = "".join(
            render_entry(e, analysis_root) for e in group_entries
        )
        sections_html += f"""
        <div class="type-group">
            <h2 class="type-header">{label}</h2>
            {entries_html}
        </div>
        """

    if not entries:
        sections_html = (
            '<div class="info-box">'
            "<p>No analysis entries registered yet. Run an analysis script or "
            "use <code>generate_analysis_index.py --backfill</code> to discover "
            "existing results.</p></div>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{study_name}</title>
    <style>
        * {{ box-sizing: border-box; }}
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
        .header-bar {{
            background: #1a5276;
            color: white;
            padding: 16px 24px;
            margin: -30px -30px 24px -30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-bar h1 {{
            margin: 0;
            font-size: 1.4em;
            border: none;
            padding: 0;
        }}
        .header-bar .timestamp {{
            font-size: 0.85em;
            opacity: 0.85;
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
        .type-header {{
            color: #1a5276;
            font-size: 1.2em;
            margin-top: 28px;
            margin-bottom: 12px;
            padding-bottom: 4px;
            border-bottom: 2px solid #2E7D32;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
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
        .entry-section {{
            margin: 10px 0;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }}
        .entry-header {{
            cursor: pointer;
            padding: 12px 16px;
            background: #fafafa;
            display: flex;
            align-items: center;
            gap: 10px;
            user-select: none;
        }}
        .entry-header:hover {{
            background: #f0f0f0;
        }}
        .entry-timestamp {{
            margin-left: auto;
            font-size: 0.85em;
            color: #888;
        }}
        .entry-body {{
            padding: 12px 16px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
            font-size: 0.9em;
        }}
        table.kv-table {{
            width: auto;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 6px 10px;
            text-align: left;
        }}
        th {{
            background-color: #2E7D32;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
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
        .figure-thumb img {{
            border-radius: 3px;
        }}
        .fig-caption {{
            font-size: 0.75em;
            color: #888;
            text-align: center;
        }}
        .notes {{
            color: #666;
            font-size: 0.9em;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
            color: #888;
            font-size: 0.85em;
        }}
        code {{
            background: #f0f0f0;
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header-bar">
        <h1>{study_name}</h1>
        <span class="timestamp">Generated: {timestamp}</span>
    </div>

    {summary_html}

    {sections_html}

    <div class="footer">
        <p>Generated by <strong>neurofaune</strong> analysis reporting</p>
        <p>Registry: <code>report_registry.json</code> &mdash;
           {len(entries)} entries</p>
    </div>
</div>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    logger.info("Generated index dashboard: %s", output_path)
    return output_path
