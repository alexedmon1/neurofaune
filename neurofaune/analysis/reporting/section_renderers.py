"""
Per-analysis-type HTML section builders for the index dashboard.

Each renderer takes an entry dict and analysis_root, and returns an
HTML string for its collapsible section in the dashboard.
"""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Max file size for base64 embedding of figures
_MAX_EMBED_BYTES = 500_000  # 500 KB


def _status_badge(status: str) -> str:
    """Render a coloured status badge."""
    colours = {
        "completed": "#2E7D32",
        "partial": "#f57c00",
        "failed": "#c62828",
    }
    bg = colours.get(status, "#666")
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:3px;'
        f'background:{bg};color:white;font-size:0.85em;font-weight:bold;">'
        f"{status.upper()}</span>"
    )


def _stat_card(value: Any, label: str) -> str:
    return (
        f'<div class="stat-card">'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-label">{label}</div>'
        f"</div>"
    )


def _figure_thumbnail(fig_rel: str, analysis_root: Path) -> str:
    """Render a figure as an embedded base64 image or relative link."""
    fig_path = analysis_root / fig_rel
    if not fig_path.exists():
        return ""

    size = fig_path.stat().st_size
    if size <= _MAX_EMBED_BYTES:
        try:
            data = fig_path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            suffix = fig_path.suffix.lower().lstrip(".")
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "svg": "image/svg+xml"}.get(suffix, "image/png")
            return (
                f'<div class="figure-thumb">'
                f'<img src="data:{mime};base64,{b64}" alt="{fig_path.name}" '
                f'style="max-width:300px;max-height:220px;margin:4px;border:1px solid #ddd;">'
                f'<div class="fig-caption">{fig_path.name}</div>'
                f"</div>"
            )
        except OSError:
            pass

    # Fallback: relative link
    return (
        f'<div class="figure-thumb">'
        f'<a href="{fig_rel}">{fig_path.name}</a>'
        f"</div>"
    )


def _figures_gallery(figures: List[str], analysis_root: Path, max_show: int = 8) -> str:
    """Render a row of figure thumbnails."""
    if not figures:
        return ""
    thumbs = []
    for fig in figures[:max_show]:
        t = _figure_thumbnail(fig, analysis_root)
        if t:
            thumbs.append(t)
    if not thumbs:
        return ""
    more = ""
    if len(figures) > max_show:
        more = f'<p style="color:#888;font-size:0.85em;">+ {len(figures) - max_show} more figures</p>'
    return (
        f'<div class="figures-row" style="display:flex;flex-wrap:wrap;gap:8px;margin:10px 0;">'
        f'{"".join(thumbs)}'
        f"</div>{more}"
    )


def _key_value_table(pairs: List[tuple]) -> str:
    """Simple 2-column table."""
    rows = "".join(
        f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>" for k, v in pairs
    )
    return f'<table class="kv-table">{rows}</table>'


# ---- Per-type renderers ---------------------------------------------------


def render_tbss(entry: Dict[str, Any], analysis_root: Path) -> str:
    """Render a TBSS analysis entry section."""
    stats = entry.get("summary_stats", {})
    cards = []
    cards.append(_stat_card(stats.get("n_subjects", "?"), "Subjects"))
    cards.append(_stat_card(", ".join(stats.get("metrics", [])), "Metrics"))
    cards.append(_stat_card(stats.get("n_permutations", "?"), "Permutations"))

    n_sig = stats.get("n_significant_contrasts", 0)
    sig_colour = "#2E7D32" if n_sig > 0 else "#666"
    cards.append(
        f'<div class="stat-card">'
        f'<div class="stat-value" style="color:{sig_colour}">{n_sig}</div>'
        f'<div class="stat-label">Significant Contrasts</div>'
        f"</div>"
    )

    gallery = _figures_gallery(entry.get("figures", []), analysis_root)
    output_link = f'<p>Output: <code>{entry.get("output_dir", "")}</code></p>'

    return (
        f'<div class="stats-grid">{"".join(cards)}</div>'
        f"{output_link}"
        f"{gallery}"
    )


def render_roi_extraction(entry: Dict[str, Any], analysis_root: Path) -> str:
    """Render an ROI extraction entry section."""
    stats = entry.get("summary_stats", {})
    cards = [
        _stat_card(stats.get("n_subjects", "?"), "Subjects"),
        _stat_card(stats.get("modality", "?").upper(), "Modality"),
        _stat_card(stats.get("n_regions", "?"), "Regions"),
        _stat_card(stats.get("n_territories", "?"), "Territories"),
    ]
    metrics_str = ", ".join(stats.get("metrics", []))
    output_link = f'<p>Output: <code>{entry.get("output_dir", "")}</code></p>'

    return (
        f'<div class="stats-grid">{"".join(cards)}</div>'
        f"<p>Metrics: {metrics_str}</p>"
        f"{output_link}"
    )


def render_covnet(entry: Dict[str, Any], analysis_root: Path) -> str:
    """Render a CovNet analysis entry section."""
    stats = entry.get("summary_stats", {})
    cards = [
        _stat_card(stats.get("n_subjects", "?"), "Subjects"),
        _stat_card(stats.get("n_bilateral_rois", "?"), "Bilateral ROIs"),
        _stat_card(", ".join(stats.get("metrics", [])), "Metrics"),
    ]
    gallery = _figures_gallery(entry.get("figures", []), analysis_root)
    output_link = f'<p>Output: <code>{entry.get("output_dir", "")}</code></p>'

    return (
        f'<div class="stats-grid">{"".join(cards)}</div>'
        f"{output_link}"
        f"{gallery}"
    )


def render_batch_qc(entry: Dict[str, Any], analysis_root: Path) -> str:
    """Render a Batch QC entry section."""
    stats = entry.get("summary_stats", {})
    cards = [
        _stat_card(stats.get("modality", "?").upper(), "Modality"),
    ]
    if "n_outliers" in stats:
        cards.append(_stat_card(stats["n_outliers"], "Outliers Flagged"))

    report_html = entry.get("report_html")
    report_link = ""
    if report_html:
        report_link = f'<p><a href="{report_html}">View detailed QC dashboard</a></p>'

    output_link = f'<p>Output: <code>{entry.get("output_dir", "")}</code></p>'

    return (
        f'<div class="stats-grid">{"".join(cards)}</div>'
        f"{report_link}"
        f"{output_link}"
    )


def render_generic(entry: Dict[str, Any], analysis_root: Path) -> str:
    """Fallback renderer for unknown analysis types."""
    stats = entry.get("summary_stats", {})
    pairs = [(k, v) for k, v in stats.items()]
    table = _key_value_table(pairs) if pairs else ""
    output_link = f'<p>Output: <code>{entry.get("output_dir", "")}</code></p>'

    return f"{table}{output_link}"


# Renderer dispatch
RENDERERS = {
    "tbss": render_tbss,
    "roi_extraction": render_roi_extraction,
    "covnet": render_covnet,
    "batch_qc": render_batch_qc,
}


def render_entry(entry: Dict[str, Any], analysis_root: Path) -> str:
    """
    Render a full collapsible section for one entry.

    Args:
        entry: Registry entry dict.
        analysis_root: Absolute path to analysis root.

    Returns:
        HTML string for this entry's section.
    """
    renderer = RENDERERS.get(entry.get("analysis_type", ""), render_generic)
    body_html = renderer(entry, analysis_root)

    badge = _status_badge(entry.get("status", "completed"))
    timestamp = entry.get("timestamp", "")
    display_name = entry.get("display_name", entry.get("entry_id", "Unknown"))

    warnings_html = ""
    warnings = entry.get("warnings", [])
    if warnings:
        items = "".join(f"<li>{w}</li>" for w in warnings)
        warnings_html = (
            f'<div class="warning-box"><strong>Warnings:</strong><ul>{items}</ul></div>'
        )

    notes_html = ""
    notes = entry.get("notes", "")
    if notes:
        notes_html = f'<p class="notes"><em>Notes: {notes}</em></p>'

    return f"""
    <details class="entry-section" open>
        <summary class="entry-header">
            {badge} <strong>{display_name}</strong>
            <span class="entry-timestamp">{timestamp}</span>
        </summary>
        <div class="entry-body">
            {body_html}
            {warnings_html}
            {notes_html}
        </div>
    </details>
    """
