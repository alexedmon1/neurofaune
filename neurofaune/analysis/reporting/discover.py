"""
Backwards-compatible discovery of existing analysis summary JSONs.

Scans known directory patterns under analysis_root and creates
registry entries for any summaries found.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurofaune.analysis.reporting.registry import load_registry, register

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON file, returning None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _rel(path: Path, root: Path) -> str:
    """Return path relative to root as a string."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _discover_roi(analysis_root: Path) -> List[Dict[str, Any]]:
    """Discover ROI extraction summaries."""
    entries = []
    roi_dir = analysis_root / "roi"
    if not roi_dir.is_dir():
        return entries

    summary_path = roi_dir / "extraction_summary.json"
    if not summary_path.exists():
        return entries

    data = _read_json(summary_path)
    if data is None:
        return entries

    modality = data.get("modality", "unknown")
    metrics_info = data.get("metrics", {})
    n_subjects = 0
    metric_names = list(metrics_info.keys())
    if metrics_info:
        first = next(iter(metrics_info.values()))
        n_subjects = first.get("n_subjects", 0)

    entries.append({
        "entry_id": f"roi_extraction_{modality}",
        "analysis_type": "roi_extraction",
        "display_name": f"ROI Extraction ({modality.upper()}: {', '.join(metric_names)})",
        "output_dir": _rel(roi_dir, analysis_root),
        "summary_stats": {
            "modality": modality,
            "n_subjects": n_subjects,
            "metrics": metric_names,
            "n_regions": next(iter(metrics_info.values()), {}).get("n_regions", 0),
            "n_territories": next(iter(metrics_info.values()), {}).get("n_territories", 0),
        },
        "source_summary_json": _rel(summary_path, analysis_root),
    })
    return entries


def _discover_tbss(analysis_root: Path) -> List[Dict[str, Any]]:
    """Discover TBSS randomise analysis summaries."""
    entries = []
    randomise_base = analysis_root / "tbss" / "randomise"
    if not randomise_base.is_dir():
        return entries

    for analysis_dir in sorted(randomise_base.iterdir()):
        if not analysis_dir.is_dir():
            continue
        summary_path = analysis_dir / "analysis_summary.json"
        if not summary_path.exists():
            continue

        data = _read_json(summary_path)
        if data is None:
            continue

        analysis_name = data.get("analysis_name", analysis_dir.name)
        n_subjects = data.get("n_subjects", 0)
        metrics = data.get("metrics", [])

        # Count significant contrasts
        n_sig = 0
        results = data.get("results", {})
        for metric_results in results.values():
            n_sig += metric_results.get("n_significant_contrasts", 0)

        # Collect figure paths
        figures = []
        for metric in metrics:
            reports_dir = analysis_dir / f"cluster_reports_{metric}"
            if reports_dir.is_dir():
                for fig in reports_dir.glob("*.png"):
                    figures.append(_rel(fig, analysis_root))

        entry_id = f"tbss_{analysis_dir.name}"
        entries.append({
            "entry_id": entry_id,
            "analysis_type": "tbss",
            "display_name": f"TBSS: {analysis_name}",
            "output_dir": _rel(analysis_dir, analysis_root),
            "summary_stats": {
                "n_subjects": n_subjects,
                "metrics": metrics,
                "n_permutations": data.get("n_permutations", 0),
                "n_significant_contrasts": n_sig,
                "n_contrasts": data.get("n_contrasts", 0),
            },
            "figures": figures,
            "source_summary_json": _rel(summary_path, analysis_root),
        })

    return entries


def _discover_covnet(analysis_root: Path) -> List[Dict[str, Any]]:
    """Discover CovNet analysis summaries."""
    entries = []
    covnet_dir = analysis_root / "covnet"
    if not covnet_dir.is_dir():
        return entries

    summary_path = covnet_dir / "covnet_summary.json"
    if not summary_path.exists():
        return entries

    data = _read_json(summary_path)
    if data is None:
        return entries

    metrics = list(data.keys())
    n_subjects = 0
    n_rois = 0
    for metric_data in data.values():
        if isinstance(metric_data, dict):
            n_subjects = max(n_subjects, metric_data.get("n_subjects", 0))
            n_rois = max(n_rois, metric_data.get("n_bilateral_rois", 0))

    # Collect figures
    figures = []
    fig_dir = covnet_dir / "figures"
    if fig_dir.is_dir():
        for fig in sorted(fig_dir.rglob("*.png")):
            figures.append(_rel(fig, analysis_root))

    entries.append({
        "entry_id": "covnet",
        "analysis_type": "covnet",
        "display_name": f"CovNet Analysis ({', '.join(metrics)})",
        "output_dir": _rel(covnet_dir, analysis_root),
        "summary_stats": {
            "metrics": metrics,
            "n_subjects": n_subjects,
            "n_bilateral_rois": n_rois,
        },
        "figures": figures[:20],  # Cap at 20 figure refs
        "source_summary_json": _rel(summary_path, analysis_root),
    })
    return entries


def _discover_classification(analysis_root: Path) -> List[Dict[str, Any]]:
    """Discover Multivariate Classification analysis summaries."""
    entries = []
    clf_dir = analysis_root / "classification"
    if not clf_dir.is_dir():
        return entries

    summary_path = clf_dir / "classification_summary.json"
    if not summary_path.exists():
        return entries

    data = _read_json(summary_path)
    if data is None:
        return entries

    metrics = data.get("metrics", [])
    feature_sets = data.get("feature_sets", [])
    n_subjects = data.get("n_subjects", 0)
    n_sig = data.get("n_significant_permanova", 0)
    best_acc = data.get("best_classification_accuracy", 0)

    # Collect figures
    figures = []
    for fig in sorted(clf_dir.rglob("*.png")):
        figures.append(_rel(fig, analysis_root))

    entries.append({
        "entry_id": "classification",
        "analysis_type": "classification",
        "display_name": f"Multivariate Classification ({', '.join(metrics)})",
        "output_dir": _rel(clf_dir, analysis_root),
        "summary_stats": {
            "metrics": metrics,
            "feature_sets": feature_sets,
            "n_subjects": n_subjects,
            "n_significant_permanova": n_sig,
            "best_classification_accuracy": best_acc,
        },
        "figures": figures[:20],
        "source_summary_json": _rel(summary_path, analysis_root),
    })
    return entries


def _discover_batch_qc(analysis_root: Path) -> List[Dict[str, Any]]:
    """
    Discover batch QC summaries.

    Batch QC lives under {study_root}/qc/ not analysis_root, but if there's
    a symlink or the user passes the study root we check both.
    """
    entries = []

    # Batch QC summaries are under qc/*_batch_summary/ in the study root.
    # Since analysis_root may be {study_root}/analysis, check parent too.
    search_roots = [analysis_root]
    if analysis_root.name == "analysis":
        search_roots.append(analysis_root.parent)

    for root in search_roots:
        qc_base = root / "qc"
        if not qc_base.is_dir():
            continue
        for summary_dir in sorted(qc_base.iterdir()):
            if not summary_dir.is_dir() or not summary_dir.name.endswith("_batch_summary"):
                continue
            modality = summary_dir.name.replace("_batch_summary", "")
            # Look for outliers.json or metrics.csv as proof of results
            metrics_csv = summary_dir / "metrics.csv"
            outliers_json = summary_dir / "outliers.json"
            summary_html = summary_dir / "summary.html"

            if not (metrics_csv.exists() or summary_html.exists()):
                continue

            stats: Dict[str, Any] = {"modality": modality}
            if outliers_json.exists():
                outlier_data = _read_json(outliers_json)
                if outlier_data is not None:
                    if isinstance(outlier_data, list):
                        stats["n_outliers"] = len(outlier_data)
                    elif isinstance(outlier_data, dict):
                        stats["n_outliers"] = len(outlier_data.get("outliers", []))

            report_html_rel = None
            if summary_html.exists():
                report_html_rel = _rel(summary_html, analysis_root)

            entries.append({
                "entry_id": f"batch_qc_{modality}",
                "analysis_type": "batch_qc",
                "display_name": f"Batch QC: {modality.upper()}",
                "output_dir": _rel(summary_dir, analysis_root),
                "summary_stats": stats,
                "report_html": report_html_rel,
                "source_summary_json": (
                    _rel(outliers_json, analysis_root) if outliers_json.exists() else None
                ),
            })

    return entries


def backfill_registry(
    analysis_root: Path,
    *,
    study_name: str = "",
    auto_generate_index: bool = True,
) -> int:
    """
    Scan analysis_root for existing summary JSONs and register them.

    Only adds entries that are not already in the registry.

    Args:
        analysis_root: Absolute path to the analysis root directory.
        study_name: Optional study name for the registry header.
        auto_generate_index: If True, regenerate index.html after backfill.

    Returns:
        Number of newly discovered entries added.
    """
    analysis_root = Path(analysis_root)
    existing = load_registry(analysis_root)
    existing_ids = set(existing["entries"].keys())

    discoveries = []
    discoveries.extend(_discover_roi(analysis_root))
    discoveries.extend(_discover_tbss(analysis_root))
    discoveries.extend(_discover_covnet(analysis_root))
    discoveries.extend(_discover_classification(analysis_root))
    discoveries.extend(_discover_batch_qc(analysis_root))

    n_added = 0
    for entry_kwargs in discoveries:
        eid = entry_kwargs["entry_id"]
        if eid in existing_ids:
            logger.debug("Skipping existing entry: %s", eid)
            continue

        # Register without auto-generating index each time
        register(
            analysis_root=analysis_root,
            study_name=study_name,
            auto_generate_index=False,
            **entry_kwargs,
        )
        n_added += 1
        logger.info("Discovered: %s", eid)

    if n_added > 0 and auto_generate_index:
        try:
            from neurofaune.analysis.reporting.index_generator import (
                generate_index_html,
            )
            generate_index_html(analysis_root)
        except Exception as exc:
            logger.warning("Failed to generate index after backfill: %s", exc)

    logger.info("Backfill complete: %d new entries added", n_added)
    return n_added
