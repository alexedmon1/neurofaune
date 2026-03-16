#!/usr/bin/env python3
"""
Generate findings summaries for all completed analyses.

Scans known summary JSON locations under --analysis-root and
--network-root, runs the appropriate summarizer for each, and
produces a cross-analysis convergence report.

Output is organized by modality:
    {output-dir}/dwi/{analysis_type}/analysis_findings.{json,txt}
    {output-dir}/msme/{analysis_type}/analysis_findings.{json,txt}
    {output-dir}/multimodal/{analysis_type}/analysis_findings.{json,txt}
    {output-dir}/cross_analysis_findings.{json,txt}

Usage:
    uv run python scripts/generate_findings_summary.py \
        --analysis-root /path/to/study/analysis \
        --network-root /path/to/study/network \
        --output-dir /path/to/study/results/findings
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.reporting.summarize import summarize_analysis, summarize_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _tbss_modality(summary_path: Path) -> str:
    """Infer modality from TBSS/voxelwise summary path.

    Paths:
      .../tbss/template/msme/p60/randomise/.../analysis_summary.json  → msme
      .../tbss/template/dwi/p90/randomise/.../analysis_summary.json   → dwi
      .../reho/randomise/.../analysis_summary.json                    → func
      .../falff/randomise/.../analysis_summary.json                   → func
      .../vbm/randomise/.../analysis_summary.json                     → anat
    """
    parts = summary_path.parts
    # Check for template/{dwi,msme} pattern
    for i, p in enumerate(parts):
        if p == "template" and i + 1 < len(parts):
            mod = parts[i + 1]
            if mod in ("dwi", "msme"):
                return mod
    # Check for top-level analysis type dirs
    for p in parts:
        if p in ("falff", "reho"):
            return "func"
        if p == "vbm":
            return "anat"
    return "other"


def _covnet_modality(stem: str, prefix: str) -> str:
    """Extract modality from a covnet summary filename stem."""
    return stem.replace(prefix, "")


def discover_summaries(analysis_root: Path, network_root: Path):
    """Find all summary JSONs. Returns {key: (analysis_type, modality, path)}."""
    found = {}

    # Randomise (TBSS, VBM, fALFF, ReHo) — one per analysis_name per source type
    for summary in sorted(analysis_root.rglob("randomise/*/analysis_summary.json")):
        mod = _tbss_modality(summary)
        rel_parts = summary.relative_to(analysis_root).parts
        source = rel_parts[0]  # falff, reho, vbm, or tbss
        if source == "tbss":
            source = rel_parts[2]  # dwi or msme
        analysis_name = summary.parent.name
        key = f"randomise_{analysis_name}_{source}"
        found[key] = ("randomise", mod, summary)

    # MVPA — one per modality
    for modality in ("dwi", "msme"):
        p = analysis_root / "mvpa" / modality / "mvpa_summary.json"
        if p.exists():
            found[f"mvpa_{modality}"] = ("mvpa", modality, p)

    # Classification — one per modality
    for modality in ("dwi", "msme", "func"):
        p = network_root / "classification" / modality / "classification_summary.json"
        if p.exists():
            found[f"classification_{modality}"] = ("classification", modality, p)

    # Regression — one per modality (and per target subfolder)
    for modality in ("dwi", "msme", "func"):
        p = network_root / "regression" / modality / "regression_summary.json"
        if p.exists():
            found[f"regression_{modality}"] = ("regression", modality, p)
        if (network_root / "regression" / modality).is_dir():
            for subdir in sorted((network_root / "regression" / modality).glob("*")):
                if subdir.is_dir():
                    sp = subdir / "regression_summary.json"
                    if sp.exists():
                        found[f"regression_{modality}_{subdir.name}"] = ("regression", modality, sp)

    # CovNet Abs-Distance (new naming + whole_network backward compat)
    for p in sorted(network_root.glob("covnet/abs_distance_summary_*.json")):
        modality = _covnet_modality(p.stem, "abs_distance_summary_")
        found[f"covnet_abs_distance_{modality}"] = ("covnet_abs_distance", modality, p)
    for p in sorted(network_root.glob("covnet/whole_network_summary_*.json")):
        modality = _covnet_modality(p.stem, "whole_network_summary_")
        key = f"covnet_abs_distance_{modality}"
        if key not in found:  # don't override new naming
            found[key] = ("covnet_abs_distance", modality, p)

    # CovNet Abs-Distance Sex (new naming + whole_network_sex backward compat)
    for p in sorted(network_root.glob("covnet/abs_distance_sex_summary_*.json")):
        modality = _covnet_modality(p.stem, "abs_distance_sex_summary_")
        found[f"covnet_abs_distance_sex_{modality}"] = ("covnet_abs_distance_sex", modality, p)
    for p in sorted(network_root.glob("covnet/whole_network_sex_summary_*.json")):
        modality = _covnet_modality(p.stem, "whole_network_sex_summary_")
        key = f"covnet_abs_distance_sex_{modality}"
        if key not in found:
            found[key] = ("covnet_abs_distance_sex", modality, p)

    # CovNet Rel-Distance (new naming + maturation_distance backward compat)
    for p in sorted(network_root.glob("covnet/rel_distance_summary_*.json")):
        modality = _covnet_modality(p.stem, "rel_distance_summary_")
        found[f"covnet_rel_distance_{modality}"] = ("covnet_rel_distance", modality, p)
    for p in sorted(network_root.glob("covnet/maturation_distance_summary_*.json")):
        modality = _covnet_modality(p.stem, "maturation_distance_summary_")
        key = f"covnet_rel_distance_{modality}"
        if key not in found:
            found[key] = ("covnet_rel_distance", modality, p)

    # CovNet Rel-Distance Sex (+ maturation_distance_sex backward compat)
    for p in sorted(network_root.glob("covnet/rel_distance_sex_summary_*.json")):
        modality = _covnet_modality(p.stem, "rel_distance_sex_summary_")
        found[f"covnet_rel_distance_sex_{modality}"] = ("covnet_rel_distance_sex", modality, p)
    for p in sorted(network_root.glob("covnet/maturation_distance_sex_summary_*.json")):
        modality = _covnet_modality(p.stem, "maturation_distance_sex_summary_")
        key = f"covnet_rel_distance_sex_{modality}"
        if key not in found:
            found[key] = ("covnet_rel_distance_sex", modality, p)

    # CovNet Territory
    for p in sorted(network_root.glob("covnet/territory_summary_*.json")):
        modality = _covnet_modality(p.stem, "territory_summary_")
        found[f"covnet_territory_{modality}"] = ("covnet_territory", modality, p)

    # CovNet NBS
    for p in sorted(network_root.glob("covnet/nbs_summary_*.json")):
        modality = _covnet_modality(p.stem, "nbs_summary_")
        found[f"covnet_nbs_{modality}"] = ("covnet_nbs", modality, p)

    # CovNet NBS Interaction
    for p in sorted(network_root.glob("covnet/nbs_interaction_summary_*.json")):
        modality = _covnet_modality(p.stem, "nbs_interaction_summary_")
        found[f"covnet_nbs_interaction_{modality}"] = ("covnet_nbs_interaction", modality, p)
    # Also check inside nbs_interaction/ subdir
    for p in sorted(network_root.glob("covnet/nbs_interaction/nbs_*_summary_*.json")):
        # e.g. nbs_interaction_summary_dwi.json
        stem = p.stem
        modality = stem.rsplit("_", 1)[-1]
        key = f"covnet_nbs_interaction_{modality}"
        if key not in found:
            found[key] = ("covnet_nbs_interaction", modality, p)

    # CovNet Graph Theory
    for p in sorted(network_root.glob("covnet/graph_theory_summary_*.json")):
        modality = _covnet_modality(p.stem, "graph_theory_summary_")
        found[f"covnet_graph_theory_{modality}"] = ("covnet_graph_theory", modality, p)

    # Edge Regression (in covnet root or covnet/edge_regression/)
    for p in sorted(network_root.glob("covnet/edge_regression_summary_*.json")):
        modality = _covnet_modality(p.stem, "edge_regression_summary_")
        found[f"edge_regression_{modality}"] = ("edge_regression", modality, p)
    for p in sorted(network_root.glob("covnet/edge_regression/edge_regression_summary_*.json")):
        modality = _covnet_modality(p.stem, "edge_regression_summary_")
        key = f"edge_regression_{modality}"
        if key not in found:
            found[key] = ("edge_regression", modality, p)

    # FC Graph Theory (lives outside covnet)
    p = network_root / "fc_graph_theory" / "fc_graph_theory_summary.json"
    if p.exists():
        found["fc_graph_theory_func"] = ("fc_graph_theory", "func", p)

    # MCCA — multimodal by definition
    p = network_root / "mcca" / "mcca_summary.json"
    if p.exists():
        found["mcca"] = ("mcca", "multimodal", p)
    else:
        mcca_dir = network_root / "mcca"
        if mcca_dir.is_dir():
            for cohort_dir in sorted(mcca_dir.glob("*")):
                sp = cohort_dir / "bilateral" / "summary.json"
                if sp.exists():
                    found[f"mcca_{cohort_dir.name}"] = ("mcca", "multimodal", sp)

    return found


def main():
    parser = argparse.ArgumentParser(
        description="Generate findings summaries for all completed analyses"
    )
    parser.add_argument(
        "--analysis-root", type=Path, required=True,
        help="Root of voxelwise analyses (e.g. /study/analysis)",
    )
    parser.add_argument(
        "--network-root", type=Path, required=True,
        help="Root of network analyses (e.g. /study/network)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for findings (e.g. /study/results/findings)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all summary JSONs
    summaries = discover_summaries(args.analysis_root, args.network_root)
    logger.info("Discovered %d analysis summaries", len(summaries))

    # Log discovery with provenance info
    import json as _json
    from datetime import datetime as _dt
    missing_prov = []
    for key, (atype, mod, path) in sorted(summaries.items()):
        prov_tag = ""
        try:
            with open(path) as _f:
                _data = _json.load(_f)
            _prov = _data.get("_provenance", {})
            if _prov:
                ts = _prov.get("timestamp", "")[:10]
                script = _prov.get("script", "")
                prov_tag = f"  [{ts} {script}]"
            else:
                missing_prov.append(key)
                prov_tag = "  [no provenance]"
        except Exception:
            pass
        logger.info("  %-40s  %-8s  %-25s%s", key, mod, atype, prov_tag)

    if missing_prov:
        logger.warning(
            "%d/%d analyses lack embedded provenance — "
            "re-run with updated scripts to record provenance",
            len(missing_prov), len(summaries),
        )

    if not summaries:
        logger.warning("No summaries found — check paths")
        sys.exit(0)

    # Summarize each, writing into {output_dir}/{modality}/{analysis_key}/
    all_findings = {}
    for key, (atype, modality, path) in sorted(summaries.items()):
        try:
            per_dir = output_dir / modality / key
            findings = summarize_analysis(atype, path, output_dir=per_dir)
            all_findings[key] = findings
            prov = findings.provenance or {}
            prov_src = prov.get("source", "")
            prov_flag = "  [mtime]" if prov_src == "file_mtime" else ""
            logger.info(
                "  %-40s  %d sig / %d trend / %d null%s",
                key, len(findings.significant), len(findings.trending),
                len(findings.null), prov_flag,
            )
        except Exception as exc:
            logger.warning("  %-40s  FAILED: %s", key, exc)

    # Cross-analysis summary at top level
    if all_findings:
        from neurofaune.reporting.summarize import _write_cross_analysis
        _write_cross_analysis(all_findings, output_dir)
        logger.info("Wrote cross-analysis summary to %s", output_dir)

    logger.info("Done. %d analyses summarized.", len(all_findings))


if __name__ == "__main__":
    main()
