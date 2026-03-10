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


def discover_summaries(analysis_root: Path, network_root: Path):
    """Find all summary JSONs. Returns {key: (analysis_type, modality, path)}."""
    found = {}

    # TBSS — one per analysis_name per source type
    for summary in sorted(analysis_root.rglob("randomise/*/analysis_summary.json")):
        mod = _tbss_modality(summary)
        # Build unique key from source: falff, reho, vbm, or dwi/msme (from template path)
        rel_parts = summary.relative_to(analysis_root).parts
        source = rel_parts[0]  # falff, reho, vbm, or tbss
        if source == "tbss":
            # tbss/template/{dwi,msme}/... → use dwi or msme as source
            source = rel_parts[2]  # dwi or msme
        analysis_name = summary.parent.name
        key = f"tbss_{analysis_name}_{source}"
        found[key] = ("tbss", mod, summary)

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
        # Also check target-specific subdirs (log_auc, auc, etc.)
        if (network_root / "regression" / modality).is_dir():
            for subdir in sorted((network_root / "regression" / modality).glob("*")):
                if subdir.is_dir():
                    sp = subdir / "regression_summary.json"
                    if sp.exists():
                        found[f"regression_{modality}_{subdir.name}"] = ("regression", modality, sp)

    # CovNet Whole-Network
    for p in sorted(network_root.glob("covnet/whole_network_summary_*.json")):
        modality = p.stem.replace("whole_network_summary_", "")
        found[f"covnet_whole_network_{modality}"] = ("covnet_whole_network", modality, p)

    # CovNet NBS
    for p in sorted(network_root.glob("covnet/nbs_summary_*.json")):
        modality = p.stem.replace("nbs_summary_", "")
        found[f"covnet_nbs_{modality}"] = ("covnet_nbs", modality, p)

    # CovNet Graph Theory
    for p in sorted(network_root.glob("covnet/graph_theory_summary_*.json")):
        modality = p.stem.replace("graph_theory_summary_", "")
        found[f"covnet_graph_theory_{modality}"] = ("covnet_graph_theory", modality, p)

    # MCCA — multimodal by definition
    p = network_root / "mcca" / "mcca_summary.json"
    if p.exists():
        found["mcca"] = ("mcca", "multimodal", p)
    else:
        for cohort_dir in sorted((network_root / "mcca").glob("*")):
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
    for key, (atype, mod, path) in sorted(summaries.items()):
        logger.info("  %-40s  %-8s  %s", key, mod, atype)

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
            logger.info(
                "  %-40s  %d sig / %d trend / %d null",
                key, len(findings.significant), len(findings.trending), len(findings.null),
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
