#!/usr/bin/env python3
"""
CovNet graph theory analysis.

Computes graph-theoretic metrics across a density sweep and tests pairwise
group differences via permutation on the area under the density curve (AUC).
This avoids multiple comparisons across density levels.

Each graph metric is run independently, so you can select which to compute.
Use --list-graph-metrics to see all available metrics.

Usage:
    # Run all graph metrics
    uv run python scripts/run_covnet_graph_theory.py \
        --roi-dir $STUDY_ROOT/network/roi \
        --output-dir $STUDY_ROOT/network/covnet \
        --modality dwi \
        --metrics FA MD AD RD \
        --exclusion-csv $STUDY_ROOT/dti_nonstandard_slices.csv \
        --n-permutations 1000 --seed 42

    # Run specific graph metrics only
    uv run python scripts/run_covnet_graph_theory.py \
        --roi-dir $STUDY_ROOT/network/roi \
        --output-dir $STUDY_ROOT/network/covnet \
        --modality dwi \
        --metrics FA MD AD RD \
        --graph-metrics global_efficiency clustering_coefficient modularity \
        --n-permutations 1000 --seed 42

    # List available graph metrics
    uv run python scripts/run_covnet_graph_theory.py --list-graph-metrics
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.network.covnet import CovNetAnalysis
from neurofaune.network.graph_theory import (
    DEFAULT_DENSITIES,
    METRIC_REGISTRY,
    list_metrics,
)
from neurofaune.analysis.progress import AnalysisProgress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CovNet graph theory analysis with density-curve AUC testing"
    )
    parser.add_argument(
        "--roi-dir", type=Path,
        help="Directory containing roi_*_wide.csv files",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Root output directory for CovNet results",
    )
    parser.add_argument(
        "--modality", type=str,
        help="Modality name (e.g. dwi, msme, func)",
    )
    parser.add_argument(
        "--metrics", nargs="+",
        help="Imaging metrics to analyse (e.g. FA MD AD RD)",
    )
    parser.add_argument(
        "--exclusion-csv", type=Path, default=None,
        help="CSV of sessions to exclude (must have subject, session columns)",
    )
    parser.add_argument(
        "--labels-csv", type=Path, default=None,
        help="SIGMA atlas labels CSV for territory mapping",
    )
    parser.add_argument(
        "--graph-metrics", nargs="+", default=None,
        help="Graph metrics to test (default: all). Use --list-graph-metrics to see options.",
    )
    parser.add_argument(
        "--densities", nargs="+", type=float, default=None,
        help=f"Network density sweep (default: {DEFAULT_DENSITIES})",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Number of permutations for AUC comparison (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Parallel workers for pairwise comparisons (default: 1)",
    )
    parser.add_argument(
        "--list-graph-metrics", action="store_true",
        help="List available graph metrics and exit",
    )

    args = parser.parse_args()

    if args.list_graph_metrics:
        print("Available graph metrics:")
        for name, (_func, desc) in METRIC_REGISTRY.items():
            print(f"  {name:30s} {desc}")
        sys.exit(0)

    # Validate required args when not just listing
    if not args.roi_dir or not args.output_dir or not args.modality or not args.metrics:
        parser.error("--roi-dir, --output-dir, --modality, and --metrics are required")

    if not args.roi_dir.exists():
        logger.error("ROI directory not found: %s", args.roi_dir)
        sys.exit(1)

    # Validate graph metric names
    gm_names = args.graph_metrics or list_metrics()
    for gm in gm_names:
        if gm not in METRIC_REGISTRY:
            logger.error(
                "Unknown graph metric %r. Available: %s", gm, list_metrics()
            )
            sys.exit(1)

    densities = args.densities or DEFAULT_DENSITIES

    args.output_dir.mkdir(parents=True, exist_ok=True)

    progress = AnalysisProgress(
        args.output_dir, "run_covnet_graph_theory.py", len(args.metrics)
    )
    all_summaries = {}
    completed = 0

    for metric in args.metrics:
        logger.info(
            "\n%s\n  Graph theory: %s / %s\n%s",
            "=" * 60, args.modality, metric, "=" * 60,
        )
        progress.update(task=metric, phase="preparing", completed=completed)

        try:
            analysis = CovNetAnalysis.prepare(
                args.roi_dir, args.exclusion_csv, args.output_dir,
                args.modality, metric, labels_csv=args.labels_csv,
            )
            analysis.save()

            progress.update(
                task=metric, phase="running graph theory", completed=completed
            )
            comparison_df = analysis.run_graph_metrics(
                graph_metrics=gm_names,
                densities=densities,
                n_perm=args.n_permutations,
                seed=args.seed,
                n_workers=args.n_workers,
            )

            n_sig = int((comparison_df["p_value"] < 0.05).sum())
            all_summaries[metric] = {
                "metric": metric,
                "n_subjects": analysis.n_subjects,
                "n_bilateral_rois": len(analysis.bilateral_region_cols),
                "graph_metrics": gm_names,
                "densities": densities,
                "n_significant": n_sig,
                "n_total_tests": len(comparison_df),
            }
            completed += 1

        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipping %s: %s", metric, e)
            continue

    summary_path = args.output_dir / f"graph_theory_summary_{args.modality}.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    progress.finish()

    # Generate findings summary
    try:
        from neurofaune.reporting.summarize import summarize_analysis
        findings = summarize_analysis("covnet_graph_theory", summary_path, output_dir=args.output_dir)
        logger.info("Findings: %s", findings.summary_text)
    except Exception as exc:
        logger.warning("Failed to generate findings summary: %s", exc)

    logger.info("\nGraph theory analysis complete. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
