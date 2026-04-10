#!/usr/bin/env python3
"""
Edge-level regression for continuous targets (example CLI wrapper).

Example scripts in scripts/ are reference wrappers. Each study should
create its own wrapper scripts that import from the library.

Usage:
    # Config-driven (recommended)
    uv run python scripts/run_edge_regression.py \\
        --config /path/to/config.yaml \\
        --modality dwi --metrics FA MD AD RD \\
        --target log_auc --auc-csv /path/to/auc_lookup.csv \\
        --n-permutations 1000 --seed 42 --force

    # Explicit paths (backwards compatible)
    uv run python scripts/run_edge_regression.py \\
        --roi-dir /path/to/network/roi \\
        --output-dir /path/to/network/edge_regression \\
        --modality dwi --metrics FA MD AD RD \\
        --target log_auc --n-permutations 1000 --seed 42
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.network.edge_regression import EdgeRegressionAnalysis
from neurofaune.analysis.progress import AnalysisProgress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_NAME = "run_edge_regression.py"
COHORTS = [None, "p30", "p60", "p90"]


def main():
    parser = argparse.ArgumentParser(
        description="Edge-level regression for continuous targets"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to study config.yaml (derives --roi-dir and --output-dir)",
    )
    parser.add_argument(
        "--roi-dir", type=Path, default=None,
        help="Directory containing roi_*_wide.csv files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for edge regression results",
    )
    parser.add_argument(
        "--modality", type=str, required=True,
        help="Modality name (e.g. dwi, msme, func)",
    )
    parser.add_argument(
        "--metrics", nargs="+", required=True,
        help="Metrics to analyse (e.g. FA MD AD RD)",
    )
    parser.add_argument(
        "--exclusion-csv", type=Path, default=None,
        help="CSV of sessions to exclude",
    )
    parser.add_argument(
        "--target", type=str, required=True,
        help="Continuous target column name (e.g. log_auc, auc)",
    )
    parser.add_argument(
        "--auc-csv", type=Path, default=None,
        help="CSV with subject, session, and target columns",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Number of permutations (default: 1000)",
    )
    parser.add_argument(
        "--nbs-threshold", type=float, default=3.0,
        help="|t|-statistic threshold (default: 3.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--sex", choices=["F", "M"], default=None,
        help="Run analysis for one sex only",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing results before running",
    )

    args = parser.parse_args()

    if not args.config and not (args.roi_dir and args.output_dir):
        parser.error("Provide --config or both --roi-dir and --output-dir")

    total_tasks = len(args.metrics) * len(COHORTS)
    all_summaries = {}
    completed = 0

    for metric in args.metrics:
        try:
            analysis = EdgeRegressionAnalysis.prepare(
                config_path=args.config,
                roi_dir=args.roi_dir,
                output_dir=args.output_dir,
                modality=args.modality,
                metric=metric,
                target=args.target,
                auc_csv=args.auc_csv,
                exclusion_csv=args.exclusion_csv,
                sex=args.sex,
                force=args.force,
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipping %s: %s", metric, e)
            continue

        progress = AnalysisProgress(
            analysis.output_dir, SCRIPT_NAME, total_tasks
        )

        metric_summary = {}
        for cohort in COHORTS:
            cohort_label = cohort if cohort else "pooled"
            progress.update(
                task=f"{metric} / {cohort_label}",
                phase="running",
                completed=completed,
            )

            result = analysis.run(
                cohort=cohort,
                n_perm=args.n_permutations,
                threshold=args.nbs_threshold,
                seed=args.seed,
            )

            if result is not None:
                n_sig = sum(
                    1 for c in result["significant_components"]
                    if c["pvalue"] < 0.05
                )
                metric_summary[cohort_label] = {
                    "n_subjects": result["n_subjects"],
                    "n_significant_components": n_sig,
                }

            completed += 1

        all_summaries[metric] = metric_summary

    # Save summary
    try:
        from neurofaune.reporting.summarize import build_provenance
        all_summaries["_provenance"] = build_provenance()
    except Exception:
        pass

    sex_suffix = f"_{args.sex}" if args.sex else ""
    summary_path = (
        analysis.output_dir
        / f"edge_regression_summary_{args.modality}{sex_suffix}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    try:
        from neurofaune.reporting.summarize import summarize_analysis
        summarize_analysis(
            "edge_regression", summary_path, output_dir=analysis.output_dir
        )
    except Exception as exc:
        logger.warning("Failed to generate findings summary: %s", exc)

    progress.finish()
    logger.info(
        "\nEdge regression complete. Results in: %s", analysis.output_dir
    )


if __name__ == "__main__":
    main()
