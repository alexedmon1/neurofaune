#!/usr/bin/env python3
"""
CovNet NBS (Network-Based Statistic) analysis.

Runs permutation-based NBS testing whether covariance network edges differ
between dose groups. For each metric, computes group correlation matrices
then tests pairwise group differences using suprathreshold edge clustering.

Usage:
    uv run python scripts/run_covnet_nbs.py \
        --roi-dir $STUDY_ROOT/network/roi \
        --output-dir $STUDY_ROOT/network/covnet \
        --modality dwi \
        --metrics FA MD AD RD \
        --exclusion-csv $STUDY_ROOT/dti_nonstandard_slices.csv \
        --n-permutations 1000 --nbs-threshold 3.0 --seed 42
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.network.covnet import CovNetAnalysis
from neurofaune.analysis.progress import AnalysisProgress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CovNet NBS (Network-Based Statistic) analysis"
    )
    parser.add_argument(
        "--roi-dir", type=Path, required=True,
        help="Directory containing roi_*_wide.csv files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Root output directory for CovNet results",
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
        help="CSV of sessions to exclude (must have subject, session columns)",
    )
    parser.add_argument(
        "--labels-csv", type=Path, default=None,
        help="SIGMA atlas labels CSV for territory mapping",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Number of permutations for NBS (default: 1000)",
    )
    parser.add_argument(
        "--nbs-threshold", type=float, default=3.0,
        help="Z-statistic threshold for suprathreshold edges (default: 3.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Parallel workers for NBS comparisons (default: 1)",
    )
    parser.add_argument(
        "--skip-cross-timepoint", action="store_true",
        help="Skip cross-timepoint comparisons (only run dose vs control within PND)",
    )
    parser.add_argument(
        "--sex", choices=["F", "M"], default=None,
        help="Run analysis for one sex only",
    )
    parser.add_argument(
        "--posthoc", action="store_true",
        help="Run post-hoc centrality and hub-vulnerability analyses on significant components",
    )

    args = parser.parse_args()

    if not args.roi_dir.exists():
        logger.error("ROI directory not found: %s", args.roi_dir)
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Comparison types
    comp_types = ["dose"]
    if not args.skip_cross_timepoint:
        comp_types.extend(["cross-timepoint", "cross-dose-timepoint"])

    progress = AnalysisProgress(
        output_dir, "run_covnet_nbs.py", len(args.metrics)
    )
    all_summaries = {}
    completed = 0

    for metric in args.metrics:
        logger.info("\n%s\n  NBS: %s / %s\n%s", "=" * 60, args.modality, metric, "=" * 60)
        progress.update(task=metric, phase="preparing", completed=completed)

        try:
            analysis = CovNetAnalysis.prepare(
                args.roi_dir, args.exclusion_csv, output_dir,
                args.modality, metric, labels_csv=args.labels_csv,
                sex=args.sex,
            )
            analysis.save()

            comparisons = analysis.resolve_comparisons(comp_types)

            progress.update(task=metric, phase="running NBS", completed=completed)
            nbs_results = analysis.run_nbs(
                comparisons, args.n_permutations, args.nbs_threshold,
                args.seed, args.n_workers,
                posthoc=args.posthoc,
            )

            n_sig = sum(
                1 for r in nbs_results.values()
                if any(c["pvalue"] < 0.05 for c in r["significant_components"])
            )
            all_summaries[metric] = {
                "metric": metric,
                "n_subjects": analysis.n_subjects,
                "n_region_rois": len(analysis.region_cols),
                "n_comparisons": len(comparisons),
                "n_significant": n_sig,
            }
            completed += 1

        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipping %s: %s", metric, e)
            continue

    # Save summary
    sex_suffix = f"_{args.sex}" if args.sex else ""
    summary_path = output_dir / f"nbs_summary_{args.modality}{sex_suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    progress.finish()

    # Generate findings summary
    try:
        from neurofaune.reporting.summarize import summarize_analysis
        findings = summarize_analysis("covnet_nbs", summary_path, output_dir=output_dir)
        logger.info("Findings: %s", findings.summary_text)
    except Exception as exc:
        logger.warning("Failed to generate findings summary: %s", exc)

    logger.info("\nNBS analysis complete. Results in: %s", output_dir)


if __name__ == "__main__":
    main()
