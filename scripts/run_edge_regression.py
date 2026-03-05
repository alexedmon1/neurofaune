#!/usr/bin/env python3
"""
Edge-level regression for continuous targets.

Tests whether pairwise ROI co-variation scales with a continuous covariate
(e.g. log-AUC) using NBS-style component extraction and permutation FWER.
This is appropriate ONLY for continuous targets — for categorical group
comparisons, use run_covnet_nbs.py instead.

Results are saved to network/edge_regression/{modality}/{metric}/{cohort}/,
separate from CovNet results.

Usage:
    uv run python scripts/run_edge_regression.py \
        --roi-dir $STUDY_ROOT/network/roi \
        --output-dir $STUDY_ROOT/network/edge_regression \
        --modality dwi \
        --metrics FA MD AD RD \
        --exclusion-csv $STUDY_ROOT/dti_nonstandard_slices.csv \
        --target log_auc \
        --auc-csv $STUDY_ROOT/network/roi/roi_FA_wide.csv \
        --n-permutations 1000 --seed 42
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.analysis.progress import AnalysisProgress
from neurofaune.network.edge_regression import run_edge_regression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Edge-level regression for continuous targets (log_auc, auc, etc.)"
    )
    parser.add_argument(
        "--roi-dir", type=Path, required=True,
        help="Directory containing roi_*_wide.csv files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
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
        help="CSV of sessions to exclude (must have subject, session columns)",
    )
    parser.add_argument(
        "--target", type=str, required=True,
        help="Continuous target column name (e.g. log_auc, auc)",
    )
    parser.add_argument(
        "--auc-csv", type=Path, default=None,
        help="CSV with subject, session, and target columns. "
             "If not provided, the target column is read from the ROI wide CSV.",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Number of permutations for NBS null distribution (default: 1000)",
    )
    parser.add_argument(
        "--nbs-threshold", type=float, default=3.0,
        help="|t|-statistic threshold for suprathreshold edges (default: 3.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if not args.roi_dir.exists():
        logger.error("ROI directory not found: %s", args.roi_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build covariate map
    covariate_map = None
    covariate_name = args.target
    if args.auc_csv is not None:
        auc_df = pd.read_csv(args.auc_csv)
        if args.target not in auc_df.columns:
            logger.error(
                "Target column %r not found in %s. Available: %s",
                args.target, args.auc_csv, list(auc_df.columns),
            )
            sys.exit(1)
        if not {"subject", "session"}.issubset(auc_df.columns):
            logger.error("AUC CSV must have 'subject' and 'session' columns")
            sys.exit(1)
        covariate_map = dict(
            zip(
                auc_df["subject"] + "_" + auc_df["session"],
                auc_df[args.target].astype(float),
            )
        )
        covariate_map = {k: v for k, v in covariate_map.items() if not np.isnan(v)}
        logger.info("Loaded %d %s values from %s", len(covariate_map), args.target, args.auc_csv)
        if args.target == "log_auc":
            covariate_name = "log(1+AUC)"

    # Save analysis configuration
    config = {
        "roi_dir": str(args.roi_dir),
        "exclusion_csv": str(args.exclusion_csv) if args.exclusion_csv else None,
        "output_dir": str(args.output_dir),
        "modality": args.modality,
        "metrics": args.metrics,
        "target": args.target,
        "auc_csv": str(args.auc_csv) if args.auc_csv else None,
        "n_permutations": args.n_permutations,
        "nbs_threshold": args.nbs_threshold,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(args.output_dir / f"edge_regression_config_{args.modality}.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run for each metric x cohort
    cohorts = [None, "p30", "p60", "p90"]
    total_tasks = len(args.metrics) * len(cohorts)
    progress = AnalysisProgress(args.output_dir, "run_edge_regression.py", total_tasks)
    all_summaries = {}
    completed = 0

    for metric in args.metrics:
        wide_csv = args.roi_dir / f"roi_{metric}_wide.csv"
        if not wide_csv.exists():
            logger.warning("Wide CSV not found: %s, skipping %s", wide_csv, metric)
            continue

        metric_summary = {}
        for cohort in cohorts:
            cohort_label = cohort if cohort else "pooled"
            progress.update(
                task=f"{metric} / {cohort_label}",
                phase="running",
                completed=completed,
            )

            result = run_edge_regression(
                wide_csv=wide_csv,
                exclusion_csv=args.exclusion_csv,
                output_dir=args.output_dir,
                modality=args.modality,
                metric=metric,
                covariate_map=covariate_map,
                covariate_name=covariate_name,
                cohort_filter=cohort,
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

    # Save overall summary
    summary_path = args.output_dir / f"edge_regression_summary_{args.modality}.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    progress.finish()
    logger.info("\nEdge regression complete. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
