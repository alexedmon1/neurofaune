#!/usr/bin/env python3
"""
CovNet whole-network similarity tests.

Tests whether entire covariance network structure differs between groups
using three complementary statistics:
  - Mantel test (Pearson r between vectorised upper triangles)
  - Frobenius distance (L2 norm of upper triangle difference)
  - Spectral divergence (L2 distance between eigenvalue spectra)

Each statistic is evaluated with a permutation test.

Usage:
    uv run python scripts/run_covnet_whole_network.py \
        --roi-dir $STUDY_ROOT/network/roi \
        --output-dir $STUDY_ROOT/network/covnet \
        --modality dwi \
        --metrics FA MD AD RD \
        --exclusion-csv $STUDY_ROOT/dti_nonstandard_slices.csv \
        --n-permutations 1000 --seed 42
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
        description="CovNet whole-network similarity tests"
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
        help="Number of permutations (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Parallel workers for comparisons (default: 1)",
    )
    parser.add_argument(
        "--skip-cross-timepoint", action="store_true",
        help="Skip cross-timepoint comparisons",
    )

    args = parser.parse_args()

    if not args.roi_dir.exists():
        logger.error("ROI directory not found: %s", args.roi_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    comp_types = ["dose"]
    if not args.skip_cross_timepoint:
        comp_types.extend(["cross-timepoint", "cross-dose-timepoint"])

    progress = AnalysisProgress(
        args.output_dir, "run_covnet_whole_network.py", len(args.metrics)
    )
    all_summaries = {}
    completed = 0

    for metric in args.metrics:
        logger.info("\n%s\n  Whole-network: %s / %s\n%s", "=" * 60, args.modality, metric, "=" * 60)
        progress.update(task=metric, phase="preparing", completed=completed)

        try:
            analysis = CovNetAnalysis.prepare(
                args.roi_dir, args.exclusion_csv, args.output_dir,
                args.modality, metric, labels_csv=args.labels_csv,
            )
            analysis.save()

            comparisons = analysis.resolve_comparisons(comp_types)

            progress.update(task=metric, phase="running whole-network", completed=completed)
            wn_df, _ = analysis.run_whole_network(
                comparisons, args.n_permutations, args.seed, args.n_workers,
            )

            n_sig = int(
                (wn_df[["mantel_p", "frobenius_p", "spectral_p"]] < 0.05)
                .any(axis=1).sum()
            )
            all_summaries[metric] = {
                "metric": metric,
                "n_subjects": analysis.n_subjects,
                "n_bilateral_rois": len(analysis.bilateral_region_cols),
                "n_comparisons": len(comparisons),
                "n_significant": n_sig,
            }
            completed += 1

        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipping %s: %s", metric, e)
            continue

    summary_path = args.output_dir / f"whole_network_summary_{args.modality}.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    progress.finish()
    logger.info("\nWhole-network analysis complete. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
