#!/usr/bin/env python3
"""
CovNet absolute network distance tests.

Directly compares covariance network structure between groups using three
complementary distance statistics:
  - Mantel test (Pearson r between vectorised upper triangles)
  - Frobenius distance (L2 norm of upper triangle difference)
  - Spectral divergence (L2 distance between eigenvalue spectra)

Each statistic is evaluated with a permutation test.

Usage:
    # Config-driven (recommended):
    uv run python scripts/run_covnet_abs_distance.py \
        --config $STUDY_ROOT/config.yaml \
        --modality dwi \
        --metrics FA MD AD RD \
        --exclusion-csv $STUDY_ROOT/dti_nonstandard_slices.csv \
        --n-permutations 1000 --seed 42

    # Explicit paths (backwards-compatible):
    uv run python scripts/run_covnet_abs_distance.py \
        --roi-dir $STUDY_ROOT/network/roi \
        --output-dir $STUDY_ROOT/network/covnet \
        --modality dwi \
        --metrics FA MD AD RD \
        --n-permutations 1000 --seed 42
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.network.covnet import CovNetAnalysis
from neurofaune.network.covnet.pipeline import resolve_covnet_paths
from neurofaune.analysis.progress import AnalysisProgress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_NAME = "run_covnet_abs_distance.py"
ANALYSIS_NAME = "abs_distance"
SUMMARY_PREFIX = "abs_distance_summary"


def main():
    parser = argparse.ArgumentParser(
        description="CovNet absolute network distance tests"
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
    parser.add_argument(
        "--sex", choices=["F", "M"], default=None,
        help="Run analysis for one sex only",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing results before running",
    )

    args = parser.parse_args()

    # Resolve paths from config or explicit arguments
    if not args.config and not (args.roi_dir and args.output_dir):
        parser.error("Either --config or both --roi-dir and --output-dir are required")

    roi_dir, covnet_root = resolve_covnet_paths(
        config_path=args.config, roi_dir=args.roi_dir, covnet_root=args.output_dir,
    )

    if not roi_dir.exists():
        logger.error("ROI directory not found: %s", roi_dir)
        sys.exit(1)

    variant = "pooled" if args.sex is None else f"sex_stratified/{args.sex}"
    progress_dir = covnet_root / ANALYSIS_NAME / variant
    progress_dir.mkdir(parents=True, exist_ok=True)

    comp_types = ["dose"]
    if not args.skip_cross_timepoint:
        comp_types.extend(["cross-timepoint", "cross-dose-timepoint"])

    progress = AnalysisProgress(progress_dir, SCRIPT_NAME, len(args.metrics))
    all_summaries = {}
    completed = 0

    for metric in args.metrics:
        logger.info("\n%s\n  Absolute distance: %s / %s\n%s", "=" * 60, args.modality, metric, "=" * 60)
        progress.update(task=metric, phase="preparing", completed=completed)

        try:
            analysis = CovNetAnalysis.prepare(
                config_path=args.config,
                roi_dir=args.roi_dir,
                covnet_root=args.output_dir,
                modality=args.modality,
                metric=metric,
                exclusion_csv=args.exclusion_csv,
                labels_csv=args.labels_csv,
                sex=args.sex,
                force=args.force,
            )
            analysis.save()

            comparisons = analysis.resolve_comparisons(comp_types)

            progress.update(task=metric, phase="running abs_distance", completed=completed)
            wn_df, _ = analysis.run_abs_distance(
                comparisons, args.n_permutations, args.seed, args.n_workers,
            )

            n_sig = int(
                (wn_df[["mantel_p", "frobenius_p", "spectral_p"]] < 0.05)
                .any(axis=1).sum()
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

    try:
        from neurofaune.reporting.summarize import build_provenance
        all_summaries["_provenance"] = build_provenance()
    except Exception:
        pass

    sex_suffix = f"_{args.sex}" if args.sex else ""
    summary_path = progress_dir / f"{SUMMARY_PREFIX}_{args.modality}{sex_suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    try:
        from neurofaune.reporting.summarize import summarize_analysis
        summarize_analysis("covnet_abs_distance", summary_path, output_dir=progress_dir)
    except Exception as exc:
        logger.warning("Failed to generate findings summary: %s", exc)

    progress.finish()
    logger.info("\nAbsolute distance analysis complete. Results in: %s", progress_dir)


if __name__ == "__main__":
    main()
