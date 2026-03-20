#!/usr/bin/env python3
"""
CovNet relative network distance analysis.

Tests whether a group's covariance network is shifted toward or away from
a reference network, relative to a comparison group. When the reference is
older controls, this tests developmental trajectory shifts:

For each (group_A, group_B, reference) triplet:
    Δ = d(A, reference) − d(B, reference)
    Δ < 0 → A is closer to reference (accelerated if reference is older)
    Δ > 0 → A is farther from reference (decelerated if reference is older)

Significance is assessed via permutation of A/B labels, keeping the
reference group fixed.

Usage:
    uv run python scripts/run_covnet_rel_distance.py \
        --roi-dir $STUDY_ROOT/network/roi \
        --output-dir $STUDY_ROOT/network/covnet \
        --modality dwi \
        --metrics FA MD AD RD \
        --exclusion-csv $STUDY_ROOT/dti_nonstandard_slices.csv \
        --n-permutations 5000 --seed 42
"""

import argparse
import json
import logging
import sys
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
        description="CovNet relative network distance analysis"
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
        "--n-permutations", type=int, default=5000,
        help="Number of permutations (default: 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Parallel workers (default: 1)",
    )
    parser.add_argument(
        "--distance-fns", nargs="+", default=None,
        help="Distance metrics (default: frobenius spectral mantel)",
    )
    parser.add_argument(
        "--sex", choices=["F", "M"], default=None,
        help="Run analysis for one sex only",
    )

    args = parser.parse_args()

    if not args.roi_dir.exists():
        logger.error("ROI directory not found: %s", args.roi_dir)
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = AnalysisProgress(
        output_dir, "run_covnet_rel_distance.py", len(args.metrics)
    )
    all_summaries = {}
    completed = 0

    for metric in args.metrics:
        logger.info(
            "\n%s\n  Relative distance: %s / %s\n%s",
            "=" * 60, args.modality, metric, "=" * 60,
        )
        progress.update(task=metric, phase="preparing", completed=completed)

        try:
            analysis = CovNetAnalysis.prepare(
                args.roi_dir, args.exclusion_csv, output_dir,
                args.modality, metric, labels_csv=args.labels_csv,
                sex=args.sex,
            )
            analysis.save()

            progress.update(
                task=metric, phase="running rel_distance",
                completed=completed,
            )
            md_df = analysis.run_rel_distance(
                n_perm=args.n_permutations,
                seed=args.seed,
                distance_fns=args.distance_fns,
                n_workers=args.n_workers,
            )

            n_accel = int((md_df["p_accelerated"] < 0.05).sum())
            n_decel = int((md_df["p_decelerated"] < 0.05).sum())
            all_summaries[metric] = {
                "metric": metric,
                "n_subjects": analysis.n_subjects,
                "n_region_rois": len(analysis.region_cols),
                "n_triplets": len(md_df) // len(args.distance_fns or ["frobenius", "spectral", "mantel"]),
                "n_accelerated": n_accel,
                "n_decelerated": n_decel,
                "n_total_tests": len(md_df),
            }
            completed += 1

        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipping %s: %s", metric, e)
            continue

    # Save summary
    try:
        from neurofaune.reporting.summarize import build_provenance
        all_summaries["_provenance"] = build_provenance()
    except Exception:
        pass

    sex_suffix = f"_{args.sex}" if args.sex else ""
    summary_path = output_dir / f"rel_distance_summary_{args.modality}{sex_suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    try:
        from neurofaune.reporting.summarize import summarize_analysis
        summarize_analysis("covnet_rel_distance", summary_path, output_dir=output_dir)
    except Exception as exc:
        logger.warning("Failed to generate findings summary: %s", exc)

    progress.finish()
    logger.info(
        "\nRelative distance analysis complete. Results in: %s",
        output_dir,
    )


if __name__ == "__main__":
    main()
