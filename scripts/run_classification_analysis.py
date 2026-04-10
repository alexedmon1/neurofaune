#!/usr/bin/env python3
"""
Multivariate group classification analysis (example CLI wrapper).

Example scripts in scripts/ are reference wrappers. Each study should
create its own wrapper scripts that import from the library.

Usage:
    # Config-driven (recommended)
    uv run python scripts/run_classification_analysis.py \\
        --config /path/to/config.yaml \\
        --metrics FA MD AD RD \\
        --feature-sets all \\
        --n-permutations 1000 --seed 42 --force

    # Explicit paths (backwards compatible)
    uv run python scripts/run_classification_analysis.py \\
        --roi-dir /path/to/network/roi \\
        --output-dir /path/to/network/classification/dwi \\
        --metrics FA MD AD RD --feature-sets all \\
        --n-permutations 1000 --seed 42
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.network.classification.pipeline import ClassificationAnalysis
from neurofaune.analysis.progress import AnalysisProgress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_NAME = "run_classification_analysis.py"
COHORTS = [None, "p30", "p60", "p90"]


def main():
    parser = argparse.ArgumentParser(
        description="Multivariate group classification analysis"
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
        help="Output directory for classification results",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["FA", "MD", "AD", "RD"],
        help="Metrics to analyse (default: FA MD AD RD)",
    )
    parser.add_argument(
        "--feature-sets", nargs="+", default=["all"],
        choices=["bilateral", "territory", "all"],
        help="Feature sets (default: all)",
    )
    parser.add_argument(
        "--exclusion-csv", type=Path, default=None,
        help="CSV of sessions to exclude",
    )
    parser.add_argument(
        "--atlas-labels", type=Path,
        default=Path(
            "/mnt/arborea/atlases/SIGMA/"
            "SIGMA_InVivo_Anatomical_Brain_Atlas_Labels.csv"
        ),
        help="SIGMA atlas labels CSV for territory weight plots",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Number of permutations (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--target", type=str, default="dose",
        help="Group variable: 'dose' or column name from wide CSV",
    )
    parser.add_argument(
        "--skip-manova", action="store_true",
        help="Skip optional MANOVA test",
    )
    parser.add_argument(
        "--skip-classification", action="store_true",
        help="Skip LOOCV classification",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing results before running",
    )

    args = parser.parse_args()

    if not args.config and not (args.roi_dir and args.output_dir):
        parser.error("Provide --config or both --roi-dir and --output-dir")

    all_summaries = {}
    n_significant_permanova = 0
    best_accuracy = 0.0
    total_n_subjects = 0

    total_tasks = len(args.metrics) * len(COHORTS) * len(args.feature_sets)
    first_analysis = None
    completed = 0

    for metric in args.metrics:
        try:
            analysis = ClassificationAnalysis.prepare(
                config_path=args.config,
                roi_dir=args.roi_dir,
                output_dir=args.output_dir,
                metric=metric,
                exclusion_csv=args.exclusion_csv,
                atlas_labels=args.atlas_labels,
                target=args.target,
                force=args.force,
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipping %s: %s", metric, e)
            continue

        if first_analysis is None:
            first_analysis = analysis
            analysis.write_design_description(
                feature_sets=args.feature_sets,
                n_permutations=args.n_permutations,
                seed=args.seed,
                skip_manova=args.skip_manova,
                skip_classification=args.skip_classification,
            )

        progress = AnalysisProgress(
            analysis.output_dir, SCRIPT_NAME, total_tasks
        )
        metric_summaries = {}

        for cohort in COHORTS:
            for feature_set in args.feature_sets:
                cohort_label = cohort or "pooled"
                key = f"{cohort_label}_{feature_set}"

                progress.update(
                    task=f"{metric} / {cohort_label} / {feature_set}",
                    phase="running",
                    completed=completed,
                )

                summary = analysis.run(
                    cohort=cohort,
                    feature_set=feature_set,
                    n_permutations=args.n_permutations,
                    seed=args.seed,
                    skip_manova=args.skip_manova,
                    skip_classification=args.skip_classification,
                )
                metric_summaries[key] = summary
                completed += 1

                if summary.get("status") == "completed":
                    total_n_subjects = max(
                        total_n_subjects, summary.get("n_samples", 0)
                    )
                    perm_p = summary.get("permanova", {}).get("p_value", 1.0)
                    if perm_p < 0.05:
                        n_significant_permanova += 1
                    acc = summary.get("classification_svm", {}).get(
                        "accuracy", 0.0
                    )
                    best_accuracy = max(best_accuracy, acc)

        all_summaries[metric] = metric_summaries

    if first_analysis is None:
        logger.error("No metrics could be prepared")
        sys.exit(1)

    overall = {
        "metrics": args.metrics,
        "feature_sets": args.feature_sets,
        "n_subjects": total_n_subjects,
        "n_significant_permanova": n_significant_permanova,
        "best_classification_accuracy": best_accuracy,
        "timestamp": datetime.now().isoformat(),
        "per_metric": all_summaries,
    }

    summary_path = first_analysis.output_dir / "classification_summary.json"
    with open(summary_path, "w") as f:
        json.dump(overall, f, indent=2, default=str)

    progress.finish()

    try:
        from neurofaune.reporting.summarize import summarize_analysis
        summarize_analysis(
            "classification", summary_path,
            output_dir=first_analysis.output_dir,
        )
    except Exception as exc:
        logger.warning("Failed to generate findings summary: %s", exc)

    logger.info(
        "\nClassification complete. Results in: %s", first_analysis.output_dir
    )


if __name__ == "__main__":
    main()
