#!/usr/bin/env python3
"""
Dose-Response Regression Analysis for ROI-level DTI metrics.

Runs cross-validated regression (SVR, Ridge, PLS) with permutation testing
per metric, cohort, and feature set. Tests whether joint ROI patterns predict
ordinal dose level (C=0, L=1, M=2, H=3), complementing classification
(discrete group discrimination) with a continuous dose-response approach.

Usage:
    uv run python scripts/run_regression_analysis.py \
        --roi-dir /mnt/arborea/bpa-rat/analysis/roi \
        --output-dir /mnt/arborea/bpa-rat/analysis/regression \
        --metrics FA MD AD RD \
        --feature-sets bilateral territory \
        --exclusion-csv /mnt/arborea/bpa-rat/dti_nonstandard_slices.csv \
        --n-permutations 1000 \
        --seed 42
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.analysis.classification.data_prep import prepare_classification_data
from neurofaune.analysis.regression.dose_response import run_regression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_regression(
    wide_csv: Path,
    metric: str,
    cohort: str,
    feature_set: str,
    exclusion_csv: Path,
    output_dir: Path,
    n_permutations: int,
    seed: int,
) -> dict:
    """Run regression pipeline for one metric/cohort/feature_set combo."""
    cohort_label = cohort if cohort else "pooled"
    combo_dir = output_dir / metric / cohort_label / feature_set

    logger.info(
        "\n%s\n  Metric: %s | Cohort: %s | Features: %s\n%s",
        "=" * 60, metric, cohort_label, feature_set, "=" * 60,
    )

    # Phase 1: Data preparation
    logger.info("[Phase 1] Preparing data...")
    try:
        data = prepare_classification_data(
            wide_csv=wide_csv,
            feature_set=feature_set,
            cohort_filter=cohort if cohort else None,
            exclusion_csv=exclusion_csv,
        )
    except ValueError as exc:
        logger.warning("Skipping %s/%s/%s: %s", metric, cohort_label, feature_set, exc)
        return {"status": "skipped", "reason": str(exc)}

    X, y = data["X"], data["y"]
    label_names = data["label_names"]
    feature_names = data["feature_names"]
    n_samples, n_features = X.shape

    if n_samples < 10:
        logger.warning("Too few samples (n=%d) — skipping", n_samples)
        return {"status": "skipped", "reason": f"n={n_samples} too small"}

    if len(np.unique(y)) < 2:
        logger.warning("Fewer than 2 groups — skipping")
        return {"status": "skipped", "reason": "fewer than 2 groups"}

    summary = {
        "status": "completed",
        "metric": metric,
        "cohort": cohort_label,
        "feature_set": feature_set,
        "n_samples": n_samples,
        "n_features": n_features,
        "label_names": label_names,
        "group_sizes": {
            name: int((y == i).sum()) for i, name in enumerate(label_names)
        },
    }

    # Phase 2: Regression (dose-response)
    logger.info("[Phase 2] Regression (LOOCV dose-response + permutation)...")
    reg_dir = combo_dir / "regression"
    y_ordinal = y.astype(float)
    reg_results = run_regression(
        X, y_ordinal, label_names, feature_names,
        n_permutations=n_permutations,
        seed=seed,
        output_dir=reg_dir,
    )

    # Serialise regression results
    reg_json = {}
    for reg_name, result in reg_results.items():
        reg_json[reg_name] = {
            "r_squared": result["r_squared"],
            "mae": result["mae"],
            "spearman_rho": result["spearman_rho"],
            "permutation_p_value": result["permutation_p_value"],
        }
        summary[f"regression_{reg_name}"] = {
            "r_squared": result["r_squared"],
            "mae": result["mae"],
            "spearman_rho": result["spearman_rho"],
            "permutation_p_value": result["permutation_p_value"],
        }

    with open(reg_dir / "regression.json", "w") as f:
        json.dump(reg_json, f, indent=2)

    # Save per-combo summary
    combo_dir.mkdir(parents=True, exist_ok=True)
    with open(combo_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def write_design_description(args: argparse.Namespace, output_path: Path) -> None:
    """Write a human-readable description of the regression analysis design."""
    lines = [
        "ANALYSIS DESCRIPTION",
        "====================",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Analysis: Dose-Response Regression",
        "",
        "DATA SOURCE",
        "-----------",
        f"ROI directory: {args.roi_dir}",
        f"Exclusion list: {args.exclusion_csv or 'None'}",
        f"Metrics: {', '.join(args.metrics)}",
        f"Feature sets: {', '.join(args.feature_sets)}",
        "",
        "EXPERIMENTAL DESIGN",
        "-------------------",
        "Dose as ordinal: C=0, L=1, M=2, H=3",
        "Cohorts analysed: p30, p60, p90, and pooled",
        "",
        "FEATURE SETS",
        "------------",
    ]

    if "bilateral" in args.feature_sets:
        lines.append("- bilateral: Bilateral-averaged region ROIs (~50 features)")
        lines.append("  L/R ROI pairs averaged, territories excluded")

    if "territory" in args.feature_sets:
        lines.append("- territory: Territory aggregate ROIs (~15 features)")
        lines.append("  Coarser anatomical groupings")

    lines.extend([
        "",
        "STATISTICAL METHODS",
        "-------------------",
        "1. Linear SVR (C=1.0)",
        "   - Support Vector Regression with linear kernel",
        "   - Leave-one-out cross-validation",
        "",
        "2. Ridge Regression (alpha=1.0)",
        "   - L2-regularised linear regression",
        "   - Leave-one-out cross-validation",
        "",
        "3. PLS Regression",
        "   - Partial Least Squares (n_components = min(n_classes-1, n_features, n-1))",
        "   - Leave-one-out cross-validation",
        "",
        "PERMUTATION TESTING",
        "-------------------",
        f"- {args.n_permutations} label shuffles per regressor",
        "- Null distribution of LOOCV R²",
        "- Empirical p-value: (n_null >= observed + 1) / (n_perm + 1)",
        "",
        "METRICS REPORTED",
        "----------------",
        "- R² (coefficient of determination)",
        "- MAE (mean absolute error)",
        "- Spearman rho (rank correlation)",
        "- Permutation p-value for R²",
        "",
        "PREPROCESSING",
        "-------------",
        "- Z-score standardisation (StandardScaler)",
        "- Median imputation for remaining NaN values",
        "- ROIs with >20% zeros or all-NaN excluded",
        "",
        "PARAMETERS",
        "----------",
        f"Permutations: {args.n_permutations}",
        f"Random seed: {args.seed}",
    ])

    output_path.write_text("\n".join(lines))
    logger.info("Saved analysis description: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Dose-Response Regression Analysis for ROI-level DTI metrics"
    )
    parser.add_argument(
        "--roi-dir", type=Path, required=True,
        help="Directory containing roi_*_wide.csv files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for regression results",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["FA", "MD", "AD", "RD"],
        help="DTI metrics to analyse (default: FA MD AD RD)",
    )
    parser.add_argument(
        "--feature-sets", nargs="+", default=["bilateral", "territory"],
        choices=["bilateral", "territory"],
        help="Feature sets to analyse (default: bilateral territory)",
    )
    parser.add_argument(
        "--exclusion-csv", type=Path, default=None,
        help="CSV of sessions to exclude (must have subject, session columns)",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Number of permutations for regression test (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.roi_dir.exists():
        logger.error("ROI directory not found: %s", args.roi_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save analysis configuration
    config = {
        "roi_dir": str(args.roi_dir),
        "exclusion_csv": str(args.exclusion_csv) if args.exclusion_csv else None,
        "output_dir": str(args.output_dir),
        "metrics": args.metrics,
        "feature_sets": args.feature_sets,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(args.output_dir / "analysis_config.json", "w") as f:
        json.dump(config, f, indent=2)

    write_design_description(args, args.output_dir / "design_description.txt")

    # Cohorts to analyse: each individually + pooled
    cohorts = [None, "p30", "p60", "p90"]

    all_summaries = {}
    best_r2 = -999.0
    best_rho = -999.0
    total_n_subjects = 0

    for metric in args.metrics:
        wide_csv = args.roi_dir / f"roi_{metric}_wide.csv"
        if not wide_csv.exists():
            logger.warning("Wide CSV not found: %s, skipping %s", wide_csv, metric)
            continue

        metric_summaries = {}

        for cohort in cohorts:
            for feature_set in args.feature_sets:
                cohort_label = cohort if cohort else "pooled"
                key = f"{cohort_label}_{feature_set}"

                summary = run_single_regression(
                    wide_csv=wide_csv,
                    metric=metric,
                    cohort=cohort,
                    feature_set=feature_set,
                    exclusion_csv=args.exclusion_csv,
                    output_dir=args.output_dir,
                    n_permutations=args.n_permutations,
                    seed=args.seed,
                )
                metric_summaries[key] = summary

                # Track global stats
                if summary.get("status") == "completed":
                    total_n_subjects = max(total_n_subjects, summary.get("n_samples", 0))
                    for reg_key in ["regression_svr", "regression_ridge", "regression_pls"]:
                        r2 = summary.get(reg_key, {}).get("r_squared", -999.0)
                        best_r2 = max(best_r2, r2)
                        rho = summary.get(reg_key, {}).get("spearman_rho", -999.0)
                        best_rho = max(best_rho, rho)

        all_summaries[metric] = metric_summaries

    # Save overall summary
    overall = {
        "metrics": args.metrics,
        "feature_sets": args.feature_sets,
        "n_subjects": total_n_subjects,
        "best_r2": best_r2 if best_r2 > -999.0 else None,
        "best_spearman_rho": best_rho if best_rho > -999.0 else None,
        "timestamp": datetime.now().isoformat(),
        "per_metric": all_summaries,
    }

    summary_path = args.output_dir / "regression_summary.json"
    with open(summary_path, "w") as f:
        json.dump(overall, f, indent=2, default=str)
    logger.info("Saved overall summary: %s", summary_path)

    # Register with unified reporting system
    try:
        from neurofaune.reporting import register as report_register

        analysis_root = args.output_dir.parent

        # Collect figure paths
        figures = []
        for fig in sorted(args.output_dir.rglob("*.png"))[:20]:
            try:
                figures.append(str(fig.relative_to(analysis_root)))
            except ValueError:
                pass

        report_register(
            analysis_root=analysis_root,
            entry_id="regression",
            analysis_type="regression",
            display_name=f"Dose-Response Regression ({', '.join(args.metrics)})",
            output_dir=str(args.output_dir.relative_to(analysis_root)),
            summary_stats={
                "metrics": args.metrics,
                "feature_sets": args.feature_sets,
                "n_subjects": total_n_subjects,
                "best_r2": round(best_r2, 3) if best_r2 > -999.0 else None,
                "best_spearman_rho": round(best_rho, 3) if best_rho > -999.0 else None,
            },
            figures=figures,
            source_summary_json=str(summary_path.relative_to(analysis_root)),
            config=config,
        )
    except Exception as exc:
        logger.warning("Failed to register with reporting system: %s", exc)

    logger.info("\nRegression analysis complete. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
