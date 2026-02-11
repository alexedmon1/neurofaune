#!/usr/bin/env python3
"""
Multivariate Group Classification and Regression Analysis for ROI-level DTI metrics.

Runs PERMANOVA, LDA, cross-validated classification (SVM + logistic),
regression (SVR + Ridge + PLS for dose-response), and PCA visualization
per metric, cohort, and feature set. Complements TBSS (mass-univariate)
and CovNet (correlation structure) with multivariate discriminative approaches.

Usage:
    uv run python scripts/run_classification_analysis.py \
        --roi-dir /mnt/arborea/bpa-rat/analysis/roi \
        --output-dir /mnt/arborea/bpa-rat/analysis/classification \
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

from neurofaune.analysis.classification.classifiers import run_classification
from neurofaune.analysis.classification.data_prep import prepare_classification_data
from neurofaune.analysis.classification.lda import run_lda
from neurofaune.analysis.classification.omnibus import run_manova, run_permanova
from neurofaune.analysis.classification.pca import run_pca
from neurofaune.analysis.classification.regression import run_regression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_analysis(
    wide_csv: Path,
    metric: str,
    cohort: str,
    feature_set: str,
    exclusion_csv: Path,
    output_dir: Path,
    n_permutations: int,
    seed: int,
    skip_manova: bool,
    skip_classification: bool,
    skip_regression: bool,
) -> dict:
    """Run full classification pipeline for one metric/cohort/feature_set combo."""
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

    # Phase 2: PERMANOVA
    logger.info("[Phase 2] PERMANOVA...")
    omnibus_dir = combo_dir / "omnibus"
    omnibus_dir.mkdir(parents=True, exist_ok=True)

    permanova = run_permanova(X, y, label_names, n_perm=min(n_permutations * 10, 9999), seed=seed)
    summary["permanova"] = {
        "pseudo_f": permanova["pseudo_f"],
        "p_value": permanova["p_value"],
        "r_squared": permanova["r_squared"],
    }

    # Save PERMANOVA results
    permanova_out = {k: v for k, v in permanova.items() if k != "null_distribution"}
    with open(omnibus_dir / "permanova.json", "w") as f:
        json.dump(permanova_out, f, indent=2)

    # Permutation null plot
    from neurofaune.analysis.classification.visualization import plot_permutation_distribution
    plot_permutation_distribution(
        permanova["null_distribution"],
        permanova["pseudo_f"],
        permanova["p_value"],
        title=f"PERMANOVA — {metric} {cohort_label} {feature_set}",
        xlabel="Pseudo-F",
        out_path=omnibus_dir / "permanova_null.png",
    )

    # Optional MANOVA
    if not skip_manova:
        logger.info("[Phase 2b] MANOVA (optional)...")
        manova = run_manova(X, y, label_names, feature_names)
        if manova is not None:
            summary["manova"] = manova
            with open(omnibus_dir / "manova.json", "w") as f:
                json.dump(manova, f, indent=2)

    # Phase 3: PCA
    logger.info("[Phase 3] PCA...")
    pca_dir = combo_dir / "pca"
    pca_results = run_pca(X, y, label_names, feature_names, pca_dir)
    summary["pca"] = {
        "n_components_95pct": pca_results["n_components_95pct"],
        "pc1_variance": float(pca_results["explained_variance_ratio"][0]),
        "pc2_variance": float(pca_results["explained_variance_ratio"][1])
        if len(pca_results["explained_variance_ratio"]) > 1 else 0.0,
    }

    # Phase 4: LDA
    logger.info("[Phase 4] LDA...")
    lda_dir = combo_dir / "lda"
    lda_results = run_lda(X, y, label_names, feature_names, lda_dir)
    summary["lda"] = {
        "n_discriminants": len(lda_results["explained_variance_ratio"]),
        "ld1_variance": float(lda_results["explained_variance_ratio"][0]),
        "top_features_ld1": lda_results["top_features"].get("LD1", [])[:5],
    }

    # Save LDA results (serialisable parts)
    lda_json = {
        "explained_variance_ratio": lda_results["explained_variance_ratio"].tolist(),
        "top_features": lda_results["top_features"],
    }
    with open(lda_dir / "results.json", "w") as f:
        json.dump(lda_json, f, indent=2)

    # Phase 5: Classification
    if not skip_classification:
        logger.info("[Phase 5] Classification (LOOCV + permutation)...")
        clf_dir = combo_dir / "classification"
        clf_results = run_classification(
            X, y, label_names, feature_names,
            n_permutations=n_permutations,
            seed=seed,
            output_dir=clf_dir,
        )

        # Serialise classification results
        clf_json = {}
        for clf_name, result in clf_results.items():
            clf_json[clf_name] = {
                "accuracy": result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "permutation_p_value": result["permutation_p_value"],
                "per_class_accuracy": result["per_class_accuracy"],
                "confusion_matrix": result["confusion_matrix"].tolist(),
            }
            summary[f"classification_{clf_name}"] = {
                "accuracy": result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "permutation_p_value": result["permutation_p_value"],
            }

        with open(clf_dir / "classification.json", "w") as f:
            json.dump(clf_json, f, indent=2)
    else:
        logger.info("[Phase 5] Skipping classification (--skip-classification)")

    # Phase 6: Regression (dose-response)
    if not skip_regression:
        logger.info("[Phase 6] Regression (LOOCV dose-response + permutation)...")
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
    else:
        logger.info("[Phase 6] Skipping regression (--skip-regression)")

    # Save per-combo summary
    with open(combo_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def write_design_description(args: argparse.Namespace, output_path: Path) -> None:
    """Write a human-readable description of the classification analysis design."""
    lines = [
        "ANALYSIS DESCRIPTION",
        "====================",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Analysis: Multivariate Group Classification",
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
        "Grouping: Dose (C, L, M, H — 4 groups)",
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
        "1. PERMANOVA (Permutational MANOVA)",
        "   - Non-parametric omnibus test using Euclidean distances",
        f"   - Permutations: up to {min(args.n_permutations * 10, 9999)}",
        "   - Reports pseudo-F, R², and permutation p-value",
        "",
    ])

    if not args.skip_manova:
        lines.extend([
            "2. MANOVA (optional, if statsmodels available)",
            "   - Parametric complement to PERMANOVA",
            "   - Pillai's trace (most robust to violations)",
            "",
        ])

    lines.extend([
        "3. PCA (Principal Component Analysis)",
        "   - Unsupervised dimensionality reduction",
        "   - PC1 vs PC2 scatter with 95% confidence ellipses",
        "   - Scree plot and feature loading charts",
        "",
        "4. LDA (Linear Discriminant Analysis)",
        "   - Supervised dimensionality reduction",
        "   - Maximises between-group separation",
        "   - 3 discriminant functions for 4 dose groups",
        "   - Structure correlations for feature interpretation",
        "",
    ])

    if not args.skip_classification:
        lines.extend([
            "5. Classification (LOOCV + permutation test)",
            "   - Linear SVM (C=1.0) and multinomial logistic regression",
            "   - Leave-one-out cross-validation (standard for n < 100)",
            f"   - Permutation test: {args.n_permutations} shuffles",
            "   - Reports accuracy, balanced accuracy, confusion matrix",
            "",
        ])

    if not args.skip_regression:
        lines.extend([
            "6. Regression (LOOCV dose-response + permutation test)",
            "   - Dose as ordinal: C=0, L=1, M=2, H=3",
            "   - Linear SVR (C=1.0), Ridge (alpha=1.0), PLS regression",
            "   - Leave-one-out cross-validation",
            f"   - Permutation test: {args.n_permutations} shuffles",
            "   - Reports R², MAE, Spearman rho, predicted vs actual scatter",
            "",
        ])

    lines.extend([
        "PREPROCESSING",
        "-------------",
        "- Z-score standardisation (StandardScaler)",
        "- Median imputation for remaining NaN values",
        "- ROIs with >20% zeros or all-NaN excluded",
        "",
        "PARAMETERS",
        "----------",
        f"Permutations (classification): {args.n_permutations}",
        f"Permutations (PERMANOVA): up to {min(args.n_permutations * 10, 9999)}",
        f"Random seed: {args.seed}",
    ])

    output_path.write_text("\n".join(lines))
    logger.info("Saved analysis description: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Multivariate Group Classification Analysis for ROI-level DTI metrics"
    )
    parser.add_argument(
        "--roi-dir", type=Path, required=True,
        help="Directory containing roi_*_wide.csv files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for classification results",
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
        help="Number of permutations for classification test (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-manova", action="store_true",
        help="Skip optional MANOVA test",
    )
    parser.add_argument(
        "--skip-classification", action="store_true",
        help="Skip LOOCV classification (only run PERMANOVA, PCA, LDA)",
    )
    parser.add_argument(
        "--skip-regression", action="store_true",
        help="Skip LOOCV regression (SVR, Ridge, PLS dose-response)",
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
        "skip_manova": args.skip_manova,
        "skip_classification": args.skip_classification,
        "skip_regression": args.skip_regression,
        "timestamp": datetime.now().isoformat(),
    }
    with open(args.output_dir / "analysis_config.json", "w") as f:
        json.dump(config, f, indent=2)

    write_design_description(args, args.output_dir / "design_description.txt")

    # Cohorts to analyse: each individually + pooled
    cohorts = [None, "p30", "p60", "p90"]

    all_summaries = {}
    n_significant_permanova = 0
    best_accuracy = 0.0
    best_regression_r2 = -999.0
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

                summary = run_single_analysis(
                    wide_csv=wide_csv,
                    metric=metric,
                    cohort=cohort,
                    feature_set=feature_set,
                    exclusion_csv=args.exclusion_csv,
                    output_dir=args.output_dir,
                    n_permutations=args.n_permutations,
                    seed=args.seed,
                    skip_manova=args.skip_manova,
                    skip_classification=args.skip_classification,
                    skip_regression=args.skip_regression,
                )
                metric_summaries[key] = summary

                # Track global stats
                if summary.get("status") == "completed":
                    total_n_subjects = max(total_n_subjects, summary.get("n_samples", 0))
                    perm_p = summary.get("permanova", {}).get("p_value", 1.0)
                    if perm_p < 0.05:
                        n_significant_permanova += 1
                    for clf_key in ["classification_svm", "classification_logistic"]:
                        acc = summary.get(clf_key, {}).get("accuracy", 0.0)
                        best_accuracy = max(best_accuracy, acc)
                    for reg_key in ["regression_svr", "regression_ridge", "regression_pls"]:
                        r2 = summary.get(reg_key, {}).get("r_squared", -999.0)
                        best_regression_r2 = max(best_regression_r2, r2)

        all_summaries[metric] = metric_summaries

    # Save overall summary
    overall = {
        "metrics": args.metrics,
        "feature_sets": args.feature_sets,
        "n_subjects": total_n_subjects,
        "n_significant_permanova": n_significant_permanova,
        "best_classification_accuracy": best_accuracy,
        "best_regression_r2": best_regression_r2 if best_regression_r2 > -999.0 else None,
        "timestamp": datetime.now().isoformat(),
        "per_metric": all_summaries,
    }

    summary_path = args.output_dir / "classification_summary.json"
    with open(summary_path, "w") as f:
        json.dump(overall, f, indent=2, default=str)
    logger.info("Saved overall summary: %s", summary_path)

    # Register with unified reporting system
    try:
        from neurofaune.analysis.reporting import register as report_register

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
            entry_id="classification",
            analysis_type="classification",
            display_name=f"Multivariate Classification ({', '.join(args.metrics)})",
            output_dir=str(args.output_dir.relative_to(analysis_root)),
            summary_stats={
                "metrics": args.metrics,
                "feature_sets": args.feature_sets,
                "n_subjects": total_n_subjects,
                "n_significant_permanova": n_significant_permanova,
                "best_classification_accuracy": round(best_accuracy, 3),
                "best_regression_r2": round(best_regression_r2, 3) if best_regression_r2 > -999.0 else None,
            },
            figures=figures,
            source_summary_json=str(summary_path.relative_to(analysis_root)),
            config=config,
        )
    except Exception as exc:
        logger.warning("Failed to register with reporting system: %s", exc)

    logger.info("\nClassification analysis complete. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
