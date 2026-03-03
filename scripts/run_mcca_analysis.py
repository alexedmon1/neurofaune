#!/usr/bin/env python3
"""
Multi-modal Canonical Correlation Analysis (MCCA) for ROI-level data.

Finds linear combinations of ROI features that maximise correlation across
modality views (e.g., DWI diffusion metrics, MSME tissue fractions, and
functional activity metrics). Tests significance via permutation and
evaluates dose-response associations in canonical variate space.

Usage:
    uv run python scripts/run_mcca_analysis.py \
        --roi-dir $STUDY_ROOT/network/roi \
        --output-dir $STUDY_ROOT/network/mcca \
        --views dwi:FA,MD,AD,RD msme:MWF,IWF,CSFF,T2 func:fALFF,ReHo,ALFF \
        --feature-set bilateral \
        --n-components 5 \
        --regs lw \
        --n-permutations 5000 --seed 42
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.analysis.progress import AnalysisProgress
from neurofaune.network.mcca import (
    load_multiview_data,
    permutation_test_mcca,
    run_mcca,
    test_dose_association,
    test_group_differences,
    test_sex_differences,
)
from neurofaune.network.mcca_visualization import (
    plot_canonical_correlations,
    plot_cross_view_loadings,
    plot_loadings_heatmap,
    plot_permutation_null,
    plot_scores_by_cohort,
    plot_scores_by_dose,
    plot_scores_by_sex,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_views(view_specs: list) -> dict:
    """Parse view specifications like 'dwi:FA,MD,AD,RD' into a dict.

    Parameters
    ----------
    view_specs : list of str
        Each element is 'view_name:metric1,metric2,...'.

    Returns
    -------
    dict mapping view_name -> list of metric names
    """
    views = {}
    for spec in view_specs:
        if ":" not in spec:
            raise ValueError(f"Invalid view spec '{spec}'. Expected 'name:metric1,metric2,...'")
        name, metrics_str = spec.split(":", 1)
        metrics = [m.strip() for m in metrics_str.split(",") if m.strip()]
        if not metrics:
            raise ValueError(f"No metrics specified for view '{name}'")
        views[name] = metrics
    return views


def run_cohort_mcca(
    roi_dir: Path,
    views: dict,
    feature_set: str,
    cohort: str,
    output_dir: Path,
    n_components: int,
    regs: str,
    n_permutations: int,
    seed: int,
    exclusion_csv: Path,
    confounds: list = None,
    target: str = "dose",
) -> dict:
    """Run MCCA pipeline for a single cohort (or pooled)."""
    cohort_label = cohort if cohort else "pooled"
    combo_dir = output_dir / cohort_label / feature_set
    combo_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "\n%s\n  MCCA | Cohort: %s | Features: %s\n%s",
        "=" * 60, cohort_label, feature_set, "=" * 60,
    )

    # Phase 1: Load multi-view data
    logger.info("[Phase 1] Loading multi-view data...")
    try:
        Xs, view_names, feature_names, metadata = load_multiview_data(
            roi_dir=roi_dir,
            views=views,
            feature_set=feature_set,
            cohort_filter=cohort if cohort else None,
            exclusion_csv=exclusion_csv,
            confounds=confounds,
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.warning("Skipping %s/%s: %s", cohort_label, feature_set, exc)
        return {"status": "skipped", "reason": str(exc)}

    n_samples = Xs[0].shape[0]
    if n_samples < 10:
        logger.warning("Too few samples (n=%d) — skipping", n_samples)
        return {"status": "skipped", "reason": f"n={n_samples} too small"}

    # Encode dose labels (always needed for coloring/group sizes)
    dose_order = ["C", "L", "M", "H"]
    dose_map = {d: i for i, d in enumerate(dose_order)}
    dose_labels = np.array([dose_map.get(d, -1) for d in metadata["dose"].values])
    valid = dose_labels >= 0
    if not valid.all():
        logger.warning("Dropping %d samples with unknown dose", (~valid).sum())
        Xs = [X[valid] for X in Xs]
        metadata = metadata[valid].reset_index(drop=True)
        dose_labels = dose_labels[valid]
        n_samples = len(dose_labels)

    label_names = [d for d in dose_order if d in set(metadata["dose"].values)]

    # Build target array for dose association test
    _session_to_auc = {
        "ses-p30": "AUC_p30",
        "ses-p60": "AUC_p60",
        "ses-p90": "AUC_p90",
    }
    if target == "auc":
        # Session-matched AUC
        auc_values = []
        for _, row in metadata.iterrows():
            auc_col = _session_to_auc.get(row["session"])
            if auc_col and auc_col in metadata.columns:
                auc_values.append(row.get(auc_col, np.nan))
            else:
                auc_values.append(np.nan)
        target_values = np.array(auc_values, dtype=float)
        valid_target = ~np.isnan(target_values)
        if not valid_target.all():
            n_drop = (~valid_target).sum()
            logger.warning("Dropping %d samples with NaN AUC", n_drop)
            Xs = [X[valid_target] for X in Xs]
            metadata = metadata[valid_target].reset_index(drop=True)
            dose_labels = dose_labels[valid_target]
            target_values = target_values[valid_target]
            n_samples = len(target_values)
        target_name = "AUC"
    else:
        target_values = dose_labels.astype(float)
        target_name = "Ordinal dose"

    summary = {
        "status": "completed",
        "cohort": cohort_label,
        "feature_set": feature_set,
        "target": target,
        "target_name": target_name,
        "n_samples": n_samples,
        "view_dims": {vn: X.shape[1] for vn, X in zip(view_names, Xs)},
        "group_sizes": {
            name: int((dose_labels == dose_map[name]).sum()) for name in label_names
        },
        "confounds_residualized": confounds if confounds else None,
    }

    # Phase 2: Fit MCCA
    logger.info("[Phase 2] Fitting MCCA (n_components=%d, regs=%s)...", n_components, regs)
    actual_n_comp = min(n_components, n_samples - 1, min(X.shape[1] for X in Xs))
    result = run_mcca(Xs, n_components=actual_n_comp, regs=regs)
    result.view_names = view_names
    result.feature_names = feature_names

    summary["n_components"] = result.n_components
    summary["canonical_correlations"] = result.canonical_correlations.tolist()

    logger.info(
        "Canonical correlations: %s",
        ", ".join(f"CV{i+1}={r:.4f}" for i, r in enumerate(result.canonical_correlations)),
    )

    # Phase 3: Permutation test
    logger.info("[Phase 3] Permutation test (%d permutations)...", n_permutations)
    perm_result = permutation_test_mcca(
        Xs, result.canonical_correlations,
        n_components=result.n_components,
        regs=regs,
        n_permutations=n_permutations,
        seed=seed,
    )
    summary["permutation_p_values"] = perm_result.p_values.tolist()

    for i in range(result.n_components):
        logger.info(
            "  CV%d: r=%.4f, p=%.4f%s",
            i + 1, result.canonical_correlations[i], perm_result.p_values[i],
            " *" if perm_result.p_values[i] < 0.05 else "",
        )

    # Phase 4: Target association (dose or AUC)
    logger.info("[Phase 4] Testing %s association in canonical variate space...", target_name)
    dose_result = test_dose_association(
        result.scores, target_values,
        n_permutations=n_permutations, seed=seed,
    )
    assoc_key = "auc_association" if target == "auc" else "dose_association"
    summary[assoc_key] = {
        f"CV{i+1}": {
            "spearman_rho": float(dose_result.spearman_rho[i]),
            "p_value": float(dose_result.p_values[i]),
        }
        for i in range(result.n_components)
    }

    for i in range(result.n_components):
        logger.info(
            "  CV%d ~ %s: rho=%.4f, p=%.4f%s",
            i + 1, target_name, dose_result.spearman_rho[i], dose_result.p_values[i],
            " *" if dose_result.p_values[i] < 0.05 else "",
        )

    # Phase 5: PERMANOVA on scores
    logger.info("[Phase 5] PERMANOVA on MCCA score space...")
    permanova = test_group_differences(
        result.scores, dose_labels,
        n_permutations=min(n_permutations, 9999), seed=seed,
    )
    summary["permanova"] = permanova
    logger.info(
        "  PERMANOVA: F=%.4f, R²=%.4f, p=%.4f",
        permanova["pseudo_f"], permanova["r_squared"], permanova["p_value"],
    )

    # Phase 5b: Sex differences
    if "sex" in metadata.columns and metadata["sex"].nunique() == 2:
        logger.info("[Phase 5b] Testing sex differences in canonical variate space...")
        sex_result = test_sex_differences(
            result.scores, metadata["sex"].values,
            n_permutations=n_permutations, seed=seed,
        )
        summary["sex_differences"] = {
            "permanova": sex_result.permanova,
            "n_male": sex_result.n_male,
            "n_female": sex_result.n_female,
            "per_component": {
                f"CV{i+1}": sex_result.per_component[i]
                for i in range(result.n_components)
            },
        }
        logger.info(
            "  Sex PERMANOVA: F=%.4f, R²=%.4f, p=%.4f",
            sex_result.permanova["pseudo_f"],
            sex_result.permanova["r_squared"],
            sex_result.permanova["p_value"],
        )

    # Phase 6: Visualisations
    logger.info("[Phase 6] Generating visualisations...")

    plot_canonical_correlations(
        result, perm_result,
        title=f"Canonical Correlations — {cohort_label}",
        out_path=combo_dir / "canonical_correlations.png",
    )

    plot_scores_by_dose(
        result, dose_labels, label_names,
        title=f"MCCA Scores by Dose — {cohort_label}",
        out_path=combo_dir / "scores_by_dose.png",
    )

    plot_scores_by_cohort(
        result, metadata["cohort"].values,
        title=f"MCCA Scores by Cohort — {cohort_label}",
        out_path=combo_dir / "scores_by_cohort.png",
    )

    if "sex" in metadata.columns and metadata["sex"].nunique() == 2:
        plot_scores_by_sex(
            result, metadata["sex"].values,
            title=f"MCCA Scores by Sex — {cohort_label}",
            out_path=combo_dir / "scores_by_sex.png",
        )

    for k, vn in enumerate(view_names):
        plot_loadings_heatmap(
            result, view_idx=k,
            title=f"Top Loadings — {vn} — {cohort_label}",
            out_path=combo_dir / f"loadings_{vn}.png",
        )

    plot_cross_view_loadings(
        result, component=0,
        title=f"Cross-View Loadings CV1 — {cohort_label}",
        out_path=combo_dir / "cross_view_loadings_cv1.png",
    )

    plot_permutation_null(
        perm_result,
        title=f"Permutation Null — {cohort_label}",
        out_path=combo_dir / "permutation_null.png",
    )

    # Save detailed results
    mcca_json = {
        "n_samples": n_samples,
        "n_components": result.n_components,
        "regularisation": regs,
        "confounds_residualized": confounds if confounds else None,
        "view_names": view_names,
        "view_dims": {vn: X.shape[1] for vn, X in zip(view_names, Xs)},
        "canonical_correlations": result.canonical_correlations.tolist(),
        "permutation_p_values": perm_result.p_values.tolist(),
        assoc_key: summary[assoc_key],
        "permanova": permanova,
        "group_sizes": summary["group_sizes"],
    }
    if "sex_differences" in summary:
        mcca_json["sex_differences"] = summary["sex_differences"]
    with open(combo_dir / "mcca_results.json", "w") as f:
        json.dump(mcca_json, f, indent=2)

    # Save per-combo summary
    with open(combo_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def write_design_description(args: argparse.Namespace, views: dict, output_path: Path) -> None:
    """Write a human-readable description of the MCCA analysis design."""
    lines = [
        "ANALYSIS DESCRIPTION",
        "====================",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Analysis: Multiset Canonical Correlation Analysis (MCCA)",
        "",
        "DATA SOURCE",
        "-----------",
        f"ROI directory: {args.roi_dir}",
        f"Exclusion list: {args.exclusion_csv or 'None'}",
        f"Feature set: {args.feature_set}",
        "",
        "VIEWS",
        "-----",
    ]

    for vn, metrics in views.items():
        lines.append(f"- {vn}: {', '.join(metrics)}")

    lines.extend([
        "",
        "EXPERIMENTAL DESIGN",
        "-------------------",
        "Grouping: Dose (C, L, M, H — 4 groups)",
        "Cohorts analysed: p30, p60, p90, and pooled",
        "",
        "STATISTICAL METHODS",
        "-------------------",
        "1. Regularised MCCA",
        f"   - Regularisation: {args.regs}",
        f"   - Components: {args.n_components}",
        "   - Generalised eigenvalue decomposition on block covariance matrices",
        "   - Subjects intersected across all views",
        "",
        "2. Permutation test for canonical correlations",
        f"   - {args.n_permutations} permutations (shuffle views 1..K, fix view 0)",
        "   - Empirical p-value per component",
        "",
        f"3. {'AUC' if getattr(args, 'target', 'dose') == 'auc' else 'Dose'} association test",
        "   - Average MCCA scores across views per component",
        f"   - Spearman correlation with {'continuous AUC (session-matched)' if getattr(args, 'target', 'dose') == 'auc' else 'ordinal dose (C=0, L=1, M=2, H=3)'}",
        f"   - {args.n_permutations} permutation p-values (two-tailed)",
        "",
        "4. PERMANOVA on MCCA score space",
        "   - Euclidean distances on average canonical variate scores",
        "   - Tests group separability in fused multi-modal space",
        "",
        "5. Sex differences test (when sex data available)",
        "   - PERMANOVA on MCCA score space by sex",
        "   - Per-CV Cohen's d with permutation p-values",
        "",
        "PREPROCESSING",
        "-------------",
        f"- Confounds residualized: {args.confounds or 'None'}",
        "- Z-score standardisation per view (independent)",
        "- Median imputation for NaN values",
        "- ROIs with >20% zeros excluded",
        "- Subject intersection across all views",
        "",
        "PARAMETERS",
        "----------",
        f"Components: {args.n_components}",
        f"Regularisation: {args.regs}",
        f"Permutations: {args.n_permutations}",
        f"Random seed: {args.seed}",
    ])

    output_path.write_text("\n".join(lines))
    logger.info("Saved analysis description: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-modal Canonical Correlation Analysis (MCCA) for ROI data"
    )
    parser.add_argument(
        "--roi-dir", type=Path, required=True,
        help="Directory containing roi_*_wide.csv files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for MCCA results",
    )
    parser.add_argument(
        "--views", nargs="+", required=True,
        help="View specifications: 'name:metric1,metric2,...' (e.g. 'dwi:FA,MD,AD,RD')",
    )
    parser.add_argument(
        "--feature-set", default="bilateral",
        choices=["bilateral", "territory"],
        help="Feature set to use (default: bilateral)",
    )
    parser.add_argument(
        "--n-components", type=int, default=5,
        help="Number of canonical components (default: 5)",
    )
    parser.add_argument(
        "--regs", default="lw",
        help="Regularisation method: 'lw' (Ledoit-Wolf), 'identity', or float (default: lw)",
    )
    parser.add_argument(
        "--target", choices=["dose", "auc"], default="dose",
        help="Target for dose association: 'dose' (ordinal C=0..H=3) or 'auc' (continuous AUC)",
    )
    parser.add_argument(
        "--confounds", nargs="*", default=None,
        help="Metadata columns to residualize before MCCA (e.g. 'sex')",
    )
    parser.add_argument(
        "--exclusion-csv", type=Path, default=None,
        help="CSV of sessions to exclude (must have subject, session columns)",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=5000,
        help="Number of permutations (default: 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Parse view specifications
    views = parse_views(args.views)
    logger.info("Views: %s", {k: v for k, v in views.items()})

    # All metrics across all views (for provenance)
    all_metrics = []
    for metrics in views.values():
        all_metrics.extend(metrics)

    # Validate inputs
    if not args.roi_dir.exists():
        logger.error("ROI directory not found: %s", args.roi_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save analysis configuration
    config = {
        "roi_dir": str(args.roi_dir),
        "exclusion_csv": str(args.exclusion_csv) if args.exclusion_csv else None,
        "confounds": args.confounds,
        "target": args.target,
        "output_dir": str(args.output_dir),
        "views": {k: v for k, v in views.items()},
        "feature_set": args.feature_set,
        "n_components": args.n_components,
        "regs": args.regs,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(args.output_dir / "analysis_config.json", "w") as f:
        json.dump(config, f, indent=2)

    write_design_description(args, views, args.output_dir / "design_description.txt")

    # Cohorts to analyse: pooled + each individually
    cohorts = [None, "p30", "p60", "p90"]

    all_summaries = {}
    total_n_subjects = 0
    n_significant_cv = 0
    n_significant_dose = 0

    progress = AnalysisProgress(args.output_dir, "run_mcca_analysis.py", len(cohorts))
    completed = 0

    for cohort in cohorts:
        cohort_label = cohort if cohort else "pooled"

        progress.update(task=cohort_label, phase="running MCCA", completed=completed)

        summary = run_cohort_mcca(
            roi_dir=args.roi_dir,
            views=views,
            feature_set=args.feature_set,
            cohort=cohort,
            output_dir=args.output_dir,
            n_components=args.n_components,
            regs=args.regs,
            n_permutations=args.n_permutations,
            seed=args.seed,
            exclusion_csv=args.exclusion_csv,
            confounds=args.confounds,
            target=args.target,
        )
        all_summaries[cohort_label] = summary
        completed += 1

        if summary.get("status") == "completed":
            total_n_subjects = max(total_n_subjects, summary.get("n_samples", 0))
            perm_ps = summary.get("permutation_p_values", [])
            n_significant_cv += sum(1 for p in perm_ps if p < 0.05)
            # Look for dose_association or auc_association
            assoc = summary.get("dose_association") or summary.get("auc_association", {})
            n_significant_dose += sum(
                1 for v in assoc.values() if v.get("p_value", 1.0) < 0.05
            )

    # Save overall summary
    overall = {
        "views": {k: v for k, v in views.items()},
        "feature_set": args.feature_set,
        "n_components": args.n_components,
        "regularisation": args.regs,
        "n_subjects_max": total_n_subjects,
        "n_significant_canonical_variates": n_significant_cv,
        "n_significant_dose_associations": n_significant_dose,
        "timestamp": datetime.now().isoformat(),
        "per_cohort": all_summaries,
    }

    summary_path = args.output_dir / "mcca_summary.json"
    with open(summary_path, "w") as f:
        json.dump(overall, f, indent=2, default=str)
    logger.info("Saved overall summary: %s", summary_path)

    progress.finish()

    # Write provenance tracking
    try:
        from neurofaune.analysis.provenance import write_roi_provenance

        write_roi_provenance(
            output_dir=args.output_dir,
            roi_dir=args.roi_dir,
            metrics=all_metrics,
            exclusion_csv=args.exclusion_csv,
            n_subjects=total_n_subjects,
            analysis_type="mcca",
            extra={
                "views": {k: v for k, v in views.items()},
                "n_components": args.n_components,
                "regularisation": args.regs,
            },
        )
    except Exception as exc:
        logger.warning("Failed to write provenance: %s", exc)

    # Register with unified reporting system
    try:
        from neurofaune.reporting import register as report_register

        analysis_root = args.output_dir.parents[1]

        figures = []
        for fig in sorted(args.output_dir.rglob("*.png"))[:30]:
            try:
                figures.append(str(fig.relative_to(analysis_root)))
            except ValueError:
                pass

        view_desc = ", ".join(f"{k}({','.join(v)})" for k, v in views.items())
        report_register(
            analysis_root=analysis_root,
            entry_id="mcca",
            analysis_type="mcca",
            display_name=f"Multi-modal CCA ({view_desc})",
            output_dir=str(args.output_dir.relative_to(analysis_root)),
            summary_stats={
                "views": list(views.keys()),
                "n_subjects": total_n_subjects,
                "n_significant_cv": n_significant_cv,
                "n_significant_dose": n_significant_dose,
            },
            figures=figures,
            source_summary_json=str(summary_path.relative_to(analysis_root)),
            config=config,
        )
    except Exception as exc:
        logger.warning("Failed to register with reporting system: %s", exc)

    logger.info("\nMCCA analysis complete. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
