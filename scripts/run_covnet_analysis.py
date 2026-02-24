#!/usr/bin/env python3
"""
Covariance Network (CovNet) Analysis for ROI-level DTI metrics.

All-in-one script: prepare data, then run NBS, territory, graph-metric, and
whole-network tests for each metric. For running individual tests separately,
use the thin wrappers (covnet_prepare.py, covnet_nbs.py, etc.).

Usage:
    uv run python scripts/run_covnet_analysis.py \
        --roi-dir /mnt/arborea/bpa-rat/analysis/roi \
        --exclusion-csv /mnt/arborea/bpa-rat/dti_nonstandard_slices.csv \
        --output-dir /mnt/arborea/bpa-rat/analysis/covnet \
        --metrics FA MD AD RD \
        --n-permutations 5000 \
        --seed 42
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.analysis.covnet.pipeline import CovNetAnalysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_metric(analysis: CovNetAnalysis, args: argparse.Namespace) -> dict:
    """Run the full CovNet pipeline for a prepared analysis."""
    metric = analysis.metric
    summary = {
        "metric": metric,
        "n_subjects": analysis.n_subjects,
        "n_bilateral_rois": len(analysis.bilateral_region_cols),
        "n_territory_rois": len(analysis.territory_cols),
        "n_groups_pnd_dose": len(analysis.group_labels),
        "group_sizes": analysis.group_sizes,
    }

    # Build comparisons
    comp_types = ["dose"]
    if not args.skip_cross_timepoint:
        comp_types.extend(["cross-timepoint", "cross-dose-timepoint"])
    comparisons = analysis.resolve_comparisons(comp_types)

    # Phase 5: NBS
    if not args.skip_nbs:
        logger.info(f"\n[Phase 5] Network-Based Statistic ({args.n_permutations} permutations)...")
        nbs_results = analysis.run_nbs(
            comparisons, args.n_permutations, args.nbs_threshold,
            args.seed, args.n_workers,
        )
        summary["nbs_significant_comparisons"] = sum(
            1 for r in nbs_results.values()
            if any(c["pvalue"] < 0.05 for c in r["significant_components"])
        )
    else:
        logger.info("\n[Phase 5] Skipping NBS (--skip-nbs)")

    # Phase 6: Territory
    logger.info("\n[Phase 6] Territory-level edge comparisons (Fisher z + FDR)...")
    analysis.run_territory(comparisons)

    # Phase 7: Graph metrics
    if not args.skip_graph:
        logger.info("\n[Phase 7] Graph metrics and permutation comparison...")
        comparison_df = analysis.run_graph_metrics(
            n_perm=args.n_permutations, seed=args.seed,
        )
        summary["graph_metrics_significant"] = int(
            (comparison_df["p_value"] < 0.05).sum()
        )
    else:
        logger.info("\n[Phase 7] Skipping graph metrics (--skip-graph)")

    # Phase 8: Whole-network
    if not args.skip_whole_network:
        logger.info(f"\n[Phase 8] Whole-network similarity tests ({args.n_permutations} permutations)...")
        wn_df, _ = analysis.run_whole_network(
            comparisons, args.n_permutations, args.seed, args.n_workers,
        )
        n_sig = int(
            (wn_df[["mantel_p", "frobenius_p", "spectral_p"]] < 0.05)
            .any(axis=1).sum()
        )
        summary["whole_network_significant"] = n_sig
    else:
        logger.info("\n[Phase 8] Skipping whole-network tests (--skip-whole-network)")

    return summary


def write_design_description(args: argparse.Namespace, output_path: Path) -> None:
    """Write a human-readable description of the CovNet analysis design."""
    lines = []

    lines.append("ANALYSIS DESCRIPTION")
    lines.append("====================")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Analysis: Covariance Network (CovNet)")
    lines.append("")

    lines.append("DATA SOURCE")
    lines.append("-----------")
    lines.append(f"ROI directory: {args.roi_dir}")
    if args.exclusion_csv:
        lines.append(f"Exclusion list: {args.exclusion_csv}")
    else:
        lines.append("Exclusion list: None")
    lines.append(f"Metrics: {', '.join(args.metrics)}")
    lines.append("")

    lines.append("EXPERIMENTAL GROUPS")
    lines.append("-------------------")
    lines.append("Primary grouping: PND x Dose (up to 12 groups)")
    lines.append("  Groups: p30_C, p30_L, p30_M, p30_H, p60_C, p60_L, "
                  "p60_M, p60_H, p90_C, p90_L, p90_M, p90_H")
    lines.append("")
    lines.append("Descriptive grouping: Sex x PND x Dose (up to 24 groups)")
    lines.append("  Used for visualization only.")
    lines.append("")
    lines.append("Territory grouping: PND x Dose at coarse anatomical level")
    lines.append("")

    lines.append("ROI PROCESSING")
    lines.append("--------------")
    lines.append("- Bilateral averaging of L/R ROI pairs")
    lines.append("- ROIs with >20% zero values excluded")
    lines.append("- ROIs with all-NaN values excluded")
    lines.append("")

    lines.append("STATISTICAL METHODS")
    lines.append("-------------------")
    lines.append("1. Spearman correlation matrices")
    lines.append("   - Pairwise complete observations (min 4 per edge)")
    lines.append("   - Computed per experimental group")
    lines.append("")

    if not args.skip_nbs:
        lines.append("2. Network-Based Statistic (NBS)")
        lines.append("   - Edge-level test: Fisher z-test between group "
                      "correlation matrices")
        lines.append(f"   - Suprathreshold edge threshold: |z| >= "
                      f"{args.nbs_threshold}")
        lines.append("   - Connected component extraction via graph analysis")
        lines.append(f"   - Permutation test: {args.n_permutations} "
                      "permutations for FWER correction")
        lines.append("   - Comparisons: each dose vs control within each PND (9)")
        if not args.skip_cross_timepoint:
            lines.append("   - Cross-timepoint: pairwise PND comparisons within each dose (12)")
            lines.append("   - Cross-dose-timepoint: dosed groups vs controls at later PNDs (9)")
        lines.append("")
    else:
        lines.append("2. Network-Based Statistic (NBS): SKIPPED")
        lines.append("")

    lines.append("3. Territory-level edge comparisons")
    lines.append("   - Fisher z-test per edge")
    lines.append("   - Benjamini-Hochberg FDR correction")
    lines.append("   - Comparisons: each dose vs control within each PND (9)")
    if not args.skip_cross_timepoint:
        lines.append("   - Cross-timepoint: pairwise PND comparisons within each dose (12)")
        lines.append("   - Cross-dose-timepoint: dosed groups vs controls at later PNDs (9)")
    lines.append("")

    if not args.skip_graph:
        lines.append("4. Graph metrics")
        lines.append("   - Metrics: clustering, centrality, small-worldness")
        lines.append("   - Densities: 0.10, 0.15, 0.20, 0.25")
        lines.append(f"   - Permutation comparison: {args.n_permutations} "
                      "permutations")
        lines.append("   - All pairwise group comparisons")
        lines.append("")
    else:
        lines.append("4. Graph metrics: SKIPPED")
        lines.append("")

    if not args.skip_whole_network:
        lines.append("5. Whole-network similarity tests")
        lines.append("   - Mantel test: Pearson r between vectorized upper triangles")
        lines.append("   - Frobenius distance: L2 norm of upper triangle difference")
        lines.append("   - Spectral divergence: L2 distance between eigenvalue spectra")
        lines.append(f"   - Permutation test: {args.n_permutations} permutations")
        lines.append("   - Comparisons: each dose vs control within each PND (9)")
        if not args.skip_cross_timepoint:
            lines.append("   - Cross-timepoint: pairwise PND comparisons within each dose (12)")
            lines.append("   - Cross-dose-timepoint: dosed groups vs controls at later PNDs (9)")
        lines.append("")
    else:
        lines.append("5. Whole-network similarity tests: SKIPPED")
        lines.append("")

    lines.append("PARAMETERS")
    lines.append("----------")
    lines.append(f"Permutations: {args.n_permutations}")
    lines.append(f"NBS threshold: {args.nbs_threshold}")
    lines.append(f"Random seed: {args.seed}")

    output_path.write_text("\n".join(lines))
    logger.info(f"Saved analysis description: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Covariance Network Analysis for ROI-level DTI metrics"
    )
    parser.add_argument(
        "--roi-dir", type=Path, required=True,
        help="Directory containing roi_*_wide.csv files",
    )
    parser.add_argument(
        "--exclusion-csv", type=Path, default=None,
        help="CSV of sessions to exclude (must have subject, session columns)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for CovNet results",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["FA", "MD", "AD", "RD"],
        help="DTI metrics to analyze (default: FA MD AD RD)",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=5000,
        help="Number of permutations for NBS and graph metric tests",
    )
    parser.add_argument(
        "--nbs-threshold", type=float, default=3.0,
        help="Z-statistic threshold for NBS suprathreshold edges (default: 3.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-nbs", action="store_true",
        help="Skip NBS analysis (matrices and territory only)",
    )
    parser.add_argument(
        "--skip-graph", action="store_true",
        help="Skip graph metrics analysis",
    )
    parser.add_argument(
        "--skip-whole-network", action="store_true",
        help="Skip whole-network similarity tests (Mantel, Frobenius, spectral)",
    )
    parser.add_argument(
        "--skip-cross-timepoint", action="store_true",
        help="Skip cross-timepoint (cross-PND) comparisons within each dose level",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel workers for NBS/whole-network comparisons (default: 1 = sequential)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.roi_dir.exists():
        logger.error(f"ROI directory not found: {args.roi_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save analysis configuration
    config = {
        "roi_dir": str(args.roi_dir),
        "exclusion_csv": str(args.exclusion_csv) if args.exclusion_csv else None,
        "output_dir": str(args.output_dir),
        "metrics": args.metrics,
        "n_permutations": args.n_permutations,
        "nbs_threshold": args.nbs_threshold,
        "seed": args.seed,
        "skip_nbs": args.skip_nbs,
        "skip_graph": args.skip_graph,
        "skip_whole_network": args.skip_whole_network,
        "skip_cross_timepoint": args.skip_cross_timepoint,
        "n_workers": args.n_workers,
        "timestamp": datetime.now().isoformat(),
    }
    with open(args.output_dir / "analysis_config.json", "w") as f:
        json.dump(config, f, indent=2)

    write_design_description(args, args.output_dir / "design_description.txt")

    # Run for each metric
    all_summaries = {}
    for metric in args.metrics:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing metric: {metric}")
        logger.info(f"{'=' * 60}")

        try:
            analysis = CovNetAnalysis.prepare(
                args.roi_dir, args.exclusion_csv, args.output_dir, metric
            )
            analysis.save()
            summary = run_single_metric(analysis, args)
            all_summaries[metric] = summary
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue

    # Save overall summary
    summary_path = args.output_dir / "covnet_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Register with unified reporting system
    try:
        from neurofaune.analysis.reporting import register as report_register

        analysis_root = args.output_dir.parent
        n_subjects = max(
            (s.get("n_subjects", 0) for s in all_summaries.values() if isinstance(s, dict)),
            default=0,
        )
        n_rois = max(
            (s.get("n_bilateral_rois", 0) for s in all_summaries.values() if isinstance(s, dict)),
            default=0,
        )

        # Collect figure paths
        figures = []
        fig_dir = args.output_dir / "figures"
        if fig_dir.is_dir():
            for fig in sorted(fig_dir.rglob("*.png"))[:20]:
                try:
                    figures.append(str(fig.relative_to(analysis_root)))
                except ValueError:
                    pass

        report_register(
            analysis_root=analysis_root,
            entry_id="covnet",
            analysis_type="covnet",
            display_name=f"CovNet Analysis ({', '.join(args.metrics)})",
            output_dir=str(args.output_dir.relative_to(analysis_root)),
            summary_stats={
                "metrics": args.metrics,
                "n_subjects": n_subjects,
                "n_bilateral_rois": n_rois,
            },
            figures=figures,
            source_summary_json=str(summary_path.relative_to(analysis_root)),
            config=config,
        )
    except Exception as exc:
        logger.warning("Failed to register with reporting system: %s", exc)

    logger.info(f"\nCovNet analysis complete. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
