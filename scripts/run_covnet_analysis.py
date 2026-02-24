#!/usr/bin/env python3
"""
Covariance Network (CovNet) Analysis for ROI-level DTI metrics.

Builds Spearman correlation matrices per experimental group, compares them
using the Network-Based Statistic (NBS) and graph-theoretic metrics. Supports
bilateral ROI averaging and territory-level analysis.

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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.analysis.covnet.matrices import (
    bilateral_average,
    compute_spearman_matrices,
    cross_dose_timepoint_comparisons,
    cross_timepoint_comparisons,
    define_groups,
    fisher_z_transform,
    load_and_prepare_data,
)
from neurofaune.analysis.covnet.nbs import (
    fisher_z_edge_test,
    run_all_comparisons,
)
from neurofaune.analysis.covnet.graph_metrics import (
    compare_metrics,
    compute_metrics,
)
from neurofaune.analysis.covnet.whole_network import (
    run_all_comparisons as run_whole_network_comparisons,
)
from neurofaune.analysis.covnet.visualization import (
    plot_all_group_heatmaps,
    plot_correlation_heatmap,
    plot_difference_matrix,
    plot_graph_metrics_comparison,
    plot_nbs_network,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def save_correlation_matrices(
    matrices: dict[str, dict], output_dir: Path, metric: str
) -> None:
    """Save per-group correlation matrices as CSV files."""
    mat_dir = output_dir / "matrices" / metric
    mat_dir.mkdir(parents=True, exist_ok=True)

    for label, data in matrices.items():
        corr_df = pd.DataFrame(
            data["corr"], index=data["rois"], columns=data["rois"]
        )
        corr_df.to_csv(mat_dir / f"{label}_corr.csv")


def save_nbs_results(
    nbs_results: dict[str, dict], output_dir: Path, metric: str
) -> None:
    """Save NBS results (test statistics, components) to disk."""
    for comp_label, result in nbs_results.items():
        nbs_dir = output_dir / "nbs" / metric / comp_label
        nbs_dir.mkdir(parents=True, exist_ok=True)

        # Test statistic matrix
        rois = result["roi_cols"]
        stat_df = pd.DataFrame(result["test_stat"], index=rois, columns=rois)
        stat_df.to_csv(nbs_dir / "test_statistics.csv")

        # Components
        components_json = {
            "group_a": result["group_a"],
            "group_b": result["group_b"],
            "n_a": result["n_a"],
            "n_b": result["n_b"],
            "components": [],
        }
        for comp in result["significant_components"]:
            comp_out = {
                "nodes": comp["nodes"],
                "node_names": [rois[n] for n in comp["nodes"]],
                "edges": comp["edges"],
                "edge_names": [(rois[u], rois[v]) for u, v in comp["edges"]],
                "size": comp["size"],
                "pvalue": comp["pvalue"],
            }
            components_json["components"].append(comp_out)

        with open(nbs_dir / "components.json", "w") as f:
            json.dump(components_json, f, indent=2)

        # Null distribution
        np.savetxt(nbs_dir / "null_distribution.txt", result["null_distribution"])


def save_territory_results(
    territory_results: dict[str, dict],
    output_dir: Path,
    metric: str,
    territory_cols: list[str],
) -> None:
    """Save territory-level Fisher z-test results."""
    terr_dir = output_dir / "territory" / metric
    terr_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for comp_label, result in territory_results.items():
        z_stats = result["z_stats"]
        p_values = result["p_values"]
        p_fdr = result["p_fdr"]
        n = len(territory_cols)

        for i in range(n):
            for j in range(i + 1, n):
                rows.append({
                    "comparison": comp_label,
                    "roi_a": territory_cols[i],
                    "roi_b": territory_cols[j],
                    "z_stat": z_stats[i, j],
                    "p_value": p_values[i, j],
                    "p_fdr": p_fdr[i, j],
                    "significant": p_fdr[i, j] < 0.05,
                })

    df = pd.DataFrame(rows)
    df.to_csv(terr_dir / "fisher_z_results.csv", index=False)
    n_sig = df["significant"].sum()
    logger.info(f"Territory {metric}: {n_sig} FDR-significant edges")


def fdr_correct_matrix(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction to a symmetric p-value matrix."""
    n = p_values.shape[0]
    # Extract upper triangle p-values
    idx = np.triu_indices(n, k=1)
    pvals = p_values[idx]
    n_tests = len(pvals)

    # BH correction
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    ranks = np.arange(1, n_tests + 1)
    adjusted = np.minimum(1.0, sorted_p * n_tests / ranks)

    # Enforce monotonicity (from largest to smallest)
    for i in range(n_tests - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Map back to original order
    p_fdr_flat = np.empty(n_tests)
    p_fdr_flat[sorted_idx] = adjusted

    # Reconstruct matrix
    p_fdr = np.ones((n, n))
    p_fdr[idx] = p_fdr_flat
    p_fdr.T[idx] = p_fdr_flat  # Symmetric
    np.fill_diagonal(p_fdr, 1.0)
    return p_fdr


def run_single_metric(
    metric: str,
    roi_dir: Path,
    exclusion_csv: Path,
    output_dir: Path,
    n_perm: int,
    nbs_threshold: float,
    seed: int,
    skip_nbs: bool,
    skip_graph: bool,
    skip_whole_network: bool = False,
    skip_cross_timepoint: bool = False,
) -> dict:
    """Run full CovNet pipeline for a single metric."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing metric: {metric}")
    logger.info(f"{'=' * 60}")

    wide_csv = roi_dir / f"roi_{metric}_wide.csv"
    if not wide_csv.exists():
        logger.warning(f"Wide CSV not found: {wide_csv}, skipping {metric}")
        return {}

    # Phase 1: Load and prepare data
    logger.info("\n[Phase 1] Loading and preparing data...")
    df, roi_cols = load_and_prepare_data(wide_csv, exclusion_csv)

    # Identify region vs territory columns
    region_cols = [c for c in roi_cols if not c.startswith("territory_")]
    territory_cols = [c for c in roi_cols if c.startswith("territory_")]

    # Save ROI selection info
    roi_info = {
        "metric": metric,
        "n_subjects": len(df),
        "n_region_rois": len(region_cols),
        "n_territory_rois": len(territory_cols),
        "region_rois": region_cols,
        "territory_rois": territory_cols,
    }

    # Phase 2: Bilateral averaging
    logger.info("\n[Phase 2] Bilateral averaging...")
    df_bilateral, bilateral_cols = bilateral_average(df, roi_cols)
    bilateral_region_cols = [c for c in bilateral_cols if not c.startswith("territory_")]
    roi_info["n_bilateral_rois"] = len(bilateral_region_cols)
    roi_info["bilateral_rois"] = bilateral_region_cols

    # Phase 3: Define groups and compute correlation matrices
    logger.info("\n[Phase 3] Computing correlation matrices...")

    # Primary: PND x dose (bilateral ROIs)
    logger.info("  PND x dose grouping (bilateral):")
    groups_pnd_dose = define_groups(df_bilateral, grouping="pnd_dose")
    matrices_pnd_dose = compute_spearman_matrices(groups_pnd_dose, bilateral_region_cols)

    # Full: sex x PND x dose (bilateral, descriptive)
    logger.info("  Full grouping (bilateral, descriptive):")
    groups_full = define_groups(df_bilateral, grouping="full")
    matrices_full = compute_spearman_matrices(groups_full, bilateral_region_cols)

    # Territory level
    logger.info("  PND x dose grouping (territory):")
    groups_territory = define_groups(df, grouping="pnd_dose")
    matrices_territory = compute_spearman_matrices(groups_territory, territory_cols)

    # Save matrices
    save_correlation_matrices(matrices_pnd_dose, output_dir, metric)
    save_correlation_matrices(matrices_full, output_dir, f"{metric}_full")
    save_correlation_matrices(matrices_territory, output_dir, f"{metric}_territory")

    # Phase 4: Visualizations — correlation heatmaps
    logger.info("\n[Phase 4] Generating heatmaps...")
    fig_dir = output_dir / "figures" / metric

    plot_all_group_heatmaps(
        matrices_pnd_dose, fig_dir / "pnd_dose_heatmaps.png",
        title_prefix=f"{metric} ",
    )
    plot_all_group_heatmaps(
        matrices_full, fig_dir / "full_heatmaps.png",
        title_prefix=f"{metric} ",
    )
    plot_all_group_heatmaps(
        matrices_territory, fig_dir / "territory_heatmaps.png",
        title_prefix=f"{metric} Territory ",
    )

    # Individual heatmaps for pnd_dose
    for label, data in matrices_pnd_dose.items():
        plot_correlation_heatmap(
            data["corr"], data["rois"],
            title=f"{metric} — {label} (n={data['n']})",
            out_path=output_dir / "matrices" / metric / f"{label}_corr_heatmap.png",
        )

    summary = {
        "metric": metric,
        "n_subjects": len(df),
        "n_bilateral_rois": len(bilateral_region_cols),
        "n_territory_rois": len(territory_cols),
        "n_groups_pnd_dose": len(groups_pnd_dose),
        "group_sizes": {k: len(v) for k, v in groups_pnd_dose.items()},
    }

    # Phase 5: NBS (bilateral ROIs)
    if not skip_nbs:
        logger.info(f"\n[Phase 5] Network-Based Statistic ({n_perm} permutations)...")
        group_arrays = {
            label: subset[bilateral_region_cols].values
            for label, subset in groups_pnd_dose.items()
        }

        nbs_results = run_all_comparisons(
            group_data=group_arrays,
            group_sizes={k: len(v) for k, v in groups_pnd_dose.items()},
            roi_cols=bilateral_region_cols,
            n_perm=n_perm,
            threshold=nbs_threshold,
            seed=seed,
        )

        # Cross-timepoint NBS comparisons
        if not skip_cross_timepoint:
            cross_comps = cross_timepoint_comparisons(list(group_arrays.keys()))
            if cross_comps:
                logger.info(f"  Running {len(cross_comps)} cross-timepoint NBS comparisons...")
                cross_nbs = run_all_comparisons(
                    group_data=group_arrays,
                    group_sizes={k: len(v) for k, v in groups_pnd_dose.items()},
                    roi_cols=bilateral_region_cols,
                    comparisons=cross_comps,
                    n_perm=n_perm,
                    threshold=nbs_threshold,
                    seed=seed,
                )
                nbs_results.update(cross_nbs)

            # Cross-dose-cross-timepoint NBS comparisons
            cross_dose_comps = cross_dose_timepoint_comparisons(list(group_arrays.keys()))
            if cross_dose_comps:
                logger.info(f"  Running {len(cross_dose_comps)} cross-dose-timepoint NBS comparisons...")
                cross_dose_nbs = run_all_comparisons(
                    group_data=group_arrays,
                    group_sizes={k: len(v) for k, v in groups_pnd_dose.items()},
                    roi_cols=bilateral_region_cols,
                    comparisons=cross_dose_comps,
                    n_perm=n_perm,
                    threshold=nbs_threshold,
                    seed=seed,
                )
                nbs_results.update(cross_dose_nbs)

        save_nbs_results(nbs_results, output_dir, metric)

        # NBS visualizations
        for comp_label, result in nbs_results.items():
            sig_comps = [c for c in result["significant_components"] if c["pvalue"] < 0.05]
            sig_edges = []
            for comp in sig_comps:
                sig_edges.extend(comp["edges"])

            plot_nbs_network(
                result["significant_components"],
                bilateral_region_cols,
                title=f"NBS {metric}: {comp_label}",
                out_path=output_dir / "nbs" / metric / comp_label / "nbs_network.png",
            )

            # Difference matrix with NBS highlights
            ga, gb = result["group_a"], result["group_b"]
            if ga in matrices_pnd_dose and gb in matrices_pnd_dose:
                plot_difference_matrix(
                    matrices_pnd_dose[ga]["corr"],
                    matrices_pnd_dose[gb]["corr"],
                    bilateral_region_cols,
                    sig_edges=sig_edges,
                    title=f"{metric} Δr: {ga} − {gb}",
                    out_path=output_dir / "nbs" / metric / comp_label / "difference_matrix.png",
                )

        n_sig_comparisons = sum(
            1 for r in nbs_results.values()
            if any(c["pvalue"] < 0.05 for c in r["significant_components"])
        )
        summary["nbs_significant_comparisons"] = n_sig_comparisons
    else:
        logger.info("\n[Phase 5] Skipping NBS (--skip-nbs)")

    # Phase 6: Territory-level Fisher z-tests with FDR
    logger.info("\n[Phase 6] Territory-level edge comparisons (Fisher z + FDR)...")
    territory_results = {}
    pnds = ["p30", "p60", "p90"]

    # Detect dose naming convention from actual group labels
    territory_labels = list(matrices_territory.keys())
    if any(k.endswith("_C") for k in territory_labels):
        control_suffix, dose_suffixes = "C", ["L", "M", "H"]
    else:
        control_suffix, dose_suffixes = "control", ["low", "medium", "high"]

    for pnd in pnds:
        control_key = f"{pnd}_{control_suffix}"
        if control_key not in matrices_territory:
            continue
        corr_ctrl = matrices_territory[control_key]["corr"]
        n_ctrl = matrices_territory[control_key]["n"]

        for dose in dose_suffixes:
            treatment_key = f"{pnd}_{dose}"
            if treatment_key not in matrices_territory:
                continue
            corr_treat = matrices_territory[treatment_key]["corr"]
            n_treat = matrices_territory[treatment_key]["n"]

            z_stats, p_values = fisher_z_edge_test(
                corr_treat, n_treat, corr_ctrl, n_ctrl
            )
            p_fdr = fdr_correct_matrix(p_values)

            comp_label = f"{treatment_key}_vs_{control_key}"
            territory_results[comp_label] = {
                "z_stats": z_stats,
                "p_values": p_values,
                "p_fdr": p_fdr,
            }

    # Cross-timepoint territory comparisons
    if not skip_cross_timepoint:
        cross_comps = cross_timepoint_comparisons(territory_labels)
        for label_a, label_b in cross_comps:
            if label_a not in matrices_territory or label_b not in matrices_territory:
                continue
            corr_a = matrices_territory[label_a]["corr"]
            n_a = matrices_territory[label_a]["n"]
            corr_b = matrices_territory[label_b]["corr"]
            n_b = matrices_territory[label_b]["n"]

            z_stats, p_values = fisher_z_edge_test(corr_a, n_a, corr_b, n_b)
            p_fdr = fdr_correct_matrix(p_values)

            comp_label = f"{label_a}_vs_{label_b}"
            territory_results[comp_label] = {
                "z_stats": z_stats,
                "p_values": p_values,
                "p_fdr": p_fdr,
            }

        # Cross-dose-cross-timepoint territory comparisons
        cross_dose_comps = cross_dose_timepoint_comparisons(territory_labels)
        for treatment_key, control_key in cross_dose_comps:
            if treatment_key not in matrices_territory or control_key not in matrices_territory:
                continue
            corr_treat = matrices_territory[treatment_key]["corr"]
            n_treat = matrices_territory[treatment_key]["n"]
            corr_ctrl = matrices_territory[control_key]["corr"]
            n_ctrl = matrices_territory[control_key]["n"]

            z_stats, p_values = fisher_z_edge_test(corr_treat, n_treat, corr_ctrl, n_ctrl)
            p_fdr = fdr_correct_matrix(p_values)

            comp_label = f"{treatment_key}_vs_{control_key}"
            territory_results[comp_label] = {
                "z_stats": z_stats,
                "p_values": p_values,
                "p_fdr": p_fdr,
            }

    if territory_results:
        save_territory_results(territory_results, output_dir, metric, territory_cols)

    # Phase 7: Graph metrics
    if not skip_graph:
        logger.info("\n[Phase 7] Graph metrics and permutation comparison...")
        graph_dir = output_dir / "graph_metrics" / metric
        graph_dir.mkdir(parents=True, exist_ok=True)

        # Compute per-group metrics at multiple densities
        densities = [0.10, 0.15, 0.20, 0.25]
        metrics_rows = []
        for label, data in matrices_pnd_dose.items():
            for d in densities:
                m = compute_metrics(data["corr"], density=d)
                m["group"] = label
                m["density"] = d
                metrics_rows.append(m)

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(graph_dir / "global_metrics.csv", index=False)
        logger.info(f"Saved global metrics: {graph_dir / 'global_metrics.csv'}")

        # Permutation comparison
        group_arrays = {
            label: subset[bilateral_region_cols].values
            for label, subset in groups_pnd_dose.items()
        }
        comparison_df = compare_metrics(
            group_arrays, bilateral_region_cols,
            densities=densities, n_perm=n_perm, seed=seed,
        )
        comparison_df.to_csv(graph_dir / "comparison_pvalues.csv", index=False)

        plot_graph_metrics_comparison(
            comparison_df,
            out_path=fig_dir / "graph_metrics_bars.png",
        )
        summary["graph_metrics_significant"] = int(
            (comparison_df["p_value"] < 0.05).sum()
        )
    else:
        logger.info("\n[Phase 7] Skipping graph metrics (--skip-graph)")

    # Phase 8: Whole-network similarity tests
    if not skip_whole_network:
        logger.info(f"\n[Phase 8] Whole-network similarity tests ({n_perm} permutations)...")
        wn_dir = output_dir / "whole_network" / metric
        wn_dir.mkdir(parents=True, exist_ok=True)

        group_arrays = {
            label: subset[bilateral_region_cols].values
            for label, subset in groups_pnd_dose.items()
        }

        wn_df, wn_nulls = run_whole_network_comparisons(
            group_data=group_arrays,
            n_perm=n_perm,
            seed=seed,
        )

        # Cross-timepoint whole-network comparisons
        if not skip_cross_timepoint:
            cross_comps = cross_timepoint_comparisons(list(group_arrays.keys()))
            if cross_comps:
                logger.info(f"  Running {len(cross_comps)} cross-timepoint whole-network comparisons...")
                cross_wn_df, cross_wn_nulls = run_whole_network_comparisons(
                    group_data=group_arrays,
                    comparisons=cross_comps,
                    n_perm=n_perm,
                    seed=seed,
                )
                wn_df = pd.concat([wn_df, cross_wn_df], ignore_index=True)
                wn_nulls.update(cross_wn_nulls)

            # Cross-dose-cross-timepoint whole-network comparisons
            cross_dose_comps = cross_dose_timepoint_comparisons(list(group_arrays.keys()))
            if cross_dose_comps:
                logger.info(f"  Running {len(cross_dose_comps)} cross-dose-timepoint whole-network comparisons...")
                cross_dose_wn_df, cross_dose_wn_nulls = run_whole_network_comparisons(
                    group_data=group_arrays,
                    comparisons=cross_dose_comps,
                    n_perm=n_perm,
                    seed=seed,
                )
                wn_df = pd.concat([wn_df, cross_dose_wn_df], ignore_index=True)
                wn_nulls.update(cross_dose_wn_nulls)

        wn_df.to_csv(wn_dir / "whole_network_results.csv", index=False)

        # Save null distributions
        null_arrays = {}
        for comp_label, dists in wn_nulls.items():
            for stat_name, arr in dists.items():
                null_arrays[f"{comp_label}__{stat_name}"] = arr
        np.savez(wn_dir / "null_distributions.npz", **null_arrays)

        n_sig = int((wn_df[["mantel_p", "frobenius_p", "spectral_p"]] < 0.05).any(axis=1).sum())
        logger.info(
            f"Whole-network: {n_sig}/{len(wn_df)} comparisons with at least "
            f"one significant statistic (p < 0.05)"
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
        "timestamp": datetime.now().isoformat(),
    }
    with open(args.output_dir / "analysis_config.json", "w") as f:
        json.dump(config, f, indent=2)

    write_design_description(args, args.output_dir / "design_description.txt")

    # Run for each metric
    all_summaries = {}
    for metric in args.metrics:
        summary = run_single_metric(
            metric=metric,
            roi_dir=args.roi_dir,
            exclusion_csv=args.exclusion_csv,
            output_dir=args.output_dir,
            n_perm=args.n_permutations,
            nbs_threshold=args.nbs_threshold,
            seed=args.seed,
            skip_nbs=args.skip_nbs,
            skip_graph=args.skip_graph,
            skip_whole_network=args.skip_whole_network,
            skip_cross_timepoint=args.skip_cross_timepoint,
        )
        all_summaries[metric] = summary

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
