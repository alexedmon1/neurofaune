"""Covariance Network (CovNet) analysis pipeline.

Provides the ``CovNetAnalysis`` class: prepare data once, then run any
combination of NBS, territory, graph-metric, or whole-network tests.
Serializable to / from disk so each test can run independently.

Typical usage::

    # Prepare and save
    analysis = CovNetAnalysis.prepare(roi_dir, exclusion_csv, output_dir, "FA")
    analysis.save()

    # Load and run individual tests
    analysis = CovNetAnalysis.load(output_dir, "FA")
    analysis.run_nbs(comparisons=analysis.resolve_comparisons(["dose"]))
    analysis.run_territory()
    analysis.run_graph_metrics()
    analysis.run_whole_network()
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neurofaune.analysis.covnet.graph_metrics import compare_metrics, compute_metrics
from neurofaune.analysis.covnet.matrices import (
    bilateral_average,
    compute_spearman_matrices,
    cross_dose_timepoint_comparisons,
    cross_timepoint_comparisons,
    default_dose_comparisons,
    define_groups,
    load_and_prepare_data,
)
from neurofaune.analysis.covnet.nbs import fisher_z_edge_test
from neurofaune.analysis.covnet.nbs import run_all_comparisons as _run_nbs_comparisons
from neurofaune.analysis.covnet.visualization import (
    plot_all_group_heatmaps,
    plot_correlation_heatmap,
    plot_difference_matrix,
    plot_graph_metrics_comparison,
    plot_nbs_network,
)
from neurofaune.analysis.covnet.whole_network import (
    run_all_comparisons as _run_whole_network_comparisons,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def fdr_correct_matrix(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction to a symmetric p-value matrix."""
    n = p_values.shape[0]
    idx = np.triu_indices(n, k=1)
    pvals = p_values[idx]
    n_tests = len(pvals)

    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    ranks = np.arange(1, n_tests + 1)
    adjusted = np.minimum(1.0, sorted_p * n_tests / ranks)

    # Enforce monotonicity (from largest to smallest)
    for i in range(n_tests - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    p_fdr_flat = np.empty(n_tests)
    p_fdr_flat[sorted_idx] = adjusted

    p_fdr = np.ones((n, n))
    p_fdr[idx] = p_fdr_flat
    p_fdr.T[idx] = p_fdr_flat
    np.fill_diagonal(p_fdr, 1.0)
    return p_fdr


def _save_matrices(
    matrices: dict[str, dict], base_dir: Path, sub_name: str
) -> None:
    """Save per-group correlation matrices as CSV files."""
    mat_dir = base_dir / "matrices" / sub_name
    mat_dir.mkdir(parents=True, exist_ok=True)
    for label, data in matrices.items():
        corr_df = pd.DataFrame(
            data["corr"], index=data["rois"], columns=data["rois"]
        )
        corr_df.to_csv(mat_dir / f"{label}_corr.csv")


def _load_matrices(mat_dir: Path) -> dict[str, dict]:
    """Load correlation matrices from CSV files."""
    matrices = {}
    if not mat_dir.exists():
        return matrices
    for csv_path in sorted(mat_dir.glob("*_corr.csv")):
        label = csv_path.stem.replace("_corr", "")
        df = pd.read_csv(csv_path, index_col=0)
        matrices[label] = {
            "corr": df.values,
            "rois": list(df.columns),
            "n": 0,  # Patched from metadata in load()
        }
    return matrices


def _save_nbs_results(
    nbs_results: dict[str, dict], nbs_base_dir: Path
) -> None:
    """Save NBS test statistics, components, and null distributions."""
    for comp_label, result in nbs_results.items():
        nbs_dir = nbs_base_dir / comp_label
        nbs_dir.mkdir(parents=True, exist_ok=True)

        rois = result["roi_cols"]
        stat_df = pd.DataFrame(result["test_stat"], index=rois, columns=rois)
        stat_df.to_csv(nbs_dir / "test_statistics.csv")

        components_json = {
            "group_a": result["group_a"],
            "group_b": result["group_b"],
            "n_a": result["n_a"],
            "n_b": result["n_b"],
            "components": [],
        }
        for comp in result["significant_components"]:
            components_json["components"].append({
                "nodes": comp["nodes"],
                "node_names": [rois[n] for n in comp["nodes"]],
                "edges": comp["edges"],
                "edge_names": [(rois[u], rois[v]) for u, v in comp["edges"]],
                "size": comp["size"],
                "pvalue": comp["pvalue"],
            })

        with open(nbs_dir / "components.json", "w") as f:
            json.dump(components_json, f, indent=2)

        np.savetxt(nbs_dir / "null_distribution.txt", result["null_distribution"])


def _save_territory_results(
    territory_results: dict[str, dict],
    terr_dir: Path,
    territory_cols: list[str],
) -> pd.DataFrame:
    """Save territory-level Fisher z results. Returns the results DataFrame."""
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
    logger.info(f"Territory: {n_sig} FDR-significant edges")
    return df


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CovNetAnalysis:
    """CovNet analysis for a single DTI metric.

    Holds prepared data (group arrays, correlation matrices) and runs
    statistical tests. Serializable to / from disk via save() / load().
    """

    def __init__(self, output_dir: Path, metric: str):
        self.output_dir = Path(output_dir)
        self.metric = metric

        # Populated by prepare() or load()
        self.n_subjects: int = 0
        self.group_arrays: dict[str, np.ndarray] = {}
        self.territory_arrays: dict[str, np.ndarray] = {}
        self.bilateral_region_cols: list[str] = []
        self.territory_cols: list[str] = []
        self.group_labels: list[str] = []
        self.group_sizes: dict[str, int] = {}
        self.matrices_pnd_dose: dict[str, dict] = {}
        self.matrices_full: dict[str, dict] = {}
        self.matrices_territory: dict[str, dict] = {}

    @property
    def metric_dir(self) -> Path:
        """Per-metric output directory: ``{output_dir}/{metric}/``."""
        return self.output_dir / self.metric

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def prepare(
        cls,
        roi_dir: Path,
        exclusion_csv: Path | None,
        output_dir: Path,
        metric: str,
    ) -> "CovNetAnalysis":
        """Load ROI data, bilateral average, compute correlation matrices.

        Returns a populated instance ready for save() and run_*() methods.
        """
        roi_dir = Path(roi_dir)
        output_dir = Path(output_dir)

        wide_csv = roi_dir / f"roi_{metric}_wide.csv"
        if not wide_csv.exists():
            raise FileNotFoundError(f"Wide CSV not found: {wide_csv}")

        inst = cls(output_dir, metric)

        # Phase 1: Load and prepare data
        logger.info(f"\n[Phase 1] Loading and preparing data for {metric}...")
        df, roi_cols = load_and_prepare_data(wide_csv, exclusion_csv)
        inst.n_subjects = len(df)
        inst.territory_cols = [c for c in roi_cols if c.startswith("territory_")]

        # Phase 2: Bilateral averaging
        logger.info("[Phase 2] Bilateral averaging...")
        df_bilateral, bilateral_cols = bilateral_average(df, roi_cols)
        inst.bilateral_region_cols = [
            c for c in bilateral_cols if not c.startswith("territory_")
        ]

        # Phase 3: Compute correlation matrices
        logger.info("[Phase 3] Computing correlation matrices...")

        logger.info("  PND x dose grouping (bilateral):")
        groups_pnd_dose = define_groups(df_bilateral, grouping="pnd_dose")
        inst.matrices_pnd_dose = compute_spearman_matrices(
            groups_pnd_dose, inst.bilateral_region_cols
        )

        logger.info("  Full grouping (bilateral, descriptive):")
        groups_full = define_groups(df_bilateral, grouping="full")
        inst.matrices_full = compute_spearman_matrices(
            groups_full, inst.bilateral_region_cols
        )

        logger.info("  PND x dose grouping (territory):")
        groups_territory = define_groups(df, grouping="pnd_dose")
        inst.matrices_territory = compute_spearman_matrices(
            groups_territory, inst.territory_cols
        )

        # Extract group arrays and metadata
        inst.group_labels = sorted(groups_pnd_dose.keys())
        inst.group_sizes = {k: len(v) for k, v in groups_pnd_dose.items()}
        inst.group_arrays = {
            label: subset[inst.bilateral_region_cols].values
            for label, subset in groups_pnd_dose.items()
        }
        inst.territory_arrays = {
            label: subset[inst.territory_cols].values
            for label, subset in groups_territory.items()
        }

        logger.info(f"Preparation complete for {metric}")
        return inst

    def save(self, save_heatmaps: bool = True) -> None:
        """Serialize prepared data to disk.

        Saves metadata, group arrays, territory arrays, and correlation
        matrices. Optionally generates heatmap figures.
        """
        prep_dir = self.metric_dir / "prep"
        prep_dir.mkdir(parents=True, exist_ok=True)

        # Metadata
        metadata = {
            "metric": self.metric,
            "n_subjects": self.n_subjects,
            "bilateral_region_cols": self.bilateral_region_cols,
            "territory_cols": self.territory_cols,
            "group_labels": self.group_labels,
            "group_sizes": self.group_sizes,
            "n_bilateral_rois": len(self.bilateral_region_cols),
            "n_territory_rois": len(self.territory_cols),
        }
        with open(prep_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Group arrays
        arr_dir = prep_dir / "group_arrays"
        arr_dir.mkdir(parents=True, exist_ok=True)
        for label, arr in self.group_arrays.items():
            np.save(arr_dir / f"{label}.npy", arr)

        # Territory arrays
        terr_dir = prep_dir / "territory_arrays"
        terr_dir.mkdir(parents=True, exist_ok=True)
        for label, arr in self.territory_arrays.items():
            np.save(terr_dir / f"{label}.npy", arr)

        # Correlation matrices as CSV
        _save_matrices(self.matrices_pnd_dose, self.metric_dir, self.metric)
        _save_matrices(self.matrices_full, self.metric_dir, f"{self.metric}_full")
        _save_matrices(
            self.matrices_territory, self.metric_dir, f"{self.metric}_territory"
        )

        # Heatmaps
        if save_heatmaps:
            self._save_heatmaps()

        logger.info(f"Saved prepared data: {prep_dir}")

    def _save_heatmaps(self) -> None:
        """Generate and save correlation heatmap figures."""
        fig_dir = self.metric_dir / "figures"

        plot_all_group_heatmaps(
            self.matrices_pnd_dose,
            fig_dir / "pnd_dose_heatmaps.png",
            title_prefix=f"{self.metric} ",
        )
        plot_all_group_heatmaps(
            self.matrices_full,
            fig_dir / "full_heatmaps.png",
            title_prefix=f"{self.metric} ",
        )
        plot_all_group_heatmaps(
            self.matrices_territory,
            fig_dir / "territory_heatmaps.png",
            title_prefix=f"{self.metric} Territory ",
        )

        for label, data in self.matrices_pnd_dose.items():
            plot_correlation_heatmap(
                data["corr"],
                data["rois"],
                title=f"{self.metric} \u2014 {label} (n={data['n']})",
                out_path=self.metric_dir
                / "matrices"
                / self.metric
                / f"{label}_corr_heatmap.png",
            )

    @classmethod
    def load(cls, output_dir: Path, metric: str) -> "CovNetAnalysis":
        """Deserialize from ``{output_dir}/{metric}/prep/``."""
        output_dir = Path(output_dir)
        inst = cls(output_dir, metric)

        meta_path = inst.metric_dir / "prep" / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Prep metadata not found: {meta_path}\n"
                f"Run CovNetAnalysis.prepare() or covnet_prepare.py first."
            )

        with open(meta_path) as f:
            metadata = json.load(f)

        inst.n_subjects = metadata.get("n_subjects", 0)
        inst.bilateral_region_cols = metadata["bilateral_region_cols"]
        inst.territory_cols = metadata["territory_cols"]
        inst.group_labels = metadata["group_labels"]
        inst.group_sizes = metadata["group_sizes"]

        # Group arrays
        arr_dir = inst.metric_dir / "prep" / "group_arrays"
        for label in inst.group_labels:
            arr_path = arr_dir / f"{label}.npy"
            if arr_path.exists():
                inst.group_arrays[label] = np.load(arr_path)
            else:
                logger.warning(f"Missing group array: {arr_path}")

        # Territory arrays
        terr_dir = inst.metric_dir / "prep" / "territory_arrays"
        if terr_dir.exists():
            for label in inst.group_labels:
                arr_path = terr_dir / f"{label}.npy"
                if arr_path.exists():
                    inst.territory_arrays[label] = np.load(arr_path)

        # Correlation matrices from CSV
        inst.matrices_pnd_dose = _load_matrices(
            inst.metric_dir / "matrices" / metric
        )
        inst.matrices_full = _load_matrices(
            inst.metric_dir / "matrices" / f"{metric}_full"
        )
        inst.matrices_territory = _load_matrices(
            inst.metric_dir / "matrices" / f"{metric}_territory"
        )

        # Patch group sizes into loaded matrices
        for matrices in (inst.matrices_pnd_dose, inst.matrices_territory):
            for label, mat_data in matrices.items():
                if label in inst.group_sizes:
                    mat_data["n"] = inst.group_sizes[label]

        return inst

    # ------------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------------

    def resolve_comparisons(
        self, comparison_types: list[str] | None = None
    ) -> list[tuple[str, str]]:
        """Convert named comparison sets to explicit group pairs.

        Parameters
        ----------
        comparison_types : list[str], optional
            Any of ``"dose"``, ``"cross-timepoint"``,
            ``"cross-dose-timepoint"``. Default: all three.
        """
        if comparison_types is None:
            comparison_types = ["dose", "cross-timepoint", "cross-dose-timepoint"]

        pairs: list[tuple[str, str]] = []
        for name in comparison_types:
            if name == "dose":
                pairs.extend(default_dose_comparisons(self.group_labels))
            elif name == "cross-timepoint":
                pairs.extend(cross_timepoint_comparisons(self.group_labels))
            elif name == "cross-dose-timepoint":
                pairs.extend(
                    cross_dose_timepoint_comparisons(self.group_labels)
                )
            else:
                raise ValueError(
                    f"Unknown comparison set: {name!r}. "
                    f"Use 'dose', 'cross-timepoint', or 'cross-dose-timepoint'."
                )
        return pairs

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def run_nbs(
        self,
        comparisons: list[tuple[str, str]] | None = None,
        n_perm: int = 5000,
        threshold: float = 3.0,
        seed: int = 42,
        n_workers: int = 1,
    ) -> dict:
        """Run NBS. Results saved to ``{metric_dir}/nbs/``.

        Parameters
        ----------
        comparisons : list of (str, str), optional
            Group pairs to compare. If None, uses all default comparisons.
        n_perm : int
            Number of permutations.
        threshold : float
            Z-statistic threshold for suprathreshold edges.
        seed : int
            Random seed.
        n_workers : int
            Parallel workers (1 = sequential).

        Returns
        -------
        dict of NBS results keyed by comparison label.
        """
        if comparisons is None:
            comparisons = self.resolve_comparisons()

        logger.info(
            f"NBS: {self.metric} ({n_perm} permutations, threshold={threshold}, "
            f"{len(comparisons)} comparisons)"
        )

        nbs_results = _run_nbs_comparisons(
            group_data=self.group_arrays,
            group_sizes=self.group_sizes,
            roi_cols=self.bilateral_region_cols,
            comparisons=comparisons,
            n_perm=n_perm,
            threshold=threshold,
            seed=seed,
            n_workers=n_workers,
        )

        # Save results
        nbs_dir = self.metric_dir / "nbs"
        _save_nbs_results(nbs_results, nbs_dir)

        # Visualizations
        for comp_label, result in nbs_results.items():
            sig_comps = [
                c for c in result["significant_components"] if c["pvalue"] < 0.05
            ]
            sig_edges = []
            for comp in sig_comps:
                sig_edges.extend(comp["edges"])

            plot_nbs_network(
                result["significant_components"],
                self.bilateral_region_cols,
                title=f"NBS {self.metric}: {comp_label}",
                out_path=nbs_dir / comp_label / "nbs_network.png",
            )

            ga, gb = result["group_a"], result["group_b"]
            if ga in self.matrices_pnd_dose and gb in self.matrices_pnd_dose:
                plot_difference_matrix(
                    self.matrices_pnd_dose[ga]["corr"],
                    self.matrices_pnd_dose[gb]["corr"],
                    self.bilateral_region_cols,
                    sig_edges=sig_edges,
                    title=f"{self.metric} \u0394r: {ga} \u2212 {gb}",
                    out_path=nbs_dir / comp_label / "difference_matrix.png",
                )

        n_sig = sum(
            1
            for r in nbs_results.values()
            if any(c["pvalue"] < 0.05 for c in r["significant_components"])
        )
        logger.info(f"NBS {self.metric}: {n_sig} significant comparisons")
        return nbs_results

    def run_territory(
        self, comparisons: list[tuple[str, str]] | None = None
    ) -> pd.DataFrame:
        """Run territory-level Fisher z + FDR.

        Results saved to ``{metric_dir}/territory/``.

        Parameters
        ----------
        comparisons : list of (str, str), optional
            Group pairs. If None, uses all default comparisons.

        Returns
        -------
        DataFrame of territory results.
        """
        if comparisons is None:
            comparisons = self.resolve_comparisons()

        logger.info(
            f"Territory analysis: {self.metric} ({len(comparisons)} comparisons)"
        )

        territory_results = {}
        for label_a, label_b in comparisons:
            if label_a not in self.matrices_territory:
                logger.warning(
                    f"Group {label_a} not in territory matrices, skipping"
                )
                continue
            if label_b not in self.matrices_territory:
                logger.warning(
                    f"Group {label_b} not in territory matrices, skipping"
                )
                continue

            corr_a = self.matrices_territory[label_a]["corr"]
            n_a = self.matrices_territory[label_a]["n"]
            corr_b = self.matrices_territory[label_b]["corr"]
            n_b = self.matrices_territory[label_b]["n"]

            z_stats, p_values = fisher_z_edge_test(corr_a, n_a, corr_b, n_b)
            p_fdr = fdr_correct_matrix(p_values)

            comp_label = f"{label_a}_vs_{label_b}"
            territory_results[comp_label] = {
                "z_stats": z_stats,
                "p_values": p_values,
                "p_fdr": p_fdr,
            }

        if territory_results:
            return _save_territory_results(
                territory_results,
                self.metric_dir / "territory",
                self.territory_cols,
            )
        return pd.DataFrame()

    def run_graph_metrics(
        self,
        densities: list[float] | None = None,
        n_perm: int = 5000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Run graph metrics. Results saved to ``{metric_dir}/graph_metrics/``.

        Parameters
        ----------
        densities : list[float], optional
            Network densities. Default [0.10, 0.15, 0.20, 0.25].
        n_perm : int
            Permutations for comparison test.
        seed : int
            Random seed.

        Returns
        -------
        DataFrame of comparison p-values.
        """
        if densities is None:
            densities = [0.10, 0.15, 0.20, 0.25]

        logger.info(
            f"Graph metrics: {self.metric} ({n_perm} permutations, "
            f"densities={densities})"
        )

        graph_dir = self.metric_dir / "graph_metrics"
        graph_dir.mkdir(parents=True, exist_ok=True)

        # Per-group metrics at multiple densities
        metrics_rows = []
        for label, data in self.matrices_pnd_dose.items():
            for d in densities:
                m = compute_metrics(data["corr"], density=d)
                m["group"] = label
                m["density"] = d
                metrics_rows.append(m)

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(graph_dir / "global_metrics.csv", index=False)
        logger.info(f"Saved global metrics: {graph_dir / 'global_metrics.csv'}")

        # Permutation comparison
        comparison_df = compare_metrics(
            self.group_arrays,
            self.bilateral_region_cols,
            densities=densities,
            n_perm=n_perm,
            seed=seed,
        )
        comparison_df.to_csv(graph_dir / "comparison_pvalues.csv", index=False)

        # Visualization
        plot_graph_metrics_comparison(
            comparison_df,
            out_path=self.metric_dir / "figures" / "graph_metrics_bars.png",
        )

        n_sig = int((comparison_df["p_value"] < 0.05).sum())
        logger.info(f"Graph metrics {self.metric}: {n_sig} significant comparisons")
        return comparison_df

    def run_whole_network(
        self,
        comparisons: list[tuple[str, str]] | None = None,
        n_perm: int = 5000,
        seed: int = 42,
        n_workers: int = 1,
    ) -> tuple[pd.DataFrame, dict]:
        """Run whole-network tests. Results saved to ``{metric_dir}/whole_network/``.

        Parameters
        ----------
        comparisons : list of (str, str), optional
            Group pairs. If None, uses all default comparisons.
        n_perm : int
            Number of permutations.
        seed : int
            Random seed.
        n_workers : int
            Parallel workers (1 = sequential).

        Returns
        -------
        (DataFrame of results, dict of null distributions)
        """
        if comparisons is None:
            comparisons = self.resolve_comparisons()

        logger.info(
            f"Whole-network: {self.metric} ({n_perm} permutations, "
            f"{len(comparisons)} comparisons)"
        )

        wn_dir = self.metric_dir / "whole_network"
        wn_dir.mkdir(parents=True, exist_ok=True)

        wn_df, wn_nulls = _run_whole_network_comparisons(
            group_data=self.group_arrays,
            comparisons=comparisons,
            n_perm=n_perm,
            seed=seed,
            n_workers=n_workers,
        )

        wn_df.to_csv(wn_dir / "whole_network_results.csv", index=False)

        # Save null distributions
        null_arrays = {}
        for comp_label, dists in wn_nulls.items():
            for stat_name, arr in dists.items():
                null_arrays[f"{comp_label}__{stat_name}"] = arr
        np.savez(wn_dir / "null_distributions.npz", **null_arrays)

        n_sig = int(
            (wn_df[["mantel_p", "frobenius_p", "spectral_p"]] < 0.05)
            .any(axis=1)
            .sum()
        )
        logger.info(
            f"Whole-network {self.metric}: {n_sig}/{len(wn_df)} comparisons "
            f"with at least one significant statistic (p < 0.05)"
        )
        return wn_df, wn_nulls
