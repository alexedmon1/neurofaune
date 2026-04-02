"""Covariance Network (CovNet) analysis pipeline.

Provides the ``CovNetAnalysis`` class: prepare data once, then run any
combination of NBS, territory, graph theory, or network distance tests.
Each test has a corresponding standalone script in ``scripts/``:

- ``run_covnet_nbs.py`` — NBS permutation testing
- ``run_covnet_territory.py`` — Territory-level Fisher z + FDR (post-hoc)
- ``run_covnet_graph_theory.py`` — Graph-theoretic metric comparisons
- ``run_covnet_abs_distance.py`` — Absolute network distance (Mantel/Frobenius/spectral)
- ``run_covnet_rel_distance.py`` — Relative network distance (shift toward/away from reference)

Edge regression (continuous targets) is separate: see
``neurofaune.network.edge_regression`` and ``run_edge_regression.py``.

Typical usage::

    analysis = CovNetAnalysis.prepare(roi_dir, exclusion_csv, covnet_root, "dwi", "FA")
    analysis.save()
    analysis.run_nbs(comparisons=analysis.resolve_comparisons(["dose"]))
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neurofaune.network.graph_theory import (
    DEFAULT_DENSITIES,
    METRIC_REGISTRY,
    compare_metric,
    compute_all_metrics,
    compute_metric_curve,
    list_metrics,
)
from .nbs import characterize_components, fisher_z_edge_test, nbs_posthoc
from .nbs import run_all_comparisons as _run_nbs_comparisons
from .visualization import (
    plot_all_group_heatmaps,
    plot_correlation_heatmap,
    plot_density_curves,
    plot_difference_matrix,
    plot_nbs_network,
)
from .whole_network import (
    run_all_comparisons as _run_abs_distance_comparisons,
    run_rel_distance as _run_rel_distance,
    run_subject_rel_distance as _run_subject_rel_distance,
    rel_distance_comparisons,
)
from neurofaune.network.matrices import (
    compute_spearman_matrices,
    cross_dose_timepoint_comparisons,
    cross_timepoint_comparisons,
    default_dose_comparisons,
    define_groups,
    load_and_prepare_data,
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
    matrices: dict[str, dict], mat_dir: Path, sub_name: str
) -> None:
    """Save per-group correlation matrices as CSV files."""
    out_dir = mat_dir / sub_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, data in matrices.items():
        corr_df = pd.DataFrame(
            data["corr"], index=data["rois"], columns=data["rois"]
        )
        corr_df.to_csv(out_dir / f"{label}_corr.csv")


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
        test_stat = result["test_stat"]
        for comp in result["significant_components"]:
            edges_with_stats = sorted(
                [
                    {
                        "edge": [u, v],
                        "edge_name": (rois[u], rois[v]),
                        "test_stat": float(test_stat[u, v]),
                    }
                    for u, v in comp["edges"]
                ],
                key=lambda e: abs(e["test_stat"]),
                reverse=True,
            )
            components_json["components"].append({
                "nodes": comp["nodes"],
                "node_names": [rois[n] for n in comp["nodes"]],
                "edges": edges_with_stats,
                "size": comp["size"],
                "pvalue": comp["pvalue"],
            })

        with open(nbs_dir / "components.json", "w") as f:
            json.dump(components_json, f, indent=2)

        np.savetxt(nbs_dir / "null_distribution.txt", result["null_distribution"])


def _save_nbs_posthoc(
    nbs_results: dict[str, dict], nbs_base_dir: Path
) -> None:
    """Run and save post-hoc centrality and hub-vulnerability analyses.

    Only runs on components with p < 0.05. Results saved to
    ``{comparison}/posthoc/`` as ``centrality.csv`` and
    ``hub_vulnerability.csv``, one file per significant component.
    """
    for comp_label, result in nbs_results.items():
        rois = result["roi_cols"]
        test_stat = result["test_stat"]
        sig_comps = [c for c in result["significant_components"] if c["pvalue"] < 0.05]
        if not sig_comps:
            continue

        posthoc_dir = nbs_base_dir / comp_label / "posthoc"
        posthoc_dir.mkdir(parents=True, exist_ok=True)

        for idx, comp in enumerate(sig_comps):
            ph = nbs_posthoc(comp, test_stat, rois)
            prefix = f"comp{idx:02d}_"

            pd.DataFrame(ph["centrality"]).to_csv(
                posthoc_dir / f"{prefix}centrality.csv", index=False
            )
            pd.DataFrame(ph["hub_vulnerability"]).to_csv(
                posthoc_dir / f"{prefix}hub_vulnerability.csv", index=False
            )

        logger.info(
            "Post-hoc saved for %s (%d significant component(s))",
            comp_label, len(sig_comps),
        )


def _save_nbs_characterization(
    characterized: list[dict],
    nbs_result: dict,
    out_dir: Path,
) -> None:
    """Save edge directionality and node centrality characterization."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "component_characterization.json", "w") as f:
        json.dump(characterized, f, indent=2)

    rows = []
    for i, comp in enumerate(characterized):
        rows.append({
            "component": i,
            "n_nodes": comp["n_nodes"],
            "n_edges": comp["n_edges"],
            "pvalue": comp["pvalue"],
            "n_increased": comp["n_increased"],
            "n_decreased": comp["n_decreased"],
            "frac_increased": comp["frac_increased"],
            "mean_z": comp["mean_z"],
            "median_z": comp["median_z"],
            "hub_nodes": "; ".join(comp["hub_nodes"]),
        })
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "component_summary.csv", index=False)

    node_rows = []
    for i, comp in enumerate(characterized):
        for node in comp["nodes"]:
            node_rows.append({
                "component": i,
                "roi": node["roi"],
                "degree": node["degree"],
                "betweenness": node["betweenness"],
                "mean_edge_z": node["mean_edge_z"],
                "n_increased": node["n_increased"],
                "n_decreased": node["n_decreased"],
            })
    if node_rows:
        pd.DataFrame(node_rows).to_csv(out_dir / "node_centrality.csv", index=False)


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


def build_territory_mapping(
    region_cols: list[str],
    labels_csv: Path,
) -> dict[str, str]:
    """Map each ROI to a hybrid territory group.

    Cortex ROIs use the "System" column (e.g. Somatosensory System,
    Hippocampus Fomation). Non-Cortex ROIs use the "Territories" column
    (e.g. Diencephalon, Brainstem).

    Parameters
    ----------
    region_cols : list[str]
        ROI column names (may have _L/_R suffixes).
    labels_csv : Path
        Path to SIGMA atlas labels CSV with columns:
        Territories, System, Region of interest.

    Returns
    -------
    dict mapping ROI column name -> territory group name.
    """
    labels_df = pd.read_csv(labels_csv)

    # Build lookup: underscore-normalised ROI base name -> (territory, system)
    # Atlas uses dots in names (e.g. "Agranular.Dysgranular.Insular.Cortex.L")
    # CSV columns use underscores (e.g. "Agranular_Dysgranular_Insular_Cortex")
    roi_info: dict[str, tuple[str, str]] = {}
    for _, row in labels_df.iterrows():
        roi_name = str(row["Region of interest"])
        territory = str(row["Territories"])
        system = str(row["System"])

        # Normalise: dots -> underscores, strip hemisphere suffix
        normalised = roi_name.replace(".", "_")
        if normalised.endswith("_L") or normalised.endswith("_R"):
            base = normalised[:-2]
        else:
            base = normalised

        # Store first occurrence (L and R have the same territory/system)
        if base not in roi_info:
            roi_info[base] = (territory, system)

    # Map each ROI to its hybrid group.
    # ROI columns may have _L/_R suffixes; strip them for atlas lookup.
    mapping: dict[str, str] = {}
    for col in region_cols:
        info = roi_info.get(col)
        if info is None and (col.endswith("_L") or col.endswith("_R")):
            info = roi_info.get(col[:-2])
        if info is None:
            logger.warning(f"ROI {col!r} not found in atlas labels, skipping")
            continue
        territory, system = info
        if territory == "Cortex":
            mapping[col] = system
        else:
            mapping[col] = territory

    n_groups = len(set(mapping.values()))
    n_mapped = len(mapping)
    logger.info(
        f"Territory mapping: {n_mapped}/{len(region_cols)} ROIs "
        f"-> {n_groups} hybrid groups"
    )
    return mapping


def compute_territory_means(
    df: pd.DataFrame,
    region_cols: list[str],
    roi_to_territory: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """Compute per-subject mean across ROIs within each territory group.

    Parameters
    ----------
    df : DataFrame
        Must contain all columns in *region_cols*.
    region_cols : list[str]
        ROI column names.
    roi_to_territory : dict[str, str]
        Mapping from ROI name to territory group (from
        ``build_territory_mapping``).

    Returns
    -------
    df_augmented : DataFrame
        Input DataFrame with new ``territory_<group>`` columns appended.
    territory_col_names : list[str]
        Sorted list of new territory column names.
    """
    # Group ROIs by territory
    groups: dict[str, list[str]] = {}
    for col in region_cols:
        grp = roi_to_territory.get(col)
        if grp is not None:
            groups.setdefault(grp, []).append(col)

    # Compute row-wise means and add as new columns
    territory_cols = []
    parts = [df]
    for grp_name in sorted(groups.keys()):
        col_name = "territory_" + grp_name.replace(" ", "_")
        territory_cols.append(col_name)
        roi_subset = groups[grp_name]
        parts.append(
            df[roi_subset].mean(axis=1, skipna=True).rename(col_name)
        )

    df_augmented = pd.concat(parts, axis=1)

    for grp_name in sorted(groups.keys()):
        col_name = "territory_" + grp_name.replace(" ", "_")
        rois = groups[grp_name]
        logger.info(f"  {col_name}: {len(rois)} ROIs")

    return df_augmented, sorted(territory_cols)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CovNetAnalysis:
    """CovNet analysis for a single modality/metric.

    Holds prepared data (group arrays, correlation matrices) and runs
    statistical tests. Serializable to / from disk via save() / load().
    """

    def __init__(self, covnet_root: Path, modality: str, metric: str):
        self.covnet_root = Path(covnet_root)
        self.modality = modality
        self.metric = metric
        self.roi_dir: Path | None = None

        # Populated by prepare() or load()
        self.n_subjects: int = 0
        self.group_arrays: dict[str, np.ndarray] = {}
        self.territory_arrays: dict[str, np.ndarray] = {}
        self.region_cols: list[str] = []
        self.territory_cols: list[str] = []
        self.group_labels: list[str] = []
        self.group_sizes: dict[str, int] = {}
        self.matrices_pnd_dose: dict[str, dict] = {}
        self.matrices_full: dict[str, dict] = {}
        self.matrices_territory: dict[str, dict] = {}
        self.roi_to_territory: dict[str, str] = {}
        self.labels_csv: str = ""

    @property
    def _variant(self) -> str:
        """Run variant: 'pooled', 'sex_stratified/F', or 'sex_stratified/M'."""
        sex = getattr(self, "sex", None)
        if sex is None:
            return "pooled"
        return f"sex_stratified/{sex}"

    def _test_dir(self, analysis: str, variant: str | None = None) -> Path:
        """Per-analysis output directory.

        Structure: ``{covnet_root}/{analysis}/{variant}/{modality}/{metric}/``

        Parameters
        ----------
        analysis : str
            Analysis type (e.g. ``"nbs"``, ``"abs_distance"``, ``"data"``).
        variant : str, optional
            Run variant override. If None, uses ``self._variant``
            (``"pooled"`` or ``"sex_stratified/{sex}"``).
        """
        v = variant if variant is not None else self._variant
        return self.covnet_root / analysis / v / self.modality / self.metric

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def prepare(
        cls,
        roi_dir: Path,
        exclusion_csv: Path | None,
        covnet_root: Path,
        modality: str,
        metric: str,
        labels_csv: Path | None = None,
        sex: str | None = None,
    ) -> "CovNetAnalysis":
        """Load ROI data, compute territory means and correlation matrices.

        Parameters
        ----------
        roi_dir : Path
            Directory containing ``roi_<metric>_wide.csv`` files.
        exclusion_csv : Path or None
            CSV of sessions to exclude.
        covnet_root : Path
            Root output directory for CovNet results.
        modality : str
            Modality name (e.g. ``"dwi"``, ``"msme"``, ``"func"``).
        metric : str
            Metric name (e.g. ``"FA"``).
        labels_csv : Path or None
            SIGMA atlas labels CSV for hybrid territory mapping.  If None,
            falls back to the default path on arborea.
        sex : str or None
            If set (``"F"`` or ``"M"``), restrict analysis to one sex.

        Returns a populated instance ready for save() and run_*() methods.
        """
        roi_dir = Path(roi_dir)
        covnet_root = Path(covnet_root)

        if labels_csv is None:
            labels_csv = Path(
                "/mnt/arborea/atlases/SIGMA/"
                "SIGMA_InVivo_Anatomical_Brain_Atlas_Labels.csv"
            )
        labels_csv = Path(labels_csv)

        wide_csv = roi_dir / f"roi_{metric}_wide.csv"
        if not wide_csv.exists():
            raise FileNotFoundError(f"Wide CSV not found: {wide_csv}")

        inst = cls(covnet_root, modality, metric)
        inst.roi_dir = roi_dir
        inst.sex = sex

        # Phase 1: Load and prepare data
        logger.info(f"\n[Phase 1] Loading and preparing data for {metric}...")
        df, roi_cols = load_and_prepare_data(wide_csv, exclusion_csv)

        # Optional sex filter
        if sex is not None:
            df = df[df["sex"] == sex].reset_index(drop=True)
            logger.info("Sex filter '%s': %d subjects remaining", sex, len(df))

        inst.n_subjects = len(df)

        # Phase 2: Separate region vs territory ROI columns
        inst.region_cols = [
            c for c in roi_cols if not c.startswith("territory_")
        ]

        # Phase 2b: Hybrid territory mapping from atlas labels
        logger.info("[Phase 2] Building hybrid territory mapping...")
        inst.roi_to_territory = build_territory_mapping(
            inst.region_cols, labels_csv
        )
        inst.labels_csv = str(labels_csv)
        # Drop old pre-aggregated territory columns before adding new ones
        old_territory_cols = [
            c for c in df.columns if c.startswith("territory_")
        ]
        if old_territory_cols:
            df = df.drop(columns=old_territory_cols)
        df, inst.territory_cols = compute_territory_means(
            df, inst.region_cols, inst.roi_to_territory
        )

        # Phase 3: Compute correlation matrices
        logger.info("[Phase 3] Computing correlation matrices...")

        logger.info("  PND x dose grouping (region):")
        groups_pnd_dose = define_groups(df, grouping="pnd_dose")
        inst.matrices_pnd_dose = compute_spearman_matrices(
            groups_pnd_dose, inst.region_cols
        )

        logger.info("  Full grouping (region, descriptive):")
        groups_full = define_groups(df, grouping="full")
        inst.matrices_full = compute_spearman_matrices(
            groups_full, inst.region_cols
        )

        logger.info("  PND x dose grouping (territory):")
        inst.matrices_territory = compute_spearman_matrices(
            groups_pnd_dose, inst.territory_cols
        )

        # Extract group arrays and metadata
        inst.group_labels = sorted(groups_pnd_dose.keys())
        inst.group_sizes = {k: len(v) for k, v in groups_pnd_dose.items()}
        inst.group_arrays = {
            label: subset[inst.region_cols].values
            for label, subset in groups_pnd_dose.items()
        }
        inst.territory_arrays = {
            label: subset[inst.territory_cols].values
            for label, subset in groups_pnd_dose.items()
        }

        logger.info(f"Preparation complete for {metric}")
        return inst

    def save(self, save_heatmaps: bool = True) -> None:
        """Serialize prepared data to disk.

        Saves metadata, group arrays, territory arrays, and correlation
        matrices. Optionally generates heatmap figures.
        """
        data_dir = self._test_dir("data")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Metadata
        metadata = {
            "metric": self.metric,
            "modality": self.modality,
            "n_subjects": self.n_subjects,
            "sex": getattr(self, "sex", None),
            "region_cols": self.region_cols,
            "territory_cols": self.territory_cols,
            "group_labels": self.group_labels,
            "group_sizes": self.group_sizes,
            "n_region_rois": len(self.region_cols),
            "n_territory_rois": len(self.territory_cols),
            "roi_to_territory": self.roi_to_territory,
            "labels_csv": self.labels_csv,
            "roi_dir": str(self.roi_dir) if self.roi_dir else "",
        }
        with open(data_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Group arrays
        arr_dir = data_dir / "group_arrays"
        arr_dir.mkdir(parents=True, exist_ok=True)
        for label, arr in self.group_arrays.items():
            np.save(arr_dir / f"{label}.npy", arr)

        # Territory arrays
        terr_dir = data_dir / "territory_arrays"
        terr_dir.mkdir(parents=True, exist_ok=True)
        for label, arr in self.territory_arrays.items():
            np.save(terr_dir / f"{label}.npy", arr)

        # Correlation matrices as CSV
        mat_dir = data_dir / "matrices"
        _save_matrices(self.matrices_pnd_dose, mat_dir, self.metric)
        _save_matrices(self.matrices_full, mat_dir, f"{self.metric}_full")
        _save_matrices(
            self.matrices_territory, mat_dir, f"{self.metric}_territory"
        )

        # Heatmaps
        if save_heatmaps:
            self._save_heatmaps()

        logger.info(f"Saved prepared data: {data_dir}")

    def _save_heatmaps(self) -> None:
        """Generate and save correlation heatmap figures."""
        fig_dir = self._test_dir("data") / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

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
                out_path=fig_dir / f"{label}_corr_heatmap.png",
            )

    @classmethod
    def load(cls, covnet_root: Path, modality: str, metric: str) -> "CovNetAnalysis":
        """Deserialize from ``{covnet_root}/prep/{modality}/{metric}/``."""
        covnet_root = Path(covnet_root)
        inst = cls(covnet_root, modality, metric)

        meta_path = inst._test_dir("data") / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Data metadata not found: {meta_path}\n"
                f"Run CovNetAnalysis.prepare() first."
            )

        with open(meta_path) as f:
            metadata = json.load(f)

        inst.sex = metadata.get("sex", None)
        inst.n_subjects = metadata.get("n_subjects", 0)
        inst.region_cols = metadata.get("region_cols", metadata.get("bilateral_region_cols", []))
        inst.territory_cols = metadata["territory_cols"]
        inst.group_labels = metadata["group_labels"]
        inst.group_sizes = metadata["group_sizes"]
        inst.roi_to_territory = metadata.get("roi_to_territory", {})
        inst.labels_csv = metadata.get("labels_csv", "")
        roi_dir_str = metadata.get("roi_dir", "")
        inst.roi_dir = Path(roi_dir_str) if roi_dir_str else None

        # Group arrays
        arr_dir = inst._test_dir("data") / "group_arrays"
        for label in inst.group_labels:
            arr_path = arr_dir / f"{label}.npy"
            if arr_path.exists():
                inst.group_arrays[label] = np.load(arr_path)
            else:
                logger.warning(f"Missing group array: {arr_path}")

        # Territory arrays
        terr_dir = inst._test_dir("data") / "territory_arrays"
        if terr_dir.exists():
            for label in inst.group_labels:
                arr_path = terr_dir / f"{label}.npy"
                if arr_path.exists():
                    inst.territory_arrays[label] = np.load(arr_path)

        # Correlation matrices from CSV
        mat_dir = inst._test_dir("data") / "matrices"
        inst.matrices_pnd_dose = _load_matrices(mat_dir / metric)
        inst.matrices_full = _load_matrices(mat_dir / f"{metric}_full")
        inst.matrices_territory = _load_matrices(mat_dir / f"{metric}_territory")

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
        posthoc: bool = False,
    ) -> dict:
        """Run NBS. Results saved to ``nbs/{modality}/{metric}/``.

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
            roi_cols=self.region_cols,
            comparisons=comparisons,
            n_perm=n_perm,
            threshold=threshold,
            seed=seed,
            n_workers=n_workers,
        )

        # Save results
        nbs_dir = self._test_dir("nbs")
        _save_nbs_results(nbs_results, nbs_dir)
        if posthoc:
            _save_nbs_posthoc(nbs_results, nbs_dir)

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
                self.region_cols,
                title=f"NBS {self.metric}: {comp_label}",
                out_path=nbs_dir / comp_label / "nbs_network.png",
            )

            ga, gb = result["group_a"], result["group_b"]
            if ga in self.matrices_pnd_dose and gb in self.matrices_pnd_dose:
                plot_difference_matrix(
                    self.matrices_pnd_dose[ga]["corr"],
                    self.matrices_pnd_dose[gb]["corr"],
                    self.region_cols,
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

    def run_nbs_posthoc(
        self,
        nbs_results: dict[str, dict] | None = None,
        p_threshold: float = 0.05,
    ) -> dict[str, list[dict]]:
        """Run post-hoc characterization of NBS components.

        For each comparison with significant components, computes edge
        directionality (increased vs. decreased covariance) and node centrality
        (degree, betweenness) within each component.

        Parameters
        ----------
        nbs_results : dict, optional
            Output from ``run_nbs()``. If None, loads saved NBS results from
            disk (reads test_statistics.csv and components.json).
        p_threshold : float
            Only characterize components with p < this value.

        Returns
        -------
        dict mapping comparison labels to lists of characterized components.
        """
        nbs_dir = self._test_dir("nbs")

        if nbs_results is None:
            nbs_results = self._load_nbs_results(nbs_dir)

        all_posthoc = {}
        for comp_label, result in nbs_results.items():
            sig_comps = [
                c for c in result["significant_components"]
                if c["pvalue"] < p_threshold
            ]
            if not sig_comps:
                continue

            characterized = characterize_components(
                result,
                roi_cols=result.get("roi_cols", self.region_cols),
                p_threshold=p_threshold,
            )
            if characterized:
                out_dir = nbs_dir / comp_label / "posthoc"
                _save_nbs_characterization(characterized, result, out_dir)
                all_posthoc[comp_label] = characterized

        logger.info(
            f"NBS post-hoc {self.metric}: characterized components in "
            f"{len(all_posthoc)} comparisons"
        )
        return all_posthoc

    def _load_nbs_results(self, nbs_dir: Path) -> dict[str, dict]:
        """Load previously saved NBS results from disk."""
        results = {}
        if not nbs_dir.exists():
            logger.warning(f"NBS directory not found: {nbs_dir}")
            return results

        for comp_dir in sorted(nbs_dir.iterdir()):
            if not comp_dir.is_dir():
                continue
            stat_path = comp_dir / "test_statistics.csv"
            comp_path = comp_dir / "components.json"
            if not stat_path.exists() or not comp_path.exists():
                continue

            stat_df = pd.read_csv(stat_path, index_col=0)
            test_stat = stat_df.values

            with open(comp_path) as f:
                comp_json = json.load(f)

            components = []
            for c in comp_json.get("components", []):
                components.append({
                    "nodes": c["nodes"],
                    "edges": [tuple(e) for e in c["edges"]],
                    "size": c["size"],
                    "pvalue": c.get("pvalue", c.get("p_value", 1.0)),
                })

            results[comp_dir.name] = {
                "test_stat": test_stat,
                "significant_components": components,
                "null_distribution": np.array([]),
                "n_a": comp_json.get("n_a", 0),
                "n_b": comp_json.get("n_b", 0),
                "roi_cols": list(stat_df.columns),
                "group_a": comp_json.get("group_a", ""),
                "group_b": comp_json.get("group_b", ""),
            }

        logger.info(f"Loaded NBS results from {len(results)} comparisons")
        return results

    def run_territory(
        self, comparisons: list[tuple[str, str]] | None = None
    ) -> pd.DataFrame:
        """Run territory-level Fisher z + FDR.

        Results saved to ``territory/{modality}/{metric}/``.

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
                self._test_dir("system_connectivity"),
                self.territory_cols,
            )
        return pd.DataFrame()

    def run_graph_metrics(
        self,
        graph_metrics: list[str] | None = None,
        densities: list[float] | None = None,
        n_perm: int = 1000,
        seed: int = 42,
        n_workers: int = 1,
    ) -> pd.DataFrame:
        """Run graph theory analysis with density-curve AUC permutation testing.

        Results saved to ``graph_metrics/{modality}/{metric}/``.

        Parameters
        ----------
        graph_metrics : list[str], optional
            Graph metrics to test. Default: all in ``METRIC_REGISTRY``.
            Use ``list_metrics()`` to see available options.
        densities : list[float], optional
            Network density sweep. Default ``DEFAULT_DENSITIES``.
        n_perm : int
            Permutations for AUC comparison test.
        seed : int
            Random seed.
        n_workers : int
            Parallel workers for pairwise comparisons (1 = sequential).

        Returns
        -------
        DataFrame of AUC comparison p-values across all requested metrics.
        """
        if densities is None:
            densities = DEFAULT_DENSITIES
        if graph_metrics is None:
            graph_metrics = list_metrics()

        logger.info(
            "Graph theory: %s (%d metrics, %d densities, %d permutations)",
            self.metric, len(graph_metrics), len(densities), n_perm,
        )

        graph_dir = self._test_dir("graph_metrics")
        graph_dir.mkdir(parents=True, exist_ok=True)

        # Per-group density curves for all metrics
        curves_rows = []
        for label, data in self.matrices_pnd_dose.items():
            all_curves = compute_all_metrics(data["corr"], densities)
            for gm_name, values in all_curves.items():
                for i, d in enumerate(densities):
                    curves_rows.append({
                        "group": label,
                        "graph_metric": gm_name,
                        "density": d,
                        "value": values[i],
                    })

        curves_df = pd.DataFrame(curves_rows)
        curves_df.to_csv(graph_dir / "density_curves.csv", index=False)
        logger.info("Saved density curves: %s", graph_dir / "density_curves.csv")

        # Permutation test per requested metric
        all_comparison_rows = []
        all_nulls = {}
        for gm_name in graph_metrics:
            logger.info("  Testing: %s", gm_name)
            comp_df, obs_curves, nulls = compare_metric(
                self.group_arrays,
                metric_name=gm_name,
                densities=densities,
                n_perm=n_perm,
                seed=seed,
                n_workers=n_workers,
            )
            all_comparison_rows.append(comp_df)
            for key, arr in nulls.items():
                all_nulls[f"{gm_name}__{key}"] = arr

        comparison_df = pd.concat(all_comparison_rows, ignore_index=True)
        comparison_df.to_csv(graph_dir / "auc_comparison.csv", index=False)

        # Save null distributions
        np.savez(graph_dir / "null_distributions.npz", **all_nulls)

        # Visualization
        plot_density_curves(
            curves_df,
            out_path=graph_dir / "graph_density_curves.png",
        )

        n_sig = int((comparison_df["p_value"] < 0.05).sum())
        logger.info(
            "Graph theory %s: %d/%d significant AUC comparisons",
            self.metric, n_sig, len(comparison_df),
        )
        return comparison_df

    def run_abs_distance(
        self,
        comparisons: list[tuple[str, str]] | None = None,
        n_perm: int = 5000,
        seed: int = 42,
        n_workers: int = 1,
    ) -> tuple[pd.DataFrame, dict]:
        """Absolute network distance. Results saved to ``abs_distance/{modality}/{metric}/``.

        Directly compares two groups' covariance networks using Mantel,
        Frobenius, and Spectral distance metrics with permutation testing.

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
            f"Absolute distance: {self.metric} ({n_perm} permutations, "
            f"{len(comparisons)} comparisons)"
        )

        wn_dir = self._test_dir("abs_distance")
        wn_dir.mkdir(parents=True, exist_ok=True)

        wn_df, wn_nulls = _run_abs_distance_comparisons(
            group_data=self.group_arrays,
            comparisons=comparisons,
            n_perm=n_perm,
            seed=seed,
            n_workers=n_workers,
        )

        wn_df.to_csv(wn_dir / "abs_distance_results.csv", index=False)

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
            f"Absolute distance {self.metric}: {n_sig}/{len(wn_df)} comparisons "
            f"with at least one significant statistic (p < 0.05)"
        )
        return wn_df, wn_nulls

    # Backward-compatible alias
    run_whole_network = run_abs_distance

    def run_rel_distance(
        self,
        triplets: list[tuple[str, str, str]] | None = None,
        n_perm: int = 5000,
        seed: int = 42,
        distance_fns: list[str] | None = None,
        n_workers: int = 1,
    ) -> pd.DataFrame:
        """Relative network distance: test whether a group is shifted toward
        or away from a reference network.

        For each (group_A, group_B, reference) triplet, computes:
            Δ = d(A, reference) − d(B, reference)
        Negative Δ = A is closer to reference than B.
        Positive Δ = A is farther from reference than B.

        When reference = older controls, Δ < 0 = accelerated, Δ > 0 = decelerated.

        Results saved to ``rel_distance/{modality}/{metric}/``.

        Parameters
        ----------
        triplets : list of (dose, early_control, late_control), optional
            If None, auto-generated from group labels.
        n_perm : int
            Number of permutations.
        seed : int
            Random seed.
        distance_fns : list[str], optional
            Distance metrics. Default: ["frobenius", "spectral", "mantel"].
        n_workers : int
            Parallel workers.

        Returns
        -------
        DataFrame with columns: dose_group, early_control, reference,
        comparison, distance_fn, d_dose_to_ref, d_ctrl_to_ref, delta,
        p_accelerated, p_decelerated, n_dose, n_ctrl, n_ref, interpretation.
        """
        if triplets is None:
            triplets = rel_distance_comparisons(self.group_labels)

        logger.info(
            f"Relative distance: {self.metric} ({n_perm} permutations, "
            f"{len(triplets)} triplets)"
        )

        md_dir = self._test_dir("rel_distance")
        md_dir.mkdir(parents=True, exist_ok=True)

        md_df, null_dists = _run_rel_distance(
            group_data=self.group_arrays,
            triplets=triplets,
            n_perm=n_perm,
            seed=seed,
            distance_fns=distance_fns,
            n_workers=n_workers,
        )

        md_df.to_csv(md_dir / "rel_distance_results.csv", index=False)
        np.savez(md_dir / "null_distributions.npz", **null_dists)

        n_accel = int((md_df["p_accelerated"] < 0.05).sum())
        n_decel = int((md_df["p_decelerated"] < 0.05).sum())
        logger.info(
            f"Relative distance {self.metric}: {n_accel} accelerated, "
            f"{n_decel} decelerated (p < 0.05) out of {len(md_df)} tests"
        )
        return md_df

    def run_subject_rel_distance(
        self,
        triplets: list[tuple[str, str, str]] | None = None,
        n_perm: int = 5000,
        seed: int = 42,
        similarity_fns: list[str] | None = None,
    ) -> tuple["pd.DataFrame", "pd.DataFrame"]:
        """Subject-level relative distance: compare individual subjects'
        similarity to a reference group.

        For each subject in dose/control groups, computes the similarity of
        that subject's ROI profile to the reference group mean.  Tests whether
        dose subjects are systematically more (or less) similar to the
        reference than controls.

        Results saved to ``subject_rel_distance/{modality}/{metric}/``.

        Returns
        -------
        summary_df, per_subject_df
        """
        if triplets is None:
            triplets = rel_distance_comparisons(self.group_labels)

        logger.info(
            f"Subject-level relative distance: {self.metric} "
            f"({n_perm} permutations, {len(triplets)} triplets)"
        )

        out_dir = self._test_dir("rel_distance", variant="subject")
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_df, per_subject_df = _run_subject_rel_distance(
            group_data=self.group_arrays,
            triplets=triplets,
            n_perm=n_perm,
            seed=seed,
            similarity_fns=similarity_fns,
        )

        summary_df.to_csv(
            out_dir / "subject_rel_distance_results.csv", index=False
        )
        per_subject_df.to_csv(
            out_dir / "subject_rel_distance_per_subject.csv", index=False
        )

        n_accel = int((summary_df["p_accelerated"] < 0.05).sum())
        n_decel = int((summary_df["p_decelerated"] < 0.05).sum())
        logger.info(
            f"Subject rel-distance {self.metric}: {n_accel} accelerated, "
            f"{n_decel} decelerated (p < 0.05) out of {len(summary_df)} tests"
        )
        return summary_df, per_subject_df

    # Backward-compatible alias
    run_maturation_distance = run_rel_distance

