"""
Visualization for covariance network analysis.

Generates correlation heatmaps, difference matrices with NBS highlights,
spring-layout network plots, and graph metric comparison charts.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


# Territory display order and colors (hybrid grouping)
# Cortex subsystems (from atlas "System" column)
# Non-cortex territories (from atlas "Territories" column)
TERRITORY_COLORS = {
    # Cortex subsystems
    "Somatosensory_System": "#e41a1c",
    "Hippocampus_Fomation": "#377eb8",
    "Olfactory_System": "#4daf4a",
    "Visual_System": "#984ea3",
    "Parietal_System": "#ff7f00",
    "Insular_System": "#a65628",
    "Auditory_System": "#f781bf",
    "Retrosplenial_System": "#66c2a5",
    "Temporal_System": "#8dd3c7",
    "Cingular_System": "#fb9a99",
    "Amygdala": "#cab2d6",
    "Motor_System": "#b2df8a",
    "Limbic_System": "#fdbf6f",
    # Non-cortex territories
    "Diencephalon": "#6a3d9a",
    "Mesencephalon": "#b15928",
    "Basal_ganglia": "#ffff99",
    "Brainstem": "#1f78b4",
    "Fiber_tract": "#33a02c",
    "Forebrain": "#e31a1c",
    "Olfactive_Bulb": "#ff7f00",
    "Cerebellum": "#999999",
    "Spinal_Cord": "#666666",
    "CSF": "#cccccc",
}


def _roi_territory_map(roi_names: list[str]) -> dict[str, str]:
    """Map ROI names to territory labels based on naming convention.

    Territory columns start with 'territory_'. Region ROIs are mapped
    based on known anatomical groupings (simplified heuristic).
    """
    mapping = {}
    for roi in roi_names:
        if roi.startswith("territory_"):
            mapping[roi] = roi.replace("territory_", "").replace(" ", "_")
        else:
            mapping[roi] = "region"
    return mapping


def plot_correlation_heatmap(
    corr: np.ndarray,
    roi_names: list[str],
    title: str,
    out_path: Path,
    territory_labels: Optional[dict[str, str]] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: tuple = None,
) -> None:
    """Plot a correlation matrix heatmap.

    Parameters
    ----------
    corr : ndarray (n_rois, n_rois)
        Correlation matrix.
    roi_names : list[str]
        ROI labels for axes.
    title : str
        Figure title.
    out_path : Path
        Output file path (png/pdf).
    territory_labels : dict, optional
        Mapping from ROI name to territory, used for divider lines.
    vmin, vmax : float
        Color scale limits.
    figsize : tuple, optional
        Figure size. Auto-scaled from number of ROIs if not provided.
    """
    n = len(roi_names)
    if figsize is None:
        size = max(8, n * 0.15)
        figsize = (size, size)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Shorten ROI labels for readability
    short_labels = _shorten_labels(roi_names)

    sns.heatmap(
        corr,
        xticklabels=short_labels,
        yticklabels=short_labels,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.6, "label": "Spearman r"},
    )

    # Add territory divider lines if territory labels provided
    if territory_labels is not None:
        _add_territory_dividers(ax, roi_names, territory_labels, n)

    ax.set_title(title, fontsize=12, pad=10)
    ax.tick_params(axis="both", labelsize=max(4, min(8, 200 // n)))
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved heatmap: {out_path}")


def plot_difference_matrix(
    corr_a: np.ndarray,
    corr_b: np.ndarray,
    roi_names: list[str],
    sig_edges: Optional[list[tuple[int, int]]] = None,
    title: str = "Correlation Difference (A - B)",
    out_path: Path = None,
    figsize: tuple = None,
) -> None:
    """Plot difference between two correlation matrices.

    Parameters
    ----------
    corr_a, corr_b : ndarray
        Correlation matrices.
    roi_names : list[str]
        ROI labels.
    sig_edges : list of (i, j) tuples, optional
        NBS-significant edges to highlight with markers.
    title : str
        Figure title.
    out_path : Path
        Output file path.
    figsize : tuple, optional
        Figure size.
    """
    diff = corr_a - corr_b
    n = len(roi_names)

    if figsize is None:
        size = max(8, n * 0.15)
        figsize = (size, size)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    short_labels = _shorten_labels(roi_names)

    # Symmetric color scale
    abs_max = np.nanmax(np.abs(diff))
    vbound = min(abs_max, 1.0)

    sns.heatmap(
        diff,
        xticklabels=short_labels,
        yticklabels=short_labels,
        cmap="RdBu_r",
        center=0,
        vmin=-vbound,
        vmax=vbound,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.6, "label": "Δr (A - B)"},
    )

    # Highlight significant edges
    if sig_edges:
        for i, j in sig_edges:
            ax.plot(j + 0.5, i + 0.5, "k*", markersize=3, alpha=0.7)
            ax.plot(i + 0.5, j + 0.5, "k*", markersize=3, alpha=0.7)

    ax.set_title(title, fontsize=12, pad=10)
    ax.tick_params(axis="both", labelsize=max(4, min(8, 200 // n)))
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved difference matrix: {out_path}")
    else:
        plt.close(fig)


def plot_nbs_network(
    components: list[dict],
    roi_names: list[str],
    title: str = "NBS Significant Network",
    out_path: Path = None,
    territory_map: Optional[dict[str, str]] = None,
    figsize: tuple = (10, 10),
) -> None:
    """Plot NBS-significant network components as a spring-layout graph.

    Parameters
    ----------
    components : list[dict]
        From ``network_based_statistic()['significant_components']``.
        Each has 'nodes', 'edges', 'pvalue'.
    roi_names : list[str]
        ROI names for node labels.
    title : str
        Figure title.
    out_path : Path
        Output file path.
    territory_map : dict, optional
        ROI name to territory, for node coloring.
    figsize : tuple
        Figure size.
    """
    # Filter to significant components only
    sig_components = [c for c in components if c.get("pvalue", 1.0) < 0.05]

    if not sig_components:
        logger.info("No significant NBS components to plot")
        if out_path:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.text(
                0.5, 0.5, "No significant components",
                ha="center", va="center", fontsize=14,
            )
            ax.set_title(title)
            ax.axis("off")
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Build graph from all significant components
    G = nx.Graph()
    for comp in sig_components:
        for u, v in comp["edges"]:
            G.add_edge(u, v)

    # Node labels
    node_labels = {}
    for node in G.nodes():
        if node < len(roi_names):
            label = roi_names[node]
            # Shorten for display
            if len(label) > 20:
                label = label[:17] + "..."
            node_labels[node] = label
        else:
            node_labels[node] = str(node)

    # Node colors by territory
    if territory_map:
        node_colors = []
        for node in G.nodes():
            roi = roi_names[node] if node < len(roi_names) else ""
            territory = territory_map.get(roi, "region")
            color = TERRITORY_COLORS.get(territory, "#cccccc")
            node_colors.append(color)
    else:
        node_colors = "#4daf4a"

    pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(G.number_of_nodes()))

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=300, alpha=0.8
    )
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, width=1.5, edge_color="#333333")
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, ax=ax, font_size=6, font_weight="bold"
    )

    # Legend for components
    for i, comp in enumerate(sig_components):
        ax.text(
            0.02, 0.98 - i * 0.04,
            f"Component {i + 1}: {comp['size']} edges, p={comp['pvalue']:.3f}",
            transform=ax.transAxes, fontsize=8, va="top",
        )

    ax.set_title(title, fontsize=12)
    ax.axis("off")

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved NBS network: {out_path}")
    else:
        plt.close(fig)


def plot_graph_metrics_comparison(
    metrics_df: pd.DataFrame,
    out_path: Path = None,
    figsize: tuple = (14, 10),
) -> None:
    """Plot graph metrics comparison across groups as grouped bar charts.

    Parameters
    ----------
    metrics_df : DataFrame
        From ``compare_metrics()``. Must have columns: group_a, group_b,
        metric, density, observed_a, observed_b, observed_diff, p_value.
    out_path : Path
        Output file path.
    figsize : tuple
        Figure size.
    """
    if metrics_df.empty:
        logger.info("No graph metrics to plot")
        return

    metrics = metrics_df["metric"].unique()
    densities = sorted(metrics_df["density"].unique())

    fig, axes = plt.subplots(
        len(metrics), 1, figsize=figsize, sharex=False
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        subset = metrics_df[metrics_df["metric"] == metric]

        # For each density, plot group differences
        for i, d in enumerate(densities):
            d_subset = subset[subset["density"] == d]
            labels = [
                f"{r['group_a']}\nvs\n{r['group_b']}"
                for _, r in d_subset.iterrows()
            ]
            diffs = d_subset["observed_diff"].values
            pvals = d_subset["p_value"].values

            x = np.arange(len(labels)) + i * 0.2
            bars = ax.bar(x, diffs, width=0.18, label=f"d={d:.2f}", alpha=0.8)

            # Mark significant comparisons
            for xi, pv in zip(x, pvals):
                if pv < 0.05:
                    ax.text(xi, 0, "*", ha="center", fontsize=12, fontweight="bold")

        ax.set_ylabel(metric.replace("_", " ").title())
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=7, loc="upper right")

        if ax == axes[-1]:
            ax.set_xticks(np.arange(len(labels)) + 0.3)
            ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")

    fig.suptitle("Graph Metric Differences Between Groups", fontsize=13)
    fig.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved graph metrics comparison: {out_path}")
    else:
        plt.close(fig)


def plot_all_group_heatmaps(
    group_matrices: dict[str, dict],
    out_path: Path,
    title_prefix: str = "",
    figsize_per_panel: tuple = (5, 5),
) -> None:
    """Plot a grid of correlation heatmaps for all groups.

    Parameters
    ----------
    group_matrices : dict[str, dict]
        From ``compute_spearman_matrices()``.
    out_path : Path
        Output file path.
    title_prefix : str
        Prefix for subplot titles (e.g., metric name).
    figsize_per_panel : tuple
        Size per subplot panel.
    """
    labels = sorted(group_matrices.keys())
    n = len(labels)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for idx, label in enumerate(labels):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        corr = group_matrices[label]["corr"]
        n_subj = group_matrices[label]["n"]

        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"{title_prefix}{label} (n={n_subj})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove empty axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Spearman r")
    fig.suptitle(f"{title_prefix}Correlation Matrices", fontsize=13)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved group heatmaps grid: {out_path}")


def _shorten_labels(names: list[str], max_len: int = 25) -> list[str]:
    """Shorten ROI labels for plot axes."""
    short = []
    for name in names:
        name = name.replace("territory_", "")
        name = name.replace("_", " ")
        if len(name) > max_len:
            name = name[:max_len - 2] + ".."
        short.append(name)
    return short


def _add_territory_dividers(
    ax, roi_names: list[str], territory_labels: dict[str, str], n: int
) -> None:
    """Add territory boundary lines to a heatmap."""
    territories = [territory_labels.get(r, "") for r in roi_names]
    prev = territories[0] if territories else ""
    for i, t in enumerate(territories[1:], 1):
        if t != prev:
            ax.axhline(i, color="black", linewidth=0.5, alpha=0.5)
            ax.axvline(i, color="black", linewidth=0.5, alpha=0.5)
        prev = t
