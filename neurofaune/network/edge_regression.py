"""
Edge-level regression for continuous targets.

Tests whether pairwise ROI co-variation (edge strength) scales with a
continuous covariate (e.g. log-AUC) using network-based regression with
NBS-style component extraction and permutation-based FWER correction.

This is a network-level analysis but is NOT a covariance network (CovNet)
analysis — CovNet tests group-level differences in correlation structure,
while edge regression tests continuous covariate associations with
individual-level edge contributions.

Appropriate for continuous targets only (AUC, log_auc, behavioural
scores, etc.). For categorical group comparisons, use NBS instead.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neurofaune.network.covnet.nbs import network_based_regression
from neurofaune.network.covnet.visualization import plot_nbs_network
from neurofaune.network.matrices import bilateral_average, load_and_prepare_data

logger = logging.getLogger(__name__)


def run_edge_regression(
    wide_csv: Path,
    exclusion_csv: Path | None,
    output_dir: Path,
    modality: str,
    metric: str,
    covariate_map: dict[str, float],
    covariate_name: str = "log(1+AUC)",
    cohort_filter: str | None = None,
    n_perm: int = 1000,
    threshold: float = 3.0,
    seed: int = 42,
) -> dict | None:
    """Run edge-level regression for one metric and cohort.

    Parameters
    ----------
    wide_csv : Path
        Path to ``roi_<metric>_wide.csv``.
    exclusion_csv : Path or None
        CSV of sessions to exclude.
    output_dir : Path
        Root output directory for edge regression results.
    modality : str
        Modality name (e.g. ``"dwi"``).
    metric : str
        Metric name (e.g. ``"FA"``).
    covariate_map : dict[str, float]
        Mapping from subject key (``"sub-Rat001_ses-p60"``) to covariate value.
    covariate_name : str
        Display name for the covariate.
    cohort_filter : str or None
        If set, restrict to this PND cohort (e.g. ``"p60"``).
    n_perm : int
        Number of permutations for NBS null distribution.
    threshold : float
        |t|-statistic threshold for suprathreshold edges.
    seed : int
        Random seed.

    Returns
    -------
    dict with NBS regression results, or None if skipped.
    """
    cohort_label = cohort_filter if cohort_filter else "pooled"

    df, roi_cols = load_and_prepare_data(wide_csv, exclusion_csv)
    df = df[df["cohort"].isin(["p30", "p60", "p90"])].copy()

    if cohort_filter:
        df = df[df["cohort"] == cohort_filter].copy()

    df_bilateral, bilateral_cols = bilateral_average(df, roi_cols)
    region_cols = [c for c in bilateral_cols if not c.startswith("territory_")]

    # Match with covariate
    df_bilateral["_key"] = df_bilateral["subject"] + "_" + df_bilateral["session"]
    matched = df_bilateral[df_bilateral["_key"].isin(covariate_map)].copy()
    matched["_cov"] = matched["_key"].map(covariate_map)
    matched = matched.dropna(subset=["_cov"])

    if len(matched) < 10:
        logger.warning(
            "Too few matched subjects (%d) for edge regression %s/%s",
            len(matched), metric, cohort_label,
        )
        return None

    X_data = matched[region_cols].values.astype(float)
    cov_arr = matched["_cov"].values.astype(float)

    logger.info(
        "Edge regression %s/%s: n=%d subjects, %d ROIs",
        metric, cohort_label, len(matched), len(region_cols),
    )

    result = network_based_regression(
        data=X_data,
        covariate=cov_arr,
        n_perm=n_perm,
        threshold=threshold,
        seed=seed,
    )

    # Save results
    reg_dir = output_dir / modality / metric / cohort_label
    reg_dir.mkdir(parents=True, exist_ok=True)

    stat_df = pd.DataFrame(
        result["test_stat"], index=region_cols, columns=region_cols
    )
    stat_df.to_csv(reg_dir / "edge_tstats.csv")

    components_json = {
        "covariate": covariate_name,
        "cohort": cohort_label,
        "n_subjects": result["n_subjects"],
        "threshold": threshold,
        "n_perm": n_perm,
        "components": [],
    }
    for comp in result["significant_components"]:
        components_json["components"].append({
            "nodes": comp["nodes"],
            "node_names": [region_cols[n] for n in comp["nodes"]],
            "edges": comp["edges"],
            "edge_names": [
                (region_cols[u], region_cols[v]) for u, v in comp["edges"]
            ],
            "size": comp["size"],
            "pvalue": comp["pvalue"],
        })

    with open(reg_dir / "components.json", "w") as f:
        json.dump(components_json, f, indent=2)

    np.savetxt(reg_dir / "null_distribution.txt", result["null_distribution"])

    # Visualization
    sig_comps = [c for c in result["significant_components"] if c["pvalue"] < 0.05]
    if sig_comps:
        plot_nbs_network(
            result["significant_components"],
            region_cols,
            title=f"Edge Regression: {metric} ~ {covariate_name} ({cohort_label})",
            out_path=reg_dir / "nbs_network.png",
        )

    return result
