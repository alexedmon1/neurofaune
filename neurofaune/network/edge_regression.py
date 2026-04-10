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

Typical usage::

    from pathlib import Path
    from neurofaune.network.edge_regression import EdgeRegressionAnalysis

    analysis = EdgeRegressionAnalysis.prepare(
        config_path=Path("config.yaml"),
        modality="dwi", metric="FA",
        target="log_auc", auc_csv=Path("auc_lookup.csv"),
        force=True,
    )
    analysis.run(cohort="p30", n_perm=1000, threshold=3.0, seed=42)
"""

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from neurofaune.network.covnet.nbs import network_based_regression
from neurofaune.network.covnet.visualization import plot_nbs_network
from neurofaune.network.matrices import load_and_prepare_data

logger = logging.getLogger(__name__)


def _resolve_edge_regression_paths(
    config_path: Path | None = None,
    roi_dir: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve ROI and edge regression output paths from config or args."""
    cfg_roi = None
    cfg_output = None

    if config_path is not None:
        from neurofaune.config import load_config, get_config_value

        config = load_config(Path(config_path))
        cfg_roi = get_config_value(config, "paths.network.roi")
        cfg_output = get_config_value(config, "paths.network.edge_regression")
        if cfg_roi is not None:
            cfg_roi = Path(cfg_roi)
        if cfg_output is not None:
            cfg_output = Path(cfg_output)

    resolved_roi = roi_dir if roi_dir is not None else cfg_roi
    resolved_output = output_dir if output_dir is not None else cfg_output

    if resolved_roi is None:
        raise ValueError(
            "ROI directory not specified. Provide config_path or roi_dir."
        )
    if resolved_output is None:
        raise ValueError(
            "Output directory not specified. Provide config_path or output_dir."
        )

    return Path(resolved_roi), Path(resolved_output)


class EdgeRegressionAnalysis:
    """Edge-level regression analysis for continuous targets.

    Follows the same pattern as ``CovNetAnalysis``: resolve paths from
    config, check for existing results before running, and provide a
    clean Python API that scripts can wrap.
    """

    def __init__(
        self,
        roi_dir: Path,
        output_dir: Path,
        modality: str,
        metric: str,
        sex: str | None = None,
        force: bool = False,
    ):
        self.roi_dir = Path(roi_dir)
        self.output_dir = Path(output_dir)
        self.modality = modality
        self.metric = metric
        self.sex = sex
        self.force = force
        self.covariate_map: dict[str, float] | None = None
        self.covariate_name: str = ""
        self.exclusion_csv: Path | None = None

    @classmethod
    def prepare(
        cls,
        config_path: Path | None = None,
        roi_dir: Path | None = None,
        output_dir: Path | None = None,
        modality: str = "",
        metric: str = "",
        target: str = "log_auc",
        auc_csv: Path | None = None,
        exclusion_csv: Path | None = None,
        sex: str | None = None,
        force: bool = False,
    ) -> "EdgeRegressionAnalysis":
        """Prepare an edge regression analysis.

        Parameters
        ----------
        config_path : Path, optional
            Study config.yaml. Derives roi_dir and output_dir from
            ``paths.network.roi`` and ``paths.network.edge_regression``.
        roi_dir, output_dir : Path, optional
            Explicit path overrides.
        modality : str
            Modality name (e.g. ``"dwi"``).
        metric : str
            Metric name (e.g. ``"FA"``).
        target : str
            Continuous target column name (e.g. ``"log_auc"``).
        auc_csv : Path, optional
            CSV with subject, session, and target columns. If None,
            the target column is read from the ROI wide CSV.
        exclusion_csv : Path, optional
            CSV of sessions to exclude.
        sex : str, optional
            If set (``"F"`` or ``"M"``), restrict to one sex.
        force : bool
            If True, delete existing results before running.
        """
        resolved_roi, resolved_output = _resolve_edge_regression_paths(
            config_path, roi_dir, output_dir
        )

        # Adjust output for sex stratification
        if sex:
            resolved_output = resolved_output / f"sex_{sex}"

        inst = cls(
            roi_dir=resolved_roi,
            output_dir=resolved_output,
            modality=modality,
            metric=metric,
            sex=sex,
            force=force,
        )
        inst.exclusion_csv = exclusion_csv

        # Build covariate map
        inst.covariate_name = target
        if auc_csv is not None:
            auc_df = pd.read_csv(auc_csv)
            if target not in auc_df.columns:
                raise ValueError(
                    f"Target column {target!r} not found in {auc_csv}. "
                    f"Available: {list(auc_df.columns)}"
                )
            if not {"subject", "session"}.issubset(auc_df.columns):
                raise ValueError(
                    "AUC CSV must have 'subject' and 'session' columns"
                )
            inst.covariate_map = dict(
                zip(
                    auc_df["subject"] + "_" + auc_df["session"],
                    auc_df[target].astype(float),
                )
            )
            inst.covariate_map = {
                k: v for k, v in inst.covariate_map.items()
                if not np.isnan(v)
            }
            logger.info(
                "Loaded %d %s values from %s",
                len(inst.covariate_map), target, auc_csv,
            )
            if target == "log_auc":
                inst.covariate_name = "log(1+AUC)"

        wide_csv = resolved_roi / f"roi_{metric}_wide.csv"
        if not wide_csv.exists():
            raise FileNotFoundError(f"Wide CSV not found: {wide_csv}")

        return inst

    def _result_dir(self, cohort: str | None = None) -> Path:
        """Output directory for a specific cohort."""
        cohort_label = cohort if cohort else "pooled"
        return self.output_dir / self.modality / self.metric / cohort_label

    def _check_or_clear(self, cohort: str | None = None) -> None:
        """Check for existing results; error unless force is set."""
        target = self._result_dir(cohort)
        if not target.exists():
            return

        result_files = [f for f in target.rglob("*") if f.is_file()]
        if not result_files:
            return

        if not self.force:
            file_list = "\n  ".join(str(f) for f in result_files[:10])
            extra = (
                f"\n  ... and {len(result_files) - 10} more"
                if len(result_files) > 10
                else ""
            )
            raise FileExistsError(
                f"Results already exist at {target} "
                f"({len(result_files)} files):\n  {file_list}{extra}\n\n"
                f"Use force=True (or --force) to delete existing results "
                f"and rerun."
            )

        logger.warning("--force: removing existing results at %s", target)
        shutil.rmtree(target)

    def run(
        self,
        cohort: str | None = None,
        n_perm: int = 1000,
        threshold: float = 3.0,
        seed: int = 42,
    ) -> dict | None:
        """Run edge regression for one cohort.

        Parameters
        ----------
        cohort : str, optional
            PND cohort filter (e.g. ``"p30"``). None = pooled.
        n_perm : int
            Number of permutations.
        threshold : float
            |t|-statistic threshold for suprathreshold edges.
        seed : int
            Random seed.

        Returns
        -------
        dict with NBS regression results, or None if skipped.
        """
        self._check_or_clear(cohort)

        wide_csv = self.roi_dir / f"roi_{self.metric}_wide.csv"
        return run_edge_regression(
            wide_csv=wide_csv,
            exclusion_csv=self.exclusion_csv,
            output_dir=self.output_dir,
            modality=self.modality,
            metric=self.metric,
            covariate_map=self.covariate_map,
            covariate_name=self.covariate_name,
            cohort_filter=cohort,
            sex_filter=self.sex,
            n_perm=n_perm,
            threshold=threshold,
            seed=seed,
        )


def run_edge_regression(
    wide_csv: Path,
    exclusion_csv: Path | None,
    output_dir: Path,
    modality: str,
    metric: str,
    covariate_map: dict[str, float] | None,
    covariate_name: str = "log(1+AUC)",
    cohort_filter: str | None = None,
    sex_filter: str | None = None,
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
    covariate_map : dict[str, float] or None
        Mapping from subject key (``"sub-Rat001_ses-p60"``) to covariate value.
        If None, the target column is read directly from the wide CSV.
    covariate_name : str
        Display name for the covariate.
    cohort_filter : str or None
        If set, restrict to this PND cohort (e.g. ``"p60"``).
    sex_filter : str or None
        If set (``"F"`` or ``"M"``), restrict to one sex.
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

    if sex_filter is not None:
        df = df[df["sex"] == sex_filter].reset_index(drop=True)
        logger.info("Sex filter '%s': %d subjects remaining", sex_filter, len(df))

    region_cols = [c for c in roi_cols if not c.startswith("territory_")]

    # Match with covariate
    if covariate_map is not None:
        df["_key"] = df["subject"] + "_" + df["session"]
        matched = df[df["_key"].isin(covariate_map)].copy()
        matched["_cov"] = matched["_key"].map(covariate_map)
    elif covariate_name in df.columns:
        matched = df.copy()
        matched["_cov"] = matched[covariate_name].astype(float)
    else:
        logger.warning(
            "No covariate_map and column '%s' not in data, skipping %s/%s",
            covariate_name, metric, cohort_label,
        )
        return None
    matched = matched.dropna(subset=["_cov"])

    if len(matched) < 10:
        logger.warning(
            "Too few matched subjects (%d) for edge regression %s/%s",
            len(matched), metric, cohort_label,
        )
        return None

    # Drop subjects with any NaN in region ROIs (out-of-FOV zeros replaced
    # with NaN by load_and_prepare_data) — regression cannot handle NaN.
    matched = matched.dropna(subset=region_cols)
    if len(matched) < 10:
        logger.warning(
            "Too few subjects (%d) after NaN drop for edge regression %s/%s",
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
