"""
Cross-validated regression with permutation testing.

Runs LOOCV with SVR, Ridge, and PLS regressors to test whether joint ROI
patterns predict a continuous or ordinal target variable. When use_pca is
set, PCA is fit inside each LOOCV fold to avoid data leakage, and model
weights are mapped back to ROI space for interpretation. PCA transforms
are pre-computed once (unsupervised) and reused across permutations.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import stats as sp_stats
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVR

from neurofaune.network.classification.visualization import (
    plot_permutation_distribution,
    plot_predicted_vs_actual,
)

logger = logging.getLogger(__name__)


def _loocv_regression(
    reg,
    X: np.ndarray,
    y: np.ndarray,
    n_pca_components: Optional[int] = None,
) -> tuple[float, float, float, np.ndarray]:
    """Run LOOCV for a regressor and return metrics + predictions.

    Parameters
    ----------
    reg : sklearn regressor
        Regressor template (will be cloned per fold).
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Target values.
    n_pca_components : int, optional
        If set, fit PCA inside each fold for dimensionality reduction.

    Returns
    -------
    r_squared : float
    mae : float
    spearman_rho : float
    y_pred : ndarray, shape (n_samples,)
    """
    loo = LeaveOneOut()
    y_pred = np.empty_like(y, dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        if n_pca_components is not None:
            pca = PCA(n_components=n_pca_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        reg_copy = clone(reg)
        reg_copy.fit(X_train, y[train_idx])
        pred = reg_copy.predict(X_test)
        # PLS returns 2D array
        y_pred[test_idx] = np.atleast_1d(pred).ravel()[0]

    return _compute_regression_metrics(y, y_pred)


def _compute_regression_metrics(
    y: np.ndarray, y_pred: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """Compute R², MAE, Spearman rho from true and predicted values."""
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mae = float(np.mean(np.abs(y - y_pred)))
    rho, _ = sp_stats.spearmanr(y, y_pred)
    spearman_rho = float(rho) if not np.isnan(rho) else 0.0

    return r_squared, mae, spearman_rho, y_pred


def _precompute_pca_folds(
    X: np.ndarray, n_components: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pre-compute PCA-transformed train/test splits for each LOOCV fold.

    PCA is unsupervised (depends only on X, not y), so transforms are
    invariant across label permutations and can be computed once.
    """
    loo = LeaveOneOut()
    folds = []
    for train_idx, test_idx in loo.split(X):
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X[train_idx])
        X_test_pca = pca.transform(X[test_idx])
        folds.append((X_train_pca, X_test_pca, train_idx, test_idx))
    return folds


def _loocv_regression_precomputed(
    reg,
    y: np.ndarray,
    pca_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[float, float, float, np.ndarray]:
    """Run LOOCV using pre-computed PCA folds. Only re-fits the regressor."""
    y_pred = np.empty_like(y, dtype=float)
    for X_train_pca, X_test_pca, train_idx, test_idx in pca_folds:
        reg_copy = clone(reg)
        reg_copy.fit(X_train_pca, y[train_idx])
        pred = reg_copy.predict(X_test_pca)
        y_pred[test_idx] = np.atleast_1d(pred).ravel()[0]

    return _compute_regression_metrics(y, y_pred)


def _determine_n_pca(X: np.ndarray, variance_threshold: float = 0.95) -> int:
    """Fit PCA on X and return n_components for the given variance threshold."""
    pca_full = PCA().fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    return int(np.searchsorted(cumvar, variance_threshold) + 1)


def _extract_roi_weights(
    reg,
    pca: PCA,
    reg_name: str,
    feature_names: Sequence[str],
) -> dict:
    """Map regressor weights back to ROI space through PCA components.

    Returns dict with roi_weights (1D array) and top_features (list of tuples).
    """
    if reg_name == "pls":
        # PLS: coef_ is (n_features_pca, 1)
        coef = reg.coef_.ravel()
    elif reg_name == "svr":
        # SVR: coef_ is (1, n_features_pca)
        coef = reg.coef_.ravel()
    else:
        # Ridge: coef_ is (n_features_pca,)
        coef = reg.coef_.ravel()

    roi_weights = (coef @ pca.components_).ravel()

    # Top features by absolute weight
    order = np.argsort(np.abs(roi_weights))[::-1]
    top_features = [(feature_names[i], float(roi_weights[i])) for i in order[:20]]

    return {"roi_weights": roi_weights, "top_features": top_features}


def run_regression(
    X: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    feature_names: Sequence[str],
    n_permutations: int = 1000,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    use_pca: bool = False,
    continuous_target: bool = False,
    dose_labels: Optional[np.ndarray] = None,
    target_name: Optional[str] = None,
) -> dict:
    """LOOCV regression with SVR, Ridge, and PLS + permutation test.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Standardised feature matrix.
    y : ndarray, shape (n_samples,)
        Target values (ordinal ints or continuous floats).
    label_names : sequence of str
        Group names (e.g. ['C', 'L', 'M', 'H']).
    feature_names : sequence of str
        Feature names.
    n_permutations : int
        Number of label permutations for p-value (default 1000).
    seed : int
        Random seed.
    output_dir : Path, optional
        Directory for output plots.
    use_pca : bool
        Whether to apply PCA dimensionality reduction (fit per LOOCV fold).
        When True, model weights are mapped back to ROI space for
        interpretation.
    continuous_target : bool
        If True, pass continuous_target to plot_predicted_vs_actual for
        proper axis handling (no jitter, continuous x-axis).
    dose_labels : ndarray, optional
        Integer dose group per sample for plot colouring when using a
        continuous target.
    target_name : str, optional
        Name of the target variable for axis labels (e.g. 'auc').

    Returns
    -------
    dict with keys per regressor ('svr', 'ridge', 'pls'):
        r_squared : float
        mae : float
        spearman_rho : float
        permutation_p_value : float — p-value for R² exceeding null
        null_distribution : ndarray — null R² values
        y_pred : ndarray — LOOCV predictions
        roi_weights : ndarray (only when use_pca=True)
        pca_n_components : int (only when use_pca=True)
        top_features : list (only when use_pca=True)
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    y_float = y.astype(float)

    # Determine PCA dimensionality if requested
    n_pca = None
    pca_folds = None
    if use_pca:
        n_pca = _determine_n_pca(X)
        n_pca = min(n_pca, X.shape[0] - 1, X.shape[1])
        logger.info("PCA reduction: %d features -> %d components (95%% variance)",
                     X.shape[1], n_pca)
        # Pre-compute PCA transforms for all LOOCV folds (unsupervised,
        # invariant to label permutations)
        logger.info("Pre-computing PCA for %d LOOCV folds...", X.shape[0])
        pca_folds = _precompute_pca_folds(X, n_pca)

    # Determine PLS n_components (min of n_samples-1, n_features, n_classes-1)
    n_classes = len(np.unique(y))
    n_feat_for_pls = n_pca if n_pca is not None else X.shape[1]
    pls_components = min(n_classes - 1, n_feat_for_pls, X.shape[0] - 1)
    pls_components = max(pls_components, 1)

    regressors = {
        "svr": SVR(kernel="linear", C=1.0),
        "ridge": Ridge(alpha=1.0),
        "pls": PLSRegression(n_components=pls_components),
    }

    results = {}
    for reg_name, reg in regressors.items():
        logger.info("Running LOOCV regression for %s...", reg_name)

        # Observed metrics
        if pca_folds is not None:
            r2, mae, rho, y_pred = _loocv_regression_precomputed(reg, y_float, pca_folds)
        else:
            r2, mae, rho, y_pred = _loocv_regression(reg, X, y_float)

        logger.info(
            "  %s: R²=%.3f, MAE=%.3f, ρ=%.3f",
            reg_name, r2, mae, rho,
        )

        # Permutation test on R² (reuse pre-computed PCA folds)
        null_r2 = np.empty(n_permutations)
        for i in range(n_permutations):
            y_perm = rng.permutation(y_float)
            try:
                if pca_folds is not None:
                    null_r2[i], _, _, _ = _loocv_regression_precomputed(reg, y_perm, pca_folds)
                else:
                    null_r2[i], _, _, _ = _loocv_regression(reg, X, y_perm)
            except (ValueError, np.linalg.LinAlgError):
                # PLS can hit numerical singularities on degenerate permutations
                null_r2[i] = np.nan

        valid_null = null_r2[~np.isnan(null_r2)]
        perm_p = float((np.sum(valid_null >= r2) + 1) / (len(valid_null) + 1))
        logger.info("  Permutation p-value (R²): %.4f (n=%d)", perm_p, n_permutations)

        results[reg_name] = {
            "r_squared": r2,
            "mae": mae,
            "spearman_rho": rho,
            "permutation_p_value": perm_p,
            "null_distribution": null_r2,
            "y_pred": y_pred,
        }

        # Weight inversion: fit PCA + regressor on full data for interpretation
        if use_pca:
            pca_interp = PCA(n_components=n_pca).fit(X)
            X_pca = pca_interp.transform(X)
            reg_full = clone(reg).fit(X_pca, y_float)
            weights = _extract_roi_weights(reg_full, pca_interp, reg_name, feature_names)
            results[reg_name]["roi_weights"] = weights["roi_weights"]
            results[reg_name]["top_features"] = weights["top_features"]
            results[reg_name]["pca_n_components"] = n_pca

        # Plots
        if output_dir is not None:
            plot_predicted_vs_actual(
                y_float, y_pred, label_names,
                r_squared=r2, spearman_rho=rho,
                title=f"{reg_name.upper()} — Predicted vs Actual",
                out_path=output_dir / f"{reg_name}_predicted_vs_actual.png",
                continuous_target=continuous_target,
                dose_labels=dose_labels,
                target_name=target_name,
            )
            plot_permutation_distribution(
                null_r2, r2, perm_p,
                title=f"{reg_name.upper()} Permutation Test",
                xlabel="LOOCV R²",
                out_path=output_dir / f"{reg_name}_permutation.png",
            )

    return results


# ---------------------------------------------------------------------------
# Path resolution helper
# ---------------------------------------------------------------------------


def _resolve_regression_paths(
    config_path: Path | None = None,
    roi_dir: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve ROI and regression output paths from config or args.

    Priority: explicit arguments override config values.

    Parameters
    ----------
    config_path : Path, optional
        Study config.yaml.  Derives roi_dir and output_dir from
        ``paths.network.roi`` and ``paths.network.regression``.
    roi_dir : Path, optional
        Explicit ROI directory override.
    output_dir : Path, optional
        Explicit regression output directory override.

    Returns
    -------
    (roi_dir, output_dir) : tuple of Path
    """
    cfg_roi = None
    cfg_output = None

    if config_path is not None:
        from neurofaune.config import load_config, get_config_value

        config = load_config(Path(config_path))
        cfg_roi = get_config_value(config, "paths.network.roi")
        cfg_output = get_config_value(config, "paths.network.regression")
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


# ---------------------------------------------------------------------------
# RegressionAnalysis
# ---------------------------------------------------------------------------


class RegressionAnalysis:
    """Multivariate dose-response regression analysis.

    Follows the same pattern as ``ClassificationAnalysis``: resolve paths
    from config, check for existing results before running, and provide a
    clean Python API that scripts can wrap.
    """

    def __init__(
        self,
        roi_dir: Path,
        output_dir: Path,
        metric: str,
        exclusion_csv: Path | None = None,
        atlas_labels: Path | None = None,
        target: str = "dose",
        force: bool = False,
    ):
        self.roi_dir = Path(roi_dir)
        self.output_dir = Path(output_dir)
        self.metric = metric
        self.exclusion_csv = exclusion_csv
        self.atlas_labels = atlas_labels
        self.target = target
        self.force = force

    @classmethod
    def prepare(
        cls,
        config_path: Path | None = None,
        roi_dir: Path | None = None,
        output_dir: Path | None = None,
        metric: str = "FA",
        exclusion_csv: Path | None = None,
        atlas_labels: Path | None = None,
        target: str = "dose",
        force: bool = False,
    ) -> "RegressionAnalysis":
        """Prepare a regression analysis.

        Parameters
        ----------
        config_path : Path, optional
            Study config.yaml.  Derives roi_dir and output_dir from
            ``paths.network.roi`` and ``paths.network.regression``.
        roi_dir, output_dir : Path, optional
            Explicit path overrides.
        metric : str
            Metric name (e.g. ``"FA"``).
        exclusion_csv : Path, optional
            CSV of sessions to exclude.
        atlas_labels : Path, optional
            SIGMA atlas labels CSV for territory mapping in weight plots.
        target : str
            Target variable: ``"dose"`` (ordinal C=0..H=3) or any column
            name from the wide CSV.
        force : bool
            If True, delete existing results before running.
        """
        resolved_roi, resolved_output = _resolve_regression_paths(
            config_path, roi_dir, output_dir
        )

        # Validate wide CSV exists
        wide_csv = resolved_roi / f"roi_{metric}_wide.csv"
        if not wide_csv.exists():
            raise FileNotFoundError(f"Wide CSV not found: {wide_csv}")

        return cls(
            roi_dir=resolved_roi,
            output_dir=resolved_output,
            metric=metric,
            exclusion_csv=exclusion_csv,
            atlas_labels=atlas_labels,
            target=target,
            force=force,
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _result_dir(
        self, cohort: str | None = None, feature_set: str = "all"
    ) -> Path:
        """Output dir for one combo: ``{output_dir}/{metric}/{cohort}/{feature_set}/``."""
        cohort_label = cohort or "pooled"
        if self.target != "dose":
            return self.output_dir / self.target / self.metric / cohort_label / feature_set
        return self.output_dir / self.metric / cohort_label / feature_set

    def _check_or_clear(
        self, cohort: str | None = None, feature_set: str = "all"
    ) -> None:
        """Check for existing results; error unless force is set.

        If ``self.force`` is True, removes the target directory for a clean
        slate.  If False and the directory has result files, raises
        ``FileExistsError``.
        """
        import shutil

        target = self._result_dir(cohort, feature_set)
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

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        cohort: str | None = None,
        feature_set: str = "all",
        n_permutations: int = 1000,
        seed: int = 42,
    ) -> dict:
        """Run regression pipeline for one metric/cohort/feature_set combo.

        Parameters
        ----------
        cohort : str, optional
            PND cohort filter (e.g. ``"p30"``).  None = pooled.
        feature_set : str
            Feature set name (``"all"``, ``"bilateral"``, ``"territory"``).
        n_permutations : int
            Number of permutations for the regression test.
        seed : int
            Random seed.

        Returns
        -------
        dict
            Summary dictionary with status, metrics, and results.
        """
        import json

        from neurofaune.network.classification.data_prep import prepare_regression_data

        self._check_or_clear(cohort, feature_set)

        cohort_label = cohort or "pooled"
        combo_dir = self._result_dir(cohort, feature_set)
        wide_csv = self.roi_dir / f"roi_{self.metric}_wide.csv"

        logger.info(
            "\n%s\n  Metric: %s | Cohort: %s | Features: %s | Target: %s\n%s",
            "=" * 60, self.metric, cohort_label, feature_set, self.target,
            "=" * 60,
        )

        # Phase 1: Data preparation
        logger.info("[Phase 1] Preparing data...")
        try:
            data = prepare_regression_data(
                wide_csv=wide_csv,
                feature_set=feature_set,
                cohort_filter=cohort if cohort else None,
                exclusion_csv=self.exclusion_csv,
                target=self.target,
            )
        except ValueError as exc:
            logger.warning(
                "Skipping %s/%s/%s: %s",
                self.metric, cohort_label, feature_set, exc,
            )
            return {"status": "skipped", "reason": str(exc)}

        X, y = data["X"], data["y"]
        label_names = data["label_names"]
        feature_names = data["feature_names"]
        dose_labels = data["dose_labels"]
        target_name = data["target_name"]
        n_samples, n_features = X.shape
        continuous = self.target != "dose"

        if n_samples < 10:
            logger.warning("Too few samples (n=%d) — skipping", n_samples)
            return {"status": "skipped", "reason": f"n={n_samples} too small"}

        if len(np.unique(y)) < 2:
            logger.warning("Fewer than 2 unique target values — skipping")
            return {"status": "skipped", "reason": "fewer than 2 unique values"}

        summary: dict = {
            "status": "completed",
            "metric": self.metric,
            "cohort": cohort_label,
            "feature_set": feature_set,
            "target": self.target,
            "target_name": target_name,
            "n_samples": n_samples,
            "n_features": n_features,
            "label_names": label_names,
            "group_sizes": {
                name: int((dose_labels == i).sum())
                for i, name in enumerate(label_names)
            },
        }

        # Phase 2: Regression
        use_pca = feature_set == "all"
        logger.info(
            "[Phase 2] Regression (LOOCV %s + permutation)...", target_name
        )
        reg_dir = combo_dir / "regression"
        reg_results = run_regression(
            X, y, label_names, feature_names,
            n_permutations=n_permutations,
            seed=seed,
            output_dir=reg_dir,
            use_pca=use_pca,
            continuous_target=continuous,
            dose_labels=dose_labels,
            target_name=target_name if continuous else None,
        )

        # Serialise regression results
        reg_json: dict = {}
        for reg_name, result in reg_results.items():
            reg_json[reg_name] = {
                "r_squared": result["r_squared"],
                "mae": result["mae"],
                "spearman_rho": result["spearman_rho"],
                "permutation_p_value": result["permutation_p_value"],
            }
            if "pca_n_components" in result:
                reg_json[reg_name]["pca_n_components"] = result[
                    "pca_n_components"
                ]
                reg_json[reg_name]["top_features"] = result["top_features"]
            summary[f"regression_{reg_name}"] = {
                "r_squared": result["r_squared"],
                "mae": result["mae"],
                "spearman_rho": result["spearman_rho"],
                "permutation_p_value": result["permutation_p_value"],
            }
            if "pca_n_components" in result:
                summary[f"regression_{reg_name}"]["pca_n_components"] = result[
                    "pca_n_components"
                ]

            # Territory-grouped weight plot
            if "roi_weights" in result and self.atlas_labels is not None:
                try:
                    from neurofaune.network.covnet.pipeline import (
                        build_territory_mapping,
                    )
                    from neurofaune.network.classification.visualization import (
                        plot_territory_weights,
                    )

                    roi_to_territory = build_territory_mapping(
                        list(feature_names), self.atlas_labels,
                    )
                    plot_territory_weights(
                        result["roi_weights"],
                        feature_names,
                        roi_to_territory,
                        title=(
                            f"{reg_name.upper()} — {self.metric} "
                            f"{cohort_label} weights"
                        ),
                        out_path=reg_dir / f"{reg_name}_territory_weights.png",
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to plot territory weights for %s: %s",
                        reg_name, exc,
                    )

        with open(reg_dir / "regression.json", "w") as f:
            json.dump(reg_json, f, indent=2)

        # Save per-combo summary
        combo_dir.mkdir(parents=True, exist_ok=True)
        with open(combo_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    # ------------------------------------------------------------------
    # Design description
    # ------------------------------------------------------------------

    def write_design_description(
        self,
        feature_sets: list[str],
        n_permutations: int,
        seed: int,
    ) -> None:
        """Write human-readable analysis description to output_dir.

        Parameters
        ----------
        feature_sets : list[str]
            Feature sets being analysed (e.g. ``["all", "bilateral"]``).
        n_permutations : int
            Number of permutations for the regression test.
        seed : int
            Random seed.
        """
        from datetime import datetime

        if self.target == "dose":
            target_desc = "Dose as ordinal: C=0, L=1, M=2, H=3"
        else:
            target_desc = f"{self.target} (continuous column from wide CSV)"

        lines = [
            "ANALYSIS DESCRIPTION",
            "====================",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis: {self.target.upper() if self.target != 'dose' else 'Dose'}-Response Regression",
            "",
            "DATA SOURCE",
            "-----------",
            f"ROI directory: {self.roi_dir}",
            f"Exclusion list: {self.exclusion_csv or 'None'}",
            f"Metric: {self.metric}",
            f"Feature sets: {', '.join(feature_sets)}",
            "",
            "EXPERIMENTAL DESIGN",
            "-------------------",
            f"Target: {target_desc}",
            "Cohorts analysed: p30, p60, p90, and pooled",
            "",
            "FEATURE SETS",
            "------------",
        ]

        if "bilateral" in feature_sets:
            lines.append(
                "- bilateral: Bilateral-averaged region ROIs (~50 features)"
            )
            lines.append("  L/R ROI pairs averaged, territories excluded")

        if "territory" in feature_sets:
            lines.append("- territory: Territory aggregate ROIs (~15 features)")
            lines.append("  Coarser anatomical groupings")

        if "all" in feature_sets:
            lines.append("- all: All individual L/R ROIs (~234 features)")
            lines.append(
                "  PCA reduction to 95% variance inside each LOOCV fold"
            )
            lines.append(
                "  Model weights mapped back to ROI space for interpretation"
            )

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
            f"- {n_permutations} label shuffles per regressor",
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
            f"Permutations: {n_permutations}",
            f"Random seed: {seed}",
        ])

        output_path = self.output_dir / "design_description.txt"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))
        logger.info("Saved analysis description: %s", output_path)
