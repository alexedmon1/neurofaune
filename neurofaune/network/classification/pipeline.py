"""Classification analysis pipeline.

Typical usage::

    from pathlib import Path
    from neurofaune.network.classification.pipeline import ClassificationAnalysis

    analysis = ClassificationAnalysis.prepare(
        config_path=Path("config.yaml"),
        metric="FA", force=True,
    )
    analysis.run(cohort="p30", feature_set="all", n_permutations=1000)
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _resolve_classification_paths(
    config_path: Path | None = None,
    roi_dir: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve ROI and classification output paths from config or args.

    Priority: explicit arguments override config values.

    Parameters
    ----------
    config_path : Path, optional
        Study config.yaml.  Derives roi_dir and output_dir from
        ``paths.network.roi`` and ``paths.network.classification``.
    roi_dir : Path, optional
        Explicit ROI directory override.
    output_dir : Path, optional
        Explicit classification output directory override.

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
        cfg_output = get_config_value(config, "paths.network.classification")
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
# ClassificationAnalysis
# ---------------------------------------------------------------------------


class ClassificationAnalysis:
    """Multivariate group classification analysis.

    Follows the same pattern as ``CovNetAnalysis`` and
    ``EdgeRegressionAnalysis``: resolve paths from config, check for
    existing results before running, and provide a clean Python API that
    scripts can wrap.
    """

    def __init__(
        self,
        roi_dir: Path,
        output_dir: Path,
        metric: str,
        exclusion_csv: Path | None = None,
        atlas_labels: Path | None = None,
        target: str = "dose",
    ):
        self.roi_dir = Path(roi_dir)
        self.output_dir = Path(output_dir)
        self.metric = metric
        self.exclusion_csv = exclusion_csv
        self.atlas_labels = atlas_labels
        self.target = target
        self.force = False

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
    ) -> "ClassificationAnalysis":
        """Prepare a classification analysis.

        Parameters
        ----------
        config_path : Path, optional
            Study config.yaml.  Derives roi_dir and output_dir from
            ``paths.network.roi`` and ``paths.network.classification``.
        roi_dir, output_dir : Path, optional
            Explicit path overrides.
        metric : str
            Metric name (e.g. ``"FA"``).
        exclusion_csv : Path, optional
            CSV of sessions to exclude.
        atlas_labels : Path, optional
            SIGMA atlas labels CSV for territory mapping in weight plots.
        target : str
            Target variable for group labels: ``"dose"`` (C/L/M/H) or any
            column name from the wide CSV.
        force : bool
            If True, delete existing results before running.
        """
        resolved_roi, resolved_output = _resolve_classification_paths(
            config_path, roi_dir, output_dir
        )

        # Validate wide CSV exists
        wide_csv = resolved_roi / f"roi_{metric}_wide.csv"
        if not wide_csv.exists():
            raise FileNotFoundError(f"Wide CSV not found: {wide_csv}")

        inst = cls(
            roi_dir=resolved_roi,
            output_dir=resolved_output,
            metric=metric,
            exclusion_csv=exclusion_csv,
            atlas_labels=atlas_labels,
            target=target,
        )
        inst.force = force
        return inst

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
        skip_manova: bool = False,
        skip_classification: bool = False,
    ) -> dict:
        """Run full classification pipeline for one metric/cohort/feature_set.

        This mirrors the ``run_single_analysis()`` function from the CLI
        script, but operates on state stored in ``self``.

        Parameters
        ----------
        cohort : str, optional
            PND cohort filter (e.g. ``"p30"``).  None = pooled.
        feature_set : str
            Feature set name (``"all"``, ``"bilateral"``, ``"territory"``).
        n_permutations : int
            Number of permutations for classification test.
        seed : int
            Random seed.
        skip_manova : bool
            Skip optional MANOVA test.
        skip_classification : bool
            Skip LOOCV classification (only run PERMANOVA, PCA, LDA).

        Returns
        -------
        dict
            Summary dictionary with status, metrics, and results.
        """
        from neurofaune.network.classification.classifiers import run_classification
        from neurofaune.network.classification.data_prep import prepare_classification_data
        from neurofaune.network.classification.lda import run_lda
        from neurofaune.network.classification.omnibus import run_manova, run_permanova
        from neurofaune.network.classification.pca import run_pca
        from neurofaune.network.classification.visualization import (
            plot_permutation_distribution,
        )

        self._check_or_clear(cohort, feature_set)

        cohort_label = cohort or "pooled"
        combo_dir = self._result_dir(cohort, feature_set)
        wide_csv = self.roi_dir / f"roi_{self.metric}_wide.csv"

        logger.info(
            "\n%s\n  Metric: %s | Cohort: %s | Features: %s\n%s",
            "=" * 60, self.metric, cohort_label, feature_set, "=" * 60,
        )

        # Phase 1: Data preparation
        logger.info("[Phase 1] Preparing data...")
        try:
            data = prepare_classification_data(
                wide_csv=wide_csv,
                feature_set=feature_set,
                cohort_filter=cohort if cohort else None,
                exclusion_csv=self.exclusion_csv,
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
        n_samples, n_features = X.shape

        if n_samples < 10:
            logger.warning("Too few samples (n=%d) — skipping", n_samples)
            return {"status": "skipped", "reason": f"n={n_samples} too small"}

        if len(np.unique(y)) < 2:
            logger.warning("Fewer than 2 groups — skipping")
            return {"status": "skipped", "reason": "fewer than 2 groups"}

        summary: dict = {
            "status": "completed",
            "metric": self.metric,
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

        permanova = run_permanova(
            X, y, label_names,
            n_perm=min(n_permutations * 10, 9999),
            seed=seed,
        )
        summary["permanova"] = {
            "pseudo_f": permanova["pseudo_f"],
            "p_value": permanova["p_value"],
            "r_squared": permanova["r_squared"],
        }

        # Save PERMANOVA results
        permanova_out = {
            k: v for k, v in permanova.items() if k != "null_distribution"
        }
        with open(omnibus_dir / "permanova.json", "w") as f:
            json.dump(permanova_out, f, indent=2)

        # Permutation null plot
        plot_permutation_distribution(
            permanova["null_distribution"],
            permanova["pseudo_f"],
            permanova["p_value"],
            title=f"PERMANOVA — {self.metric} {cohort_label} {feature_set}",
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
            if len(pca_results["explained_variance_ratio"]) > 1
            else 0.0,
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
            "explained_variance_ratio": lda_results[
                "explained_variance_ratio"
            ].tolist(),
            "top_features": lda_results["top_features"],
        }
        with open(lda_dir / "results.json", "w") as f:
            json.dump(lda_json, f, indent=2)

        # Phase 5: Classification
        use_pca = feature_set == "all"
        if not skip_classification:
            logger.info("[Phase 5] Classification (LOOCV + permutation)...")
            clf_dir = combo_dir / "classification"
            clf_results = run_classification(
                X, y, label_names, feature_names,
                n_permutations=n_permutations,
                seed=seed,
                output_dir=clf_dir,
                use_pca=use_pca,
            )

            # Serialise classification results
            clf_json: dict = {}
            for clf_name, result in clf_results.items():
                clf_json[clf_name] = {
                    "accuracy": result["accuracy"],
                    "balanced_accuracy": result["balanced_accuracy"],
                    "permutation_p_value": result["permutation_p_value"],
                    "per_class_accuracy": result["per_class_accuracy"],
                    "confusion_matrix": result["confusion_matrix"].tolist(),
                }
                if "pca_n_components" in result:
                    clf_json[clf_name]["pca_n_components"] = result[
                        "pca_n_components"
                    ]
                    clf_json[clf_name]["top_features"] = result["top_features"]
                summary[f"classification_{clf_name}"] = {
                    "accuracy": result["accuracy"],
                    "balanced_accuracy": result["balanced_accuracy"],
                    "permutation_p_value": result["permutation_p_value"],
                }
                if "pca_n_components" in result:
                    summary[f"classification_{clf_name}"]["pca_n_components"] = (
                        result["pca_n_components"]
                    )

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
                                f"{clf_name.upper()} — {self.metric} "
                                f"{cohort_label} weights"
                            ),
                            out_path=clf_dir / f"{clf_name}_territory_weights.png",
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to plot territory weights for %s: %s",
                            clf_name, exc,
                        )

            with open(clf_dir / "classification.json", "w") as f:
                json.dump(clf_json, f, indent=2)
        else:
            logger.info(
                "[Phase 5] Skipping classification (--skip-classification)"
            )

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
        skip_manova: bool = False,
        skip_classification: bool = False,
    ) -> None:
        """Write human-readable analysis description to output_dir.

        Parameters
        ----------
        feature_sets : list[str]
            Feature sets being analysed (e.g. ``["all", "bilateral"]``).
        n_permutations : int
            Number of permutations for the classification test.
        seed : int
            Random seed.
        skip_manova : bool
            Whether MANOVA was skipped.
        skip_classification : bool
            Whether classification was skipped.
        """
        lines = [
            "ANALYSIS DESCRIPTION",
            "====================",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Analysis: Multivariate Group Classification",
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
            f"Grouping: {self.target} "
            f"({'C, L, M, H — 4 groups' if self.target == 'dose' else 'from wide CSV column'})",
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
            "1. PERMANOVA (Permutational MANOVA)",
            "   - Non-parametric omnibus test using Euclidean distances",
            f"   - Permutations: up to {min(n_permutations * 10, 9999)}",
            "   - Reports pseudo-F, R², and permutation p-value",
            "",
        ])

        if not skip_manova:
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

        if not skip_classification:
            lines.extend([
                "5. Classification (LOOCV + permutation test)",
                "   - Linear SVM (C=1.0) and multinomial logistic regression",
                "   - Leave-one-out cross-validation (standard for n < 100)",
                f"   - Permutation test: {n_permutations} shuffles",
                "   - Reports accuracy, balanced accuracy, confusion matrix",
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
            f"Permutations (classification): {n_permutations}",
            f"Permutations (PERMANOVA): up to {min(n_permutations * 10, 9999)}",
            f"Random seed: {seed}",
            "",
            "NOTE: For dose-response regression (SVR, Ridge, PLS), see",
            "run_regression_analysis.py",
        ])

        output_path = self.output_dir / "design_description.txt"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))
        logger.info("Saved analysis description: %s", output_path)
