"""MVPA analysis pipeline.

Typical usage::

    from pathlib import Path
    from neurofaune.analysis.mvpa.pipeline import MVPAAnalysis

    analysis = MVPAAnalysis.prepare(
        config_path=Path("config.yaml"),
        modality="dwi", metric="FA", force=True,
    )
    analysis.run(cohort="p30", design="per_pnd_p30", n_permutations=1000)
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _resolve_mvpa_paths(
    config_path: Path | None = None,
    output_dir: Path | None = None,
    derivatives_dir: Path | None = None,
    design_dir: Path | None = None,
    mask: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Resolve MVPA paths from config or explicit arguments.

    Priority: explicit arguments override config values.

    Parameters
    ----------
    config_path : Path, optional
        Study config.yaml.  Derives output_dir from
        ``paths.analysis.mvpa``, derivatives_dir from
        ``paths.derivatives``, and mask from
        ``atlas.study_space.brain_mask``.
    output_dir : Path, optional
        Explicit output directory override.
    derivatives_dir : Path, optional
        Explicit derivatives directory override.
    design_dir : Path, optional
        Explicit design directory override.
    mask : Path, optional
        Explicit brain mask NIfTI override.

    Returns
    -------
    (output_dir, derivatives_dir, design_dir, mask) : tuple of Path
    """
    cfg_output = None
    cfg_derivatives = None
    cfg_design = None
    cfg_mask = None

    if config_path is not None:
        from neurofaune.config import load_config, get_config_value

        config = load_config(Path(config_path))
        cfg_output = get_config_value(config, "paths.analysis.mvpa")
        cfg_derivatives = get_config_value(config, "paths.derivatives")
        cfg_mask = get_config_value(config, "atlas.study_space.brain_mask")
        if cfg_output is not None:
            cfg_output = Path(cfg_output)
            # Default design_dir is a subdirectory of mvpa output
            cfg_design = cfg_output / "designs"
        if cfg_derivatives is not None:
            cfg_derivatives = Path(cfg_derivatives)
        if cfg_mask is not None:
            cfg_mask = Path(cfg_mask)

    resolved_output = output_dir if output_dir is not None else cfg_output
    resolved_derivatives = derivatives_dir if derivatives_dir is not None else cfg_derivatives
    resolved_design = design_dir if design_dir is not None else cfg_design
    resolved_mask = mask if mask is not None else cfg_mask

    if resolved_output is None:
        raise ValueError(
            "Output directory not specified. Provide config_path or output_dir."
        )
    if resolved_derivatives is None:
        raise ValueError(
            "Derivatives directory not specified. Provide config_path or "
            "derivatives_dir."
        )
    if resolved_design is None:
        raise ValueError(
            "Design directory not specified. Provide config_path or design_dir."
        )
    if resolved_mask is None:
        raise ValueError(
            "Brain mask not specified. Provide config_path or mask."
        )

    return (
        Path(resolved_output),
        Path(resolved_derivatives),
        Path(resolved_design),
        Path(resolved_mask),
    )


def encode_labels(labels, analysis_type):
    """Encode labels for sklearn: strings to integers or floats.

    For classification: unique string labels to integer codes.
    For regression: labels are already numeric.

    Returns (encoded_labels, label_names).
    """
    if analysis_type == "classification":
        unique_labels = sorted(set(labels))
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        encoded = np.array([label_map[lbl] for lbl in labels])
        return encoded, unique_labels
    else:
        encoded = np.array(labels, dtype=float)
        return encoded, None


def discover_designs(
    design_dir: Path,
    skip_regression: bool = False,
    searchlight_only: bool = False,
) -> tuple[Dict[str, Path], Dict[str, Path]]:
    """Discover and classify designs in a design directory.

    Parameters
    ----------
    design_dir : Path
        Directory containing design subdirectories.
    skip_regression : bool
        If True, return empty regression designs.
    searchlight_only : bool
        If True, return empty categorical designs (only regression).

    Returns
    -------
    (categorical_designs, regression_designs) : tuple of dicts
        Each maps design_name -> design_dir Path.
    """
    all_designs = {}
    for d in sorted(design_dir.iterdir()):
        if d.is_dir() and (d / "design_summary.json").exists():
            all_designs[d.name] = d

    categorical_designs = {}
    regression_designs = {}

    for k, v in all_designs.items():
        if "_regression_" in k or k.endswith("_regression_pooled"):
            regression_designs[k] = v
        elif k.startswith("per_pnd_"):
            categorical_designs[k] = v
        elif k == "pooled":
            # Pooled classification mixes cohorts where dose categories
            # have different meanings -- skip it
            logger.info(
                "Skipping pooled categorical design "
                "(dose groups are cohort-specific)"
            )
            continue
        else:
            if "regression" in k or "response" in k:
                regression_designs[k] = v
            else:
                categorical_designs[k] = v

    if searchlight_only:
        categorical_designs = {}
    if skip_regression:
        regression_designs = {}

    return categorical_designs, regression_designs


# ---------------------------------------------------------------------------
# MVPAAnalysis
# ---------------------------------------------------------------------------


class MVPAAnalysis:
    """Multi-Voxel Pattern Analysis pipeline.

    Whole-brain decoding and searchlight mapping using voxel-level
    SIGMA-space metrics. Follows the same pattern as
    ``ClassificationAnalysis`` and ``RegressionAnalysis``: resolve paths
    from config, check for existing results before running, and provide a
    clean Python API that scripts can wrap.

    Typical usage::

        analysis = MVPAAnalysis.prepare(
            config_path=Path("config.yaml"),
            metrics=["FA", "MD"],
            force=True,
        )
        analysis.run(metric="FA", design_name="per_pnd_p30",
                     analysis_type="classification")
    """

    def __init__(
        self,
        output_dir: Path,
        derivatives_dir: Path,
        design_dir: Path,
        mask_path: Path,
        metrics: List[str],
        n_permutations: int = 1000,
        cv_folds: int = 5,
        screening_percentile: int = 20,
        seed: int = 42,
        run_searchlight: bool = False,
        searchlight_only: bool = False,
        searchlight_radius: float = 2.0,
        searchlight_cv_folds: int = 3,
        searchlight_n_jobs: int = 1,
        searchlight_n_perm_fwer: int = 100,
        skip_regression: bool = False,
        force: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.derivatives_dir = Path(derivatives_dir)
        self.design_dir = Path(design_dir)
        self.mask_path = Path(mask_path)
        self.metrics = metrics
        self.n_permutations = n_permutations
        self.cv_folds = cv_folds
        self.screening_percentile = screening_percentile
        self.seed = seed
        self.run_searchlight = run_searchlight
        self.searchlight_only = searchlight_only
        self.searchlight_radius = searchlight_radius
        self.searchlight_cv_folds = searchlight_cv_folds
        self.searchlight_n_jobs = searchlight_n_jobs
        self.searchlight_n_perm_fwer = searchlight_n_perm_fwer
        self.skip_regression = skip_regression
        self.force = force
        self._mask_img: Optional[nib.Nifti1Image] = None

    @property
    def mask_img(self) -> nib.Nifti1Image:
        """Lazy-load brain mask."""
        if self._mask_img is None:
            self._mask_img = nib.load(self.mask_path)
            logger.info(
                "Loaded brain mask: %s (shape: %s)",
                self.mask_path, self._mask_img.shape,
            )
        return self._mask_img

    @classmethod
    def prepare(
        cls,
        config_path: Path | None = None,
        output_dir: Path | None = None,
        derivatives_dir: Path | None = None,
        design_dir: Path | None = None,
        mask: Path | None = None,
        metrics: List[str] | None = None,
        n_permutations: int = 1000,
        cv_folds: int = 5,
        screening_percentile: int = 20,
        seed: int = 42,
        run_searchlight: bool = False,
        searchlight_only: bool = False,
        searchlight_radius: float = 2.0,
        searchlight_cv_folds: int = 3,
        searchlight_n_jobs: int = 1,
        searchlight_n_perm_fwer: int = 100,
        skip_regression: bool = False,
        force: bool = False,
    ) -> "MVPAAnalysis":
        """Prepare an MVPA analysis.

        Parameters
        ----------
        config_path : Path, optional
            Study config.yaml.  Derives output_dir from
            ``paths.analysis.mvpa``, derivatives_dir from
            ``paths.derivatives``, and mask from
            ``atlas.study_space.brain_mask``.
        output_dir : Path, optional
            Explicit output directory override.
        derivatives_dir : Path, optional
            Explicit derivatives directory override.
        design_dir : Path, optional
            Explicit design directory override.
        mask : Path, optional
            Explicit brain mask NIfTI override.
        metrics : list of str, optional
            Metrics to analyse (default: FA, MD, AD, RD).
        n_permutations : int
            Permutations for whole-brain decoding p-value (default: 1000).
        cv_folds : int
            Cross-validation folds for whole-brain (default: 5).
        screening_percentile : int
            ANOVA feature screening percentile (default: 20).
        seed : int
            Random seed.
        run_searchlight : bool
            Enable searchlight mapping.
        searchlight_only : bool
            Run searchlight only (skip whole-brain decoding).
        searchlight_radius : float
            Searchlight sphere radius in mm (default: 2.0).
        searchlight_cv_folds : int
            Cross-validation folds for searchlight (default: 3).
        searchlight_n_jobs : int
            Parallel jobs for searchlight (default: 1).
        searchlight_n_perm_fwer : int
            Label permutations for FWER correction (default: 100).
        skip_regression : bool
            Skip regression designs, only run classification.
        force : bool
            If True, delete existing results before running.
        """
        resolved_output, resolved_derivatives, resolved_design, resolved_mask = (
            _resolve_mvpa_paths(
                config_path, output_dir, derivatives_dir, design_dir, mask,
            )
        )

        # Validate inputs
        if not resolved_derivatives.exists():
            raise FileNotFoundError(
                f"Derivatives directory not found: {resolved_derivatives}"
            )
        if not resolved_design.exists():
            raise FileNotFoundError(
                f"Design directory not found: {resolved_design}"
            )
        if not resolved_mask.exists():
            raise FileNotFoundError(
                f"Brain mask not found: {resolved_mask}"
            )

        if metrics is None:
            metrics = ["FA", "MD", "AD", "RD"]

        # searchlight_only implies run_searchlight
        if searchlight_only:
            run_searchlight = True

        return cls(
            output_dir=resolved_output,
            derivatives_dir=resolved_derivatives,
            design_dir=resolved_design,
            mask_path=resolved_mask,
            metrics=metrics,
            n_permutations=n_permutations,
            cv_folds=cv_folds,
            screening_percentile=screening_percentile,
            seed=seed,
            run_searchlight=run_searchlight,
            searchlight_only=searchlight_only,
            searchlight_radius=searchlight_radius,
            searchlight_cv_folds=searchlight_cv_folds,
            searchlight_n_jobs=searchlight_n_jobs,
            searchlight_n_perm_fwer=searchlight_n_perm_fwer,
            skip_regression=skip_regression,
            force=force,
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _result_dir(
        self, metric: str, design_name: str, analysis_type: str,
    ) -> Path:
        """Output dir for one combo: ``{output_dir}/{metric}/{design}/{type}/``."""
        return self.output_dir / metric / design_name / analysis_type

    def _check_or_clear(
        self, metric: str, design_name: str, analysis_type: str,
    ) -> None:
        """Check for existing results; error unless force is set.

        If ``self.force`` is True, removes the target directory for a clean
        slate.  If False and the directory has result files, raises
        ``FileExistsError``.
        """
        target = self._result_dir(metric, design_name, analysis_type)
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
        metric: str,
        design_name: str,
        analysis_type: str,
    ) -> dict:
        """Run MVPA for one metric + design + analysis_type combination.

        If ``self.searchlight_only`` is True, skips whole-brain decoding
        and only runs searchlight. This is the appropriate mode for
        continuous targets where whole-brain Decoder is not suitable.

        Parameters
        ----------
        metric : str
            DTI/MSME metric name (e.g. ``"FA"``, ``"MD"``).
        design_name : str
            Design subdirectory name (e.g. ``"per_pnd_p30"``).
        analysis_type : str
            ``"classification"`` or ``"regression"``.

        Returns
        -------
        dict
            Summary dictionary with status, metrics, and results.
        """
        from neurofaune.analysis.mvpa.data_loader import (
            align_data_to_design,
            discover_sigma_images,
            load_design,
            load_mvpa_data,
        )
        from neurofaune.analysis.mvpa.searchlight import run_searchlight
        from neurofaune.analysis.mvpa.visualization import (
            plot_decoding_scores,
            plot_glass_brain,
            plot_regression_brain,
            plot_searchlight_map,
            plot_weight_map,
        )
        from neurofaune.analysis.mvpa.whole_brain import run_whole_brain_decoding

        self._check_or_clear(metric, design_name, analysis_type)

        combo_label = f"{metric}/{design_name}/{analysis_type}"
        logger.info("\n%s\n  %s\n%s", "=" * 60, combo_label, "=" * 60)

        # Discover images for this metric
        images = discover_sigma_images(self.derivatives_dir, metric)
        if not images:
            logger.warning("No SIGMA-space %s images found, skipping", metric)
            return {"status": "skipped", "reason": f"no {metric} images"}

        # Load design and align
        design_path = self.design_dir / design_name
        try:
            design = load_design(design_path)
        except FileNotFoundError as exc:
            logger.warning("Design not found: %s, skipping", exc)
            return {"status": "skipped", "reason": str(exc)}

        aligned = align_data_to_design(images, design)
        if aligned["n_matched"] < 5:
            logger.warning(
                "Too few matched subjects (%d), skipping %s",
                aligned["n_matched"], combo_label,
            )
            return {
                "status": "skipped",
                "reason": f"n={aligned['n_matched']} too small",
            }

        # Load 4D data
        mvpa_data = load_mvpa_data(aligned["image_info"], mask_img=self.mask_img)
        images_4d = mvpa_data["images_4d"]

        # Encode labels
        labels = aligned["labels"]
        encoded_labels, label_names = encode_labels(labels, analysis_type)

        # Check label diversity
        unique_labels = np.unique(encoded_labels)
        if len(unique_labels) < 2:
            logger.warning(
                "Fewer than 2 unique labels, skipping %s", combo_label
            )
            return {"status": "skipped", "reason": "fewer than 2 labels"}

        n_samples = len(encoded_labels)
        combo_dir = self._result_dir(metric, design_name, analysis_type)

        summary: dict = {
            "status": "completed",
            "metric": metric,
            "design": design_name,
            "analysis_type": analysis_type,
            "n_subjects": n_samples,
            "n_unique_labels": int(len(unique_labels)),
        }
        if label_names:
            summary["label_names"] = label_names
            summary["group_sizes"] = {
                name: int((encoded_labels == i).sum())
                for i, name in enumerate(label_names)
            }

        # --- Whole-brain decoding (skip for searchlight_only) ---
        if not self.searchlight_only:
            logger.info("[Phase 1] Whole-brain decoding...")
            wb_dir = combo_dir / "whole_brain"
            wb_results = run_whole_brain_decoding(
                images_4d=images_4d,
                labels=encoded_labels,
                mask_img=self.mask_img,
                analysis_type=analysis_type,
                n_permutations=self.n_permutations,
                cv_folds=min(self.cv_folds, n_samples),
                screening_percentile=self.screening_percentile,
                seed=self.seed,
                output_dir=wb_dir,
            )

            summary["whole_brain"] = {
                "mean_score": wb_results["mean_score"],
                "std_score": wb_results["std_score"],
                "permutation_p": wb_results["permutation_p"],
                "score_label": wb_results["score_label"],
            }

            # Visualizations for whole-brain
            logger.info("[Phase 2] Whole-brain visualizations...")
            score_label = (
                "Accuracy" if analysis_type == "classification" else "R\u00b2"
            )

            plot_weight_map(
                wb_results["weight_img"], self.mask_img,
                wb_dir / "weight_map.png",
                title=(
                    f"Decoder Weights \u2014 {metric} {design_name} "
                    f"({analysis_type})"
                ),
            )
            plot_glass_brain(
                wb_results["weight_img"],
                wb_dir / "glass_brain.png",
                title=f"Glass Brain \u2014 {metric} {design_name}",
            )
            plot_decoding_scores(
                wb_results["fold_scores"],
                wb_results["null_distribution"],
                wb_results["mean_score"],
                wb_results["permutation_p"],
                wb_dir / "decoding_scores.png",
                title=(
                    f"Decoding \u2014 {metric} {design_name} ({analysis_type})"
                ),
                metric_label=score_label,
            )

            if analysis_type == "regression":
                plot_regression_brain(
                    wb_results["weight_img"],
                    wb_dir / "regression_weights.png",
                    title=(
                        f"Regression Weights \u2014 {metric} {design_name}"
                    ),
                    bg_img=self.mask_img,
                )

        # --- Searchlight ---
        if self.run_searchlight or self.searchlight_only:
            phase = "[Phase 1]" if self.searchlight_only else "[Phase 3]"
            logger.info("%s Searchlight (Ridge R\u00b2)...", phase)
            sl_dir = combo_dir / "searchlight"
            sl_results = run_searchlight(
                images_4d=images_4d,
                labels=encoded_labels,
                mask_img=self.mask_img,
                analysis_type=analysis_type,
                radius=self.searchlight_radius,
                cv_folds=min(self.searchlight_cv_folds, n_samples),
                n_jobs=self.searchlight_n_jobs,
                seed=self.seed,
                output_dir=sl_dir,
                n_perm_fwer=self.searchlight_n_perm_fwer,
            )

            summary["searchlight"] = {
                "mean_score": sl_results["mean_score"],
                "threshold_fwer": sl_results["threshold_fwer"],
                "n_significant_voxels": sl_results["n_significant_voxels"],
                "radius": sl_results["radius"],
            }

            # Searchlight visualization
            threshold = sl_results["threshold_fwer"] or 0.0
            plot_searchlight_map(
                sl_results["searchlight_img"],
                threshold,
                sl_dir / "searchlight_map.png",
                title=(
                    f"Searchlight \u2014 {metric} {design_name} "
                    f"({analysis_type})"
                ),
                bg_img=self.mask_img,
            )

        # Save per-combo summary
        combo_dir.mkdir(parents=True, exist_ok=True)
        with open(combo_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    # ------------------------------------------------------------------
    # Design description
    # ------------------------------------------------------------------

    def write_design_description(self) -> None:
        """Write human-readable analysis description to output_dir."""
        lines = [
            "ANALYSIS DESCRIPTION",
            "====================",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Analysis: MVPA (Multi-Voxel Pattern Analysis)",
            f"Mode: {'Searchlight-only' if self.searchlight_only else 'Full (decoding + searchlight)'}",
            "",
            "DATA SOURCE",
            "-----------",
            f"Derivatives root: {self.derivatives_dir}",
            f"Design directory: {self.design_dir}",
            f"Brain mask: {self.mask_path}",
            f"Metrics: {', '.join(self.metrics)}",
            "",
        ]

        if not self.searchlight_only:
            lines.extend([
                "WHOLE-BRAIN DECODING",
                "--------------------",
                "- PCA dimensionality reduction (95% variance threshold)",
                "- PCA pre-computed once per CV fold, reused across permutations",
                f"- Cross-validation: StratifiedKFold({self.cv_folds}) / KFold({self.cv_folds})",
                f"- Permutation test: {self.n_permutations} shuffles for empirical p",
                "- Classification: LinearSVC (dual=False)",
                "- Regression: Ridge (alpha=1.0)",
                "- Weight inversion: coef @ pca.components_ \u2192 voxel space",
                "",
            ])

        if self.run_searchlight:
            cv_type = (
                "KFold" if self.searchlight_only else "StratifiedKFold/KFold"
            )
            lines.extend([
                "SEARCHLIGHT MAPPING",
                "-------------------",
                f"- Sphere radius: {self.searchlight_radius} mm (scaled space)",
                f"- Cross-validation: {cv_type}({self.searchlight_cv_folds})",
                "- Classification: LinearSVC, scoring=accuracy",
                "- Regression: Ridge (alpha=1.0), scoring=R\u00b2",
                f"- Max-statistic FWER correction ({self.searchlight_n_perm_fwer} label permutations, p<0.05)",
                f"- Parallel jobs: {self.searchlight_n_jobs}",
                "",
            ])

        lines.extend([
            "ANALYSIS MODES",
            "--------------",
        ])
        if not self.searchlight_only:
            lines.extend([
                "1. Classification: categorical dose groups (C, L, M, H)",
                "   - Metric: accuracy",
            ])
        if not self.skip_regression:
            lines.extend([
                "2. Regression: continuous/ordinal target",
                "   - Metric: R\u00b2",
            ])

        lines.extend([
            "",
            "PARAMETERS",
            "----------",
            f"Random seed: {self.seed}",
            f"Screening percentile: {self.screening_percentile}",
        ])

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "design_description.txt"
        output_path.write_text("\n".join(lines))
        logger.info("Saved design description: %s", output_path)
