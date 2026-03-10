#!/usr/bin/env python3
"""
MVPA (Multi-Voxel Pattern Analysis) runner for SIGMA-space metrics.

Runs whole-brain decoding and/or searchlight mapping per metric,
cohort, and analysis type. Uses PCA + LinearSVC/Ridge for whole-brain
decoding and nilearn SearchLight for spatial mapping, on individual
SIGMA-space NIfTIs discovered from the derivatives tree.

For continuous targets (log_auc, auc, etc.): use --searchlight-only to
skip whole-brain decoding and run searchlight Ridge regression with R²
scoring and FWER correction.

Usage:
    # Classification + regression:
    uv run python scripts/run_mvpa_analysis.py \
        --design-dir $STUDY_ROOT/analysis/mvpa/designs \
        --derivatives-root $STUDY_ROOT/derivatives \
        --output-dir $STUDY_ROOT/analysis/mvpa \
        --mask $STUDY_ROOT/atlas/SIGMA_study_space/SIGMA_InVivo_Brain_Mask.nii.gz \
        --metrics FA MD AD RD \
        --n-permutations 1000

    # Searchlight-only for continuous targets:
    uv run python scripts/run_mvpa_analysis.py \
        --design-dir $STUDY_ROOT/analysis/mvpa/designs \
        --derivatives-root $STUDY_ROOT/derivatives \
        --output-dir $STUDY_ROOT/analysis/mvpa \
        --mask $STUDY_ROOT/atlas/SIGMA_study_space/SIGMA_InVivo_Brain_Mask.nii.gz \
        --metrics FA MD AD RD \
        --searchlight-only --searchlight-radius 2.0 --searchlight-n-jobs 4
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.analysis.progress import AnalysisProgress
from neurofaune.analysis.mvpa.data_loader import (
    align_data_to_design,
    discover_sigma_images,
    load_mvpa_data,
)
from neurofaune.analysis.mvpa.searchlight import run_searchlight
from neurofaune.analysis.mvpa.visualization import (
    plot_decoding_scores,
    plot_regression_brain,
    plot_glass_brain,
    plot_searchlight_map,
    plot_weight_map,
)
from neurofaune.analysis.mvpa.whole_brain import run_whole_brain_decoding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


def run_single_mvpa(
    metric,
    design_name,
    design_dir,
    derivatives_root,
    mask_img,
    analysis_type,
    output_dir,
    n_permutations,
    cv_folds,
    screening_percentile,
    seed,
    run_sl,
    sl_radius,
    sl_cv_folds,
    sl_n_jobs,
    sl_n_perm_fwer,
    searchlight_only=False,
):
    """Run MVPA for one metric + design + analysis_type combination.

    If searchlight_only=True, skips whole-brain decoding and only runs
    searchlight. This is the appropriate mode for continuous targets
    where whole-brain Decoder is not suitable.
    """
    combo_label = f"{metric}/{design_name}/{analysis_type}"
    logger.info("\n%s\n  %s\n%s", "=" * 60, combo_label, "=" * 60)

    # Discover images for this metric
    images = discover_sigma_images(derivatives_root, metric)
    if not images:
        logger.warning("No SIGMA-space %s images found, skipping", metric)
        return {"status": "skipped", "reason": f"no {metric} images"}

    # Load design and align
    from neurofaune.analysis.mvpa.data_loader import load_design
    try:
        design = load_design(design_dir)
    except FileNotFoundError as exc:
        logger.warning("Design not found: %s, skipping", exc)
        return {"status": "skipped", "reason": str(exc)}

    aligned = align_data_to_design(images, design)
    if aligned["n_matched"] < 5:
        logger.warning(
            "Too few matched subjects (%d), skipping %s",
            aligned["n_matched"], combo_label,
        )
        return {"status": "skipped", "reason": f"n={aligned['n_matched']} too small"}

    # Load 4D data
    mvpa_data = load_mvpa_data(aligned["image_info"], mask_img=mask_img)
    images_4d = mvpa_data["images_4d"]

    # Encode labels
    labels = aligned["labels"]
    encoded_labels, label_names = encode_labels(labels, analysis_type)

    # Check label diversity
    unique_labels = np.unique(encoded_labels)
    if len(unique_labels) < 2:
        logger.warning("Fewer than 2 unique labels, skipping %s", combo_label)
        return {"status": "skipped", "reason": "fewer than 2 labels"}

    n_samples = len(encoded_labels)
    combo_dir = output_dir / metric / design_name / analysis_type

    summary = {
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
    if not searchlight_only:
        logger.info("[Phase 1] Whole-brain decoding...")
        wb_dir = combo_dir / "whole_brain"
        wb_results = run_whole_brain_decoding(
            images_4d=images_4d,
            labels=encoded_labels,
            mask_img=mask_img,
            analysis_type=analysis_type,
            n_permutations=n_permutations,
            cv_folds=min(cv_folds, n_samples),
            screening_percentile=screening_percentile,
            seed=seed,
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
        score_label = "Accuracy" if analysis_type == "classification" else "R\u00b2"

        plot_weight_map(
            wb_results["weight_img"], mask_img,
            wb_dir / "weight_map.png",
            title=f"Decoder Weights \u2014 {metric} {design_name} ({analysis_type})",
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
            title=f"Decoding \u2014 {metric} {design_name} ({analysis_type})",
            metric_label=score_label,
        )

        if analysis_type == "regression":
            plot_regression_brain(
                wb_results["weight_img"],
                wb_dir / "regression_weights.png",
                title=f"Regression Weights \u2014 {metric} {design_name}",
                bg_img=mask_img,
            )

    # --- Searchlight ---
    if run_sl or searchlight_only:
        phase = "[Phase 1]" if searchlight_only else "[Phase 3]"
        logger.info("%s Searchlight (Ridge R\u00b2)...", phase)
        sl_dir = combo_dir / "searchlight"
        sl_results = run_searchlight(
            images_4d=images_4d,
            labels=encoded_labels,
            mask_img=mask_img,
            analysis_type=analysis_type,
            radius=sl_radius,
            cv_folds=min(sl_cv_folds, n_samples),
            n_jobs=sl_n_jobs,
            seed=seed,
            output_dir=sl_dir,
            n_perm_fwer=sl_n_perm_fwer,
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
            title=f"Searchlight \u2014 {metric} {design_name} ({analysis_type})",
            bg_img=mask_img,
        )

    return summary


def write_design_description(args, output_path):
    """Write a human-readable description of the MVPA analysis."""
    searchlight_only = getattr(args, "searchlight_only", False)

    lines = [
        "ANALYSIS DESCRIPTION",
        "====================",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Analysis: MVPA (Multi-Voxel Pattern Analysis)",
        f"Mode: {'Searchlight-only' if searchlight_only else 'Full (decoding + searchlight)'}",
        "",
        "DATA SOURCE",
        "-----------",
        f"Derivatives root: {args.derivatives_root}",
        f"Design directory: {args.design_dir}",
        f"Brain mask: {args.mask}",
        f"Metrics: {', '.join(args.metrics)}",
        "",
    ]

    if not searchlight_only:
        lines.extend([
            "WHOLE-BRAIN DECODING",
            "--------------------",
            "- PCA dimensionality reduction (95% variance threshold)",
            "- PCA pre-computed once per CV fold, reused across permutations",
            f"- Cross-validation: StratifiedKFold({args.cv_folds}) / KFold({args.cv_folds})",
            f"- Permutation test: {args.n_permutations} shuffles for empirical p",
            "- Classification: LinearSVC (dual=False)",
            "- Regression: Ridge (alpha=1.0)",
            "- Weight inversion: coef @ pca.components_ → voxel space",
            "",
        ])

    if args.run_searchlight:
        cv_type = "KFold" if searchlight_only else "StratifiedKFold/KFold"
        lines.extend([
            "SEARCHLIGHT MAPPING",
            "-------------------",
            f"- Sphere radius: {args.searchlight_radius} mm (scaled space)",
            f"- Cross-validation: {cv_type}({args.searchlight_cv_folds})",
            "- Classification: LinearSVC, scoring=accuracy",
            "- Regression: Ridge (alpha=1.0), scoring=R\u00b2",
            f"- Max-statistic FWER correction ({getattr(args, 'searchlight_n_perm_fwer', 100)} label permutations, p<0.05)",
            f"- Parallel jobs: {args.searchlight_n_jobs}",
            "",
        ])

    lines.extend([
        "ANALYSIS MODES",
        "--------------",
    ])
    if not searchlight_only:
        lines.extend([
            "1. Classification: categorical dose groups (C, L, M, H)",
            "   - Metric: accuracy",
        ])
    if not args.skip_regression:
        lines.extend([
            "2. Regression: continuous/ordinal target",
            "   - Metric: R\u00b2",
        ])

    lines.extend([
        "",
        "PARAMETERS",
        "----------",
        f"Random seed: {args.seed}",
        f"Screening percentile: {args.screening_percentile}",
    ])

    output_path.write_text("\n".join(lines))
    logger.info("Saved design description: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="MVPA analysis of SIGMA-space DTI metrics"
    )
    parser.add_argument(
        "--design-dir", type=Path, required=True,
        help="Directory containing design subdirs (per_pnd_p30, pooled, etc.)",
    )
    parser.add_argument(
        "--derivatives-root", type=Path, required=True,
        help="Path to derivatives directory with SIGMA-space NIfTIs",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for MVPA results",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["FA", "MD", "AD", "RD"],
        help="DTI metrics to analyse (default: FA MD AD RD)",
    )
    parser.add_argument(
        "--mask", type=Path, required=True,
        help="SIGMA brain mask NIfTI for all subjects",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Permutations for whole-brain decoding p-value (default: 1000)",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Cross-validation folds for whole-brain (default: 5)",
    )
    parser.add_argument(
        "--screening-percentile", type=int, default=20,
        help="ANOVA feature screening percentile (default: 20)",
    )
    parser.add_argument(
        "--run-searchlight", action="store_true",
        help="Enable searchlight mapping (computationally expensive)",
    )
    parser.add_argument(
        "--searchlight-radius", type=float, default=2.0,
        help="Searchlight sphere radius in mm (default: 2.0)",
    )
    parser.add_argument(
        "--searchlight-cv-folds", type=int, default=3,
        help="Cross-validation folds for searchlight (default: 3)",
    )
    parser.add_argument(
        "--searchlight-n-jobs", type=int, default=1,
        help="Parallel jobs for searchlight (default: 1)",
    )
    parser.add_argument(
        "--searchlight-n-perm-fwer", type=int, default=100,
        help="Number of label permutations for FWER correction (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-regression", action="store_true",
        help="Skip regression designs, only run classification",
    )
    parser.add_argument(
        "--searchlight-only", action="store_true",
        help="Run searchlight only (skip whole-brain decoding). "
             "Appropriate for continuous targets where Decoder is not suitable.",
    )

    args = parser.parse_args()

    # --searchlight-only implies --run-searchlight
    if args.searchlight_only:
        args.run_searchlight = True

    # Validate inputs
    if not args.derivatives_root.exists():
        logger.error("Derivatives root not found: %s", args.derivatives_root)
        sys.exit(1)
    if not args.design_dir.exists():
        logger.error("Design directory not found: %s", args.design_dir)
        sys.exit(1)
    if not args.mask.exists():
        logger.error("Mask not found: %s", args.mask)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load mask
    mask_img = nib.load(args.mask)
    logger.info("Loaded brain mask: %s (shape: %s)", args.mask, mask_img.shape)

    # Save analysis configuration
    config = {
        "design_dir": str(args.design_dir),
        "derivatives_root": str(args.derivatives_root),
        "output_dir": str(args.output_dir),
        "mask": str(args.mask),
        "metrics": args.metrics,
        "n_permutations": args.n_permutations,
        "cv_folds": args.cv_folds,
        "screening_percentile": args.screening_percentile,
        "run_searchlight": args.run_searchlight,
        "searchlight_only": args.searchlight_only,
        "searchlight_radius": args.searchlight_radius,
        "searchlight_cv_folds": args.searchlight_cv_folds,
        "searchlight_n_jobs": args.searchlight_n_jobs,
        "seed": args.seed,
        "skip_regression": args.skip_regression,
        "timestamp": datetime.now().isoformat(),
    }
    with open(args.output_dir / "analysis_config.json", "w") as f:
        json.dump(config, f, indent=2)

    write_design_description(args, args.output_dir / "design_description.txt")

    # Discover available designs
    design_dirs = {}
    for d in sorted(args.design_dir.iterdir()):
        if d.is_dir() and (d / "design_summary.json").exists():
            design_dirs[d.name] = d

    if not design_dirs:
        logger.error("No designs found in %s", args.design_dir)
        sys.exit(1)

    logger.info("Found %d designs: %s", len(design_dirs), list(design_dirs.keys()))

    # Separate categorical and regression designs
    categorical_designs = {}
    regression_designs = {}
    for k, v in design_dirs.items():
        if "_regression_" in k or k.endswith("_regression_pooled"):
            regression_designs[k] = v
        elif k.startswith("per_pnd_"):
            categorical_designs[k] = v
        elif k == "pooled":
            # Pooled classification mixes cohorts where dose categories
            # have different meanings — skip it
            logger.info("Skipping pooled categorical design (dose groups are cohort-specific)")
            continue
        else:
            # Unknown — treat as regression if name suggests it
            if "regression" in k or "response" in k:
                regression_designs[k] = v
            else:
                categorical_designs[k] = v

    all_summaries = {}
    best_accuracy = 0.0
    best_r2 = -999.0
    total_n_subjects = 0
    n_searchlight_sig = 0

    # In searchlight-only mode, skip classification designs
    n_cat = 0 if args.searchlight_only else len(categorical_designs)
    n_resp = 0 if args.skip_regression else len(regression_designs)
    total_tasks = len(args.metrics) * (n_cat + n_resp)
    progress = AnalysisProgress(args.output_dir, "run_mvpa_analysis.py", total_tasks)
    completed = 0

    for metric in args.metrics:
        metric_summaries = {}

        # Classification analyses (categorical designs) — skip in searchlight-only mode
        if not args.searchlight_only:
            for design_name, design_path in categorical_designs.items():
                progress.update(
                    task=f"{metric} / {design_name} / classification",
                    phase="running",
                    completed=completed,
                )

                summary = run_single_mvpa(
                    metric=metric,
                    design_name=design_name,
                    design_dir=design_path,
                    derivatives_root=args.derivatives_root,
                    mask_img=mask_img,
                    analysis_type="classification",
                    output_dir=args.output_dir,
                    n_permutations=args.n_permutations,
                    cv_folds=args.cv_folds,
                    screening_percentile=args.screening_percentile,
                    seed=args.seed,
                    run_sl=args.run_searchlight,
                    sl_radius=args.searchlight_radius,
                    sl_cv_folds=args.searchlight_cv_folds,
                    sl_n_jobs=args.searchlight_n_jobs,
                    sl_n_perm_fwer=args.searchlight_n_perm_fwer,
                )
                metric_summaries[f"{design_name}_classification"] = summary
                completed += 1

                if summary.get("status") == "completed":
                    total_n_subjects = max(total_n_subjects, summary.get("n_subjects", 0))
                    wb = summary.get("whole_brain", {})
                    if wb.get("score_label") == "accuracy":
                        best_accuracy = max(best_accuracy, wb.get("mean_score", 0))
                    sl = summary.get("searchlight", {})
                    n_searchlight_sig += sl.get("n_significant_voxels", 0)

        # Regression designs
        if not args.skip_regression:
            for design_name, design_path in regression_designs.items():
                progress.update(
                    task=f"{metric} / {design_name} / regression",
                    phase="running",
                    completed=completed,
                )

                summary = run_single_mvpa(
                    metric=metric,
                    design_name=design_name,
                    design_dir=design_path,
                    derivatives_root=args.derivatives_root,
                    mask_img=mask_img,
                    analysis_type="regression",
                    output_dir=args.output_dir,
                    n_permutations=args.n_permutations,
                    cv_folds=args.cv_folds,
                    screening_percentile=args.screening_percentile,
                    seed=args.seed,
                    run_sl=args.run_searchlight,
                    sl_radius=args.searchlight_radius,
                    sl_cv_folds=args.searchlight_cv_folds,
                    sl_n_jobs=args.searchlight_n_jobs,
                    sl_n_perm_fwer=args.searchlight_n_perm_fwer,
                    searchlight_only=args.searchlight_only,
                )
                metric_summaries[f"{design_name}_regression"] = summary
                completed += 1

                if summary.get("status") == "completed":
                    total_n_subjects = max(total_n_subjects, summary.get("n_subjects", 0))
                    wb = summary.get("whole_brain", {})
                    if wb.get("score_label") == "r2":
                        best_r2 = max(best_r2, wb.get("mean_score", -999))
                    sl = summary.get("searchlight", {})
                    n_searchlight_sig += sl.get("n_significant_voxels", 0)

        all_summaries[metric] = metric_summaries

    # Save overall summary
    overall = {
        "metrics": args.metrics,
        "n_subjects": total_n_subjects,
        "best_whole_brain_accuracy": round(best_accuracy, 3),
        "best_whole_brain_r2": round(best_r2, 3) if best_r2 > -999.0 else None,
        "searchlight_significant_voxels": n_searchlight_sig,
        "run_searchlight": args.run_searchlight,
        "n_permutations": args.n_permutations,
        "timestamp": datetime.now().isoformat(),
        "per_metric": all_summaries,
    }

    summary_path = args.output_dir / "mvpa_summary.json"
    with open(summary_path, "w") as f:
        json.dump(overall, f, indent=2, default=str)
    logger.info("Saved overall summary: %s", summary_path)

    progress.finish()

    # Write provenance tracking (MVPA uses NIfTIs, not ROI CSVs — track mask hash)
    try:
        from neurofaune.analysis.provenance import sha256_file

        prov = {
            "analysis_type": "mvpa",
            "mask": str(args.mask),
            "mask_sha256": sha256_file(args.mask),
            "design_dir": str(args.design_dir),
            "metrics": args.metrics,
            "n_subjects": total_n_subjects,
            "n_permutations": args.n_permutations,
            "date_created": datetime.now().isoformat(),
        }
        prov_path = args.output_dir / "provenance.json"
        with open(prov_path, "w") as f:
            json.dump(prov, f, indent=2)
        logger.info("Wrote provenance.json → %s", prov_path)
    except Exception as exc:
        logger.warning("Failed to write provenance: %s", exc)

    # Register with unified reporting system
    try:
        from neurofaune.reporting import register as report_register

        analysis_root = args.output_dir.parent

        figures = []
        for fig in sorted(args.output_dir.rglob("*.png"))[:20]:
            try:
                figures.append(str(fig.relative_to(analysis_root)))
            except ValueError:
                pass

        report_register(
            analysis_root=analysis_root,
            entry_id="mvpa",
            analysis_type="mvpa",
            display_name=f"MVPA ({', '.join(args.metrics)})",
            output_dir=str(args.output_dir.relative_to(analysis_root)),
            summary_stats={
                "metrics": args.metrics,
                "n_subjects": total_n_subjects,
                "best_whole_brain_accuracy": round(best_accuracy, 3),
                "best_whole_brain_r2": round(best_r2, 3) if best_r2 > -999.0 else None,
                "searchlight_significant_voxels": n_searchlight_sig,
            },
            figures=figures,
            source_summary_json=str(summary_path.relative_to(analysis_root)),
            config=config,
        )
    except Exception as exc:
        logger.warning("Failed to register with reporting system: %s", exc)

    # Generate findings summary
    try:
        from neurofaune.reporting.summarize import summarize_analysis
        findings = summarize_analysis("mvpa", summary_path, output_dir=args.output_dir)
        logger.info("Findings: %s", findings.summary_text)
    except Exception as exc:
        logger.warning("Failed to generate findings summary: %s", exc)

    logger.info("\nMVPA analysis complete. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
