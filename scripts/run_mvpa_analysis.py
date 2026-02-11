#!/usr/bin/env python3
"""
MVPA (Multi-Voxel Pattern Analysis) runner for SIGMA-space DTI metrics.

Runs whole-brain decoding and optional searchlight mapping per metric,
cohort, and analysis type (classification + dose-response). Uses nilearn
Decoder and SearchLight on individual SIGMA-space NIfTIs discovered from
the derivatives tree.

Usage:
    uv run python scripts/run_mvpa_analysis.py \
        --design-dir /mnt/arborea/bpa-rat/analysis/mvpa/designs \
        --derivatives-root /mnt/arborea/bpa-rat/derivatives \
        --output-dir /mnt/arborea/bpa-rat/analysis/mvpa \
        --mask /mnt/arborea/bpa-rat/atlas/SIGMA_study_space/SIGMA_InVivo_Brain_Template_Masked.nii.gz \
        --metrics FA MD AD RD \
        --n-permutations 1000

    # With searchlight (computationally expensive):
    uv run python scripts/run_mvpa_analysis.py \
        --design-dir ... --derivatives-root ... --output-dir ... --mask ... \
        --run-searchlight --searchlight-radius 2.0 --searchlight-n-jobs 4
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

from neurofaune.analysis.mvpa.data_loader import (
    align_data_to_design,
    discover_sigma_images,
    load_mvpa_data,
)
from neurofaune.analysis.mvpa.searchlight import run_searchlight
from neurofaune.analysis.mvpa.visualization import (
    plot_decoding_scores,
    plot_dose_response_brain,
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
    For dose-response: labels are already numeric (ordinal).

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
):
    """Run MVPA for one metric + design + analysis_type combination."""
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

    # --- Whole-brain decoding (always runs) ---
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

    if analysis_type == "dose_response":
        plot_dose_response_brain(
            wb_results["weight_img"],
            wb_dir / "dose_response_weights.png",
            title=f"Dose-Response Weights \u2014 {metric} {design_name}",
            bg_img=mask_img,
        )

    # --- Searchlight (optional) ---
    if run_sl:
        logger.info("[Phase 3] Searchlight...")
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
    lines = [
        "ANALYSIS DESCRIPTION",
        "====================",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Analysis: MVPA (Multi-Voxel Pattern Analysis)",
        "",
        "DATA SOURCE",
        "-----------",
        f"Derivatives root: {args.derivatives_root}",
        f"Design directory: {args.design_dir}",
        f"Brain mask: {args.mask}",
        f"Metrics: {', '.join(args.metrics)}",
        "",
        "WHOLE-BRAIN DECODING",
        "--------------------",
        "- nilearn Decoder with ANOVA feature screening",
        f"- Screening percentile: {args.screening_percentile}% (top voxels within mask)",
        f"- Cross-validation: StratifiedKFold({args.cv_folds})",
        f"- Permutation test: {args.n_permutations} shuffles for empirical p",
        "- Classification: Linear SVM (L1 penalty)",
        "- Dose-response: Ridge regression (alpha=1.0)",
        "",
    ]

    if args.run_searchlight:
        lines.extend([
            "SEARCHLIGHT MAPPING",
            "-------------------",
            f"- Sphere radius: {args.searchlight_radius} mm (scaled space)",
            f"- Cross-validation: StratifiedKFold({args.searchlight_cv_folds})",
            "- Max-statistic FWER correction (100 label permutations, p<0.05)",
            f"- Parallel jobs: {args.searchlight_n_jobs}",
            "",
        ])

    lines.extend([
        "ANALYSIS MODES",
        "--------------",
        "1. Classification: categorical dose groups (C, L, M, H)",
        "   - Metric: accuracy",
    ])
    if not args.skip_dose_response:
        lines.extend([
            "2. Dose-response: ordinal dose (0, 1, 2, 3)",
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
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-dose-response", action="store_true",
        help="Skip dose-response regression, only run classification",
    )

    args = parser.parse_args()

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
        "searchlight_radius": args.searchlight_radius,
        "searchlight_cv_folds": args.searchlight_cv_folds,
        "searchlight_n_jobs": args.searchlight_n_jobs,
        "seed": args.seed,
        "skip_dose_response": args.skip_dose_response,
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

    # Separate categorical and dose-response designs
    categorical_designs = {
        k: v for k, v in design_dirs.items()
        if not k.startswith("dose_response")
    }
    dose_response_designs = {
        k: v for k, v in design_dirs.items()
        if k.startswith("dose_response")
    }

    all_summaries = {}
    best_accuracy = 0.0
    best_r2 = -999.0
    total_n_subjects = 0
    n_searchlight_sig = 0

    for metric in args.metrics:
        metric_summaries = {}

        # Classification analyses (categorical designs)
        for design_name, design_path in categorical_designs.items():
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
                sl_n_perm_fwer=100,
            )
            metric_summaries[f"{design_name}_classification"] = summary

            if summary.get("status") == "completed":
                total_n_subjects = max(total_n_subjects, summary.get("n_subjects", 0))
                wb = summary.get("whole_brain", {})
                if wb.get("score_label") == "accuracy":
                    best_accuracy = max(best_accuracy, wb.get("mean_score", 0))
                sl = summary.get("searchlight", {})
                n_searchlight_sig += sl.get("n_significant_voxels", 0)

        # Dose-response analyses
        if not args.skip_dose_response:
            for design_name, design_path in dose_response_designs.items():
                summary = run_single_mvpa(
                    metric=metric,
                    design_name=design_name,
                    design_dir=design_path,
                    derivatives_root=args.derivatives_root,
                    mask_img=mask_img,
                    analysis_type="dose_response",
                    output_dir=args.output_dir,
                    n_permutations=args.n_permutations,
                    cv_folds=args.cv_folds,
                    screening_percentile=args.screening_percentile,
                    seed=args.seed,
                    run_sl=args.run_searchlight,
                    sl_radius=args.searchlight_radius,
                    sl_cv_folds=args.searchlight_cv_folds,
                    sl_n_jobs=args.searchlight_n_jobs,
                    sl_n_perm_fwer=100,
                )
                metric_summaries[f"{design_name}_dose_response"] = summary

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

    # Register with unified reporting system
    try:
        from neurofaune.analysis.reporting import register as report_register

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

    logger.info("\nMVPA analysis complete. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
