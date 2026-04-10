#!/usr/bin/env python3
"""
MVPA (Multi-Voxel Pattern Analysis) runner — thin CLI wrapper.

Runs whole-brain decoding and/or searchlight mapping per metric,
cohort, and analysis type. Delegates all logic to MVPAAnalysis.

Usage:
    # Config-driven (recommended)
    uv run python scripts/run_mvpa_analysis.py \
        --config /path/to/config.yaml \
        --metrics FA MD AD RD \
        --n-permutations 1000

    # Explicit paths (backwards compatible)
    uv run python scripts/run_mvpa_analysis.py \
        --design-dir $STUDY_ROOT/analysis/mvpa/designs \
        --derivatives-dir $STUDY_ROOT/derivatives \
        --output-dir $STUDY_ROOT/analysis/mvpa \
        --mask $STUDY_ROOT/atlas/SIGMA_study_space/SIGMA_InVivo_Brain_Mask.nii.gz \
        --metrics FA MD AD RD \
        --n-permutations 1000

    # Searchlight-only for continuous targets:
    uv run python scripts/run_mvpa_analysis.py \
        --config /path/to/config.yaml \
        --metrics FA MD AD RD \
        --searchlight-only --searchlight-radius 2.0 --searchlight-n-jobs 4
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.analysis.mvpa.pipeline import MVPAAnalysis, discover_designs
from neurofaune.analysis.progress import AnalysisProgress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_NAME = "run_mvpa_analysis.py"


def main():
    parser = argparse.ArgumentParser(
        description="MVPA analysis of SIGMA-space metrics"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to study config.yaml (derives paths automatically)",
    )
    parser.add_argument(
        "--design-dir", type=Path, default=None,
        help="Directory containing design subdirs (per_pnd_p30, pooled, etc.)",
    )
    parser.add_argument(
        "--derivatives-dir", type=Path, default=None,
        help="Path to derivatives directory with SIGMA-space NIfTIs",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for MVPA results",
    )
    parser.add_argument(
        "--mask", type=Path, default=None,
        help="SIGMA brain mask NIfTI for all subjects",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["FA", "MD", "AD", "RD"],
        help="DTI metrics to analyse (default: FA MD AD RD)",
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
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing results before running",
    )

    args = parser.parse_args()

    if not args.config and not (
        args.derivatives_dir and args.output_dir and args.mask
    ):
        parser.error(
            "Provide --config or all of --derivatives-dir, --output-dir, "
            "--design-dir, and --mask"
        )

    # Prepare analysis
    try:
        analysis = MVPAAnalysis.prepare(
            config_path=args.config,
            output_dir=args.output_dir,
            derivatives_dir=args.derivatives_dir,
            design_dir=args.design_dir,
            mask=args.mask,
            metrics=args.metrics,
            n_permutations=args.n_permutations,
            cv_folds=args.cv_folds,
            screening_percentile=args.screening_percentile,
            seed=args.seed,
            run_searchlight=args.run_searchlight,
            searchlight_only=args.searchlight_only,
            searchlight_radius=args.searchlight_radius,
            searchlight_cv_folds=args.searchlight_cv_folds,
            searchlight_n_jobs=args.searchlight_n_jobs,
            searchlight_n_perm_fwer=args.searchlight_n_perm_fwer,
            skip_regression=args.skip_regression,
            force=args.force,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to prepare analysis: %s", e)
        sys.exit(1)

    analysis.output_dir.mkdir(parents=True, exist_ok=True)

    # Save analysis configuration
    config = {
        "design_dir": str(analysis.design_dir),
        "derivatives_dir": str(analysis.derivatives_dir),
        "output_dir": str(analysis.output_dir),
        "mask": str(analysis.mask_path),
        "metrics": analysis.metrics,
        "n_permutations": analysis.n_permutations,
        "cv_folds": analysis.cv_folds,
        "screening_percentile": analysis.screening_percentile,
        "run_searchlight": analysis.run_searchlight,
        "searchlight_only": analysis.searchlight_only,
        "searchlight_radius": analysis.searchlight_radius,
        "searchlight_cv_folds": analysis.searchlight_cv_folds,
        "searchlight_n_jobs": analysis.searchlight_n_jobs,
        "seed": analysis.seed,
        "skip_regression": analysis.skip_regression,
        "timestamp": datetime.now().isoformat(),
    }
    with open(analysis.output_dir / "analysis_config.json", "w") as f:
        json.dump(config, f, indent=2)

    analysis.write_design_description()

    # Discover available designs
    categorical_designs, regression_designs = discover_designs(
        analysis.design_dir,
        skip_regression=analysis.skip_regression,
        searchlight_only=analysis.searchlight_only,
    )

    if not categorical_designs and not regression_designs:
        logger.error("No designs found in %s", analysis.design_dir)
        sys.exit(1)

    logger.info(
        "Found %d categorical + %d regression designs",
        len(categorical_designs), len(regression_designs),
    )

    all_summaries = {}
    best_accuracy = 0.0
    best_r2 = -999.0
    total_n_subjects = 0
    n_searchlight_sig = 0

    n_cat = len(categorical_designs)
    n_reg = len(regression_designs)
    total_tasks = len(analysis.metrics) * (n_cat + n_reg)
    progress = AnalysisProgress(analysis.output_dir, SCRIPT_NAME, total_tasks)
    completed = 0

    for metric in analysis.metrics:
        metric_summaries = {}

        # Classification analyses (categorical designs)
        for design_name in categorical_designs:
            progress.update(
                task=f"{metric} / {design_name} / classification",
                phase="running",
                completed=completed,
            )

            summary = analysis.run(
                metric=metric,
                design_name=design_name,
                analysis_type="classification",
            )
            metric_summaries[f"{design_name}_classification"] = summary
            completed += 1

            if summary.get("status") == "completed":
                total_n_subjects = max(
                    total_n_subjects, summary.get("n_subjects", 0)
                )
                wb = summary.get("whole_brain", {})
                if wb.get("score_label") == "accuracy":
                    best_accuracy = max(best_accuracy, wb.get("mean_score", 0))
                sl = summary.get("searchlight", {})
                n_searchlight_sig += sl.get("n_significant_voxels", 0)

        # Regression designs
        for design_name in regression_designs:
            progress.update(
                task=f"{metric} / {design_name} / regression",
                phase="running",
                completed=completed,
            )

            summary = analysis.run(
                metric=metric,
                design_name=design_name,
                analysis_type="regression",
            )
            metric_summaries[f"{design_name}_regression"] = summary
            completed += 1

            if summary.get("status") == "completed":
                total_n_subjects = max(
                    total_n_subjects, summary.get("n_subjects", 0)
                )
                wb = summary.get("whole_brain", {})
                if wb.get("score_label") == "r2":
                    best_r2 = max(best_r2, wb.get("mean_score", -999))
                sl = summary.get("searchlight", {})
                n_searchlight_sig += sl.get("n_significant_voxels", 0)

        all_summaries[metric] = metric_summaries

    # Save overall summary
    overall = {
        "metrics": analysis.metrics,
        "n_subjects": total_n_subjects,
        "best_whole_brain_accuracy": round(best_accuracy, 3),
        "best_whole_brain_r2": (
            round(best_r2, 3) if best_r2 > -999.0 else None
        ),
        "searchlight_significant_voxels": n_searchlight_sig,
        "run_searchlight": analysis.run_searchlight,
        "n_permutations": analysis.n_permutations,
        "timestamp": datetime.now().isoformat(),
        "per_metric": all_summaries,
    }

    summary_path = analysis.output_dir / "mvpa_summary.json"
    with open(summary_path, "w") as f:
        json.dump(overall, f, indent=2, default=str)
    logger.info("Saved overall summary: %s", summary_path)

    progress.finish()

    # Write provenance tracking
    try:
        from neurofaune.analysis.provenance import sha256_file

        prov = {
            "analysis_type": "mvpa",
            "mask": str(analysis.mask_path),
            "mask_sha256": sha256_file(analysis.mask_path),
            "design_dir": str(analysis.design_dir),
            "metrics": analysis.metrics,
            "n_subjects": total_n_subjects,
            "n_permutations": analysis.n_permutations,
            "date_created": datetime.now().isoformat(),
        }
        prov_path = analysis.output_dir / "provenance.json"
        with open(prov_path, "w") as f:
            json.dump(prov, f, indent=2)
        logger.info("Wrote provenance.json -> %s", prov_path)
    except Exception as exc:
        logger.warning("Failed to write provenance: %s", exc)

    # Register with unified reporting system
    try:
        from neurofaune.reporting import register as report_register

        analysis_root = analysis.output_dir.parent

        figures = []
        for fig in sorted(analysis.output_dir.rglob("*.png"))[:20]:
            try:
                figures.append(str(fig.relative_to(analysis_root)))
            except ValueError:
                pass

        report_register(
            analysis_root=analysis_root,
            entry_id="mvpa",
            analysis_type="mvpa",
            display_name=f"MVPA ({', '.join(analysis.metrics)})",
            output_dir=str(analysis.output_dir.relative_to(analysis_root)),
            summary_stats={
                "metrics": analysis.metrics,
                "n_subjects": total_n_subjects,
                "best_whole_brain_accuracy": round(best_accuracy, 3),
                "best_whole_brain_r2": (
                    round(best_r2, 3) if best_r2 > -999.0 else None
                ),
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
        findings = summarize_analysis(
            "mvpa", summary_path, output_dir=analysis.output_dir
        )
        logger.info("Findings: %s", findings.summary_text)
    except Exception as exc:
        logger.warning("Failed to generate findings summary: %s", exc)

    logger.info(
        "\nMVPA analysis complete. Results in: %s", analysis.output_dir
    )


if __name__ == "__main__":
    main()
