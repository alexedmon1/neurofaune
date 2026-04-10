#!/usr/bin/env python3
"""
Multi-modal Canonical Correlation Analysis (MCCA) for ROI-level data (CLI wrapper).

Example scripts in scripts/ are reference wrappers. Each study should
create its own wrapper scripts that import from the library.

Usage:
    # Config-driven (recommended)
    uv run python scripts/run_mcca_analysis.py \
        --config /path/to/config.yaml \
        --views dwi:FA,MD,AD,RD msme:MWF,IWF,CSFF,T2 func:fALFF,ReHo,ALFF \
        --n-components 5 --regs lw \
        --n-permutations 5000 --seed 42 --force

    # Explicit paths (backwards compatible)
    uv run python scripts/run_mcca_analysis.py \
        --roi-dir /path/to/network/roi \
        --output-dir /path/to/network/mcca \
        --views dwi:FA,MD,AD,RD msme:MWF,IWF,CSFF,T2 func:fALFF,ReHo,ALFF \
        --n-components 5 --regs lw \
        --n-permutations 5000 --seed 42
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.network.mcca import MCCAAnalysis, parse_views
from neurofaune.analysis.progress import AnalysisProgress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_NAME = "run_mcca_analysis.py"
COHORTS = [None, "p30", "p60", "p90"]


def main():
    parser = argparse.ArgumentParser(
        description="Multi-modal Canonical Correlation Analysis (MCCA) for ROI data"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to study config.yaml (derives --roi-dir and --output-dir)",
    )
    parser.add_argument(
        "--roi-dir", type=Path, default=None,
        help="Directory containing roi_*_wide.csv files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for MCCA results",
    )
    parser.add_argument(
        "--views", nargs="+", required=True,
        help="View specifications: 'name:metric1,metric2,...' (e.g. 'dwi:FA,MD,AD,RD')",
    )
    parser.add_argument(
        "--feature-sets", nargs="+", default=["bilateral"],
        choices=["bilateral", "territory"],
        help="Feature sets (default: bilateral)",
    )
    parser.add_argument(
        "--n-components", type=int, default=5,
        help="Number of canonical components (default: 5)",
    )
    parser.add_argument(
        "--regs", default="lw",
        help="Regularisation method: 'lw' (Ledoit-Wolf), 'identity', or float (default: lw)",
    )
    parser.add_argument(
        "--target", choices=["dose", "auc", "log_auc"], default="dose",
        help="Target for dose association: 'dose' (ordinal C=0..H=3), 'auc' (continuous AUC), or 'log_auc' (log-transformed AUC)",
    )
    parser.add_argument(
        "--auc-csv", type=Path, default=None,
        help="Path to AUC lookup CSV with subject, session, auc, log_auc columns (used when --target is auc or log_auc)",
    )
    parser.add_argument(
        "--confounds", nargs="*", default=None,
        help="Metadata columns to residualize before MCCA (e.g. 'sex')",
    )
    parser.add_argument(
        "--exclusion-csv", type=Path, default=None,
        help="CSV of sessions to exclude (must have subject, session columns)",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=5000,
        help="Number of permutations (default: 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing results before running",
    )

    args = parser.parse_args()

    if not args.config and not (args.roi_dir and args.output_dir):
        parser.error("Provide --config or both --roi-dir and --output-dir")

    # Parse view specifications
    views = parse_views(args.views)
    logger.info("Views: %s", {k: v for k, v in views.items()})

    # All metrics across all views (for provenance)
    all_metrics = []
    for metrics in views.values():
        all_metrics.extend(metrics)

    # Prepare analysis
    try:
        analysis = MCCAAnalysis.prepare(
            config_path=args.config,
            roi_dir=args.roi_dir,
            output_dir=args.output_dir,
            views=views,
            exclusion_csv=args.exclusion_csv,
            confounds=args.confounds,
            target=args.target,
            auc_csv=args.auc_csv,
            force=args.force,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to prepare analysis: %s", e)
        sys.exit(1)

    # Save analysis configuration
    config = {
        "roi_dir": str(analysis.roi_dir),
        "exclusion_csv": str(args.exclusion_csv) if args.exclusion_csv else None,
        "confounds": args.confounds,
        "target": args.target,
        "output_dir": str(analysis.output_dir),
        "views": {k: v for k, v in views.items()},
        "feature_sets": args.feature_sets,
        "n_components": args.n_components,
        "regs": args.regs,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    analysis.output_dir.mkdir(parents=True, exist_ok=True)
    with open(analysis.output_dir / "analysis_config.json", "w") as f:
        json.dump(config, f, indent=2)

    analysis.write_design_description(
        feature_sets=args.feature_sets,
        n_components=args.n_components,
        regs=args.regs,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    # Run across cohorts and feature sets
    total_tasks = len(COHORTS) * len(args.feature_sets)
    all_summaries = {}
    total_n_subjects = 0
    n_significant_cv = 0
    n_significant_dose = 0

    progress = AnalysisProgress(analysis.output_dir, SCRIPT_NAME, total_tasks)
    completed = 0

    for cohort in COHORTS:
        for feature_set in args.feature_sets:
            cohort_label = cohort or "pooled"
            key = f"{cohort_label}_{feature_set}"

            progress.update(
                task=f"{cohort_label} / {feature_set}",
                phase="running MCCA",
                completed=completed,
            )

            summary = analysis.run(
                cohort=cohort,
                feature_set=feature_set,
                n_components=args.n_components,
                regs=args.regs,
                n_permutations=args.n_permutations,
                seed=args.seed,
            )
            all_summaries[key] = summary
            completed += 1

            if summary.get("status") == "completed":
                total_n_subjects = max(total_n_subjects, summary.get("n_samples", 0))
                perm_ps = summary.get("permutation_p_values", [])
                n_significant_cv += sum(1 for p in perm_ps if p < 0.05)
                assoc = summary.get("dose_association") or summary.get("auc_association", {})
                n_significant_dose += sum(
                    1 for v in assoc.values() if v.get("p_value", 1.0) < 0.05
                )

    # Save overall summary
    overall = {
        "views": {k: v for k, v in views.items()},
        "feature_sets": args.feature_sets,
        "n_components": args.n_components,
        "regularisation": args.regs,
        "n_subjects_max": total_n_subjects,
        "n_significant_canonical_variates": n_significant_cv,
        "n_significant_dose_associations": n_significant_dose,
        "timestamp": datetime.now().isoformat(),
        "per_cohort": all_summaries,
    }

    summary_path = analysis.output_dir / "mcca_summary.json"
    with open(summary_path, "w") as f:
        json.dump(overall, f, indent=2, default=str)
    logger.info("Saved overall summary: %s", summary_path)

    progress.finish()

    # Write provenance tracking
    try:
        from neurofaune.analysis.provenance import write_roi_provenance

        write_roi_provenance(
            output_dir=analysis.output_dir,
            roi_dir=analysis.roi_dir,
            metrics=all_metrics,
            exclusion_csv=args.exclusion_csv,
            n_subjects=total_n_subjects,
            analysis_type="mcca",
            extra={
                "views": {k: v for k, v in views.items()},
                "n_components": args.n_components,
                "regularisation": args.regs,
            },
        )
    except Exception as exc:
        logger.warning("Failed to write provenance: %s", exc)

    # Register with unified reporting system
    try:
        from neurofaune.reporting import register as report_register

        analysis_root = analysis.output_dir.parents[1]

        figures = []
        for fig in sorted(analysis.output_dir.rglob("*.png"))[:30]:
            try:
                figures.append(str(fig.relative_to(analysis_root)))
            except ValueError:
                pass

        view_desc = ", ".join(f"{k}({','.join(v)})" for k, v in views.items())
        report_register(
            analysis_root=analysis_root,
            entry_id="mcca",
            analysis_type="mcca",
            display_name=f"Multi-modal CCA ({view_desc})",
            output_dir=str(analysis.output_dir.relative_to(analysis_root)),
            summary_stats={
                "views": list(views.keys()),
                "n_subjects": total_n_subjects,
                "n_significant_cv": n_significant_cv,
                "n_significant_dose": n_significant_dose,
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
        findings = summarize_analysis("mcca", summary_path, output_dir=analysis.output_dir)
        logger.info("Findings: %s", findings.summary_text)
    except Exception as exc:
        logger.warning("Failed to generate findings summary: %s", exc)

    logger.info("\nMCCA analysis complete. Results in: %s", analysis.output_dir)


if __name__ == "__main__":
    main()
