#!/usr/bin/env python3
"""
Per-subject functional connectivity graph theory analysis.

Computes graph-theoretic metrics on pre-computed per-subject FC matrices
(from fc_matrices.npy), then tests group differences via permutation on
per-subject density-curve AUC values.

Usage:
    uv run python scripts/run_fc_graph_theory.py \
        --connectome-dir $STUDY_ROOT/network/connectome/func \
        --output-dir $STUDY_ROOT/network/fc_graph_theory \
        --n-permutations 1000 --seed 42

    # Specific graph metrics and cohort
    uv run python scripts/run_fc_graph_theory.py \
        --connectome-dir $STUDY_ROOT/network/connectome/func \
        --output-dir $STUDY_ROOT/network/fc_graph_theory \
        --graph-metrics global_efficiency modularity small_worldness \
        --cohorts p30 p60 p90 \
        --n-permutations 1000 --seed 42
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.network.fc_graph_theory import (
    build_groups,
    compute_subject_aucs,
    load_fc_data,
    permutation_test_groups,
)
from neurofaune.network.graph_theory import (
    DEFAULT_DENSITIES,
    METRIC_REGISTRY,
    list_metrics,
)
from neurofaune.analysis.progress import AnalysisProgress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Per-subject FC graph theory with permutation testing"
    )
    parser.add_argument(
        "--connectome-dir", type=Path, required=True,
        help="Directory with fc_matrices.npy, fc_subjects.csv, fc_roi_labels.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--graph-metrics", nargs="+", default=None,
        help="Graph metrics to compute (default: all). Use --list-graph-metrics to see options.",
    )
    parser.add_argument(
        "--densities", nargs="+", type=float, default=None,
        help=f"Network density sweep (default: {DEFAULT_DENSITIES})",
    )
    parser.add_argument(
        "--exclusion-csv", type=Path, default=None,
        help="CSV with subject, session columns listing sessions to exclude",
    )
    parser.add_argument(
        "--cohorts", nargs="+", default=None,
        help="Cohorts to analyse (default: pooled + p30 p60 p90)",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Number of permutations for group comparison (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--list-graph-metrics", action="store_true",
        help="List available graph metrics and exit",
    )

    args = parser.parse_args()

    if args.list_graph_metrics:
        print("Available graph metrics:")
        for name, (_func, desc) in METRIC_REGISTRY.items():
            print(f"  {name:30s} {desc}")
        sys.exit(0)

    if not args.connectome_dir.exists():
        logger.error("Connectome directory not found: %s", args.connectome_dir)
        sys.exit(1)

    gm_names = args.graph_metrics or list_metrics()
    for gm in gm_names:
        if gm not in METRIC_REGISTRY:
            logger.error("Unknown graph metric %r. Available: %s", gm, list_metrics())
            sys.exit(1)

    densities = args.densities or DEFAULT_DENSITIES
    cohorts = args.cohorts or [None, "p30", "p60", "p90"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "connectome_dir": str(args.connectome_dir),
        "exclusion_csv": str(args.exclusion_csv) if args.exclusion_csv else None,
        "output_dir": str(args.output_dir),
        "graph_metrics": gm_names,
        "densities": densities,
        "cohorts": [c or "pooled" for c in cohorts],
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(args.output_dir / "analysis_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load data
    logger.info("Loading FC matrices...")
    fc_matrices, subjects_df, roi_labels = load_fc_data(
        args.connectome_dir, exclusion_csv=args.exclusion_csv,
    )

    # Compute per-subject graph metric AUCs (done once, reused across cohorts)
    logger.info("Computing per-subject graph metrics across density sweep...")
    auc_df, curves = compute_subject_aucs(fc_matrices, gm_names, densities)

    # Save per-subject AUCs
    auc_out = auc_df.copy()
    auc_out.insert(0, "subject", subjects_df["subject"].values)
    auc_out.insert(1, "session", subjects_df["session"].values)
    auc_out.insert(2, "dose", subjects_df["dose"].values)
    if "sex" in subjects_df.columns:
        auc_out.insert(3, "sex", subjects_df["sex"].values)
    auc_out.to_csv(args.output_dir / "subject_graph_aucs.csv", index=False)
    logger.info("Saved per-subject AUCs: %s", args.output_dir / "subject_graph_aucs.csv")

    # Save density curves
    np.savez_compressed(
        args.output_dir / "density_curves.npz",
        densities=np.array(densities),
        **{m: curves[m] for m in gm_names},
    )

    # Group comparisons per cohort
    progress = AnalysisProgress(
        args.output_dir, "run_fc_graph_theory.py", len(cohorts)
    )
    all_results = {}
    completed = 0

    for cohort in cohorts:
        cohort_label = cohort or "pooled"
        logger.info(
            "\n%s\n  FC Graph Theory — %s\n%s",
            "=" * 60, cohort_label, "=" * 60,
        )
        progress.update(task=cohort_label, phase="permutation testing", completed=completed)

        group_labels, group_names, mask = build_groups(subjects_df, cohort)

        if len(group_names) < 2:
            logger.warning("Fewer than 2 groups for %s, skipping", cohort_label)
            completed += 1
            continue

        n_per_group = {
            name: int((group_labels == i).sum())
            for i, name in enumerate(group_names)
            if i in set(group_labels)
        }
        logger.info("  Groups: %s", n_per_group)

        results_df = permutation_test_groups(
            auc_df[mask],
            group_labels,
            group_names,
            n_permutations=args.n_permutations,
            seed=args.seed,
        )

        # Save per-cohort results
        cohort_dir = args.output_dir / cohort_label
        cohort_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(cohort_dir / "pairwise_tests.csv", index=False)

        n_sig = int((results_df["p_value"] < 0.05).sum())
        n_tests = len(results_df)
        logger.info("  %s: %d/%d significant (p < 0.05)", cohort_label, n_sig, n_tests)

        all_results[cohort_label] = {
            "n_subjects": int(mask.sum()),
            "groups": n_per_group,
            "n_significant": n_sig,
            "n_tests": n_tests,
        }
        completed += 1

    # Save summary
    summary = {
        "graph_metrics": gm_names,
        "densities": densities,
        "n_subjects_total": len(subjects_df),
        "n_rois": fc_matrices.shape[1],
        "cohorts": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(args.output_dir / "fc_graph_theory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    progress.finish()
    logger.info("\nFC graph theory analysis complete. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
