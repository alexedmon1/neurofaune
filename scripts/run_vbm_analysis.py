#!/usr/bin/env python3
"""
VBM Voxel-Wise Analysis with FSL randomise

Runs FSL randomise with 3D TFCE for each design matrix, subsetting the 4D
tissue volumes to match each design's subject order.

Prerequisites:
    - Completed VBM prep (prepare_vbm.py) with 4D volumes + analysis mask
    - Design matrices in {vbm_dir}/designs/{analysis}/ with:
        design.mat, design.con, subject_order.txt, design_summary.json

Usage:
    # Run categorical analyses
    PYTHONUNBUFFERED=1 uv run python scripts/run_vbm_analysis.py \
        --vbm-dir $STUDY_ROOT/analysis/vbm \
        --analyses per_pnd_p30 per_pnd_p60 per_pnd_p90 \
        --n-permutations 5000 --seed 42

    # Run dose-response analyses
    PYTHONUNBUFFERED=1 uv run python scripts/run_vbm_analysis.py \
        --vbm-dir $STUDY_ROOT/analysis/vbm \
        --analyses dose_response_p30 dose_response_p60 dose_response_p90 \
        --n-permutations 5000 --seed 42
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from neurofaune.analysis.progress import AnalysisProgress
from neurofaune.analysis.randomise_analysis import (
    VBMAnalysis,
    validate_provenance,
)


ALL_ANALYSES = [
    'per_pnd_p30', 'per_pnd_p60', 'per_pnd_p90',
    'dose_response_p30', 'dose_response_p60', 'dose_response_p90',
]


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"vbm_analysis_{timestamp}.log"

    logger = logging.getLogger("neurofaune.vbm")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def main():
    parser = argparse.ArgumentParser(
        description="Run VBM voxel-wise analysis with FSL randomise (3D TFCE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Runs FSL randomise with 3D TFCE for each design matrix. Each design has a
different subject subset (per-PND), so the master 4D volumes from VBM prep
are subsetted to match each design's subject_order.txt.

Output structure:
  {vbm_dir}/randomise/{analysis}/
    data/all_{GM,WM}.nii.gz               Subsetted 4D volumes
    randomise_{GM,WM}/                     randomise outputs
    cluster_reports_{GM,WM}/               Cluster reports with atlas labels
    analysis_summary.json
        """
    )

    parser.add_argument('--vbm-dir', type=Path, required=True,
                        help='VBM directory with stats/ and designs/')
    parser.add_argument('--config', type=Path,
                        help='Config file (for SIGMA atlas labels)')
    parser.add_argument('--analyses', nargs='+', default=ALL_ANALYSES,
                        help=f'Analyses to run (default: {ALL_ANALYSES})')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='Tissues to analyze (auto-detected from vbm_config.json)')
    parser.add_argument('--n-permutations', type=int, default=5000,
                        help='Number of permutations (default: 5000)')
    parser.add_argument('--cluster-threshold', type=float, default=0.95,
                        help='Cluster threshold (default: 0.95 = p<0.05)')
    parser.add_argument('--min-cluster-size', type=int, default=10,
                        help='Minimum cluster size in voxels (default: 10)')
    parser.add_argument('--seed', type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--force', action='store_true',
                        help='Delete existing results before running')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip metrics that already have randomise output')
    parser.add_argument('--parcellation', type=Path,
                        help='Override SIGMA parcellation NIfTI')

    args = parser.parse_args()

    # Prepare the analysis object (handles config auto-discovery)
    try:
        analysis = VBMAnalysis.prepare(
            config_path=args.config,
            analysis_dir=args.vbm_dir,
            force=args.force,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Auto-detect metrics from vbm_config.json if not specified
    if args.metrics is None:
        args.metrics = analysis.detect_metrics()

    logger = setup_logging(analysis.analysis_dir)
    # Share the configured logger with the analysis object
    analysis.logger = logger

    logger.info("=" * 80)
    logger.info("VBM VOXEL-WISE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"VBM dir: {analysis.analysis_dir}")
    logger.info(f"Analyses: {args.analyses}")
    logger.info(f"Tissues: {args.metrics}")
    logger.info(f"Permutations: {args.n_permutations}")
    logger.info(f"TFCE: 3D (-T)")

    if analysis.config:
        logger.info(f"Config loaded")

    # Validate provenance for each analysis before running
    for name in args.analyses:
        try:
            validate_provenance(analysis.analysis_dir, name, logger)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

    # Run each analysis
    results = {}
    progress = AnalysisProgress(
        analysis.analysis_dir, "run_vbm_analysis.py", len(args.analyses)
    )
    completed = 0
    failed = 0

    for name in args.analyses:
        progress.update(
            task=name, phase="running randomise",
            completed=completed, failed=failed,
        )

        try:
            analysis._check_or_clear(name)
            result = analysis.run(
                analysis_name=name,
                metrics=args.metrics,
                n_permutations=args.n_permutations,
                cluster_threshold=args.cluster_threshold,
                min_cluster_size=args.min_cluster_size,
                seed=args.seed,
                skip_existing=args.skip_existing,
                parcellation_override=args.parcellation,
            )
            results[name] = result
            completed += 1
        except Exception as e:
            logger.error(f"\nFAILED: {name}: {e}")
            results[name] = {'success': False, 'error': str(e)}
            failed += 1
            continue

    # Register with unified reporting
    try:
        from neurofaune.reporting import register as report_register

        analysis_root = analysis.analysis_dir.parent
        for name, result in results.items():
            if not result.get('success'):
                continue
            output_dir = analysis.analysis_dir / "randomise" / name
            summary_json = output_dir / "analysis_summary.json"

            n_sig = 0
            for metric_results in result.get('results', {}).values():
                summary = metric_results.get('summary', {})
                for c in summary.get('contrasts', []):
                    if c.get('significant'):
                        n_sig += 1

            report_register(
                analysis_root=analysis_root,
                entry_id=f"vbm_{name}",
                analysis_type="vbm",
                display_name=f"VBM: {name}",
                output_dir=str(output_dir.relative_to(analysis_root)),
                summary_stats={
                    "n_subjects": result.get('n_subjects', 0),
                    "metrics": args.metrics,
                    "n_permutations": args.n_permutations,
                    "n_significant_contrasts": n_sig,
                },
                source_summary_json=(
                    str(summary_json.relative_to(analysis_root))
                    if summary_json.exists() else None
                ),
                auto_generate_index=name == list(results.keys())[-1],
            )
    except Exception as exc:
        logger.warning("Failed to register with reporting system: %s", exc)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ALL ANALYSES COMPLETE")
    logger.info("=" * 80)

    for name, result in results.items():
        if result['success']:
            logger.info(f"  {name}: {result['n_subjects']} subjects - OK")
        else:
            logger.info(
                f"  {name}: FAILED - {result.get('error', 'unknown')}"
            )

    n_ok = sum(1 for r in results.values() if r['success'])
    logger.info(f"\n{n_ok}/{len(results)} analyses completed successfully")

    progress.finish()

    logger.info("\nTo check significance:")
    logger.info(
        f"  fslstats {analysis.analysis_dir}/randomise/*/randomise_*/"
        "randomise_tfce_corrp_tstat*.nii.gz -R"
    )

    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == '__main__':
    main()
