#!/usr/bin/env python3
"""
TBSS Statistical Analysis Workflow

Runs group-level statistical analysis on prepared TBSS data using FSL randomise.

This workflow:
1. Validates prepared TBSS data (skeletonised volumes, manifest)
2. Loads pre-generated design matrices (from neuroaider or manual creation)
3. Runs FSL randomise with TFCE for each metric
4. Extracts significant clusters with SIGMA atlas labels
5. Generates HTML reports

Prerequisites:
- Completed TBSS data preparation (prepare_tbss.py)
- Design matrix files (design.mat, design.con) from neuroaider

Usage:
    python -m neurofaune.analysis.tbss.run_tbss_stats \\
        --tbss-dir /study/analysis/tbss/ \\
        --design-dir /study/designs/model1/ \\
        --analysis-name dose_response \\
        --metrics FA MD AD RD
"""

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from neurofaune.analysis.stats.randomise_wrapper import run_randomise, summarize_results
from neurofaune.analysis.stats.cluster_report import generate_reports_for_all_contrasts
from neurofaune.config import load_config, get_config_value


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"tbss_stats_{timestamp}.log"

    logger = logging.getLogger("neurofaune.tbss")
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


def validate_prepared_data(tbss_dir: Path, metrics: List[str]) -> Dict:
    """
    Validate that TBSS data preparation completed successfully.

    Args:
        tbss_dir: Directory containing prepared TBSS data
        metrics: Metrics to validate

    Returns:
        Dictionary with paths to required files

    Raises:
        FileNotFoundError: If required files are missing
    """
    tbss_dir = Path(tbss_dir)

    if not tbss_dir.exists():
        raise FileNotFoundError(f"TBSS directory not found: {tbss_dir}")

    manifest_file = tbss_dir / "subject_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Subject manifest not found: {manifest_file}\n"
            "Run prepare_tbss.py first."
        )

    with open(manifest_file) as f:
        manifest = json.load(f)

    stats_dir = tbss_dir / "stats"
    if not stats_dir.exists():
        raise FileNotFoundError(f"Stats directory not found: {stats_dir}")

    # Validate skeleton files
    skeleton_mask = stats_dir / "mean_FA_skeleton_mask.nii.gz"
    if not skeleton_mask.exists():
        raise FileNotFoundError(f"Skeleton mask not found: {skeleton_mask}")

    # Validate skeletonised data for each metric
    skeletonised_files = {}
    for metric in metrics:
        skel_file = stats_dir / f"all_{metric}_skeletonised.nii.gz"
        if not skel_file.exists():
            raise FileNotFoundError(
                f"Skeletonised {metric} not found: {skel_file}\n"
                f"Run prepare_tbss.py with --metrics {metric}"
            )
        skeletonised_files[metric] = skel_file

    return {
        'tbss_dir': tbss_dir,
        'stats_dir': stats_dir,
        'manifest': manifest,
        'manifest_file': manifest_file,
        'skeleton_mask': skeleton_mask,
        'skeletonised_files': skeletonised_files,
        'n_subjects': manifest['subjects_included']
    }


def validate_design_files(design_dir: Path, n_subjects: int) -> Dict:
    """
    Validate design matrix and contrast files.

    Args:
        design_dir: Directory containing design.mat, design.con
        n_subjects: Expected number of subjects (rows in design matrix)

    Returns:
        Dictionary with design file paths and metadata

    Raises:
        FileNotFoundError: If required design files are missing
        ValueError: If design dimensions don't match data
    """
    design_dir = Path(design_dir)

    design_mat = design_dir / 'design.mat'
    design_con = design_dir / 'design.con'

    if not design_mat.exists():
        raise FileNotFoundError(f"Design matrix not found: {design_mat}")
    if not design_con.exists():
        raise FileNotFoundError(f"Contrast file not found: {design_con}")

    # Parse design matrix dimensions
    n_points = None
    n_waves = None
    with open(design_mat) as f:
        for line in f:
            if '/NumPoints' in line:
                n_points = int(line.split()[-1])
            elif '/NumWaves' in line:
                n_waves = int(line.split()[-1])

    if n_points and n_points != n_subjects:
        raise ValueError(
            f"Design matrix has {n_points} rows but TBSS data has {n_subjects} subjects.\n"
            "Ensure the design matrix matches the subject list used in prepare_tbss.py."
        )

    # Parse contrast dimensions
    n_contrasts = None
    with open(design_con) as f:
        for line in f:
            if '/NumContrasts' in line:
                n_contrasts = int(line.split()[-1])

    # Load design summary if available
    design_summary_file = design_dir / 'design_summary.json'
    contrast_names = None
    if design_summary_file.exists():
        with open(design_summary_file) as f:
            design_summary = json.load(f)
        contrast_names = design_summary.get('contrasts', None)

    if contrast_names is None and n_contrasts:
        contrast_names = [f'contrast_{i+1}' for i in range(n_contrasts)]

    return {
        'design_mat': design_mat,
        'design_con': design_con,
        'n_subjects': n_points,
        'n_predictors': n_waves,
        'n_contrasts': n_contrasts,
        'contrast_names': contrast_names
    }


def run_tbss_statistical_analysis(
    tbss_dir: Path,
    design_dir: Path,
    output_dir: Path,
    analysis_name: str,
    metrics: List[str] = None,
    n_permutations: int = 5000,
    tfce: bool = True,
    cluster_threshold: float = 0.95,
    min_cluster_size: int = 10,
    seed: Optional[int] = None,
    config: Optional[Dict] = None
) -> Dict:
    """
    Run statistical analysis on prepared TBSS data.

    Args:
        tbss_dir: Directory containing prepared TBSS data
        design_dir: Directory with pre-generated design matrices
        output_dir: Output directory for this analysis
        analysis_name: Name for this analysis run
        metrics: Metrics to analyze (default: ['FA', 'MD', 'AD', 'RD'])
        n_permutations: Number of permutations for randomise
        tfce: Use TFCE (default: True)
        cluster_threshold: Threshold for cluster extraction (0.95 = p<0.05)
        min_cluster_size: Minimum cluster size in voxels
        seed: Random seed for reproducibility
        config: Optional config dict for SIGMA atlas paths

    Returns:
        Dictionary with analysis results
    """
    if metrics is None:
        metrics = ['FA', 'MD', 'AD', 'RD']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info("TBSS STATISTICAL ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Analysis: {analysis_name}")
    logger.info(f"TBSS data: {tbss_dir}")
    logger.info(f"Design: {design_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Permutations: {n_permutations}")
    logger.info(f"TFCE: {tfce}")

    # Step 1: Validate prepared data
    logger.info("\n[Step 1] Validating prepared TBSS data...")
    prepared = validate_prepared_data(tbss_dir, metrics)
    logger.info(f"  Subjects: {prepared['n_subjects']}")
    logger.info(f"  Metrics: {list(prepared['skeletonised_files'].keys())}")

    # Step 2: Validate design files
    logger.info("\n[Step 2] Validating design matrices...")
    design = validate_design_files(design_dir, prepared['n_subjects'])
    logger.info(f"  Predictors: {design['n_predictors']}")
    logger.info(f"  Contrasts: {design['n_contrasts']}")
    logger.info(f"  Names: {design['contrast_names']}")

    # Copy design files to output
    shutil.copy(design['design_mat'], output_dir / 'design.mat')
    shutil.copy(design['design_con'], output_dir / 'design.con')
    design_summary = design_dir / 'design_summary.json'
    if design_summary.exists():
        shutil.copy(design_summary, output_dir / 'design_summary.json')

    # Step 3: Run randomise for each metric
    logger.info("\n[Step 3] Running FSL randomise...")
    all_results = {}

    for metric in metrics:
        logger.info(f"\n  --- {metric} ---")
        metric_output = output_dir / f"randomise_{metric}"

        randomise_result = run_randomise(
            input_file=prepared['skeletonised_files'][metric],
            design_mat=design['design_mat'],
            contrast_con=design['design_con'],
            output_dir=metric_output,
            mask=prepared['skeleton_mask'],
            n_permutations=n_permutations,
            tfce=tfce,
            seed=seed
        )

        all_results[metric] = {
            'randomise': randomise_result,
            'summary': summarize_results(metric_output, cluster_threshold)
        }

    # Step 4: Extract clusters and generate reports
    logger.info("\n[Step 4] Extracting clusters and generating reports...")

    # Find SIGMA parcellation for labeling
    sigma_parcellation = None
    if config:
        study_root = Path(get_config_value(config, 'paths.study_root', default=''))
        parc_path = study_root / 'atlas' / 'SIGMA_study_space' / 'SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz'
        if parc_path.exists():
            sigma_parcellation = parc_path

    for metric in metrics:
        metric_output = output_dir / f"randomise_{metric}"
        reports_dir = output_dir / f"cluster_reports_{metric}"

        cluster_results = generate_reports_for_all_contrasts(
            randomise_output_dir=metric_output,
            output_dir=reports_dir,
            contrast_names=design['contrast_names'],
            sigma_parcellation=sigma_parcellation,
            threshold=cluster_threshold,
            min_cluster_size=min_cluster_size
        )

        all_results[metric]['clusters'] = cluster_results

    # Step 5: Summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)

    for metric in metrics:
        summary = all_results[metric]['summary']
        n_sig = sum(1 for c in summary['contrasts'] if c['significant'])
        logger.info(f"\n  {metric}:")
        for contrast in summary['contrasts']:
            status = "SIGNIFICANT" if contrast['significant'] else "not significant"
            logger.info(
                f"    {contrast['type']}{contrast['contrast_number']}: "
                f"{contrast['n_significant_voxels']} voxels ({status})"
            )

    # Save analysis summary
    summary_file = output_dir / 'analysis_summary.json'
    summary_data = {
        'analysis_name': analysis_name,
        'date': datetime.now().isoformat(),
        'tbss_dir': str(tbss_dir),
        'design_dir': str(design_dir),
        'n_subjects': prepared['n_subjects'],
        'metrics': metrics,
        'n_permutations': n_permutations,
        'tfce': tfce,
        'results': {
            metric: {
                'n_significant_contrasts': sum(
                    1 for c in all_results[metric]['summary']['contrasts']
                    if c['significant']
                ),
                'contrasts': all_results[metric]['summary']['contrasts']
            }
            for metric in metrics
        }
    }
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)

    return {
        'success': True,
        'output_dir': str(output_dir),
        'results': all_results,
        'summary_file': str(summary_file)
    }


def main():
    """Command-line interface for TBSS statistical analysis."""
    parser = argparse.ArgumentParser(
        description="Run TBSS statistical analysis using FSL randomise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis with pre-generated design from neuroaider
  uv run python -m neurofaune.analysis.tbss.run_tbss_stats \\
      --tbss-dir /study/analysis/tbss/ \\
      --design-dir /study/designs/dose_response/ \\
      --analysis-name dose_p60 \\
      --metrics FA MD AD RD

  # Quick test with fewer permutations
  uv run python -m neurofaune.analysis.tbss.run_tbss_stats \\
      --tbss-dir /study/analysis/tbss/ \\
      --design-dir /study/designs/dose_response/ \\
      --analysis-name test_run \\
      --n-permutations 100 \\
      --seed 42
        """
    )

    parser.add_argument('--tbss-dir', type=Path, required=True,
                        help='Directory containing prepared TBSS data')
    parser.add_argument('--design-dir', type=Path, required=True,
                        help='Directory with design.mat and design.con')
    parser.add_argument('--analysis-name', type=str, required=True,
                        help='Name for this analysis run')
    parser.add_argument('--output-dir', type=Path,
                        help='Output directory (default: tbss-dir/randomise/analysis-name)')
    parser.add_argument('--config', type=Path,
                        help='Config file (for SIGMA atlas labels)')
    parser.add_argument('--metrics', nargs='+', default=['FA', 'MD', 'AD', 'RD'],
                        help='Metrics to analyze (default: FA MD AD RD)')
    parser.add_argument('--n-permutations', type=int, default=5000,
                        help='Number of permutations (default: 5000)')
    parser.add_argument('--no-tfce', action='store_true',
                        help='Disable TFCE')
    parser.add_argument('--cluster-threshold', type=float, default=0.95,
                        help='Cluster threshold (default: 0.95 = p<0.05)')
    parser.add_argument('--min-cluster-size', type=int, default=10,
                        help='Minimum cluster size (default: 10)')
    parser.add_argument('--seed', type=int, help='Random seed')

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.tbss_dir / 'randomise' / args.analysis_name

    # Load config if provided
    config = None
    if args.config:
        config = load_config(args.config)

    result = run_tbss_statistical_analysis(
        tbss_dir=args.tbss_dir,
        design_dir=args.design_dir,
        output_dir=output_dir,
        analysis_name=args.analysis_name,
        metrics=args.metrics,
        n_permutations=args.n_permutations,
        tfce=not args.no_tfce,
        cluster_threshold=args.cluster_threshold,
        min_cluster_size=args.min_cluster_size,
        seed=args.seed,
        config=config
    )

    exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
