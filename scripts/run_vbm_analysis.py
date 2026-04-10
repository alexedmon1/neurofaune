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
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from neurofaune.analysis.progress import AnalysisProgress
from neurofaune.analysis.stats.randomise_wrapper import run_randomise, summarize_results
from neurofaune.analysis.stats.cluster_report import generate_reports_for_all_contrasts
from neurofaune.config import load_config, get_config_value


VBM_TISSUES = ['GM', 'WM']

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


def load_subject_list(path: Path) -> List[str]:
    """Load a subject list file (one subject per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def validate_provenance(
    vbm_dir: Path,
    analysis_name: str,
    logger: logging.Logger,
) -> None:
    """
    Validate that a design's provenance matches the current subject list.

    Computes SHA256 of subject_list.txt and compares against the hash
    stored in the design's provenance.json.
    """
    provenance_file = vbm_dir / 'designs' / analysis_name / 'provenance.json'

    if not provenance_file.exists():
        logger.warning(
            f"  No provenance.json for {analysis_name} - "
            "skipping hash validation"
        )
        return

    with open(provenance_file) as f:
        provenance = json.load(f)

    expected_hash = provenance.get('subject_list_sha256')
    if not expected_hash:
        logger.warning(f"  provenance.json for {analysis_name} has no hash - skipping")
        return

    subject_list_path = vbm_dir / 'subject_list.txt'
    h = hashlib.sha256()
    with open(subject_list_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    current_hash = h.hexdigest()

    if current_hash != expected_hash:
        raise ValueError(
            f"Subject list mismatch for design '{analysis_name}'!\n"
            f"  Current subject_list.txt SHA256:  {current_hash[:16]}...\n"
            f"  Design provenance expected:       {expected_hash[:16]}...\n"
            f"  The design was built for a different subject list.\n"
            f"  Re-run prepare_vbm_designs.py to update."
        )

    logger.info(
        f"  Provenance OK for {analysis_name} "
        f"(hash: {current_hash[:16]}..., "
        f"n_design={provenance.get('n_subjects_in_design', '?')})"
    )


def subset_4d_volume(
    input_4d: Path,
    master_subjects: List[str],
    design_subjects: List[str],
    output_path: Path,
    logger: logging.Logger,
) -> Path:
    """Extract volumes from a 4D NIfTI to match a design's subject order."""
    master_index = {subj: i for i, subj in enumerate(master_subjects)}

    indices = []
    missing = []
    for subj in design_subjects:
        if subj in master_index:
            indices.append(master_index[subj])
        else:
            missing.append(subj)

    if missing:
        raise ValueError(
            f"{len(missing)} design subjects not found in master list: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    img = nib.load(input_4d)
    data = img.get_fdata()

    if data.shape[3] != len(master_subjects):
        raise ValueError(
            f"4D volume has {data.shape[3]} volumes but master list has "
            f"{len(master_subjects)} subjects"
        )

    subset_data = data[:, :, :, indices].astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(subset_data, img.affine, img.header), output_path)

    logger.info(
        f"  Subsetted {input_4d.name}: {data.shape[3]} -> {subset_data.shape[3]} volumes"
    )
    return output_path


def _metric_randomise_complete(output_dir: Path, metric: str) -> bool:
    """Check if randomise output already exists for a metric."""
    metric_dir = output_dir / f"randomise_{metric}"
    if not metric_dir.is_dir():
        return False
    corrp_files = list(metric_dir.glob("randomise_tfce_corrp_*.nii.gz"))
    return len(corrp_files) > 0


def run_single_analysis(
    vbm_dir: Path,
    analysis_name: str,
    metrics: List[str],
    master_subjects: List[str],
    n_permutations: int,
    cluster_threshold: float,
    min_cluster_size: int,
    seed: Optional[int],
    config: Optional[Dict],
    logger: logging.Logger,
    skip_existing: bool = False,
    parcellation_override: Optional[Path] = None,
) -> Dict:
    """
    Run randomise for a single analysis (design).

    Uses 3D TFCE (-T) for volumetric VBM data (not 2D TFCE).
    """
    design_dir = vbm_dir / "designs" / analysis_name
    stats_dir = vbm_dir / "stats"
    output_dir = vbm_dir / "randomise" / analysis_name

    logger.info("=" * 70)
    logger.info(f"ANALYSIS: {analysis_name}")
    logger.info("=" * 70)

    # Step 1: Load design subject order
    subject_order_file = design_dir / "subject_order.txt"
    if not subject_order_file.exists():
        raise FileNotFoundError(
            f"subject_order.txt not found in {design_dir}"
        )
    design_subjects = load_subject_list(subject_order_file)
    n_design = len(design_subjects)
    logger.info(f"Design subjects: {n_design}")

    # Validate design matrix dimensions
    design_mat = design_dir / "design.mat"
    design_con = design_dir / "design.con"
    if not design_mat.exists():
        raise FileNotFoundError(f"design.mat not found: {design_mat}")
    if not design_con.exists():
        raise FileNotFoundError(f"design.con not found: {design_con}")

    n_points = None
    n_waves = None
    with open(design_mat) as f:
        for line in f:
            if '/NumPoints' in line:
                n_points = int(line.split()[-1])
            elif '/NumWaves' in line:
                n_waves = int(line.split()[-1])

    if n_points != n_design:
        raise ValueError(
            f"design.mat has NumPoints={n_points} but subject_order.txt has "
            f"{n_design} subjects"
        )
    logger.info(f"Design matrix: {n_points} subjects x {n_waves} predictors")

    n_contrasts = None
    with open(design_con) as f:
        for line in f:
            if '/NumContrasts' in line:
                n_contrasts = int(line.split()[-1])
    logger.info(f"Contrasts: {n_contrasts}")

    # Load contrast names from design summary
    contrast_names = None
    design_summary_file = design_dir / "design_summary.json"
    if design_summary_file.exists():
        with open(design_summary_file) as f:
            design_summary = json.load(f)
        contrast_names = design_summary.get('contrasts', None)

    # Step 2: Subset 4D volumes
    logger.info("\nSubsetting 4D volumes...")
    analysis_mask = stats_dir / "analysis_mask.nii.gz"
    if not analysis_mask.exists():
        raise FileNotFoundError(
            f"Analysis mask not found: {analysis_mask}\n"
            "Run prepare_vbm.py first."
        )

    subset_dir = output_dir / "data"
    subset_dir.mkdir(parents=True, exist_ok=True)
    metric_files = {}

    for metric in metrics:
        master_4d = stats_dir / f"all_{metric}.nii.gz"
        if not master_4d.exists():
            raise FileNotFoundError(
                f"Master 4D not found: {master_4d}\n"
                "Run prepare_vbm.py first."
            )

        subset_file = subset_dir / f"all_{metric}.nii.gz"
        if subset_file.exists():
            existing_shape = nib.load(subset_file).shape
            if len(existing_shape) == 4 and existing_shape[3] == n_design:
                logger.info(
                    f"  {metric}: using existing subset ({existing_shape[3]} volumes)"
                )
                metric_files[metric] = subset_file
                continue
            else:
                logger.info(
                    f"  {metric}: existing subset has wrong shape "
                    f"{existing_shape}, re-creating"
                )

        subset_4d_volume(
            input_4d=master_4d,
            master_subjects=master_subjects,
            design_subjects=design_subjects,
            output_path=subset_file,
            logger=logger,
        )
        metric_files[metric] = subset_file

    # Step 3: Run randomise for each metric (3D TFCE for volumetric data)
    logger.info(f"\nRunning FSL randomise ({n_permutations} permutations, 3D TFCE)...")
    all_results = {}

    for metric in metrics:
        logger.info(f"\n  --- {metric} ---")
        metric_output = output_dir / f"randomise_{metric}"

        if skip_existing and _metric_randomise_complete(output_dir, metric):
            logger.info(f"  {metric}: randomise output exists, skipping (--skip-existing)")
            all_results[metric] = {
                'randomise': {'skipped': True},
                'summary': summarize_results(metric_output, cluster_threshold),
            }
            continue

        randomise_result = run_randomise(
            input_file=metric_files[metric],
            design_mat=design_mat,
            contrast_con=design_con,
            output_dir=metric_output,
            mask=analysis_mask,
            n_permutations=n_permutations,
            tfce=True,
            tfce_2d=False,  # 3D TFCE for VBM (not skeleton-based)
            seed=seed,
        )

        all_results[metric] = {
            'randomise': randomise_result,
            'summary': summarize_results(metric_output, cluster_threshold),
        }

    # Step 4: Extract clusters and generate reports
    logger.info("\nExtracting clusters...")

    sigma_parcellation = None
    if parcellation_override and parcellation_override.exists():
        sigma_parcellation = parcellation_override
    elif config:
        study_root = Path(get_config_value(config, 'paths.study_root', default=''))
        parc_path = (
            study_root / 'atlas' / 'SIGMA_study_space'
            / 'SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz'
        )
        if parc_path.exists():
            sigma_parcellation = parc_path

    for metric in metrics:
        metric_output = output_dir / f"randomise_{metric}"
        reports_dir = output_dir / f"cluster_reports_{metric}"

        cluster_results = generate_reports_for_all_contrasts(
            randomise_output_dir=metric_output,
            output_dir=reports_dir,
            contrast_names=contrast_names,
            sigma_parcellation=sigma_parcellation,
            threshold=cluster_threshold,
            min_cluster_size=min_cluster_size,
        )
        all_results[metric]['clusters'] = cluster_results

    # Summary
    logger.info(f"\n{analysis_name} results:")
    for metric in metrics:
        summary = all_results[metric]['summary']
        for contrast in summary['contrasts']:
            status = "SIGNIFICANT" if contrast['significant'] else "ns"
            logger.info(
                f"  {metric} {contrast['type']}{contrast['contrast_number']}: "
                f"{contrast['n_significant_voxels']} voxels ({status})"
            )

    # Save analysis summary
    summary_file = output_dir / 'analysis_summary.json'
    summary_data = {
        'analysis_name': analysis_name,
        'analysis_type': 'vbm',
        'date': datetime.now().isoformat(),
        'n_subjects': n_design,
        'n_predictors': n_waves,
        'n_contrasts': n_contrasts,
        'contrast_names': contrast_names,
        'metrics': metrics,
        'n_permutations': n_permutations,
        'tfce_mode': '3D',
        'results': {
            metric: {
                'n_significant_contrasts': sum(
                    1 for c in all_results[metric]['summary']['contrasts']
                    if c['significant']
                ),
                'contrasts': all_results[metric]['summary']['contrasts'],
            }
            for metric in metrics
        },
    }
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)

    return {
        'success': True,
        'analysis_name': analysis_name,
        'n_subjects': n_design,
        'output_dir': str(output_dir),
        'results': all_results,
    }


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
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip metrics that already have randomise output')
    parser.add_argument('--parcellation', type=Path,
                        help='Override SIGMA parcellation NIfTI')

    args = parser.parse_args()

    vbm_dir = args.vbm_dir
    if not vbm_dir.exists():
        print(f"ERROR: VBM directory not found: {vbm_dir}", file=sys.stderr)
        sys.exit(1)

    # Auto-detect metrics from vbm_config.json
    if args.metrics is None:
        config_file = vbm_dir / 'vbm_config.json'
        if config_file.exists():
            with open(config_file) as f:
                vbm_cfg = json.load(f)
            args.metrics = vbm_cfg.get('tissues', VBM_TISSUES)
        else:
            args.metrics = VBM_TISSUES

    logger = setup_logging(vbm_dir)

    logger.info("=" * 80)
    logger.info("VBM VOXEL-WISE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"VBM dir: {vbm_dir}")
    logger.info(f"Analyses: {args.analyses}")
    logger.info(f"Tissues: {args.metrics}")
    logger.info(f"Permutations: {args.n_permutations}")
    logger.info(f"TFCE: 3D (-T)")

    # Load config
    config = None
    config_path = args.config
    if config_path is None:
        for parent in vbm_dir.resolve().parents:
            candidate = parent / 'config.yaml'
            if candidate.exists():
                config_path = candidate
                break
    if config_path and config_path.exists():
        config = load_config(config_path)
        logger.info(f"Config loaded: {config_path}")
    elif args.config:
        logger.warning(f"Config not found: {args.config}")

    # Load master subject list
    master_list_file = vbm_dir / "subject_list.txt"
    if not master_list_file.exists():
        logger.error(f"Master subject list not found: {master_list_file}")
        logger.error("Run prepare_vbm.py first.")
        sys.exit(1)
    master_subjects = load_subject_list(master_list_file)
    logger.info(f"Master subject list: {len(master_subjects)} subjects")

    # Verify stats directory
    stats_dir = vbm_dir / "stats"
    if not stats_dir.exists():
        logger.error(f"Stats directory not found: {stats_dir}")
        logger.error("Run prepare_vbm.py first.")
        sys.exit(1)

    # Validate provenance for each analysis before running
    for analysis in args.analyses:
        try:
            validate_provenance(vbm_dir, analysis, logger)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

    # Run each analysis
    results = {}
    progress = AnalysisProgress(vbm_dir, "run_vbm_analysis.py", len(args.analyses))
    completed = 0
    failed = 0

    for analysis in args.analyses:
        progress.update(task=analysis, phase="running randomise", completed=completed, failed=failed)

        try:
            result = run_single_analysis(
                vbm_dir=vbm_dir,
                analysis_name=analysis,
                metrics=args.metrics,
                master_subjects=master_subjects,
                n_permutations=args.n_permutations,
                cluster_threshold=args.cluster_threshold,
                min_cluster_size=args.min_cluster_size,
                seed=args.seed,
                config=config,
                logger=logger,
                skip_existing=args.skip_existing,
                parcellation_override=args.parcellation,
            )
            results[analysis] = result
            completed += 1
        except Exception as e:
            logger.error(f"\nFAILED: {analysis}: {e}")
            results[analysis] = {'success': False, 'error': str(e)}
            failed += 1
            continue

    # Register with unified reporting
    try:
        from neurofaune.reporting import register as report_register

        analysis_root = vbm_dir.parent  # vbm_dir is .../analysis/vbm
        for analysis, result in results.items():
            if not result.get('success'):
                continue
            output_dir = vbm_dir / "randomise" / analysis
            summary_json = output_dir / "analysis_summary.json"

            n_sig = 0
            for metric_results in result.get('results', {}).values():
                summary = metric_results.get('summary', {})
                for c in summary.get('contrasts', []):
                    if c.get('significant'):
                        n_sig += 1

            report_register(
                analysis_root=analysis_root,
                entry_id=f"vbm_{analysis}",
                analysis_type="vbm",
                display_name=f"VBM: {analysis}",
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
                auto_generate_index=analysis == list(results.keys())[-1],
            )
    except Exception as exc:
        logger.warning("Failed to register with reporting system: %s", exc)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ALL ANALYSES COMPLETE")
    logger.info("=" * 80)

    for analysis, result in results.items():
        if result['success']:
            logger.info(f"  {analysis}: {result['n_subjects']} subjects - OK")
        else:
            logger.info(f"  {analysis}: FAILED - {result.get('error', 'unknown')}")

    n_ok = sum(1 for r in results.values() if r['success'])
    logger.info(f"\n{n_ok}/{len(results)} analyses completed successfully")

    progress.finish()

    logger.info("\nTo check significance:")
    logger.info(
        f"  fslstats {vbm_dir}/randomise/*/randomise_*/randomise_tfce_corrp_tstat*.nii.gz -R"
    )

    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == '__main__':
    main()
