#!/usr/bin/env python3
"""
FSL Randomise Wrapper

Executes FSL's randomise for nonparametric permutation testing with:
- Threshold-Free Cluster Enhancement (TFCE)
- Multiple comparison correction
- Comprehensive logging and error handling

Adapted from neurovrai for rodent-specific TBSS analysis.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np


class RandomiseError(Exception):
    """Raised when randomise execution fails"""
    pass


def check_fsl_available() -> bool:
    """Check if FSL randomise is accessible"""
    try:
        result = subprocess.run(
            ['which', 'randomise'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def validate_inputs(
    input_file: Path,
    design_mat: Path,
    contrast_con: Path,
    mask: Optional[Path] = None
):
    """
    Validate that all required input files exist and have correct dimensions.

    Raises:
        RandomiseError: If any required files are missing or malformed
    """
    if not input_file.exists():
        raise RandomiseError(f"Input file not found: {input_file}")
    if not design_mat.exists():
        raise RandomiseError(f"Design matrix not found: {design_mat}")
    if not contrast_con.exists():
        raise RandomiseError(f"Contrast file not found: {contrast_con}")
    if mask is not None and not mask.exists():
        raise RandomiseError(f"Mask file not found: {mask}")

    try:
        img = nib.load(input_file)
        if len(img.shape) != 4:
            raise RandomiseError(
                f"Input file must be 4D volume, got shape {img.shape}"
            )
    except RandomiseError:
        raise
    except Exception as e:
        raise RandomiseError(f"Failed to load input file: {e}")

    if mask is not None:
        try:
            mask_img = nib.load(mask)
            if len(mask_img.shape) != 3:
                raise RandomiseError(
                    f"Mask must be 3D volume, got shape {mask_img.shape}"
                )
        except RandomiseError:
            raise
        except Exception as e:
            raise RandomiseError(f"Failed to load mask: {e}")


def run_randomise(
    input_file: Path,
    design_mat: Path,
    contrast_con: Path,
    output_dir: Path,
    mask: Optional[Path] = None,
    n_permutations: int = 5000,
    tfce: bool = True,
    voxel_threshold: Optional[float] = None,
    demean: bool = False,
    variance_smoothing: Optional[float] = None,
    seed: Optional[int] = None
) -> Dict:
    """
    Execute FSL randomise with specified parameters.

    Args:
        input_file: 4D input volume (e.g., all_FA_skeletonised.nii.gz)
        design_mat: FSL design matrix (.mat file)
        contrast_con: FSL contrast matrix (.con file)
        output_dir: Output directory for results
        mask: Optional binary mask (3D volume)
        n_permutations: Number of permutations (default: 5000)
        tfce: Use Threshold-Free Cluster Enhancement (default: True)
        voxel_threshold: Cluster-forming threshold (only if tfce=False)
        demean: Demean data temporally
        variance_smoothing: Variance smoothing in mm
        seed: Random seed for reproducibility

    Returns:
        Dictionary with execution results and output file paths

    Raises:
        RandomiseError: If randomise execution fails
    """
    logger = logging.getLogger("neurofaune.tbss")

    if not check_fsl_available():
        raise RandomiseError(
            "FSL randomise not found. Ensure FSL is installed and $FSLDIR is set."
        )

    validate_inputs(input_file, design_mat, contrast_con, mask)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_basename = output_dir / "randomise"

    cmd = [
        'randomise',
        '-i', str(input_file),
        '-o', str(output_basename),
        '-d', str(design_mat),
        '-t', str(contrast_con),
        '-n', str(n_permutations),
    ]

    if mask is not None:
        cmd.extend(['-m', str(mask)])

    if tfce:
        cmd.append('--T2')
    elif voxel_threshold is not None:
        cmd.extend(['-c', str(voxel_threshold)])

    if demean:
        cmd.append('-D')

    if variance_smoothing is not None:
        cmd.extend(['-v', str(variance_smoothing)])

    if seed is not None:
        cmd.append(f'--seed={seed}')

    logger.info("Executing FSL randomise")
    logger.info(f"  Input: {input_file}")
    logger.info(f"  Design: {design_mat}")
    logger.info(f"  Contrasts: {contrast_con}")
    logger.info(f"  Permutations: {n_permutations}")
    logger.info(f"  TFCE: {tfce}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Command: {' '.join(cmd)}")

    start_time = time.time()
    log_file = output_dir / "randomise.log"

    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )

        elapsed = time.time() - start_time
        logger.info(f"Randomise completed in {elapsed:.1f} seconds")

    except subprocess.CalledProcessError as e:
        logger.error(f"Randomise failed with exit code {e.returncode}")
        logger.error(f"Check log file: {log_file}")
        raise RandomiseError(f"Randomise execution failed: {e}")

    output_files = _collect_output_files(output_dir)

    logger.info(f"Output files: {sum(len(v) for v in output_files.values() if isinstance(v, list))} NIfTI files")

    return {
        'success': True,
        'elapsed_time': elapsed,
        'n_permutations': n_permutations,
        'output_dir': str(output_dir),
        'log_file': str(log_file),
        'output_files': output_files
    }


def _collect_output_files(output_dir: Path) -> Dict[str, List[Path]]:
    """Collect and categorize randomise output files."""
    output_files = {
        'tstat': [],
        'tstat_tfce': [],
        'tstat_corrp': [],
        'fstat': [],
        'fstat_tfce': [],
        'fstat_corrp': []
    }

    for f in sorted(output_dir.glob('randomise_*.nii.gz')):
        name = f.name
        if 'tstat' in name and 'tfce' in name and 'corrp' in name:
            output_files['tstat_corrp'].append(f)
        elif 'tstat' in name and 'tfce' in name:
            output_files['tstat_tfce'].append(f)
        elif 'tstat' in name:
            output_files['tstat'].append(f)
        elif 'fstat' in name and 'tfce' in name and 'corrp' in name:
            output_files['fstat_corrp'].append(f)
        elif 'fstat' in name and 'tfce' in name:
            output_files['fstat_tfce'].append(f)
        elif 'fstat' in name:
            output_files['fstat'].append(f)

    return output_files


def get_significant_voxels(
    corrp_file: Path,
    threshold: float = 0.95
) -> Dict:
    """
    Extract significant voxels from corrected p-value map.

    FSL randomise outputs 1-p values, so threshold at 0.95 = p < 0.05.

    Args:
        corrp_file: Path to randomise corrected p-value file
        threshold: Threshold for significance (default: 0.95 for p<0.05)

    Returns:
        Dictionary with significant voxel statistics
    """
    if not corrp_file.exists():
        raise RandomiseError(f"Corrected p-value file not found: {corrp_file}")

    img = nib.load(corrp_file)
    data = img.get_fdata()

    sig_mask = data >= threshold
    n_significant = int(np.sum(sig_mask))

    if n_significant > 0:
        sig_values = data[sig_mask]
        max_val = float(np.max(sig_values))
        mean_val = float(np.mean(sig_values))
    else:
        max_val = 0.0
        mean_val = 0.0

    return {
        'n_significant_voxels': n_significant,
        'max_corrp': max_val,
        'mean_corrp': mean_val,
        'threshold': threshold,
        'significant': n_significant > 0
    }


def summarize_results(
    output_dir: Path,
    threshold: float = 0.95
) -> Dict:
    """
    Summarize randomise results across all contrasts.

    Args:
        output_dir: Directory containing randomise outputs
        threshold: Significance threshold (default: 0.95 for p<0.05)

    Returns:
        Summary dictionary with per-contrast significant findings
    """
    logger = logging.getLogger("neurofaune.tbss")
    output_dir = Path(output_dir)

    summary = {'contrasts': [], 'threshold': threshold}

    corrp_files = sorted(output_dir.glob('*_corrp_*.nii.gz'))

    for corrp_file in corrp_files:
        name = corrp_file.stem.replace('.nii', '')
        parts = name.split('_')

        contrast_type = None
        contrast_num = None
        for part in parts:
            if 'tstat' in part:
                contrast_type = 'tstat'
                contrast_num = part.replace('tstat', '')
            elif 'fstat' in part:
                contrast_type = 'fstat'
                contrast_num = part.replace('fstat', '')

        sig_info = get_significant_voxels(corrp_file, threshold)

        summary['contrasts'].append({
            'file': str(corrp_file),
            'type': contrast_type,
            'contrast_number': contrast_num,
            'n_significant_voxels': sig_info['n_significant_voxels'],
            'max_corrp': sig_info['max_corrp'],
            'significant': sig_info['significant']
        })

    logger.info(f"Randomise summary (threshold={threshold}, p<{1-threshold}):")
    for contrast in summary['contrasts']:
        status = "SIGNIFICANT" if contrast['significant'] else "not significant"
        logger.info(
            f"  {contrast['type']}{contrast['contrast_number']}: "
            f"{contrast['n_significant_voxels']} voxels ({status})"
        )

    return summary
