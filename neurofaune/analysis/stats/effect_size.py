#!/usr/bin/env python3
"""
Voxelwise Effect Size Maps for FSL Randomise Results

Computes Cohen's d and partial eta-squared maps from randomise t-stat and
F-stat outputs. Requires the design matrix (.mat) and contrast (.con) files
to properly scale Cohen's d via the GLM contrast variance factor.

Cohen's d for GLM contrast c:
    d = t * sqrt(c' (X'X)^{-1} c)

Partial eta-squared:
    t-contrasts: eta_p^2 = t^2 / (t^2 + df)
    F-tests:     eta_p^2 = F * df1 / (F * df1 + df2)
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

logger = logging.getLogger("neurofaune.tbss")


def read_fsl_vest(path: Path) -> np.ndarray:
    """Read an FSL VEST-format matrix (.mat, .con, .fts)."""
    lines = path.read_text().strip().split('\n')
    matrix_lines = []
    in_matrix = False
    for line in lines:
        if line.strip().startswith('/Matrix'):
            in_matrix = True
            continue
        if in_matrix and line.strip():
            matrix_lines.append([float(x) for x in line.split()])
    return np.array(matrix_lines)


def _parse_vest_header(path: Path) -> Dict[str, int]:
    """Parse /NumWaves and /NumPoints from VEST header."""
    header = {}
    for line in path.read_text().strip().split('\n'):
        line = line.strip()
        if line.startswith('/NumWaves'):
            header['n_waves'] = int(line.split()[-1])
        elif line.startswith('/NumPoints'):
            header['n_points'] = int(line.split()[-1])
        elif line.startswith('/NumContrasts'):
            header['n_contrasts'] = int(line.split()[-1])
    return header


def compute_contrast_variance_factors(
    design_mat: np.ndarray,
    contrast_mat: np.ndarray
) -> np.ndarray:
    """
    Compute c'(X'X)^{-1}c for each contrast row.

    This is the variance inflation factor that converts t-stats to Cohen's d:
        d = t * sqrt(c'(X'X)^{-1}c)

    Args:
        design_mat: Design matrix X (n_obs x n_predictors)
        contrast_mat: Contrast matrix C (n_contrasts x n_predictors)

    Returns:
        Array of variance factors, one per contrast
    """
    XtX = design_mat.T @ design_mat
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
        logger.warning("Design matrix is singular, using pseudo-inverse for effect size")

    factors = np.array([
        float(c @ XtX_inv @ c.T)
        for c in contrast_mat
    ])
    return factors


def compute_cohens_d_map(
    tstat_file: Path,
    variance_factor: float,
    output_file: Optional[Path] = None
) -> Path:
    """
    Compute Cohen's d map from a t-statistic map.

    d = t * sqrt(c'(X'X)^{-1}c)

    Args:
        tstat_file: Path to t-statistic NIfTI
        variance_factor: c'(X'X)^{-1}c for this contrast
        output_file: Output path (default: alongside tstat with _cohend suffix)

    Returns:
        Path to saved Cohen's d map
    """
    img = nib.load(tstat_file)
    tdata = img.get_fdata()

    d_map = tdata * np.sqrt(variance_factor)

    if output_file is None:
        output_file = tstat_file.parent / tstat_file.name.replace(
            '_tstat', '_cohend'
        )

    out_img = nib.Nifti1Image(d_map.astype(np.float32), img.affine, img.header)
    nib.save(out_img, output_file)
    return output_file


def compute_partial_etasq_from_tstat(
    tstat_file: Path,
    df_error: int,
    output_file: Optional[Path] = None
) -> Path:
    """
    Compute partial eta-squared map from t-statistic.

    eta_p^2 = t^2 / (t^2 + df)

    Args:
        tstat_file: Path to t-statistic NIfTI
        df_error: Residual degrees of freedom (n - rank(X))
        output_file: Output path (default: alongside tstat with _etasq suffix)

    Returns:
        Path to saved partial eta-squared map
    """
    img = nib.load(tstat_file)
    tdata = img.get_fdata()

    t2 = tdata ** 2
    etasq = t2 / (t2 + df_error)

    if output_file is None:
        output_file = tstat_file.parent / tstat_file.name.replace(
            '_tstat', '_etasq'
        )

    out_img = nib.Nifti1Image(etasq.astype(np.float32), img.affine, img.header)
    nib.save(out_img, output_file)
    return output_file


def compute_partial_etasq_from_fstat(
    fstat_file: Path,
    df_numerator: int,
    df_error: int,
    output_file: Optional[Path] = None
) -> Path:
    """
    Compute partial eta-squared map from F-statistic.

    eta_p^2 = F * df1 / (F * df1 + df2)

    Args:
        fstat_file: Path to F-statistic NIfTI
        df_numerator: Numerator degrees of freedom (number of contrasts in F-test)
        df_error: Residual degrees of freedom (n - rank(X))
        output_file: Output path (default: alongside fstat with _etasq suffix)

    Returns:
        Path to saved partial eta-squared map
    """
    img = nib.load(fstat_file)
    fdata = img.get_fdata()

    etasq = (fdata * df_numerator) / (fdata * df_numerator + df_error)

    if output_file is None:
        output_file = fstat_file.parent / fstat_file.name.replace(
            '_fstat', '_etasq_f'
        )

    out_img = nib.Nifti1Image(etasq.astype(np.float32), img.affine, img.header)
    nib.save(out_img, output_file)
    return output_file


def _extract_contrast_number(filename: str) -> Optional[int]:
    """Extract contrast number from randomise filename like randomise_tstat2.nii.gz."""
    m = re.search(r'_(?:tstat|fstat)(\d+)', filename)
    return int(m.group(1)) if m else None


def _count_ftest_contrasts(fts_mat: np.ndarray, ftest_idx: int) -> int:
    """Count number of included contrasts (1s) in an F-test row."""
    return int(np.sum(fts_mat[ftest_idx]))


def generate_effect_size_maps(
    randomise_output_dir: Path,
    design_mat_file: Path,
    design_con_file: Path,
    design_fts_file: Optional[Path] = None
) -> Dict[str, List[Path]]:
    """
    Generate all effect size maps for a randomise output directory.

    Computes:
    - Cohen's d maps for each t-stat (using proper GLM scaling)
    - Partial eta-squared maps for each t-stat and F-stat

    Args:
        randomise_output_dir: Directory with randomise outputs
        design_mat_file: Path to FSL design matrix (.mat)
        design_con_file: Path to FSL contrast file (.con)
        design_fts_file: Optional path to F-test file (.fts)

    Returns:
        Dictionary with 'cohens_d', 'etasq_t', 'etasq_f' lists of output paths
    """
    randomise_output_dir = Path(randomise_output_dir)

    # Read design files
    X = read_fsl_vest(design_mat_file)
    C = read_fsl_vest(design_con_file)

    n_obs, n_pred = X.shape
    df_error = n_obs - np.linalg.matrix_rank(X)

    # Compute contrast variance factors for Cohen's d
    variance_factors = compute_contrast_variance_factors(X, C)

    # Read F-test spec if available
    fts_mat = None
    if design_fts_file and design_fts_file.exists():
        fts_mat = read_fsl_vest(design_fts_file)

    output_paths = {'cohens_d': [], 'etasq_t': [], 'etasq_f': []}

    # Process t-stat files
    tstat_files = sorted(randomise_output_dir.glob('randomise_tstat*.nii.gz'))
    for tstat_file in tstat_files:
        contrast_num = _extract_contrast_number(tstat_file.name)
        if contrast_num is None:
            continue

        # contrast_num is 1-based, array is 0-based
        idx = contrast_num - 1
        if idx >= len(variance_factors):
            logger.warning(
                f"Contrast {contrast_num} exceeds design ({len(variance_factors)} contrasts), "
                f"skipping effect size for {tstat_file.name}"
            )
            continue

        # Cohen's d
        d_path = compute_cohens_d_map(tstat_file, variance_factors[idx])
        output_paths['cohens_d'].append(d_path)
        logger.info(f"  Cohen's d: {d_path.name} (variance factor={variance_factors[idx]:.4f})")

        # Partial eta-squared from t
        etasq_path = compute_partial_etasq_from_tstat(tstat_file, df_error)
        output_paths['etasq_t'].append(etasq_path)
        logger.info(f"  Partial eta-sq: {etasq_path.name} (df_error={df_error})")

    # Process F-stat files
    fstat_files = sorted(randomise_output_dir.glob('randomise_fstat*.nii.gz'))
    for fstat_file in fstat_files:
        ftest_num = _extract_contrast_number(fstat_file.name)
        if ftest_num is None:
            continue

        idx = ftest_num - 1

        # Determine numerator df from F-test spec
        if fts_mat is not None and idx < len(fts_mat):
            df_num = _count_ftest_contrasts(fts_mat, idx)
        else:
            # Fallback: assume 1 df
            df_num = 1
            logger.warning(
                f"No .fts file for F-stat {ftest_num}, assuming df_numerator=1"
            )

        etasq_path = compute_partial_etasq_from_fstat(fstat_file, df_num, df_error)
        output_paths['etasq_f'].append(etasq_path)
        logger.info(
            f"  Partial eta-sq (F): {etasq_path.name} "
            f"(df_num={df_num}, df_error={df_error})"
        )

    n_total = sum(len(v) for v in output_paths.values())
    logger.info(f"Generated {n_total} effect size maps in {randomise_output_dir}")

    return output_paths
