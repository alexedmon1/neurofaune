#!/usr/bin/env python3
"""
Regional Homogeneity (ReHo) Analysis for Resting-State fMRI

ReHo measures the similarity/synchronization of time series in a local region
using Kendall's coefficient of concordance (KCC). Higher ReHo values indicate
more synchronized activity in that region.

Handles partial-coverage fMRI (e.g., 9-slice acquisitions) by allowing
reduced neighborhoods at volume boundaries.

References:
- Zang et al. (2004). Regional homogeneity approach to fMRI data analysis.
  NeuroImage, 22(1), 394-400.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Optional
from scipy import stats


def _kendall_w(timeseries: np.ndarray) -> float:
    """
    Compute Kendall's coefficient of concordance (W).

    Parameters
    ----------
    timeseries : ndarray, shape (n_voxels, n_timepoints)
        Time series from neighborhood voxels.

    Returns
    -------
    float
        KCC value between 0 (no agreement) and 1 (perfect agreement).
    """
    k, n = timeseries.shape
    if k < 2:
        return 0.0

    # Rank each voxel's time series
    ranks = np.empty_like(timeseries, dtype=np.float64)
    for i in range(k):
        ranks[i] = stats.rankdata(timeseries[i])

    # Sum ranks across voxels for each timepoint
    rank_sums = ranks.sum(axis=0)

    # Kendall's W = 12 * SS / (K^2 * (N^3 - N))
    ss = np.sum((rank_sums - rank_sums.mean()) ** 2)
    W = (12.0 * ss) / (k ** 2 * (n ** 3 - n))

    return float(W)


def _get_neighborhood_offsets(neighborhood: int) -> list:
    """Return list of (dx, dy, dz) offsets for the given neighborhood size."""
    if neighborhood == 7:
        return [
            (0, 0, 0),
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ]
    elif neighborhood == 19:
        return [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if abs(dx) + abs(dy) + abs(dz) <= 2
        ]
    else:  # 27
        return [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
        ]


def compute_reho_map(func_file: Path,
                     mask_file: Path,
                     output_dir: Path,
                     subject: str,
                     session: str,
                     neighborhood: int = 27) -> dict:
    """
    Compute ReHo (Regional Homogeneity) map for whole brain.

    Parameters
    ----------
    func_file : Path
        Preprocessed 4D functional image (bandpass FILTERED).
    mask_file : Path
        Brain mask.
    output_dir : Path
        Directory to save output map.
    subject : str
        Subject ID (e.g., 'sub-Rat49').
    session : str
        Session ID (e.g., 'ses-p90').
    neighborhood : int
        Neighborhood size: 7 (faces), 19 (faces+edges), or 27 (full cube).

    Returns
    -------
    dict
        Dictionary with keys:
        - reho_file: Path to ReHo map
        - statistics: dict with ReHo summary statistics
    """
    func_file = Path(func_file)
    mask_file = Path(mask_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Computing ReHo (Regional Homogeneity)")
    print("=" * 60)
    print(f"  Input: {func_file.name}")
    print(f"  Neighborhood: {neighborhood} voxels")

    # Load data
    func_img = nib.load(func_file)
    func_data = func_img.get_fdata()

    if func_data.ndim != 4:
        raise ValueError(f"Expected 4D functional data, got {func_data.ndim}D")

    nx, ny, nz, nt = func_data.shape
    print(f"  Dimensions: {nx} x {ny} x {nz} x {nt} timepoints")

    mask_data = nib.load(mask_file).get_fdata().astype(bool)
    n_brain = int(np.sum(mask_data))
    print(f"  Brain voxels: {n_brain}")

    # Precompute voxel standard deviations for quick zero-variance check
    std_map = np.std(func_data, axis=3)

    offsets = _get_neighborhood_offsets(neighborhood)

    # Initialize ReHo map
    reho_data = np.zeros((nx, ny, nz), dtype=np.float32)

    brain_voxels = np.argwhere(mask_data)
    print(f"  Computing KCC for {n_brain} voxels...")

    for idx, (x, y, z) in enumerate(brain_voxels):
        if (idx + 1) % 5000 == 0:
            print(f"    Progress: {idx + 1}/{n_brain} ({100 * (idx + 1) / n_brain:.0f}%)")

        # Collect valid neighbor timeseries (bounds checking handles edge slices)
        neighbors = []
        for dx, dy, dz in offsets:
            xi, yi, zi = x + dx, y + dy, z + dz
            if 0 <= xi < nx and 0 <= yi < ny and 0 <= zi < nz:
                if mask_data[xi, yi, zi] and std_map[xi, yi, zi] > 0:
                    neighbors.append(func_data[xi, yi, zi, :])

        if len(neighbors) < 2:
            continue

        reho_data[x, y, z] = _kendall_w(np.array(neighbors))

    print("  ReHo computation complete")

    # Save
    reho_file = output_dir / f"{subject}_{session}_desc-ReHo_bold.nii.gz"

    hdr = func_img.header.copy()
    hdr.set_data_shape((nx, ny, nz))
    nib.save(nib.Nifti1Image(reho_data, func_img.affine, hdr), reho_file)
    print(f"  Saved ReHo: {reho_file.name}")

    # Statistics
    brain_reho = reho_data[mask_data]
    statistics = {
        'reho': {
            'mean': float(np.mean(brain_reho)),
            'std': float(np.std(brain_reho)),
            'min': float(np.min(brain_reho)),
            'max': float(np.max(brain_reho)),
        },
        'parameters': {
            'neighborhood': neighborhood,
            'n_brain_voxels': n_brain,
            'n_timepoints': nt,
        },
    }

    print(f"\n  ReHo — mean: {statistics['reho']['mean']:.4f}, "
          f"std: {statistics['reho']['std']:.4f}, "
          f"range: [{statistics['reho']['min']:.4f}, {statistics['reho']['max']:.4f}]")
    print("=" * 60)

    return {
        'reho_file': reho_file,
        'statistics': statistics,
    }


def compute_reho_zscore(reho_file: Path,
                        mask_file: Path,
                        output_dir: Path,
                        subject: str,
                        session: str) -> dict:
    """
    Standardize ReHo map to z-scores within the brain mask.

    Parameters
    ----------
    reho_file : Path
        Path to ReHo map.
    mask_file : Path
        Brain mask.
    output_dir : Path
        Directory to save z-scored map.
    subject : str
        Subject ID.
    session : str
        Session ID.

    Returns
    -------
    dict
        Dictionary with key:
        - reho_zscore_file: Path to z-scored ReHo map
    """
    reho_file = Path(reho_file)
    mask_file = Path(mask_file)
    output_dir = Path(output_dir)

    print("  Standardizing ReHo to z-scores...")

    reho_img = nib.load(reho_file)
    reho_data = reho_img.get_fdata()
    mask_data = nib.load(mask_file).get_fdata().astype(bool)

    brain_reho = reho_data[mask_data]
    mean_reho = np.mean(brain_reho)
    std_reho = np.std(brain_reho)

    zscore_data = np.zeros_like(reho_data, dtype=np.float32)
    if std_reho > 0:
        zscore_data[mask_data] = ((brain_reho - mean_reho) / std_reho).astype(np.float32)

    reho_zscore_file = output_dir / f"{subject}_{session}_desc-ReHozscore_bold.nii.gz"
    nib.save(nib.Nifti1Image(zscore_data, reho_img.affine, reho_img.header), reho_zscore_file)

    print(f"  Saved z-scored ReHo: {reho_zscore_file.name}")

    return {
        'reho_zscore_file': reho_zscore_file,
    }
