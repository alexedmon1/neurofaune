"""
ROI-to-ROI functional connectivity from SIGMA-space BOLD timeseries.

Extracts mean timeseries per atlas ROI, computes Pearson correlation
matrices, and applies Fisher z-transform for group-level analysis.
"""

from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np


def extract_roi_timeseries(
    bold_4d: Path,
    atlas: Path,
    mask: Optional[Path] = None,
    min_voxels: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mean timeseries for each ROI in the atlas.

    Parameters
    ----------
    bold_4d : Path
        4D BOLD timeseries in SIGMA space.
    atlas : Path
        SIGMA anatomical parcellation (integer labels).
    mask : Path, optional
        Brain mask to intersect with atlas ROIs.
    min_voxels : int
        Minimum voxels required for an ROI to be included.

    Returns
    -------
    timeseries : ndarray, shape (n_timepoints, n_rois)
        Mean timeseries per ROI.
    roi_labels : ndarray, shape (n_rois,)
        Integer labels of included ROIs (sorted).
    """
    bold_img = nib.load(bold_4d)
    bold_data = bold_img.get_fdata()
    atlas_data = nib.load(atlas).get_fdata().astype(int)

    if mask is not None:
        mask_data = nib.load(mask).get_fdata().astype(bool)
        atlas_data = atlas_data * mask_data

    # Find all non-zero labels
    all_labels = np.unique(atlas_data)
    all_labels = all_labels[all_labels > 0]

    n_timepoints = bold_data.shape[3]
    timeseries_list = []
    kept_labels = []

    for label in all_labels:
        roi_mask = atlas_data == label
        n_voxels = roi_mask.sum()
        if n_voxels < min_voxels:
            continue

        roi_ts = bold_data[roi_mask, :].mean(axis=0)
        timeseries_list.append(roi_ts)
        kept_labels.append(label)

    timeseries = np.column_stack(timeseries_list)  # (n_timepoints, n_rois)
    roi_labels = np.array(kept_labels, dtype=int)

    print(f"  Extracted timeseries: {timeseries.shape[1]} ROIs, "
          f"{timeseries.shape[0]} timepoints "
          f"(skipped {len(all_labels) - len(kept_labels)} ROIs < {min_voxels} voxels)")

    return timeseries, roi_labels


def compute_fc_matrix(timeseries: np.ndarray) -> np.ndarray:
    """
    Compute functional connectivity matrix (Pearson r -> Fisher z).

    Parameters
    ----------
    timeseries : ndarray, shape (n_timepoints, n_rois)
        Mean timeseries per ROI.

    Returns
    -------
    fc_z : ndarray, shape (n_rois, n_rois)
        Fisher z-transformed correlation matrix. Diagonal is zero.
    """
    r_matrix = np.corrcoef(timeseries, rowvar=False)

    # Clip to avoid arctanh(1) = inf
    np.fill_diagonal(r_matrix, 0.0)
    r_clipped = np.clip(r_matrix, -0.9999, 0.9999)

    fc_z = np.arctanh(r_clipped)
    return fc_z


def save_fc_matrix(
    fc_matrix: np.ndarray,
    roi_labels: np.ndarray,
    output_path: Path,
) -> Tuple[Path, Path]:
    """
    Save FC matrix as .npy and ROI labels as .tsv.

    Parameters
    ----------
    fc_matrix : ndarray, shape (n_rois, n_rois)
        Functional connectivity matrix.
    roi_labels : ndarray, shape (n_rois,)
        Integer labels for each ROI.
    output_path : Path
        Base output path (without extension). Produces:
        - {output_path}.npy  — the matrix
        - {output_path}_labels.tsv — ROI label mapping

    Returns
    -------
    npy_path : Path
        Path to saved .npy file.
    tsv_path : Path
        Path to saved .tsv label file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    npy_path = output_path.with_suffix(".npy")
    np.save(npy_path, fc_matrix)

    tsv_path = output_path.parent / (output_path.stem + "_labels.tsv")
    with open(tsv_path, "w") as f:
        f.write("index\tlabel\n")
        for i, label in enumerate(roi_labels):
            f.write(f"{i}\t{label}\n")

    print(f"  FC matrix: {npy_path.name} ({fc_matrix.shape[0]}x{fc_matrix.shape[1]})")
    print(f"  Labels:    {tsv_path.name}")

    return npy_path, tsv_path
