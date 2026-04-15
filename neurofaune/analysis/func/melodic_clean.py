#!/usr/bin/env python3
"""
MELODIC RSN Cleaning Tool.

Identifies resting-state networks (RSNs) from group MELODIC ICA output,
produces a cleaned 4D IC map containing only RSN components, writes an
index JSON, and generates a mosaic figure.

Two modes
---------
auto
    Scores each component on spatial and spectral features:
    - Low-frequency power ratio (RSNs are low-frequency dominated)
    - Edge fraction (artefacts cluster at brain edges)
    - CSF/background fraction (noise components bleed outside brain)
    - Spatial smoothness (artefacts tend to be rougher)
    Components with noise score below ``--noise-threshold`` are kept as RSNs.

manual
    User supplies ``--components 1,3,7,12`` (1-based indices). Those
    components are kept unconditionally.

In both modes the tool writes:
- ``melodic_IC_RSN.nii.gz``     — 4D volume (RSN components only)
- ``rsn_index.json``            — component metadata and classification
- ``rsn_mosaic.png``            — slice mosaic of RSN spatial maps

Usage
-----
  # Automated:
  uv run python scripts/run_melodic_clean.py \\
      --melodic-dir /mnt/arborea/bpa-rat/analysis/melodic/p90 \\
      --mask /mnt/arborea/bpa-rat/atlas/SIGMA_study_space/SIGMA_InVivo_Brain_Mask.nii.gz \\
      --tr 0.5 \\
      --mode auto

  # Manual:
  uv run python scripts/run_melodic_clean.py \\
      --melodic-dir /mnt/arborea/bpa-rat/analysis/melodic/p90 \\
      --mask /mnt/arborea/bpa-rat/atlas/SIGMA_study_space/SIGMA_InVivo_Brain_Mask.nii.gz \\
      --mode manual --components 1,4,7,9,12,15
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage, signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _load_melodic_outputs(melodic_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, nib.Nifti1Image]:
    """Load IC maps, mixing matrix, frequency matrix from MELODIC directory."""
    ic_img = nib.load(melodic_dir / "melodic_IC.nii.gz")
    ic_data = ic_img.get_fdata()  # (x, y, z, n_components)

    mix_file = melodic_dir / "melodic_mix"
    ft_file = melodic_dir / "melodic_FTmix"

    mix = np.loadtxt(mix_file) if mix_file.exists() else None
    ftmix = np.loadtxt(ft_file) if ft_file.exists() else None

    if mix is not None and mix.ndim == 1:
        mix = mix[:, np.newaxis]
    if ftmix is not None and ftmix.ndim == 1:
        ftmix = ftmix[:, np.newaxis]

    return ic_data, mix, ftmix, ic_img


def _low_freq_ratio(ftmix: np.ndarray, ic_idx: int, tr: float,
                    low_band: Tuple[float, float] = (0.01, 0.1)) -> float:
    """
    Fraction of spectral power in the RSN low-frequency band.

    MELODIC's FTmix rows correspond to frequencies from 0 to Nyquist.
    Higher ratio → more signal-like.
    """
    if ftmix is None:
        return np.nan

    col = ftmix[:, ic_idx] if ftmix.ndim > 1 else ftmix
    n_freqs = len(col)
    nyquist = 0.5 / tr
    freqs = np.linspace(0, nyquist, n_freqs)

    low_mask = (freqs >= low_band[0]) & (freqs <= low_band[1])
    total_power = np.sum(col ** 2)
    if total_power == 0:
        return 0.0
    return float(np.sum(col[low_mask] ** 2) / total_power)


def _edge_fraction(spatial_map: np.ndarray, brain_mask: np.ndarray,
                   z_thresh: float = 1.5, erode_iters: int = 2) -> float:
    """
    Fraction of active voxels that lie at the brain edge.

    High edge fraction → likely motion / susceptibility artefact.
    """
    brain_eroded = ndimage.binary_erosion(brain_mask, iterations=erode_iters)
    edge_mask = brain_mask & ~brain_eroded

    comp_binary = (np.abs(spatial_map) > z_thresh) & brain_mask
    n_comp = np.sum(comp_binary)
    if n_comp == 0:
        return 0.0
    return float(np.sum(comp_binary & edge_mask) / n_comp)


def _background_fraction(spatial_map: np.ndarray, brain_mask: np.ndarray,
                          z_thresh: float = 1.5) -> float:
    """
    Fraction of active voxels that fall outside the brain mask.

    High background fraction → noise / ringing artefact.
    """
    comp_active = np.abs(spatial_map) > z_thresh
    n_active = np.sum(comp_active)
    if n_active == 0:
        return 0.0
    return float(np.sum(comp_active & ~brain_mask) / n_active)


def _spatial_smoothness(spatial_map: np.ndarray, brain_mask: np.ndarray) -> float:
    """
    Mean absolute gradient within the brain mask (lower = smoother).

    RSNs tend to be spatially smooth; artefacts tend to be rough.
    Returns the normalised roughness (0 = perfectly smooth).
    """
    masked = spatial_map * brain_mask
    grad = np.sqrt(
        ndimage.sobel(masked, axis=0) ** 2
        + ndimage.sobel(masked, axis=1) ** 2
        + ndimage.sobel(masked, axis=2) ** 2
    )
    denom = np.std(masked[brain_mask]) if np.any(brain_mask) else 1.0
    if denom == 0:
        return 0.0
    return float(np.mean(grad[brain_mask]) / denom)


def _spatial_extent(spatial_map: np.ndarray, brain_mask: np.ndarray,
                    z_thresh: float = 1.5) -> float:
    """Fraction of brain voxels that are active (above threshold)."""
    n_brain = np.sum(brain_mask)
    if n_brain == 0:
        return 0.0
    n_active = np.sum((np.abs(spatial_map) > z_thresh) & brain_mask)
    return float(n_active / n_brain)


# ---------------------------------------------------------------------------
# Automated classifier
# ---------------------------------------------------------------------------

def classify_rsn_components(
    melodic_dir: Path,
    brain_mask_file: Path,
    tr: float,
    low_freq_thresh: float = 0.3,
    edge_thresh: float = 0.4,
    bg_thresh: float = 0.1,
    roughness_thresh: float = 1.5,
    noise_score_thresh: float = 1.5,
) -> Dict:
    """
    Automatically classify MELODIC components as RSN or noise.

    Scoring (each criterion contributes 1.0 to the noise score):

    - low-frequency ratio < ``low_freq_thresh``  → +1.0  (not RSN-like)
    - edge fraction > ``edge_thresh``             → +1.0  (artefact)
    - background fraction > ``bg_thresh``         → +1.0  (outside brain)
    - spatial roughness > ``roughness_thresh``    → +0.5  (not smooth)

    Components with ``noise_score < noise_score_thresh`` are labelled RSN.

    Parameters
    ----------
    melodic_dir : Path
        MELODIC output directory (contains ``melodic_IC.nii.gz``, etc.).
    brain_mask_file : Path
        Brain mask in the same space as the IC maps.
    tr : float
        Repetition time in seconds (needed for frequency axis).
    low_freq_thresh : float
        Minimum low-frequency power ratio to be considered signal.
    edge_thresh : float
        Maximum edge fraction tolerated for RSNs.
    bg_thresh : float
        Maximum background fraction tolerated for RSNs.
    roughness_thresh : float
        Maximum roughness tolerated for RSNs.
    noise_score_thresh : float
        Components with noise score below this are kept as RSNs.

    Returns
    -------
    dict
        ``n_components``, ``rsn_indices`` (0-based), ``components`` list
        with per-component features and classification.
    """
    ic_data, mix, ftmix, ic_img = _load_melodic_outputs(melodic_dir)
    brain_mask = nib.load(brain_mask_file).get_fdata().astype(bool)

    n_components = ic_data.shape[3]
    logger.info("Classifying %d components (auto mode)...", n_components)

    components = []
    rsn_indices = []

    for idx in range(n_components):
        spatial_map = ic_data[:, :, :, idx]

        low_freq = _low_freq_ratio(ftmix, idx, tr) if ftmix is not None else np.nan
        edge_frac = _edge_fraction(spatial_map, brain_mask)
        bg_frac = _background_fraction(spatial_map, brain_mask)
        roughness = _spatial_smoothness(spatial_map, brain_mask)
        extent = _spatial_extent(spatial_map, brain_mask)

        # Noise scoring
        noise_score = 0.0
        if not np.isnan(low_freq) and low_freq < low_freq_thresh:
            noise_score += 1.0
        if edge_frac > edge_thresh:
            noise_score += 1.0
        if bg_frac > bg_thresh:
            noise_score += 1.0
        if roughness > roughness_thresh:
            noise_score += 0.5

        is_rsn = noise_score < noise_score_thresh

        comp = {
            "index_0based": idx,
            "index_1based": idx + 1,
            "label": "RSN" if is_rsn else "noise",
            "noise_score": round(noise_score, 3),
            "features": {
                "low_freq_ratio": round(float(low_freq), 3) if not np.isnan(low_freq) else None,
                "edge_fraction": round(edge_frac, 3),
                "background_fraction": round(bg_frac, 3),
                "spatial_roughness": round(roughness, 3),
                "spatial_extent": round(extent, 3),
            },
        }
        components.append(comp)
        if is_rsn:
            rsn_indices.append(idx)
            logger.info(
                "  IC%02d → RSN  (score=%.2f | lfr=%.2f | edge=%.2f | bg=%.2f)",
                idx + 1, noise_score, low_freq if not np.isnan(low_freq) else -1,
                edge_frac, bg_frac,
            )
        else:
            logger.info(
                "  IC%02d → noise (score=%.2f | lfr=%.2f | edge=%.2f | bg=%.2f)",
                idx + 1, noise_score, low_freq if not np.isnan(low_freq) else -1,
                edge_frac, bg_frac,
            )

    logger.info(
        "Classification complete: %d RSNs / %d noise out of %d total",
        len(rsn_indices), n_components - len(rsn_indices), n_components,
    )

    return {
        "mode": "auto",
        "n_components": n_components,
        "rsn_indices": rsn_indices,
        "n_rsn": len(rsn_indices),
        "thresholds": {
            "low_freq_thresh": low_freq_thresh,
            "edge_thresh": edge_thresh,
            "bg_thresh": bg_thresh,
            "roughness_thresh": roughness_thresh,
            "noise_score_thresh": noise_score_thresh,
        },
        "components": components,
    }


# ---------------------------------------------------------------------------
# Manual selection
# ---------------------------------------------------------------------------

def select_manual_components(
    melodic_dir: Path,
    component_indices: List[int],
) -> Dict:
    """
    Build classification result from a manually supplied component list.

    Parameters
    ----------
    melodic_dir : Path
        MELODIC output directory.
    component_indices : list of int
        1-based component indices to keep as RSNs.

    Returns
    -------
    dict
        Same structure as ``classify_rsn_components`` output.
    """
    ic_img = nib.load(melodic_dir / "melodic_IC.nii.gz")
    n_components = ic_img.shape[3]

    rsn_indices_0 = []
    for idx_1 in component_indices:
        idx_0 = idx_1 - 1
        if idx_0 < 0 or idx_0 >= n_components:
            raise ValueError(
                f"Component {idx_1} is out of range (1–{n_components})"
            )
        rsn_indices_0.append(idx_0)

    components = []
    for idx in range(n_components):
        is_rsn = idx in rsn_indices_0
        components.append({
            "index_0based": idx,
            "index_1based": idx + 1,
            "label": "RSN" if is_rsn else "noise",
            "noise_score": None,
            "features": None,
        })

    logger.info(
        "Manual selection: keeping %d / %d components as RSNs: %s",
        len(rsn_indices_0), n_components,
        [i + 1 for i in rsn_indices_0],
    )

    return {
        "mode": "manual",
        "n_components": n_components,
        "rsn_indices": rsn_indices_0,
        "n_rsn": len(rsn_indices_0),
        "components": components,
    }


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def build_rsn_volume(
    melodic_dir: Path,
    rsn_indices: List[int],
    output_path: Path,
) -> Path:
    """
    Extract RSN components and save as a new 4D NIfTI.

    Parameters
    ----------
    melodic_dir : Path
        MELODIC output directory.
    rsn_indices : list of int
        0-based indices of components to keep.
    output_path : Path
        Where to write the cleaned 4D volume.

    Returns
    -------
    Path
        Path to the written file.
    """
    ic_img = nib.load(melodic_dir / "melodic_IC.nii.gz")
    ic_data = ic_img.get_fdata()

    rsn_data = ic_data[:, :, :, rsn_indices]
    rsn_img = nib.Nifti1Image(rsn_data.astype(np.float32), ic_img.affine, ic_img.header)
    nib.save(rsn_img, output_path)
    logger.info("Saved RSN volume (%d components): %s", len(rsn_indices), output_path)
    return output_path


def build_rsn_mosaic(
    melodic_dir: Path,
    rsn_indices: List[int],
    output_path: Path,
    bg_image: Optional[Path] = None,
    z_thresh: float = 1.5,
    n_slices: int = 5,
) -> Path:
    """
    Generate a mosaic figure of RSN spatial maps.

    Lays out one row per RSN component, each row showing ``n_slices``
    axial slices through the peak activation. The IC spatial maps are
    overlaid (hot/cold colormap) on a background anatomical if provided,
    otherwise on the IC map itself.

    Parameters
    ----------
    melodic_dir : Path
        MELODIC output directory.
    rsn_indices : list of int
        0-based indices of RSN components to plot.
    output_path : Path
        Where to save the PNG figure.
    bg_image : Path, optional
        Background anatomical image (e.g. SIGMA template). If None,
        a maximum-intensity projection of the IC maps is used.
    z_thresh : float
        Z-score threshold for IC map display.
    n_slices : int
        Number of axial slices per component row.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap

    ic_img = nib.load(melodic_dir / "melodic_IC.nii.gz")
    ic_data = ic_img.get_fdata()

    n_rsn = len(rsn_indices)
    if n_rsn == 0:
        raise ValueError("No RSN components to plot")

    # Load background
    if bg_image and Path(bg_image).exists():
        bg_data = nib.load(bg_image).get_fdata()
        bg_data = (bg_data - bg_data.min()) / (bg_data.max() - bg_data.min() + 1e-8)
    else:
        # Use mean of absolute IC maps as background
        bg_data = np.mean(np.abs(ic_data), axis=3)
        bg_data = (bg_data - bg_data.min()) / (bg_data.max() - bg_data.min() + 1e-8)

    # Colourmap: blue→transparent→red (hot/cold symmetric)
    cmap_pos = plt.cm.get_cmap("Reds")
    cmap_neg = plt.cm.get_cmap("Blues_r")

    fig_height = max(2.0 * n_rsn, 4)
    fig = plt.figure(figsize=(n_slices * 2.2, fig_height), facecolor="black")
    gs = gridspec.GridSpec(n_rsn, n_slices, figure=fig, hspace=0.05, wspace=0.03)

    for row, ic_idx in enumerate(rsn_indices):
        spatial_map = ic_data[:, :, :, ic_idx]

        # Choose slices spanning the peak activation
        z_profile = np.max(np.abs(spatial_map), axis=(0, 1))
        peak_z = int(np.argmax(z_profile))
        half = n_slices // 2
        z_indices = np.clip(
            np.linspace(max(0, peak_z - 10), min(ic_data.shape[2] - 1, peak_z + 10), n_slices),
            0, ic_data.shape[2] - 1,
        ).astype(int)

        for col, z in enumerate(z_indices):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(
                np.rot90(bg_data[:, :, z]),
                cmap="gray", vmin=0, vmax=1, aspect="auto",
            )
            ic_slice = spatial_map[:, :, z]
            # Positive overlay
            pos = np.ma.masked_where(ic_slice < z_thresh, ic_slice)
            ax.imshow(
                np.rot90(pos),
                cmap="Reds", vmin=z_thresh, vmax=max(z_thresh * 2, np.max(ic_slice)),
                alpha=0.8, aspect="auto",
            )
            # Negative overlay
            neg = np.ma.masked_where(ic_slice > -z_thresh, -ic_slice)
            ax.imshow(
                np.rot90(neg),
                cmap="Blues", vmin=z_thresh, vmax=max(z_thresh * 2, np.max(-ic_slice)),
                alpha=0.8, aspect="auto",
            )
            ax.axis("off")

            if col == 0:
                ax.set_ylabel(
                    f"IC{ic_idx + 1:02d}",
                    color="white", fontsize=7, rotation=0,
                    labelpad=28, va="center",
                )

    fig.suptitle("RSN Components", color="white", fontsize=10, y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    logger.info("Saved RSN mosaic (%d components): %s", n_rsn, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def clean_melodic(
    melodic_dir: Path,
    brain_mask_file: Path,
    tr: float,
    mode: str = "auto",
    component_indices: Optional[List[int]] = None,
    bg_image: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    # auto-mode thresholds
    low_freq_thresh: float = 0.3,
    edge_thresh: float = 0.4,
    bg_thresh: float = 0.1,
    roughness_thresh: float = 1.5,
    noise_score_thresh: float = 1.5,
) -> Dict:
    """
    Identify RSN components, write cleaned outputs, and return a summary.

    Parameters
    ----------
    melodic_dir : Path
        MELODIC output directory.
    brain_mask_file : Path
        Brain mask in IC map space.
    tr : float
        Repetition time in seconds.
    mode : ``'auto'`` or ``'manual'``
        Classification mode.
    component_indices : list of int, optional
        Required when ``mode='manual'``. 1-based component indices to keep.
    bg_image : Path, optional
        Background anatomical image for the mosaic figure.
    output_dir : Path, optional
        Where to write outputs. Defaults to ``melodic_dir``.
    low_freq_thresh, edge_thresh, bg_thresh, roughness_thresh, noise_score_thresh
        Thresholds for auto-mode classifier (see ``classify_rsn_components``).

    Returns
    -------
    dict
        Classification result + paths to output files.
    """
    melodic_dir = Path(melodic_dir)
    output_dir = Path(output_dir) if output_dir else melodic_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (melodic_dir / "melodic_IC.nii.gz").exists():
        raise FileNotFoundError(
            f"melodic_IC.nii.gz not found in {melodic_dir}"
        )

    if mode == "auto":
        result = classify_rsn_components(
            melodic_dir=melodic_dir,
            brain_mask_file=brain_mask_file,
            tr=tr,
            low_freq_thresh=low_freq_thresh,
            edge_thresh=edge_thresh,
            bg_thresh=bg_thresh,
            roughness_thresh=roughness_thresh,
            noise_score_thresh=noise_score_thresh,
        )
    elif mode == "manual":
        if not component_indices:
            raise ValueError("component_indices required for manual mode")
        result = select_manual_components(melodic_dir, component_indices)
    else:
        raise ValueError(f"mode must be 'auto' or 'manual', got '{mode}'")

    rsn_indices = result["rsn_indices"]

    if not rsn_indices:
        logger.warning("No RSN components identified — outputs will be empty")

    # Write cleaned 4D volume
    rsn_vol_path = output_dir / "melodic_IC_RSN.nii.gz"
    if rsn_indices:
        build_rsn_volume(melodic_dir, rsn_indices, rsn_vol_path)
        result["rsn_volume"] = str(rsn_vol_path)
    else:
        result["rsn_volume"] = None

    # Write mosaic figure
    mosaic_path = output_dir / "rsn_mosaic.png"
    if rsn_indices:
        build_rsn_mosaic(
            melodic_dir=melodic_dir,
            rsn_indices=rsn_indices,
            output_path=mosaic_path,
            bg_image=bg_image,
        )
        result["rsn_mosaic"] = str(mosaic_path)
    else:
        result["rsn_mosaic"] = None

    # Write index JSON
    index_path = output_dir / "rsn_index.json"
    index_path.write_text(json.dumps(result, indent=2))
    result["rsn_index"] = str(index_path)

    logger.info(
        "RSN cleaning complete: %d components kept → %s",
        len(rsn_indices), output_dir,
    )
    return result
