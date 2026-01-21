"""
Registration quality control metrics and visualizations.

This module provides:
- Dice coefficient calculation
- Correlation metrics
- Overlay visualizations for registration QC
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def compute_dice_coefficient(
    mask1: np.ndarray,
    mask2: np.ndarray
) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Parameters
    ----------
    mask1, mask2 : np.ndarray
        Binary masks (will be binarized if not already)

    Returns
    -------
    float
        Dice coefficient (0 to 1, higher is better)
    """
    mask1 = (mask1 > 0).astype(bool)
    mask2 = (mask2 > 0).astype(bool)

    intersection = np.sum(mask1 & mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)

    if sum_masks == 0:
        return 0.0

    return 2.0 * intersection / sum_masks


def compute_correlation(
    img1: np.ndarray,
    img2: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute Pearson correlation between two images.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Images to compare
    mask : np.ndarray, optional
        Binary mask to restrict correlation calculation

    Returns
    -------
    float
        Pearson correlation coefficient (-1 to 1)
    """
    if mask is not None:
        mask = mask > 0
        v1 = img1[mask].flatten()
        v2 = img2[mask].flatten()
    else:
        # Use non-zero voxels
        combined_mask = (img1 > 0) | (img2 > 0)
        v1 = img1[combined_mask].flatten()
        v2 = img2[combined_mask].flatten()

    if len(v1) == 0 or len(v2) == 0:
        return 0.0

    # Normalize
    v1 = (v1 - np.mean(v1)) / (np.std(v1) + 1e-10)
    v2 = (v2 - np.mean(v2)) / (np.std(v2) + 1e-10)

    return np.corrcoef(v1, v2)[0, 1]


def compute_registration_metrics(
    moving_file: Path,
    fixed_file: Path,
    warped_file: Path,
    moving_mask: Optional[Path] = None,
    fixed_mask: Optional[Path] = None
) -> Dict[str, float]:
    """
    Compute comprehensive registration QC metrics.

    Parameters
    ----------
    moving_file : Path
        Original moving image (before registration)
    fixed_file : Path
        Fixed/reference image
    warped_file : Path
        Moving image after registration (in fixed space)
    moving_mask : Path, optional
        Brain mask for moving image
    fixed_mask : Path, optional
        Brain mask for fixed image

    Returns
    -------
    dict
        Dictionary with:
        - correlation_before: Correlation before registration
        - correlation_after: Correlation after registration
        - correlation_improvement: Improvement in correlation
        - dice_masks: Dice coefficient of brain masks (if provided)
    """
    # Load images
    moving_img = nib.load(moving_file).get_fdata()
    fixed_img = nib.load(fixed_file).get_fdata()
    warped_img = nib.load(warped_file).get_fdata()

    # Load masks if provided
    fixed_mask_data = None
    if fixed_mask and Path(fixed_mask).exists():
        fixed_mask_data = nib.load(fixed_mask).get_fdata()

    # Compute correlations
    corr_before = compute_correlation(moving_img, fixed_img, fixed_mask_data)
    corr_after = compute_correlation(warped_img, fixed_img, fixed_mask_data)

    metrics = {
        'correlation_before': float(corr_before),
        'correlation_after': float(corr_after),
        'correlation_improvement': float(corr_after - corr_before),
    }

    # Compute Dice if masks provided
    if moving_mask and fixed_mask:
        if Path(moving_mask).exists() and Path(fixed_mask).exists():
            moving_mask_data = nib.load(moving_mask).get_fdata()
            fixed_mask_data = nib.load(fixed_mask).get_fdata()
            metrics['dice_masks'] = float(compute_dice_coefficient(moving_mask_data, fixed_mask_data))

    return metrics


def create_edge_overlay(
    background: np.ndarray,
    overlay: np.ndarray,
    threshold_pct: float = 95
) -> np.ndarray:
    """
    Create edge overlay for registration QC.

    Extracts edges from overlay image and displays on background.

    Parameters
    ----------
    background : np.ndarray
        Background image (e.g., fixed image)
    overlay : np.ndarray
        Overlay image (e.g., warped moving image)
    threshold_pct : float
        Percentile threshold for edge detection

    Returns
    -------
    np.ndarray
        Edge mask
    """
    from scipy import ndimage

    # Compute gradient magnitude
    gradient = np.zeros_like(overlay)
    for axis in range(3):
        gradient += ndimage.sobel(overlay, axis=axis) ** 2
    gradient = np.sqrt(gradient)

    # Threshold to get edges
    threshold = np.percentile(gradient[gradient > 0], threshold_pct)
    edges = gradient > threshold

    return edges


def create_checkerboard(
    img1: np.ndarray,
    img2: np.ndarray,
    n_tiles: int = 8
) -> np.ndarray:
    """
    Create checkerboard pattern for registration comparison.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Images to compare (must be same shape)
    n_tiles : int
        Number of tiles per dimension

    Returns
    -------
    np.ndarray
        Checkerboard image
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have same shape: {img1.shape} vs {img2.shape}")

    shape = img1.shape
    checkerboard = np.zeros_like(img1)

    tile_size = [s // n_tiles for s in shape]

    for i in range(n_tiles):
        for j in range(n_tiles):
            for k in range(n_tiles):
                # Determine which image to use based on checkerboard pattern
                use_img1 = (i + j + k) % 2 == 0

                # Calculate slice bounds
                si = slice(i * tile_size[0], (i + 1) * tile_size[0])
                sj = slice(j * tile_size[1], (j + 1) * tile_size[1])
                sk = slice(k * tile_size[2], (k + 1) * tile_size[2])

                if use_img1:
                    checkerboard[si, sj, sk] = img1[si, sj, sk]
                else:
                    checkerboard[si, sj, sk] = img2[si, sj, sk]

    return checkerboard


def generate_registration_qc_figure(
    fixed_file: Path,
    warped_file: Path,
    output_file: Path,
    fixed_mask: Optional[Path] = None,
    title: str = "Registration QC",
    slice_indices: Optional[List[int]] = None,
    n_slices: int = 9,
    dpi: int = 150
) -> Path:
    """
    Generate registration QC figure with overlays.

    Creates a figure with:
    - Row 1: Fixed image slices
    - Row 2: Warped image slices
    - Row 3: Edge overlay (warped edges on fixed)
    - Row 4: Checkerboard pattern

    Parameters
    ----------
    fixed_file : Path
        Fixed/reference image
    warped_file : Path
        Warped moving image
    output_file : Path
        Output figure path
    fixed_mask : Path, optional
        Brain mask for fixed image (improves visualization)
    title : str
        Figure title
    slice_indices : list, optional
        Specific slice indices to show (auto-selected if None)
    n_slices : int
        Number of slices to show (default: 9)
    dpi : int
        Figure DPI

    Returns
    -------
    Path
        Path to saved figure
    """
    # Load images
    fixed_img = nib.load(fixed_file)
    warped_img = nib.load(warped_file)

    fixed_data = fixed_img.get_fdata()
    warped_data = warped_img.get_fdata()

    # Ensure same shape
    if fixed_data.shape != warped_data.shape:
        print(f"Warning: Shape mismatch - fixed {fixed_data.shape} vs warped {warped_data.shape}")
        # Crop to smaller shape
        min_shape = tuple(min(f, w) for f, w in zip(fixed_data.shape, warped_data.shape))
        fixed_data = fixed_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        warped_data = warped_data[:min_shape[0], :min_shape[1], :min_shape[2]]

    # Load mask if provided
    mask_data = None
    if fixed_mask and Path(fixed_mask).exists():
        mask_data = nib.load(fixed_mask).get_fdata()
        if mask_data.shape != fixed_data.shape:
            mask_data = mask_data[:fixed_data.shape[0], :fixed_data.shape[1], :fixed_data.shape[2]]

    # Select slices (coronal view - axis 2 for typical rodent data)
    n_total_slices = fixed_data.shape[2]

    if slice_indices is None:
        # Auto-select evenly spaced slices, avoiding edges
        start = n_total_slices // 10
        end = n_total_slices - start
        slice_indices = np.linspace(start, end, n_slices, dtype=int).tolist()

    # Normalize images for display
    def normalize(img, mask=None):
        if mask is not None:
            brain_vals = img[mask > 0]
            if len(brain_vals) > 0:
                vmin, vmax = np.percentile(brain_vals, [2, 98])
            else:
                vmin, vmax = np.percentile(img[img > 0], [2, 98])
        else:
            vmin, vmax = np.percentile(img[img > 0], [2, 98]) if np.any(img > 0) else (0, 1)
        return np.clip((img - vmin) / (vmax - vmin + 1e-10), 0, 1)

    fixed_norm = normalize(fixed_data, mask_data)
    warped_norm = normalize(warped_data, mask_data)

    # Create edge overlay
    edges = create_edge_overlay(fixed_data, warped_data)

    # Create checkerboard
    checkerboard = create_checkerboard(fixed_norm, warped_norm, n_tiles=8)

    # Create figure
    fig, axes = plt.subplots(4, len(slice_indices), figsize=(2 * len(slice_indices), 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    row_labels = ['Fixed', 'Warped', 'Edge Overlay', 'Checkerboard']

    for row_idx, (row_data, label) in enumerate([
        (fixed_norm, 'Fixed'),
        (warped_norm, 'Warped'),
        (None, 'Edge Overlay'),  # Special handling
        (checkerboard, 'Checkerboard')
    ]):
        for col_idx, slice_idx in enumerate(slice_indices):
            ax = axes[row_idx, col_idx]

            if label == 'Edge Overlay':
                # Show fixed with warped edges overlaid - rotate for proper coronal display
                fixed_slice = np.rot90(fixed_norm[:, :, slice_idx], k=1)
                ax.imshow(fixed_slice, cmap='gray', origin='lower')
                edge_slice = np.rot90(edges[:, :, slice_idx], k=1)
                # Create red overlay for edges
                edge_rgba = np.zeros((*edge_slice.shape, 4))
                edge_rgba[edge_slice > 0] = [1, 0, 0, 0.7]  # Red with alpha
                ax.imshow(edge_rgba, origin='lower')
            else:
                # Rotate 90° for proper coronal display (dorsal up)
                data_slice = np.rot90(row_data[:, :, slice_idx], k=1)
                ax.imshow(data_slice, cmap='gray', origin='lower')

            ax.axis('off')

            # Add labels
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10, fontweight='bold')
            if row_idx == 0:
                ax.set_title(f'Slice {slice_idx}', fontsize=9)

    plt.tight_layout()

    # Save figure
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Registration QC figure saved to: {output_file}")
    return output_file


def generate_atlas_overlay_figure(
    anatomical_file: Path,
    atlas_file: Path,
    output_file: Path,
    title: str = "Atlas Overlay",
    n_slices: int = 9,
    alpha: float = 0.4,
    dpi: int = 150
) -> Path:
    """
    Generate figure showing atlas labels overlaid on anatomical image.

    Parameters
    ----------
    anatomical_file : Path
        Anatomical image (background)
    atlas_file : Path
        Atlas labels in anatomical space
    output_file : Path
        Output figure path
    title : str
        Figure title
    n_slices : int
        Number of slices to show
    alpha : float
        Overlay transparency (0-1)
    dpi : int
        Figure DPI

    Returns
    -------
    Path
        Path to saved figure
    """
    # Load images
    anat_img = nib.load(anatomical_file)
    atlas_img = nib.load(atlas_file)

    anat_data = anat_img.get_fdata()
    atlas_data = atlas_img.get_fdata()

    # Ensure same shape
    if anat_data.shape != atlas_data.shape:
        print(f"Warning: Shape mismatch - anat {anat_data.shape} vs atlas {atlas_data.shape}")
        min_shape = tuple(min(a, b) for a, b in zip(anat_data.shape, atlas_data.shape))
        anat_data = anat_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        atlas_data = atlas_data[:min_shape[0], :min_shape[1], :min_shape[2]]

    # Select slices
    n_total_slices = anat_data.shape[2]
    start = n_total_slices // 10
    end = n_total_slices - start
    slice_indices = np.linspace(start, end, n_slices, dtype=int).tolist()

    # Normalize anatomical
    vmin, vmax = np.percentile(anat_data[anat_data > 0], [2, 98]) if np.any(anat_data > 0) else (0, 1)
    anat_norm = np.clip((anat_data - vmin) / (vmax - vmin + 1e-10), 0, 1)

    # Create colormap for atlas
    n_labels = int(atlas_data.max())
    np.random.seed(42)  # Reproducible colors
    colors = np.random.rand(n_labels + 1, 3)
    colors[0] = [0, 0, 0]  # Background is black

    # Create figure
    n_cols = min(n_slices, 5)
    n_rows = (n_slices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    fig.suptitle(f"{title}\n({n_labels} regions)", fontsize=14, fontweight='bold')

    axes = axes.flatten() if n_slices > 1 else [axes]

    for idx, slice_idx in enumerate(slice_indices):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Show anatomical - rotate 90° for proper coronal display (dorsal up)
        anat_slice = np.rot90(anat_norm[:, :, slice_idx], k=1)
        ax.imshow(anat_slice, cmap='gray', origin='lower')

        # Overlay atlas with colors - same rotation
        atlas_slice = np.rot90(atlas_data[:, :, slice_idx].astype(int), k=1)
        atlas_rgb = colors[atlas_slice]

        # Create RGBA with alpha for non-zero regions
        atlas_rgba = np.zeros((*atlas_slice.shape, 4))
        atlas_rgba[..., :3] = atlas_rgb
        atlas_rgba[..., 3] = (atlas_slice > 0).astype(float) * alpha

        ax.imshow(atlas_rgba, origin='lower')
        ax.set_title(f'Slice {slice_idx}', fontsize=10)
        ax.axis('off')

    # Hide empty axes
    for idx in range(len(slice_indices), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Atlas overlay figure saved to: {output_file}")
    return output_file


def generate_template_qc_report(
    template_file: Path,
    sigma_file: Path,
    warped_template_file: Path,
    output_dir: Path,
    cohort: str,
    sigma_mask: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive QC report for template-to-SIGMA registration.

    Parameters
    ----------
    template_file : Path
        Study template
    sigma_file : Path
        SIGMA atlas template
    warped_template_file : Path
        Study template warped to SIGMA space
    output_dir : Path
        Output directory for QC files
    cohort : str
        Cohort name (for labeling)
    sigma_mask : Path, optional
        SIGMA brain mask

    Returns
    -------
    dict
        Dictionary with:
        - metrics: Registration metrics (correlation, Dice)
        - figures: Paths to QC figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute metrics
    print(f"\nComputing registration QC metrics for {cohort}...")

    metrics = compute_registration_metrics(
        moving_file=template_file,
        fixed_file=sigma_file,
        warped_file=warped_template_file,
        fixed_mask=sigma_mask
    )

    print(f"  Correlation (before): {metrics['correlation_before']:.3f}")
    print(f"  Correlation (after):  {metrics['correlation_after']:.3f}")
    print(f"  Improvement:          {metrics['correlation_improvement']:.3f}")
    if 'dice_masks' in metrics:
        print(f"  Dice coefficient:     {metrics['dice_masks']:.3f}")

    # Generate figures
    figures = {}

    # Registration overlay figure
    overlay_fig = output_dir / f'registration_qc_{cohort}.png'
    generate_registration_qc_figure(
        fixed_file=sigma_file,
        warped_file=warped_template_file,
        output_file=overlay_fig,
        fixed_mask=sigma_mask,
        title=f"Template-to-SIGMA Registration: {cohort}"
    )
    figures['registration_overlay'] = overlay_fig

    # Save metrics to JSON
    import json
    metrics_file = output_dir / f'registration_metrics_{cohort}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    figures['metrics_json'] = metrics_file

    return {
        'metrics': metrics,
        'figures': figures
    }
