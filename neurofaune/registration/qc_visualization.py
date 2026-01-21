"""
QC visualization for registration results.

Generates visual quality control figures for:
1. Slice correspondence - showing how partial-coverage modalities align with T2w
2. Template registration - showing subject-to-template alignment quality
3. Atlas propagation - showing how SIGMA labels map to subject space
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid X11 errors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from neurofaune.registration.slice_correspondence import SliceCorrespondenceResult


def create_checkerboard(img1: np.ndarray, img2: np.ndarray, n_tiles: int = 8) -> np.ndarray:
    """
    Create a checkerboard pattern combining two images.

    Useful for visualizing registration quality - misalignment shows as
    discontinuities at tile boundaries.
    """
    # Normalize images to 0-1
    img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
    img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)

    # Create checkerboard pattern
    h, w = img1.shape[:2]
    tile_h = h // n_tiles
    tile_w = w // n_tiles

    result = np.zeros_like(img1_norm)

    for i in range(n_tiles):
        for j in range(n_tiles):
            y_start = i * tile_h
            y_end = (i + 1) * tile_h if i < n_tiles - 1 else h
            x_start = j * tile_w
            x_end = (j + 1) * tile_w if j < n_tiles - 1 else w

            if (i + j) % 2 == 0:
                result[y_start:y_end, x_start:x_end] = img1_norm[y_start:y_end, x_start:x_end]
            else:
                result[y_start:y_end, x_start:x_end] = img2_norm[y_start:y_end, x_start:x_end]

    return result


def create_edge_overlay(
    base_img: np.ndarray,
    overlay_img: np.ndarray,
    threshold: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create edge overlay for registration visualization.

    Returns RGB image with base in grayscale and overlay edges in color.
    """
    from scipy.ndimage import sobel

    # Normalize base image
    base_norm = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)

    # Compute edges on overlay
    overlay_norm = (overlay_img - overlay_img.min()) / (overlay_img.max() - overlay_img.min() + 1e-8)
    edge_x = sobel(overlay_norm, axis=0)
    edge_y = sobel(overlay_norm, axis=1)
    edges = np.sqrt(edge_x**2 + edge_y**2)
    edges = edges / (edges.max() + 1e-8)

    # Create RGB output
    rgb = np.stack([base_norm, base_norm, base_norm], axis=-1)

    # Add red edges
    edge_mask = edges > threshold
    rgb[edge_mask, 0] = 1.0  # Red channel
    rgb[edge_mask, 1] = 0.3  # Reduce green
    rgb[edge_mask, 2] = 0.3  # Reduce blue

    return rgb, edges, edge_mask


def plot_slice_correspondence(
    partial_data: np.ndarray,
    full_data: np.ndarray,
    result: SliceCorrespondenceResult,
    output_file: Optional[Path] = None,
    title: str = "Slice Correspondence QC"
) -> plt.Figure:
    """
    Create QC visualization for slice correspondence results.

    Shows:
    - Top row: Partial slices with their matched T2w slices
    - Middle row: Correlation profile across search space
    - Bottom row: Per-slice correlation values

    Parameters
    ----------
    partial_data : np.ndarray
        Partial coverage 3D volume
    full_data : np.ndarray
        Full coverage 3D volume (T2w)
    result : SliceCorrespondenceResult
        Slice correspondence result
    output_file : Path, optional
        Save figure to this path
    title : str
        Figure title

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    n_partial = partial_data.shape[2]

    # Select slices to display (evenly spaced)
    n_display = min(6, n_partial)
    display_indices = np.linspace(0, n_partial - 1, n_display).astype(int)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, n_display + 1, height_ratios=[1, 1, 0.5, 0.5],
                           width_ratios=[1] * n_display + [0.05])

    # Normalize data for display
    partial_norm = partial_data / (partial_data.max() + 1e-8)
    full_norm = full_data / (full_data.max() + 1e-8)

    # Row 1: Partial slices
    for col, partial_idx in enumerate(display_indices):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(partial_norm[:, :, partial_idx].T, cmap='gray', origin='lower',
                  aspect='auto')
        ax.set_title(f'Partial [{partial_idx}]', fontsize=10)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Partial Volume', fontsize=12, fontweight='bold')

    # Row 2: Matched T2w slices
    for col, partial_idx in enumerate(display_indices):
        ax = fig.add_subplot(gs[1, col])
        full_idx = result.slice_mapping[partial_idx]
        ax.imshow(full_norm[:, :, full_idx].T, cmap='gray', origin='lower',
                  aspect='auto')

        # Get correlation for this slice
        corr = result.intensity_correlations[partial_idx] if partial_idx < len(result.intensity_correlations) else 0
        ax.set_title(f'T2w [{full_idx}]\nr={corr:.2f}', fontsize=10)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Matched T2w', fontsize=12, fontweight='bold')

    # Row 3: Per-slice correlation profile
    ax_corr = fig.add_subplot(gs[2, :-1])
    correlations = result.intensity_correlations
    x = range(len(correlations))

    # Color bars by correlation quality
    colors = ['red' if c < 0.3 else 'orange' if c < 0.5 else 'green' for c in correlations]
    ax_corr.bar(x, correlations, color=colors, alpha=0.7, edgecolor='black')
    ax_corr.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (0.5)')
    ax_corr.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
    ax_corr.set_xlabel('Partial Slice Index')
    ax_corr.set_ylabel('Correlation')
    ax_corr.set_title(f'Per-Slice Correlation (Mean: {np.mean(correlations):.3f})')
    ax_corr.set_ylim(0, 1)
    ax_corr.legend(loc='lower right', fontsize=8)

    # Row 4: Summary info
    ax_info = fig.add_subplot(gs[3, :-1])
    ax_info.axis('off')

    info_text = (
        f"Correspondence Result:\n"
        f"  Partial slices 0-{n_partial-1} → T2w slices {result.start_slice}-{result.end_slice}\n"
        f"  Method: {result.method_used}\n"
        f"  Intensity confidence: {result.intensity_confidence:.3f}\n"
        f"  Landmark confidence: {result.landmark_confidence:.3f}\n"
        f"  Combined confidence: {result.combined_confidence:.3f}\n"
    )

    if result.landmarks_found:
        info_text += f"  Landmarks: {result.landmarks_found}\n"

    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_file}")

    return fig


def plot_registration_quality(
    fixed_data: np.ndarray,
    moving_data: np.ndarray,
    warped_data: np.ndarray,
    output_file: Optional[Path] = None,
    title: str = "Registration QC",
    slice_axis: int = 2
) -> plt.Figure:
    """
    Create QC visualization for registration quality.

    Shows:
    - Row 1: Fixed (template), Moving (subject), Warped (subject in template space)
    - Row 2: Checkerboard before/after registration
    - Row 3: Edge overlay before/after

    Parameters
    ----------
    fixed_data : np.ndarray
        Fixed/reference image (template)
    moving_data : np.ndarray
        Original moving image (subject)
    warped_data : np.ndarray
        Warped moving image (subject in template space)
    output_file : Path, optional
        Save figure to this path
    title : str
        Figure title
    slice_axis : int
        Axis along which to take slices (0=sagittal, 1=coronal, 2=axial)

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Select slices to display
    n_slices = fixed_data.shape[slice_axis]
    slice_indices = np.linspace(5, n_slices - 5, 6).astype(int)

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 6, height_ratios=[1, 1, 1, 1])

    # Normalize for display
    def norm(data):
        return (data - data.min()) / (data.max() - data.min() + 1e-8)

    fixed_norm = norm(fixed_data)
    moving_norm = norm(moving_data)
    warped_norm = norm(warped_data)

    # Row 1: Fixed, Moving, Warped
    for col, slice_idx in enumerate(slice_indices):
        if slice_axis == 0:
            f_slice = fixed_norm[slice_idx, :, :]
            w_slice = warped_norm[slice_idx, :, :]
        elif slice_axis == 1:
            f_slice = fixed_norm[:, slice_idx, :]
            w_slice = warped_norm[:, slice_idx, :]
        else:
            f_slice = fixed_norm[:, :, slice_idx]
            w_slice = warped_norm[:, :, slice_idx]

        # Fixed
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(f_slice.T, cmap='gray', origin='lower', aspect='auto')
        ax.set_title(f'z={slice_idx}', fontsize=10)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Template\n(Fixed)', fontsize=11, fontweight='bold')

        # Warped
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(w_slice.T, cmap='gray', origin='lower', aspect='auto')
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Subject\n(Warped)', fontsize=11, fontweight='bold')

    # Row 2: Checkerboard
    for col, slice_idx in enumerate(slice_indices):
        if slice_axis == 0:
            f_slice = fixed_norm[slice_idx, :, :]
            w_slice = warped_norm[slice_idx, :, :]
        elif slice_axis == 1:
            f_slice = fixed_norm[:, slice_idx, :]
            w_slice = warped_norm[:, slice_idx, :]
        else:
            f_slice = fixed_norm[:, :, slice_idx]
            w_slice = warped_norm[:, :, slice_idx]

        checker = create_checkerboard(f_slice, w_slice, n_tiles=8)

        ax = fig.add_subplot(gs[2, col])
        ax.imshow(checker.T, cmap='gray', origin='lower', aspect='auto')
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Checkerboard\n(Alignment)', fontsize=11, fontweight='bold')

    # Row 3: Edge overlay
    for col, slice_idx in enumerate(slice_indices):
        if slice_axis == 0:
            f_slice = fixed_norm[slice_idx, :, :]
            w_slice = warped_norm[slice_idx, :, :]
        elif slice_axis == 1:
            f_slice = fixed_norm[:, slice_idx, :]
            w_slice = warped_norm[:, slice_idx, :]
        else:
            f_slice = fixed_norm[:, :, slice_idx]
            w_slice = warped_norm[:, :, slice_idx]

        rgb, _, _ = create_edge_overlay(w_slice, f_slice, threshold=0.1)

        ax = fig.add_subplot(gs[3, col])
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower', aspect='auto')
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Template Edges\non Subject', fontsize=11, fontweight='bold')

    # Compute registration metrics
    # Correlation between warped and fixed
    mask = (fixed_norm > 0.05) | (warped_norm > 0.05)
    if mask.sum() > 100:
        from scipy.stats import pearsonr
        corr, _ = pearsonr(fixed_norm[mask].flatten(), warped_norm[mask].flatten())
    else:
        corr = 0.0

    fig.suptitle(f'{title}\nCorrelation: {corr:.3f}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_file}")

    return fig


def plot_slice_correspondence_detailed(
    partial_data: np.ndarray,
    full_data: np.ndarray,
    result: SliceCorrespondenceResult,
    output_file: Optional[Path] = None,
    title: str = "Detailed Slice Correspondence"
) -> plt.Figure:
    """
    Create detailed side-by-side visualization showing each partial slice
    next to its matched T2w slice.

    Parameters
    ----------
    partial_data : np.ndarray
        Partial coverage 3D volume
    full_data : np.ndarray
        Full coverage 3D volume (T2w)
    result : SliceCorrespondenceResult
        Slice correspondence result
    output_file : Path, optional
        Save figure to this path
    title : str
        Figure title

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    n_partial = partial_data.shape[2]

    # Calculate grid size
    n_cols = 4  # 2 pairs of (partial, T2w) per row
    n_rows = (n_partial + 1) // 2  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(20, 3 * n_rows))

    # Normalize
    partial_norm = partial_data / (partial_data.max() + 1e-8)
    full_norm = full_data / (full_data.max() + 1e-8)

    for partial_idx in range(n_partial):
        row = partial_idx // 2
        col_base = (partial_idx % 2) * 4

        full_idx = result.slice_mapping[partial_idx]
        corr = result.intensity_correlations[partial_idx] if partial_idx < len(result.intensity_correlations) else 0

        # Partial slice
        ax_p = axes[row, col_base] if n_rows > 1 else axes[col_base]
        ax_p.imshow(partial_norm[:, :, partial_idx].T, cmap='gray', origin='lower', aspect='auto')
        ax_p.set_title(f'Partial [{partial_idx}]', fontsize=9)
        ax_p.axis('off')

        # Matched T2w slice
        ax_t = axes[row, col_base + 1] if n_rows > 1 else axes[col_base + 1]
        ax_t.imshow(full_norm[:, :, full_idx].T, cmap='gray', origin='lower', aspect='auto')

        # Color code by correlation quality
        color = 'green' if corr > 0.6 else 'orange' if corr > 0.4 else 'red'
        ax_t.set_title(f'T2w [{full_idx}] r={corr:.2f}', fontsize=9, color=color)
        ax_t.axis('off')

    # Hide unused axes
    for i in range(n_partial, n_rows * 2):
        row = i // 2
        col_base = (i % 2) * 4
        if n_rows > 1:
            axes[row, col_base].axis('off')
            axes[row, col_base + 1].axis('off')

    # Add summary
    fig.suptitle(
        f'{title}\n'
        f'Partial → T2w: slices {result.start_slice}-{result.end_slice} | '
        f'Mean r={np.mean(result.intensity_correlations):.3f} | '
        f'Confidence: {result.combined_confidence:.2f}',
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_file}")

    return fig


def create_registration_report(
    subject: str,
    session: str,
    preproc_t2w: Path,
    template: Path,
    warped_to_template: Path,
    output_dir: Path,
    slice_correspondence_result: Optional[SliceCorrespondenceResult] = None,
    partial_data: Optional[np.ndarray] = None,
    modality: str = 'anat'
) -> Dict[str, Path]:
    """
    Generate complete registration QC report.

    Parameters
    ----------
    subject : str
        Subject ID
    session : str
        Session ID
    preproc_t2w : Path
        Preprocessed T2w image
    template : Path
        Cohort template
    warped_to_template : Path
        Subject T2w warped to template space
    output_dir : Path
        Output directory for QC figures
    slice_correspondence_result : SliceCorrespondenceResult, optional
        Slice correspondence result (for partial modalities)
    partial_data : np.ndarray, optional
        Partial coverage data (for slice correspondence QC)
    modality : str
        Modality name

    Returns
    -------
    dict
        Dictionary with paths to generated QC figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}

    print(f"\nGenerating registration QC for {subject} {session}...")

    # Load data
    t2w_data = nib.load(preproc_t2w).get_fdata()
    template_data = nib.load(template).get_fdata()
    warped_data = nib.load(warped_to_template).get_fdata()

    # 1. Registration quality figure
    reg_qc_file = output_dir / f'{subject}_{session}_registration_qc.png'
    plot_registration_quality(
        fixed_data=template_data,
        moving_data=t2w_data,
        warped_data=warped_data,
        output_file=reg_qc_file,
        title=f'{subject} {session} → Template Registration'
    )
    reports['registration_qc'] = reg_qc_file
    plt.close()

    # 2. Slice correspondence figure (if provided)
    if slice_correspondence_result is not None and partial_data is not None:
        corr_qc_file = output_dir / f'{subject}_{session}_{modality}_slice_correspondence.png'
        plot_slice_correspondence(
            partial_data=partial_data,
            full_data=t2w_data,
            result=slice_correspondence_result,
            output_file=corr_qc_file,
            title=f'{subject} {session} {modality.upper()} Slice Correspondence'
        )
        reports['slice_correspondence'] = corr_qc_file
        plt.close()

        # Detailed view
        detail_file = output_dir / f'{subject}_{session}_{modality}_slice_detail.png'
        plot_slice_correspondence_detailed(
            partial_data=partial_data,
            full_data=t2w_data,
            result=slice_correspondence_result,
            output_file=detail_file,
            title=f'{subject} {session} {modality.upper()} Slice Detail'
        )
        reports['slice_detail'] = detail_file
        plt.close()

    print(f"  Generated {len(reports)} QC figures")

    return reports
