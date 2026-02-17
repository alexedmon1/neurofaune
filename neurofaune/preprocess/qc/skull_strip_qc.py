"""
Shared skull stripping QC utilities.

Provides reusable functions for generating skull strip QC figures, metrics,
and HTML snippets that can be embedded in any modality's QC report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List

# Modality-specific contour colors
MODALITY_COLORS = {
    'anat': 'red',
    'dwi': 'cyan',
    'func': 'lime',
    'msme': 'orange',
}


def calculate_skull_strip_metrics(
    original_data: np.ndarray,
    brain_data: np.ndarray,
    mask_data: np.ndarray,
    voxel_sizes: Optional[tuple] = None,
    skull_strip_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate skull stripping quality metrics.

    Parameters
    ----------
    original_data : np.ndarray
        Original (pre-skull-strip) volume data.
    brain_data : np.ndarray
        Brain-extracted volume data.
    mask_data : np.ndarray
        Binary brain mask (bool or 0/1).
    voxel_sizes : tuple, optional
        Voxel dimensions in mm (x, y, z) for volume calculation.
    skull_strip_info : dict, optional
        Info dict from the skull_strip dispatcher (method, extraction_ratio, etc.).

    Returns
    -------
    dict
        Metrics dictionary with brain volume, SNR, quality flags, etc.
    """
    mask_bool = mask_data > 0

    metrics = {}

    # Brain volume
    metrics['brain_volume_voxels'] = int(np.sum(mask_bool))
    if voxel_sizes is not None:
        voxel_vol = float(np.prod(voxel_sizes))
        metrics['brain_volume_mm3'] = float(metrics['brain_volume_voxels'] * voxel_vol)

    # Brain-to-total ratio
    total_nonzero = np.sum(original_data > 0)
    metrics['brain_to_total_ratio'] = (
        float(np.sum(mask_bool) / total_nonzero) if total_nonzero > 0 else 0.0
    )

    # SNR estimate (mean / std within brain)
    brain_signal = original_data[mask_bool]
    if len(brain_signal) > 0:
        metrics['snr_estimate'] = float(
            np.mean(brain_signal) / (np.std(brain_signal) + 1e-10)
        )
    else:
        metrics['snr_estimate'] = 0.0

    # Quality flags
    metrics['potential_over_stripping'] = metrics['brain_to_total_ratio'] < 0.15
    metrics['potential_under_stripping'] = metrics['brain_to_total_ratio'] > 0.5

    # Merge dispatcher info
    if skull_strip_info:
        metrics['method'] = skull_strip_info.get('method', 'unknown')
        metrics['extraction_ratio'] = skull_strip_info.get('extraction_ratio', None)

    return metrics


def plot_slicesdir_mosaic(
    original_data: np.ndarray,
    mask_data: np.ndarray,
    subject: str,
    session: str,
    modality: str,
    figures_dir: Path,
    max_cols: int = 8,
    max_slices: int = 60,
) -> Path:
    """
    Generate a slicesdir-style mosaic showing mask contours on ALL axial slices.

    Parameters
    ----------
    original_data : np.ndarray
        3D volume (original image).
    mask_data : np.ndarray
        3D binary mask.
    subject, session : str
        Identifiers for filename and title.
    modality : str
        One of 'anat', 'dwi', 'func', 'msme' (determines contour color).
    figures_dir : Path
        Output directory for figure.
    max_cols : int
        Maximum columns in the grid.
    max_slices : int
        If more slices than this, sample evenly.

    Returns
    -------
    Path
        Path to saved PNG figure.
    """
    mask_bool = mask_data > 0
    n_slices = original_data.shape[2]

    # Determine which slices to show
    if n_slices <= max_slices:
        slice_indices = np.arange(n_slices)
    else:
        slice_indices = np.linspace(0, n_slices - 1, max_slices, dtype=int)

    n_show = len(slice_indices)
    n_cols = min(max_cols, n_show)
    n_rows = int(np.ceil(n_show / n_cols))

    color = MODALITY_COLORS.get(modality, 'red')
    vmax = np.percentile(original_data[original_data > 0], 99) if np.any(original_data > 0) else 1.0

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows))
    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, slice_idx in enumerate(slice_indices):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # Show skull-stripped image (original * mask) so quality is immediately visible
        img_slice = np.rot90(original_data[:, :, slice_idx] * mask_bool[:, :, slice_idx])
        ax.imshow(img_slice, cmap='gray', vmin=0, vmax=vmax, interpolation='nearest')

        ax.text(
            0.02, 0.98, str(slice_idx), transform=ax.transAxes,
            fontsize=7, color='white', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5),
        )
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n_show, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')

    fig.suptitle(
        f'Skull Strip Mosaic ({modality.upper()}): {subject} {session}',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = figures_dir / f'{subject}_{session}_skull_strip_mosaic.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def plot_mask_edge_triplanar(
    original_data: np.ndarray,
    mask_data: np.ndarray,
    subject: str,
    session: str,
    modality: str,
    figures_dir: Path,
) -> Path:
    """
    Sagittal/coronal/axial center slices with mask contours.

    Skips sagittal and coronal views when n_slices < 7 (partial-coverage
    modalities like DWI/func/MSME).

    Parameters
    ----------
    original_data : np.ndarray
        3D volume.
    mask_data : np.ndarray
        3D binary mask.
    subject, session : str
        Identifiers.
    modality : str
        Modality label.
    figures_dir : Path
        Output directory.

    Returns
    -------
    Path
        Path to saved PNG figure.
    """
    mask_bool = mask_data > 0
    n_slices = original_data.shape[2]
    vmax = np.percentile(original_data[original_data > 0], 99) if np.any(original_data > 0) else 1.0

    show_all = n_slices >= 7

    if show_all:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        mid_x = original_data.shape[0] // 2
        mid_y = original_data.shape[1] // 2
        mid_z = original_data.shape[2] // 2

        # Show skull-stripped images so quality is immediately visible
        # Sagittal
        axes[0].imshow(np.rot90(original_data[mid_x, :, :] * mask_bool[mid_x, :, :]), cmap='gray', vmin=0, vmax=vmax)
        axes[0].set_title('Sagittal')
        axes[0].axis('off')

        # Coronal
        axes[1].imshow(np.rot90(original_data[:, mid_y, :] * mask_bool[:, mid_y, :]), cmap='gray', vmin=0, vmax=vmax)
        axes[1].set_title('Coronal')
        axes[1].axis('off')

        # Axial
        axes[2].imshow(np.rot90(original_data[:, :, mid_z] * mask_bool[:, :, mid_z]), cmap='gray', vmin=0, vmax=vmax)
        axes[2].set_title('Axial')
        axes[2].axis('off')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        mid_z = original_data.shape[2] // 2
        ax.imshow(np.rot90(original_data[:, :, mid_z] * mask_bool[:, :, mid_z]), cmap='gray', vmin=0, vmax=vmax)
        ax.set_title('Axial (center)')
        ax.axis('off')

    fig.suptitle(
        f'Mask Edge ({modality.upper()}): {subject} {session}',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()

    output_file = figures_dir / f'{subject}_{session}_skull_strip_triplanar.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def skull_strip_html_section(
    metrics: Dict[str, Any],
    figures: List[Path],
    skull_strip_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate an HTML snippet (div.section) for skull strip QC.

    This returns a fragment to be inserted into an existing HTML report,
    NOT a full HTML document.

    Parameters
    ----------
    metrics : dict
        From calculate_skull_strip_metrics().
    figures : list of Path
        Figure file paths (mosaic, triplanar).
    skull_strip_info : dict, optional
        Extra info from skull_strip dispatcher.

    Returns
    -------
    str
        HTML snippet string.
    """
    method = metrics.get('method', skull_strip_info.get('method', 'unknown') if skull_strip_info else 'unknown')
    extraction_ratio = metrics.get('extraction_ratio',
                                    skull_strip_info.get('extraction_ratio') if skull_strip_info else None)

    brain_vol = metrics.get('brain_volume_voxels', 0)
    brain_vol_mm3 = metrics.get('brain_volume_mm3')
    ratio = metrics.get('brain_to_total_ratio', 0)
    snr = metrics.get('snr_estimate', 0)

    # Quality assessment
    quality = 'GOOD'
    quality_class = 'good'
    if metrics.get('potential_over_stripping', False):
        quality = 'WARNING - Possible over-stripping'
        quality_class = 'warning'
    elif metrics.get('potential_under_stripping', False):
        quality = 'WARNING - Possible under-stripping'
        quality_class = 'warning'

    html = '<div class="section">\n'
    html += '    <h2>Skull Stripping</h2>\n'

    # Metric cards
    html += '    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0;">\n'

    html += f'        <div class="metric"><div class="metric-label">Method</div><div class="metric-value">{method}</div></div>\n'

    if extraction_ratio is not None:
        html += f'        <div class="metric"><div class="metric-label">Extraction Ratio</div><div class="metric-value">{extraction_ratio:.3f}</div></div>\n'

    vol_str = f'{brain_vol:,} voxels'
    if brain_vol_mm3 is not None:
        vol_str += f' ({brain_vol_mm3:.1f} mm³)'
    html += f'        <div class="metric"><div class="metric-label">Brain Volume</div><div class="metric-value">{vol_str}</div></div>\n'

    html += f'        <div class="metric"><div class="metric-label">Brain/Total Ratio</div><div class="metric-value">{ratio:.2%}</div></div>\n'
    html += f'        <div class="metric"><div class="metric-label">SNR Estimate</div><div class="metric-value">{snr:.2f}</div></div>\n'
    html += f'        <div class="metric"><div class="metric-label">Quality</div><div class="metric-value {quality_class}">{quality}</div></div>\n'

    html += '    </div>\n'

    # Figures
    for fig_path in figures:
        rel_path = f'figures/{fig_path.name}'
        fig_title = fig_path.stem.replace('_', ' ').title()
        html += f'    <h3>{fig_title}</h3>\n'
        html += f'    <img src="{rel_path}" alt="{fig_title}" style="max-width:100%;border:1px solid #ddd;border-radius:5px;">\n'

    html += '</div>\n'
    return html
