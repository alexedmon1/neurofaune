"""
Anatomical T2w preprocessing quality control.

This module generates QC visualizations for anatomical preprocessing:
- Skull stripping (brain mask overlays)
- Tissue segmentation (GM, WM, CSF probability maps)
- Overall preprocessing quality metrics
"""

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from neurofaune.preprocess.qc.skull_strip_qc import plot_slicesdir_mosaic


def generate_skull_strip_qc(
    subject: str,
    session: str,
    t2w_file: Path,
    brain_file: Path,
    mask_file: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Generate skull stripping QC report.

    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier
    t2w_file : Path
        Original T2w image (before skull stripping)
    brain_file : Path
        Brain-extracted T2w image
    mask_file : Path
        Binary brain mask
    output_dir : Path
        Output directory for QC reports

    Returns
    -------
    dict
        Dictionary with QC metrics and figure paths
    """
    print(f"Generating skull stripping QC for {subject} {session}...")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Load images
    t2w_img = nib.load(t2w_file)
    t2w_data = t2w_img.get_fdata()

    brain_img = nib.load(brain_file)
    brain_data = brain_img.get_fdata()

    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata() > 0

    # Calculate metrics
    metrics = _calculate_skull_strip_metrics(t2w_data, brain_data, mask_data)

    # Generate visualizations
    figures = []

    # 1. Mask overlay on T2w (axial slices)
    overlay_fig = _plot_mask_overlay(
        t2w_data, mask_data, subject, session, figures_dir
    )
    figures.append(overlay_fig)

    # 2. Brain extraction comparison (before/after)
    comparison_fig = _plot_extraction_comparison(
        t2w_data, brain_data, mask_data, subject, session, figures_dir
    )
    figures.append(comparison_fig)

    # 3. Mask edge overlay (to check for over/under-stripping)
    edge_fig = _plot_mask_edge(
        t2w_data, mask_data, subject, session, figures_dir
    )
    figures.append(edge_fig)

    # 4. Full-slice mosaic (slicesdir-style, all slices)
    mosaic_fig = plot_slicesdir_mosaic(
        t2w_data, mask_data, subject, session, 'anat', figures_dir
    )
    figures.append(mosaic_fig)

    print(f"  Skull stripping QC complete: {len(figures)} figures generated")

    return {
        'metrics': metrics,
        'figures': figures
    }


def generate_segmentation_qc(
    subject: str,
    session: str,
    t2w_file: Path,
    gm_file: Path,
    wm_file: Path,
    csf_file: Path,
    mask_file: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Generate tissue segmentation QC report.

    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier
    t2w_file : Path
        Preprocessed T2w image
    gm_file : Path
        Gray matter probability map
    wm_file : Path
        White matter probability map
    csf_file : Path
        CSF probability map
    mask_file : Path
        Brain mask
    output_dir : Path
        Output directory for QC reports

    Returns
    -------
    dict
        Dictionary with QC metrics and figure paths
    """
    print(f"Generating segmentation QC for {subject} {session}...")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Load images
    t2w_data = nib.load(t2w_file).get_fdata()
    gm_data = nib.load(gm_file).get_fdata()
    wm_data = nib.load(wm_file).get_fdata()
    csf_data = nib.load(csf_file).get_fdata()
    mask_data = nib.load(mask_file).get_fdata() > 0

    # Calculate metrics
    metrics = _calculate_segmentation_metrics(gm_data, wm_data, csf_data, mask_data)

    # Generate visualizations
    figures = []

    # 1. Tissue probability overlays
    tissue_fig = _plot_tissue_overlays(
        t2w_data, gm_data, wm_data, csf_data, subject, session, figures_dir
    )
    figures.append(tissue_fig)

    # 2. Tissue probability histograms
    hist_fig = _plot_tissue_histograms(
        gm_data, wm_data, csf_data, mask_data, subject, session, figures_dir
    )
    figures.append(hist_fig)

    # 3. Combined segmentation view
    seg_fig = _plot_segmentation_montage(
        t2w_data, gm_data, wm_data, csf_data, subject, session, figures_dir
    )
    figures.append(seg_fig)

    print(f"  Segmentation QC complete: {len(figures)} figures generated")

    return {
        'metrics': metrics,
        'figures': figures
    }


def generate_anatomical_qc_report(
    subject: str,
    session: str,
    t2w_file: Path,
    brain_file: Path,
    mask_file: Path,
    gm_file: Optional[Path] = None,
    wm_file: Optional[Path] = None,
    csf_file: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Generate comprehensive anatomical preprocessing QC report.

    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier
    t2w_file : Path
        Original T2w image
    brain_file : Path
        Brain-extracted T2w image
    mask_file : Path
        Binary brain mask
    gm_file : Path, optional
        Gray matter probability map
    wm_file : Path, optional
        White matter probability map
    csf_file : Path, optional
        CSF probability map
    output_dir : Path, optional
        Output directory for QC report

    Returns
    -------
    Path
        Path to HTML QC report
    """
    print(f"\n{'='*80}")
    print(f"Generating Anatomical QC Report: {subject} {session}")
    print(f"{'='*80}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    all_metrics = {}
    all_figures = []

    # Generate skull stripping QC
    skull_strip_results = generate_skull_strip_qc(
        subject, session, t2w_file, brain_file, mask_file, output_dir
    )
    all_metrics['skull_stripping'] = skull_strip_results['metrics']
    all_figures.extend(skull_strip_results['figures'])

    # Generate segmentation QC if tissue maps provided
    if gm_file and wm_file and csf_file:
        if gm_file.exists() and wm_file.exists() and csf_file.exists():
            seg_results = generate_segmentation_qc(
                subject, session, brain_file, gm_file, wm_file, csf_file,
                mask_file, output_dir
            )
            all_metrics['segmentation'] = seg_results['metrics']
            all_figures.extend(seg_results['figures'])

    # Create HTML report
    html_report = _create_html_report(
        subject, session, all_metrics, all_figures, output_dir
    )

    # Save metrics to JSON
    metrics_file = output_dir / f'{subject}_{session}_anat_qc_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n  QC report saved: {html_report}")
    print(f"  Metrics saved: {metrics_file}")

    return html_report


# =============================================================================
# Metric Calculation Functions
# =============================================================================

def _calculate_skull_strip_metrics(
    t2w: np.ndarray,
    brain: np.ndarray,
    mask: np.ndarray
) -> Dict[str, float]:
    """Calculate skull stripping quality metrics."""
    metrics = {}

    # Brain volume (in voxels)
    metrics['brain_volume_voxels'] = int(np.sum(mask))

    # Brain/total ratio
    total_nonzero = np.sum(t2w > 0)
    metrics['brain_to_total_ratio'] = float(np.sum(mask) / total_nonzero) if total_nonzero > 0 else 0

    # Signal statistics within brain
    brain_signal = t2w[mask]
    metrics['mean_brain_intensity'] = float(np.mean(brain_signal))
    metrics['std_brain_intensity'] = float(np.std(brain_signal))

    # SNR estimate (mean / std within brain)
    metrics['snr_estimate'] = float(np.mean(brain_signal) / (np.std(brain_signal) + 1e-10))

    # Check for potential issues
    # Very small brain (possible over-stripping)
    metrics['potential_over_stripping'] = metrics['brain_to_total_ratio'] < 0.15
    # Very large brain (possible under-stripping)
    metrics['potential_under_stripping'] = metrics['brain_to_total_ratio'] > 0.5

    return metrics


def _calculate_segmentation_metrics(
    gm: np.ndarray,
    wm: np.ndarray,
    csf: np.ndarray,
    mask: np.ndarray
) -> Dict[str, float]:
    """Calculate tissue segmentation quality metrics."""
    metrics = {}

    # Volume fractions
    total_brain = np.sum(mask)
    metrics['gm_volume_fraction'] = float(np.sum(gm[mask] > 0.5) / total_brain)
    metrics['wm_volume_fraction'] = float(np.sum(wm[mask] > 0.5) / total_brain)
    metrics['csf_volume_fraction'] = float(np.sum(csf[mask] > 0.5) / total_brain)

    # Mean probabilities
    metrics['gm_mean_probability'] = float(np.mean(gm[mask]))
    metrics['wm_mean_probability'] = float(np.mean(wm[mask]))
    metrics['csf_mean_probability'] = float(np.mean(csf[mask]))

    # Check probability sum (should be ~1 for proper segmentation)
    prob_sum = gm + wm + csf
    metrics['mean_probability_sum'] = float(np.mean(prob_sum[mask]))
    metrics['probability_sum_in_range'] = 0.9 < metrics['mean_probability_sum'] < 1.1

    # GM/WM ratio (typical value ~1.2-1.5 for rat brain)
    wm_vol = np.sum(wm[mask] > 0.5)
    metrics['gm_wm_ratio'] = float(np.sum(gm[mask] > 0.5) / wm_vol) if wm_vol > 0 else 0

    return metrics


# =============================================================================
# Visualization Functions
# =============================================================================

def _plot_mask_overlay(
    t2w: np.ndarray,
    mask: np.ndarray,
    subject: str,
    session: str,
    output_dir: Path,
    n_slices: int = 9
) -> Path:
    """Plot brain mask overlay on T2w image."""
    # Select evenly spaced axial slices
    z_dim = t2w.shape[2]
    slice_indices = np.linspace(int(z_dim * 0.1), int(z_dim * 0.9), n_slices, dtype=int)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    # Create red mask colormap
    mask_cmap = LinearSegmentedColormap.from_list('mask', [(1, 0, 0, 0), (1, 0, 0, 0.5)])

    for idx, slice_idx in enumerate(slice_indices):
        ax = axes[idx]

        # Plot T2w
        t2w_slice = np.rot90(t2w[:, :, slice_idx])
        ax.imshow(t2w_slice, cmap='gray', vmin=0, vmax=np.percentile(t2w, 99))

        # Overlay mask edges
        mask_slice = np.rot90(mask[:, :, slice_idx].astype(float))
        ax.contour(mask_slice, levels=[0.5], colors='red', linewidths=1.5)

        ax.set_title(f'Slice {slice_idx}')
        ax.axis('off')

    fig.suptitle(f'Brain Mask Overlay: {subject} {session}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / f'{subject}_{session}_mask_overlay.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def _plot_extraction_comparison(
    t2w: np.ndarray,
    brain: np.ndarray,
    mask: np.ndarray,
    subject: str,
    session: str,
    output_dir: Path
) -> Path:
    """Plot before/after skull stripping comparison."""
    # Select 3 representative slices
    z_dim = t2w.shape[2]
    slice_indices = [int(z_dim * 0.3), int(z_dim * 0.5), int(z_dim * 0.7)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    vmax = np.percentile(t2w, 99)

    for col, slice_idx in enumerate(slice_indices):
        # Original T2w
        axes[0, col].imshow(np.rot90(t2w[:, :, slice_idx]), cmap='gray', vmin=0, vmax=vmax)
        axes[0, col].set_title(f'Original (slice {slice_idx})')
        axes[0, col].axis('off')

        # Brain extracted
        axes[1, col].imshow(np.rot90(brain[:, :, slice_idx]), cmap='gray', vmin=0, vmax=vmax)
        axes[1, col].set_title(f'Brain Extracted')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Original T2w', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Skull Stripped', fontsize=12, fontweight='bold')

    fig.suptitle(f'Skull Stripping Comparison: {subject} {session}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / f'{subject}_{session}_skull_strip_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def _plot_mask_edge(
    t2w: np.ndarray,
    mask: np.ndarray,
    subject: str,
    session: str,
    output_dir: Path
) -> Path:
    """Plot mask edge overlay for checking over/under-stripping."""
    from scipy import ndimage

    # Select middle slice from each orientation
    mid_x = t2w.shape[0] // 2
    mid_y = t2w.shape[1] // 2
    mid_z = t2w.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmax = np.percentile(t2w, 99)

    # Sagittal
    axes[0].imshow(np.rot90(t2w[mid_x, :, :]), cmap='gray', vmin=0, vmax=vmax)
    mask_slice = np.rot90(mask[mid_x, :, :].astype(float))
    axes[0].contour(mask_slice, levels=[0.5], colors='lime', linewidths=2)
    axes[0].set_title('Sagittal')
    axes[0].axis('off')

    # Coronal
    axes[1].imshow(np.rot90(t2w[:, mid_y, :]), cmap='gray', vmin=0, vmax=vmax)
    mask_slice = np.rot90(mask[:, mid_y, :].astype(float))
    axes[1].contour(mask_slice, levels=[0.5], colors='lime', linewidths=2)
    axes[1].set_title('Coronal')
    axes[1].axis('off')

    # Axial
    axes[2].imshow(np.rot90(t2w[:, :, mid_z]), cmap='gray', vmin=0, vmax=vmax)
    mask_slice = np.rot90(mask[:, :, mid_z].astype(float))
    axes[2].contour(mask_slice, levels=[0.5], colors='lime', linewidths=2)
    axes[2].set_title('Axial')
    axes[2].axis('off')

    fig.suptitle(f'Mask Edge (3 Orientations): {subject} {session}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / f'{subject}_{session}_mask_edge.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def _plot_tissue_overlays(
    t2w: np.ndarray,
    gm: np.ndarray,
    wm: np.ndarray,
    csf: np.ndarray,
    subject: str,
    session: str,
    output_dir: Path
) -> Path:
    """Plot tissue probability maps overlaid on T2w."""
    # Select 3 representative axial slices
    z_dim = t2w.shape[2]
    slice_indices = [int(z_dim * 0.3), int(z_dim * 0.5), int(z_dim * 0.7)]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    vmax = np.percentile(t2w, 99)

    for col, slice_idx in enumerate(slice_indices):
        t2w_slice = np.rot90(t2w[:, :, slice_idx])
        gm_slice = np.rot90(gm[:, :, slice_idx])
        wm_slice = np.rot90(wm[:, :, slice_idx])
        csf_slice = np.rot90(csf[:, :, slice_idx])

        # GM overlay (red)
        axes[0, col].imshow(t2w_slice, cmap='gray', vmin=0, vmax=vmax)
        gm_masked = np.ma.masked_where(gm_slice < 0.2, gm_slice)
        axes[0, col].imshow(gm_masked, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        axes[0, col].set_title(f'Slice {slice_idx}')
        axes[0, col].axis('off')

        # WM overlay (blue)
        axes[1, col].imshow(t2w_slice, cmap='gray', vmin=0, vmax=vmax)
        wm_masked = np.ma.masked_where(wm_slice < 0.2, wm_slice)
        axes[1, col].imshow(wm_masked, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
        axes[1, col].axis('off')

        # CSF overlay (cyan)
        axes[2, col].imshow(t2w_slice, cmap='gray', vmin=0, vmax=vmax)
        csf_masked = np.ma.masked_where(csf_slice < 0.2, csf_slice)
        axes[2, col].imshow(csf_masked, cmap='GnBu', alpha=0.6, vmin=0, vmax=1)
        axes[2, col].axis('off')

    axes[0, 0].set_ylabel('Gray Matter', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('White Matter', fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('CSF', fontsize=12, fontweight='bold')

    fig.suptitle(f'Tissue Probability Overlays: {subject} {session}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / f'{subject}_{session}_tissue_overlays.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def _plot_tissue_histograms(
    gm: np.ndarray,
    wm: np.ndarray,
    csf: np.ndarray,
    mask: np.ndarray,
    subject: str,
    session: str,
    output_dir: Path
) -> Path:
    """Plot tissue probability histograms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # GM histogram
    axes[0].hist(gm[mask].flatten(), bins=50, color='red', alpha=0.7, edgecolor='darkred')
    axes[0].set_xlabel('Probability')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Gray Matter')
    axes[0].axvline(x=np.mean(gm[mask]), color='darkred', linestyle='--',
                    label=f'Mean: {np.mean(gm[mask]):.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # WM histogram
    axes[1].hist(wm[mask].flatten(), bins=50, color='blue', alpha=0.7, edgecolor='darkblue')
    axes[1].set_xlabel('Probability')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('White Matter')
    axes[1].axvline(x=np.mean(wm[mask]), color='darkblue', linestyle='--',
                    label=f'Mean: {np.mean(wm[mask]):.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # CSF histogram
    axes[2].hist(csf[mask].flatten(), bins=50, color='cyan', alpha=0.7, edgecolor='darkcyan')
    axes[2].set_xlabel('Probability')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('CSF')
    axes[2].axvline(x=np.mean(csf[mask]), color='darkcyan', linestyle='--',
                    label=f'Mean: {np.mean(csf[mask]):.3f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'Tissue Probability Distributions: {subject} {session}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / f'{subject}_{session}_tissue_histograms.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def _plot_segmentation_montage(
    t2w: np.ndarray,
    gm: np.ndarray,
    wm: np.ndarray,
    csf: np.ndarray,
    subject: str,
    session: str,
    output_dir: Path,
    n_slices: int = 6
) -> Path:
    """Create combined segmentation montage with RGB overlay."""
    # Select evenly spaced axial slices
    z_dim = t2w.shape[2]
    slice_indices = np.linspace(int(z_dim * 0.2), int(z_dim * 0.8), n_slices, dtype=int)

    fig, axes = plt.subplots(2, n_slices // 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, slice_idx in enumerate(slice_indices):
        ax = axes[idx]

        # Create RGB segmentation overlay
        t2w_slice = np.rot90(t2w[:, :, slice_idx])
        gm_slice = np.rot90(gm[:, :, slice_idx])
        wm_slice = np.rot90(wm[:, :, slice_idx])
        csf_slice = np.rot90(csf[:, :, slice_idx])

        # Normalize T2w for background
        t2w_norm = t2w_slice / (np.percentile(t2w_slice, 99) + 1e-10)
        t2w_norm = np.clip(t2w_norm, 0, 1)

        # Create RGB image: R=GM, G=WM, B=CSF
        rgb = np.zeros((*t2w_slice.shape, 3))
        rgb[:, :, 0] = gm_slice  # Red = GM
        rgb[:, :, 1] = wm_slice  # Green = WM (actually will show as combination)
        rgb[:, :, 2] = csf_slice  # Blue = CSF

        # Blend with T2w background
        alpha = 0.6
        blended = np.zeros((*t2w_slice.shape, 3))
        for c in range(3):
            blended[:, :, c] = (1 - alpha) * t2w_norm + alpha * rgb[:, :, c]

        ax.imshow(np.clip(blended, 0, 1))
        ax.set_title(f'Slice {slice_idx}')
        ax.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='GM'),
        Patch(facecolor='green', label='WM'),
        Patch(facecolor='blue', label='CSF')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12)

    fig.suptitle(f'Tissue Segmentation (RGB): {subject} {session}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    output_file = output_dir / f'{subject}_{session}_segmentation_montage.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


# =============================================================================
# HTML Report Generation
# =============================================================================

def _create_html_report(
    subject: str,
    session: str,
    metrics: Dict[str, Any],
    figures: List[Path],
    output_dir: Path
) -> Path:
    """Create HTML QC report for anatomical preprocessing."""

    # Extract metrics
    ss_metrics = metrics.get('skull_stripping', {})
    seg_metrics = metrics.get('segmentation', {})

    # Assess quality
    ss_quality = "GOOD"
    ss_quality_class = "good"
    if ss_metrics.get('potential_over_stripping', False):
        ss_quality = "WARNING - Possible over-stripping"
        ss_quality_class = "warning"
    elif ss_metrics.get('potential_under_stripping', False):
        ss_quality = "WARNING - Possible under-stripping"
        ss_quality_class = "warning"

    seg_quality = "GOOD"
    seg_quality_class = "good"
    if seg_metrics and not seg_metrics.get('probability_sum_in_range', True):
        seg_quality = "WARNING - Probability sum out of range"
        seg_quality_class = "warning"

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Anatomical QC Report - {subject} {session}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #666;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        .summary {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-weight: bold;
            color: #555;
            font-size: 0.9em;
        }}
        .metric-value {{
            color: #000;
            font-size: 1.2em;
            margin-top: 5px;
        }}
        .good {{
            color: #27ae60;
            font-weight: bold;
        }}
        .warning {{
            color: #f39c12;
            font-weight: bold;
        }}
        .error {{
            color: #e74c3c;
            font-weight: bold;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Anatomical Preprocessing QC Report</h1>
        <p><strong>Subject:</strong> {subject}</p>
        <p><strong>Session:</strong> {session}</p>

        <div class="summary">
            <h2>Quality Summary</h2>

            <div class="metric">
                <div class="metric-label">Skull Stripping</div>
                <div class="metric-value {ss_quality_class}">{ss_quality}</div>
            </div>

            <div class="metric">
                <div class="metric-label">Brain Volume</div>
                <div class="metric-value">{ss_metrics.get('brain_volume_voxels', 'N/A'):,} voxels</div>
            </div>

            <div class="metric">
                <div class="metric-label">Brain/Total Ratio</div>
                <div class="metric-value">{ss_metrics.get('brain_to_total_ratio', 0):.2%}</div>
            </div>

            <div class="metric">
                <div class="metric-label">SNR Estimate</div>
                <div class="metric-value">{ss_metrics.get('snr_estimate', 0):.2f}</div>
            </div>
        </div>
"""

    # Add segmentation metrics if available
    if seg_metrics:
        html_content += f"""
        <div class="summary">
            <h2>Tissue Segmentation</h2>

            <div class="metric">
                <div class="metric-label">Segmentation Quality</div>
                <div class="metric-value {seg_quality_class}">{seg_quality}</div>
            </div>

            <div class="metric">
                <div class="metric-label">GM Volume</div>
                <div class="metric-value">{seg_metrics.get('gm_volume_fraction', 0):.1%}</div>
            </div>

            <div class="metric">
                <div class="metric-label">WM Volume</div>
                <div class="metric-value">{seg_metrics.get('wm_volume_fraction', 0):.1%}</div>
            </div>

            <div class="metric">
                <div class="metric-label">CSF Volume</div>
                <div class="metric-value">{seg_metrics.get('csf_volume_fraction', 0):.1%}</div>
            </div>

            <div class="metric">
                <div class="metric-label">GM/WM Ratio</div>
                <div class="metric-value">{seg_metrics.get('gm_wm_ratio', 0):.2f}</div>
            </div>

            <div class="metric">
                <div class="metric-label">Probability Sum</div>
                <div class="metric-value">{seg_metrics.get('mean_probability_sum', 0):.3f}</div>
            </div>
        </div>
"""

    # Add figures section
    html_content += """
        <h2>Quality Control Figures</h2>
"""

    for fig_path in figures:
        if fig_path.exists():
            rel_path = f"figures/{fig_path.name}"
            fig_title = fig_path.stem.replace(f'{subject}_{session}_', '').replace('_', ' ').title()
            html_content += f"""
        <h3>{fig_title}</h3>
        <img src="{rel_path}" alt="{fig_title}">
"""

    # Add quality criteria section
    html_content += """
        <h2>Quality Criteria</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Expected Range</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>Brain/Total Ratio</td>
                <td>15% - 50%</td>
                <td>&lt;15%: over-stripping, &gt;50%: under-stripping</td>
            </tr>
            <tr>
                <td>SNR Estimate</td>
                <td>&gt; 5</td>
                <td>Higher is better; &lt;5 indicates low image quality</td>
            </tr>
            <tr>
                <td>Probability Sum</td>
                <td>0.9 - 1.1</td>
                <td>Should be ~1.0 for proper segmentation</td>
            </tr>
            <tr>
                <td>GM/WM Ratio</td>
                <td>1.2 - 1.5 (rat)</td>
                <td>Species-dependent; unusual values may indicate issues</td>
            </tr>
        </table>

        <hr>
        <p style="text-align: center; color: #888; font-size: 0.9em;">
            Generated by Neurofaune anatomical preprocessing pipeline
        </p>
    </div>
</body>
</html>
"""

    # Save HTML report
    report_path = output_dir / f'{subject}_{session}_anat_qc.html'
    with open(report_path, 'w') as f:
        f.write(html_content)

    return report_path
