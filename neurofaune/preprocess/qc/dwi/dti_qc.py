"""
DTI metrics quality control.

This module generates QC visualizations for DTI scalar maps (FA, MD, AD, RD).
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional
import json


def generate_dti_qc_report(
    subject: str,
    session: str,
    fa_file: Path,
    md_file: Path,
    ad_file: Path,
    rd_file: Path,
    brain_mask: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Generate comprehensive QC report for DTI metrics.

    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier
    fa_file : Path
        FA map
    md_file : Path
        MD map
    ad_file : Path
        AD map
    rd_file : Path
        RD map
    brain_mask : Path
        Brain mask
    output_dir : Path
        Output directory for QC reports

    Returns
    -------
    dict
        Dictionary with QC metrics and report paths
    """
    print(f"\n{'='*80}")
    print(f"Generating DTI QC Report: {subject} {session}")
    print(f"{'='*80}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Load DTI maps
    fa_img = nib.load(fa_file)
    fa = fa_img.get_fdata()

    md_img = nib.load(md_file)
    md = md_img.get_fdata()

    ad_img = nib.load(ad_file)
    ad = ad_img.get_fdata()

    rd_img = nib.load(rd_file)
    rd = rd_img.get_fdata()

    mask_img = nib.load(brain_mask)
    mask = mask_img.get_fdata() > 0

    # Calculate summary statistics
    qc_metrics = _calculate_dti_statistics(fa, md, ad, rd, mask)

    # Generate visualizations
    figure_paths = []

    # Histogram plots
    hist_fig = _plot_dti_histograms(fa, md, ad, rd, mask, subject, session, figures_dir)
    figure_paths.append(hist_fig)

    # Slice montage for each metric
    for metric_name, metric_data, cmap, vmin, vmax in [
        ('FA', fa, 'hot', 0, 1),
        ('MD', md, 'viridis', 0, 0.003),
        ('AD', ad, 'plasma', 0, 0.004),
        ('RD', rd, 'cividis', 0, 0.003)
    ]:
        montage_fig = _plot_slice_montage(
            metric_data,
            metric_name,
            subject,
            session,
            figures_dir,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        figure_paths.append(montage_fig)

    # Scatter plots (FA vs MD, etc.)
    scatter_fig = _plot_dti_scatter(fa, md, ad, rd, mask, subject, session, figures_dir)
    figure_paths.append(scatter_fig)

    # Create HTML report
    html_report = _create_dti_html_report(
        subject,
        session,
        qc_metrics,
        figure_paths,
        output_dir
    )

    # Save metrics to JSON
    metrics_file = output_dir / f'{subject}_{session}_dti_qc_metrics.json'
    with open(metrics_file, 'w') as f:
        # Convert numpy types to native Python types
        metrics_serializable = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                for k, v in qc_metrics.items()}
        json.dump(metrics_serializable, f, indent=2)

    print(f"\n✓ DTI QC report saved: {html_report}")
    print(f"✓ Metrics saved: {metrics_file}")

    return {
        'html_report': html_report,
        'metrics_file': metrics_file,
        'metrics': qc_metrics,
        'figures': figure_paths
    }


def _calculate_dti_statistics(
    fa: np.ndarray,
    md: np.ndarray,
    ad: np.ndarray,
    rd: np.ndarray,
    mask: np.ndarray
) -> Dict[str, float]:
    """Calculate summary statistics for DTI metrics."""
    metrics = {}

    for name, data in [('fa', fa), ('md', md), ('ad', ad), ('rd', rd)]:
        masked_data = data[mask]
        metrics[f'{name}_mean'] = float(np.mean(masked_data))
        metrics[f'{name}_median'] = float(np.median(masked_data))
        metrics[f'{name}_std'] = float(np.std(masked_data))
        metrics[f'{name}_min'] = float(np.min(masked_data))
        metrics[f'{name}_max'] = float(np.max(masked_data))
        metrics[f'{name}_p25'] = float(np.percentile(masked_data, 25))
        metrics[f'{name}_p75'] = float(np.percentile(masked_data, 75))

    return metrics


def _plot_dti_histograms(
    fa: np.ndarray,
    md: np.ndarray,
    ad: np.ndarray,
    rd: np.ndarray,
    mask: np.ndarray,
    subject: str,
    session: str,
    output_dir: Path
) -> Path:
    """Plot histograms for all DTI metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # FA
    axes[0, 0].hist(fa[mask], bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('FA')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Fractional Anisotropy')
    axes[0, 0].axvline(np.mean(fa[mask]), color='darkred', linestyle='--', label=f'Mean: {np.mean(fa[mask]):.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MD
    axes[0, 1].hist(md[mask], bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('MD (mm²/s)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Mean Diffusivity')
    axes[0, 1].axvline(np.mean(md[mask]), color='darkblue', linestyle='--', label=f'Mean: {np.mean(md[mask]):.6f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # AD
    axes[1, 0].hist(ad[mask], bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('AD (mm²/s)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Axial Diffusivity')
    axes[1, 0].axvline(np.mean(ad[mask]), color='darkgreen', linestyle='--', label=f'Mean: {np.mean(ad[mask]):.6f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # RD
    axes[1, 1].hist(rd[mask], bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('RD (mm²/s)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Radial Diffusivity')
    axes[1, 1].axvline(np.mean(rd[mask]), color='darkviolet', linestyle='--', label=f'Mean: {np.mean(rd[mask]):.6f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'DTI Metrics Distribution: {subject} {session}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / f'{subject}_{session}_dti_histograms.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def _plot_slice_montage(
    data: np.ndarray,
    metric_name: str,
    subject: str,
    session: str,
    output_dir: Path,
    cmap: str = 'viridis',
    vmin: float = None,
    vmax: float = None,
    n_slices: int = 9
) -> Path:
    """Create slice montage for a DTI metric."""
    # Select evenly spaced slices
    z_dim = data.shape[2]
    slice_indices = np.linspace(0, z_dim - 1, n_slices, dtype=int)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for idx, slice_idx in enumerate(slice_indices):
        im = axes[idx].imshow(np.rot90(data[:, :, slice_idx]), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[idx].set_title(f'Slice {slice_idx}')
        axes[idx].axis('off')

    fig.suptitle(f'{metric_name} Map: {subject} {session}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label(metric_name)

    output_file = output_dir / f'{subject}_{session}_{metric_name}_montage.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def _plot_dti_scatter(
    fa: np.ndarray,
    md: np.ndarray,
    ad: np.ndarray,
    rd: np.ndarray,
    mask: np.ndarray,
    subject: str,
    session: str,
    output_dir: Path
) -> Path:
    """Create scatter plots showing relationships between DTI metrics."""
    # Downsample for visualization
    sample_size = min(10000, np.sum(mask))
    mask_indices = np.where(mask.flatten())[0]
    sample_indices = np.random.choice(mask_indices, size=sample_size, replace=False)

    fa_flat = fa.flatten()[sample_indices]
    md_flat = md.flatten()[sample_indices]
    ad_flat = ad.flatten()[sample_indices]
    rd_flat = rd.flatten()[sample_indices]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # FA vs MD
    axes[0].scatter(fa_flat, md_flat, alpha=0.1, s=1, c='blue')
    axes[0].set_xlabel('FA')
    axes[0].set_ylabel('MD (mm²/s)')
    axes[0].set_title('FA vs MD')
    axes[0].grid(True, alpha=0.3)

    # FA vs AD
    axes[1].scatter(fa_flat, ad_flat, alpha=0.1, s=1, c='green')
    axes[1].set_xlabel('FA')
    axes[1].set_ylabel('AD (mm²/s)')
    axes[1].set_title('FA vs AD')
    axes[1].grid(True, alpha=0.3)

    # FA vs RD
    axes[2].scatter(fa_flat, rd_flat, alpha=0.1, s=1, c='red')
    axes[2].set_xlabel('FA')
    axes[2].set_ylabel('RD (mm²/s)')
    axes[2].set_title('FA vs RD')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'DTI Metrics Relationships: {subject} {session}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / f'{subject}_{session}_dti_scatter.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def _create_dti_html_report(
    subject: str,
    session: str,
    metrics: Dict[str, Any],
    figures: List[Path],
    output_dir: Path
) -> Path:
    """Create HTML QC report for DTI metrics."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DTI QC Report: {subject} {session}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
            .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
            .metric-label {{ font-weight: bold; color: #34495e; }}
            .metric-value {{ color: #2c3e50; font-size: 1.1em; }}
            .warning {{ color: #e74c3c; font-weight: bold; }}
            .good {{ color: #27ae60; font-weight: bold; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #34495e; color: white; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>DTI Metrics QC Report</h1>
            <p>Subject: {subject} | Session: {session}</p>
        </div>

        <div class="section">
            <h2>Summary Statistics</h2>
            <h3>Fractional Anisotropy (FA)</h3>
            <table>
                <tr>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Range</th>
                </tr>
                <tr>
                    <td>{metrics['fa_mean']:.3f}</td>
                    <td>{metrics['fa_median']:.3f}</td>
                    <td>{metrics['fa_std']:.3f}</td>
                    <td>[{metrics['fa_min']:.3f}, {metrics['fa_max']:.3f}]</td>
                </tr>
            </table>

            <h3>Mean Diffusivity (MD)</h3>
            <table>
                <tr>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Range</th>
                </tr>
                <tr>
                    <td>{metrics['md_mean']:.6f} mm²/s</td>
                    <td>{metrics['md_median']:.6f} mm²/s</td>
                    <td>{metrics['md_std']:.6f} mm²/s</td>
                    <td>[{metrics['md_min']:.6f}, {metrics['md_max']:.6f}] mm²/s</td>
                </tr>
            </table>

            <h3>Axial Diffusivity (AD)</h3>
            <table>
                <tr>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Range</th>
                </tr>
                <tr>
                    <td>{metrics['ad_mean']:.6f} mm²/s</td>
                    <td>{metrics['ad_median']:.6f} mm²/s</td>
                    <td>{metrics['ad_std']:.6f} mm²/s</td>
                    <td>[{metrics['ad_min']:.6f}, {metrics['ad_max']:.6f}] mm²/s</td>
                </tr>
            </table>

            <h3>Radial Diffusivity (RD)</h3>
            <table>
                <tr>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Range</th>
                </tr>
                <tr>
                    <td>{metrics['rd_mean']:.6f} mm²/s</td>
                    <td>{metrics['rd_median']:.6f} mm²/s</td>
                    <td>{metrics['rd_std']:.6f} mm²/s</td>
                    <td>[{metrics['rd_min']:.6f}, {metrics['rd_max']:.6f}] mm²/s</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>Quality Control Figures</h2>
    """

    # Add figures
    for fig_path in figures:
        if fig_path.exists():
            # Use relative path from output_dir
            rel_path = fig_path.relative_to(output_dir.parent)
            html_content += f'<img src="../{rel_path.parent.name}/{rel_path.name}" alt="{fig_path.stem}">\n'

    html_content += """
        </div>
    </body>
    </html>
    """

    # Save HTML report
    report_path = output_dir / f'{subject}_{session}_dti_qc.html'
    with open(report_path, 'w') as f:
        f.write(html_content)

    return report_path
