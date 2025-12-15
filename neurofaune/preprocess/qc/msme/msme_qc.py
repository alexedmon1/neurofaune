"""MSME T2 mapping and MWF quality control."""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
import json


def generate_msme_qc_report(
    subject: str,
    session: str,
    mwf_file: Path,
    iwf_file: Path,
    csf_file: Path,
    t2_file: Path,
    brain_mask: Path,
    te_values: np.ndarray,
    sample_data: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """Generate QC report for MSME T2 mapping and MWF."""
    print(f"\n{'='*80}")
    print(f"Generating MSME QC Report: {subject} {session}")
    print(f"{'='*80}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Load data
    mwf = nib.load(mwf_file).get_fdata()
    iwf = nib.load(iwf_file).get_fdata()
    csf = nib.load(csf_file).get_fdata()
    t2 = nib.load(t2_file).get_fdata()
    mask = nib.load(brain_mask).get_fdata() > 0

    # Calculate statistics
    metrics = {}
    for name, data in [('mwf', mwf), ('iwf', iwf), ('csf', csf), ('t2', t2)]:
        masked_data = data[mask]
        metrics[f'{name}_mean'] = float(np.mean(masked_data))
        metrics[f'{name}_std'] = float(np.std(masked_data))
        metrics[f'{name}_median'] = float(np.median(masked_data))
        metrics[f'{name}_min'] = float(np.min(masked_data))
        metrics[f'{name}_max'] = float(np.max(masked_data))

    # Generate visualizations
    figures = []

    # Histograms
    hist_fig = _plot_histograms(mwf, iwf, csf, t2, mask, subject, session, figures_dir)
    figures.append(hist_fig)

    # T2 decay curves
    if sample_data and len(sample_data['signals']) > 0:
        t2_curves_fig = _plot_t2_curves(
            te_values,
            sample_data,
            subject,
            session,
            figures_dir
        )
        figures.append(t2_curves_fig)

    # NNLS spectra
    if sample_data and len(sample_data['spectra']) > 0:
        nnls_fig = _plot_nnls_spectra(
            sample_data,
            subject,
            session,
            figures_dir
        )
        figures.append(nnls_fig)

    # Slice montages
    for name, data, cmap, vmin, vmax in [
        ('MWF', mwf, 'hot', 0, 0.3),
        ('IWF', iwf, 'viridis', 0, 1),
        ('CSF', csf, 'Blues', 0, 1),
        ('T2', t2, 'plasma', 0, 100)
    ]:
        montage = _plot_montage(data, name, subject, session, figures_dir, cmap, vmin, vmax)
        figures.append(montage)

    # Create HTML report
    html_report = _create_html_report(subject, session, metrics, figures, output_dir)

    # Save metrics
    metrics_file = output_dir / f'{subject}_{session}_msme_qc_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ QC report: {html_report}")

    return {
        'html_report': html_report,
        'metrics_file': metrics_file,
        'metrics': metrics,
        'figures': figures
    }


def _plot_t2_curves(te_values, sample_data, subject, session, output_dir):
    """Plot T2 decay curves from sample voxels."""
    n_samples = len(sample_data['signals'])

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for idx in range(min(n_samples, 10)):
        signal = sample_data['signals'][idx]
        voxel = sample_data['voxels'][idx]
        mwf = sample_data['mwf_values'][idx] if idx < len(sample_data['mwf_values']) else 0

        ax = axes[idx]
        ax.plot(te_values, signal, 'o-', linewidth=2, markersize=4)
        ax.set_xlabel('TE (ms)')
        ax.set_ylabel('Signal Intensity')
        ax.set_title(f'Voxel {voxel}\nMWF={mwf:.3f}')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    # Hide unused subplots
    for idx in range(n_samples, 10):
        axes[idx].axis('off')

    fig.suptitle(f'T2 Decay Curves: {subject} {session}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / f'{subject}_{session}_t2_curves.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def _plot_nnls_spectra(sample_data, subject, session, output_dir):
    """Plot NNLS T2 spectra from sample voxels with compartment shading."""
    n_samples = len(sample_data['spectra'])
    t2_dist = sample_data['t2_dist']

    # Define compartment boundaries
    mw_cutoff = np.where(t2_dist < 25)[0][-1] + 1
    iw_cutoff = np.where(t2_dist < 40)[0][-1] + 1

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx in range(min(n_samples, 10)):
        spectrum = sample_data['spectra'][idx]
        voxel = sample_data['voxels'][idx]
        mwf = sample_data['mwf_values'][idx] if idx < len(sample_data['mwf_values']) else 0

        ax = axes[idx]

        # Plot spectrum
        ax.plot(t2_dist, spectrum, linewidth=2, color='black')
        ax.fill_between(t2_dist, spectrum, alpha=0.3, color='black')

        # Shade compartments
        ax.axvspan(t2_dist[0], t2_dist[mw_cutoff-1], alpha=0.2, color='red', label='Myelin')
        ax.axvspan(t2_dist[mw_cutoff], t2_dist[iw_cutoff-1], alpha=0.2, color='blue', label='IW')
        ax.axvspan(t2_dist[iw_cutoff], t2_dist[-1], alpha=0.2, color='cyan', label='CSF')

        ax.set_xlabel('T2 (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Voxel {voxel}\nMWF={mwf:.3f}')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')

        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Hide unused subplots
    for idx in range(n_samples, 10):
        axes[idx].axis('off')

    fig.suptitle(f'NNLS T2 Spectra: {subject} {session}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / f'{subject}_{session}_nnls_spectra.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def _plot_histograms(mwf, iwf, csf, t2, mask, subject, session, output_dir):
    """Plot histograms for all MSME metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(mwf[mask], bins=50, color='red', alpha=0.7)
    axes[0, 0].set_title('Myelin Water Fraction')
    axes[0, 0].set_xlabel('MWF')

    axes[0, 1].hist(iwf[mask], bins=50, color='blue', alpha=0.7)
    axes[0, 1].set_title('Intra/Extra-cellular Water Fraction')
    axes[0, 1].set_xlabel('IWF')

    axes[1, 0].hist(csf[mask], bins=50, color='cyan', alpha=0.7)
    axes[1, 0].set_title('CSF Fraction')
    axes[1, 0].set_xlabel('CSF')

    axes[1, 1].hist(t2[mask], bins=50, color='purple', alpha=0.7)
    axes[1, 1].set_title('T2 Relaxation Time')
    axes[1, 1].set_xlabel('T2 (ms)')

    fig.suptitle(f'MSME Metrics Distribution: {subject} {session}', fontsize=14)
    plt.tight_layout()

    output_file = output_dir / f'{subject}_{session}_msme_histograms.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def _plot_montage(data, name, subject, session, output_dir, cmap, vmin, vmax):
    """Create slice montage."""
    z_dim = data.shape[2]
    slice_indices = np.linspace(0, z_dim - 1, 9, dtype=int)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for idx, slice_idx in enumerate(slice_indices):
        im = axes[idx].imshow(np.rot90(data[:, :, slice_idx]), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[idx].set_title(f'Slice {slice_idx}')
        axes[idx].axis('off')

    fig.suptitle(f'{name}: {subject} {session}', fontsize=14)
    plt.tight_layout()
    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)

    output_file = output_dir / f'{subject}_{session}_{name}_montage.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def _create_html_report(subject, session, metrics, figures, output_dir):
    """Create HTML QC report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MSME QC: {subject} {session}</title>
        <style>
            body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
            .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
            .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px; border-bottom: 1px solid #ddd; }}
            th {{ background: #34495e; color: white; }}
            img {{ max-width: 100%; margin: 10px 0; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>MSME T2 Mapping QC Report</h1>
            <p>Subject: {subject} | Session: {session}</p>
        </div>
        <div class="section">
            <h2>Summary Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Mean</th><th>Std</th><th>Range</th></tr>
                <tr><td>MWF</td><td>{metrics['mwf_mean']:.3f}</td><td>{metrics['mwf_std']:.3f}</td><td>[{metrics['mwf_min']:.3f}, {metrics['mwf_max']:.3f}]</td></tr>
                <tr><td>IWF</td><td>{metrics['iwf_mean']:.3f}</td><td>{metrics['iwf_std']:.3f}</td><td>[{metrics['iwf_min']:.3f}, {metrics['iwf_max']:.3f}]</td></tr>
                <tr><td>CSF</td><td>{metrics['csf_mean']:.3f}</td><td>{metrics['csf_std']:.3f}</td><td>[{metrics['csf_min']:.3f}, {metrics['csf_max']:.3f}]</td></tr>
                <tr><td>T2 (ms)</td><td>{metrics['t2_mean']:.1f}</td><td>{metrics['t2_std']:.1f}</td><td>[{metrics['t2_min']:.1f}, {metrics['t2_max']:.1f}]</td></tr>
            </table>
        </div>
        <div class="section">
            <h2>Quality Control Figures</h2>
    """

    for fig in figures:
        if fig.exists():
            rel_path = fig.relative_to(output_dir.parent)
            html_content += f'<img src="../{rel_path.parent.name}/{rel_path.name}">\n'

    html_content += "</div></body></html>"

    report_path = output_dir / f'{subject}_{session}_msme_qc.html'
    with open(report_path, 'w') as f:
        f.write(html_content)

    return report_path
