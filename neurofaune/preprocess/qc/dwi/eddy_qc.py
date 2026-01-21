"""
Eddy correction and motion QC for DWI data.

This module generates quality control visualizations for eddy current correction
and subject motion in diffusion MRI data.
"""

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid X11 errors
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import json


def generate_eddy_qc_report(
    subject: str,
    session: str,
    dwi_preproc: Path,
    eddy_params: Optional[Path] = None,
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Generate comprehensive QC report for eddy correction.

    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier
    dwi_preproc : Path
        Eddy-corrected DWI file
    eddy_params : Path, optional
        Eddy parameter file (.eddy_parameters)
    output_dir : Path
        Output directory for QC reports

    Returns
    -------
    dict
        Dictionary with QC metrics and report paths
    """
    print(f"\n{'='*80}")
    print(f"Generating Eddy QC Report: {subject} {session}")
    print(f"{'='*80}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    qc_metrics = {}
    figure_paths = []

    # Load DWI data
    img = nib.load(dwi_preproc)
    data = img.get_fdata()
    n_volumes = data.shape[3]

    # Generate motion plots if parameters available
    if eddy_params and eddy_params.exists():
        motion_metrics, motion_fig = _plot_motion_parameters(
            eddy_params,
            subject,
            session,
            figures_dir
        )
        qc_metrics.update(motion_metrics)
        figure_paths.append(motion_fig)

    # Generate signal intensity plots
    signal_fig = _plot_signal_intensity(
        data,
        subject,
        session,
        figures_dir
    )
    figure_paths.append(signal_fig)

    # Generate volume-wise QC
    volume_metrics, volume_fig = _plot_volume_metrics(
        data,
        subject,
        session,
        figures_dir
    )
    qc_metrics.update(volume_metrics)
    figure_paths.append(volume_fig)

    # Create HTML report
    html_report = _create_eddy_html_report(
        subject,
        session,
        qc_metrics,
        figure_paths,
        output_dir
    )

    # Save metrics to JSON
    metrics_file = output_dir / f'{subject}_{session}_eddy_qc_metrics.json'
    with open(metrics_file, 'w') as f:
        # Convert numpy types to native Python types
        metrics_serializable = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                for k, v in qc_metrics.items()}
        json.dump(metrics_serializable, f, indent=2)

    print(f"\n✓ Eddy QC report saved: {html_report}")
    print(f"✓ Metrics saved: {metrics_file}")

    return {
        'html_report': html_report,
        'metrics_file': metrics_file,
        'metrics': qc_metrics,
        'figures': figure_paths
    }


def _plot_motion_parameters(
    params_file: Path,
    subject: str,
    session: str,
    output_dir: Path
) -> tuple:
    """Plot motion parameters from eddy correction."""
    # Load eddy parameters (rotations and translations)
    params = np.loadtxt(params_file)

    # Extract translations (columns 3-5) and rotations (columns 0-2)
    translations = params[:, 3:6]  # in mm
    rotations = params[:, 0:3] * 180 / np.pi  # convert to degrees

    # Calculate framewise displacement (FD)
    fd = np.zeros(len(params))
    fd[1:] = np.sum(np.abs(np.diff(translations, axis=0)), axis=1) + \
             np.sum(np.abs(np.diff(rotations, axis=0)), axis=1) * 50 / 180  # assume 50mm head radius

    # Calculate summary metrics
    metrics = {
        'mean_fd': float(np.mean(fd)),
        'max_fd': float(np.max(fd)),
        'mean_translation': float(np.mean(np.abs(translations))),
        'max_translation': float(np.max(np.abs(translations))),
        'mean_rotation': float(np.mean(np.abs(rotations))),
        'max_rotation': float(np.max(np.abs(rotations))),
        'n_high_motion_volumes': int(np.sum(fd > 0.5))  # FD > 0.5mm threshold
    }

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot translations
    axes[0].plot(translations[:, 0], label='X (mm)', alpha=0.7)
    axes[0].plot(translations[:, 1], label='Y (mm)', alpha=0.7)
    axes[0].plot(translations[:, 2], label='Z (mm)', alpha=0.7)
    axes[0].set_ylabel('Translation (mm)')
    axes[0].set_title(f'Motion Parameters: {subject} {session}')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot rotations
    axes[1].plot(rotations[:, 0], label='Pitch (deg)', alpha=0.7)
    axes[1].plot(rotations[:, 1], label='Roll (deg)', alpha=0.7)
    axes[1].plot(rotations[:, 2], label='Yaw (deg)', alpha=0.7)
    axes[1].set_ylabel('Rotation (degrees)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Plot framewise displacement
    axes[2].plot(fd, color='darkred', alpha=0.7)
    axes[2].axhline(y=0.5, color='red', linestyle='--', label='FD threshold (0.5mm)')
    axes[2].set_xlabel('Volume')
    axes[2].set_ylabel('Framewise Displacement (mm)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f'{subject}_{session}_motion_params.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return metrics, output_file


def _plot_signal_intensity(
    data: np.ndarray,
    subject: str,
    session: str,
    output_dir: Path
) -> Path:
    """Plot signal intensity across volumes."""
    # Calculate mean signal per volume
    mean_signal = np.mean(data, axis=(0, 1, 2))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mean_signal, marker='o', markersize=3, linewidth=1, alpha=0.7)
    ax.set_xlabel('Volume')
    ax.set_ylabel('Mean Signal Intensity')
    ax.set_title(f'DWI Signal Intensity: {subject} {session}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f'{subject}_{session}_signal_intensity.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def _plot_volume_metrics(
    data: np.ndarray,
    subject: str,
    session: str,
    output_dir: Path
) -> tuple:
    """Calculate and plot volume-wise quality metrics."""
    n_volumes = data.shape[3]

    # Calculate SNR estimate per volume (mean / std within brain)
    # Use simple thresholding for brain mask
    brain_mask = np.mean(data, axis=3) > (0.1 * np.max(data))

    snr_volumes = []
    for i in range(n_volumes):
        vol = data[..., i]
        brain_signal = vol[brain_mask]
        if len(brain_signal) > 0:
            snr = np.mean(brain_signal) / (np.std(brain_signal) + 1e-10)
            snr_volumes.append(snr)
        else:
            snr_volumes.append(0)

    snr_volumes = np.array(snr_volumes)

    metrics = {
        'mean_snr': float(np.mean(snr_volumes)),
        'min_snr': float(np.min(snr_volumes)),
        'snr_std': float(np.std(snr_volumes))
    }

    # Plot SNR per volume
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(snr_volumes, marker='o', markersize=3, linewidth=1, alpha=0.7, color='green')
    ax.set_xlabel('Volume')
    ax.set_ylabel('SNR Estimate')
    ax.set_title(f'Volume-wise SNR: {subject} {session}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f'{subject}_{session}_volume_snr.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return metrics, output_file


def _create_eddy_html_report(
    subject: str,
    session: str,
    metrics: Dict[str, Any],
    figures: List[Path],
    output_dir: Path
) -> Path:
    """Create HTML QC report for eddy correction."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Eddy QC Report: {subject} {session}</title>
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
            <h1>DWI Eddy Correction QC Report</h1>
            <p>Subject: {subject} | Session: {session}</p>
        </div>

        <div class="section">
            <h2>Summary Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Status</th>
                </tr>
    """

    # Add motion metrics if available
    if 'mean_fd' in metrics:
        status = '<span class="good">PASS</span>' if metrics['mean_fd'] < 0.3 else '<span class="warning">CHECK</span>'
        html_content += f"""
                <tr>
                    <td>Mean Framewise Displacement</td>
                    <td>{metrics['mean_fd']:.3f} mm</td>
                    <td>{status}</td>
                </tr>
        """

        status = '<span class="good">PASS</span>' if metrics['max_fd'] < 0.5 else '<span class="warning">CHECK</span>'
        html_content += f"""
                <tr>
                    <td>Maximum Framewise Displacement</td>
                    <td>{metrics['max_fd']:.3f} mm</td>
                    <td>{status}</td>
                </tr>
        """

        status = '<span class="good">PASS</span>' if metrics['n_high_motion_volumes'] == 0 else '<span class="warning">CHECK</span>'
        html_content += f"""
                <tr>
                    <td>High Motion Volumes (FD > 0.5mm)</td>
                    <td>{metrics['n_high_motion_volumes']}</td>
                    <td>{status}</td>
                </tr>
        """

    # Add SNR metrics if available
    if 'mean_snr' in metrics:
        status = '<span class="good">PASS</span>' if metrics['mean_snr'] > 10 else '<span class="warning">CHECK</span>'
        html_content += f"""
                <tr>
                    <td>Mean SNR</td>
                    <td>{metrics['mean_snr']:.2f}</td>
                    <td>{status}</td>
                </tr>
        """

    html_content += """
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
    report_path = output_dir / f'{subject}_{session}_eddy_qc.html'
    with open(report_path, 'w') as f:
        f.write(html_content)

    return report_path
