"""
Motion QC for functional MRI preprocessing.

Generates quality control reports for fMRI motion correction.
"""

import json
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_framewise_displacement(motion_params: np.ndarray, radius: float = 50.0) -> np.ndarray:
    """
    Calculate framewise displacement from motion parameters.
    
    FD = sum of absolute derivatives of 6 motion parameters
    Rotational displacements are converted to mm using head radius.
    
    Parameters
    ----------
    motion_params : np.ndarray
        Motion parameters array (n_timepoints x 6)
        Columns: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    radius : float
        Head radius in mm for converting rotations (default: 50mm for rodents)
    
    Returns
    -------
    np.ndarray
        Framewise displacement (n_timepoints - 1,)
    """
    # Convert rotations (radians) to mm displacement
    motion_mm = motion_params.copy()
    motion_mm[:, :3] = motion_mm[:, :3] * radius
    
    # Calculate absolute derivatives
    diff = np.abs(np.diff(motion_mm, axis=0))
    
    # Sum across all parameters
    fd = np.sum(diff, axis=1)
    
    return fd


def calculate_dvars(bold_file: Path, mask_file: Path) -> np.ndarray:
    """
    Calculate DVARS (spatial root mean square of temporal derivative).
    
    DVARS measures how much the brain image changes from one volume to the next.
    
    Parameters
    ----------
    bold_file : Path
        4D BOLD timeseries
    mask_file : Path
        Brain mask
    
    Returns
    -------
    np.ndarray
        DVARS (n_timepoints - 1,)
    """
    # Load data
    bold_img = nib.load(bold_file)
    bold_data = bold_img.get_fdata()
    
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata().astype(bool)
    
    # Extract timeseries within mask
    timeseries = bold_data[mask_data, :]
    
    # Calculate temporal derivative
    diff = np.diff(timeseries, axis=1)
    
    # Calculate RMS across voxels
    dvars = np.sqrt(np.mean(diff ** 2, axis=0))
    
    return dvars


def generate_motion_qc_report(
    subject: str,
    session: str,
    motion_params_file: Path,
    bold_file: Path,
    mask_file: Path,
    output_dir: Path,
    threshold_fd: float = 0.5
) -> Path:
    """
    Generate comprehensive motion QC report for fMRI.
    
    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier
    motion_params_file : Path
        FSL motion parameters file (.par)
    bold_file : Path
        Preprocessed BOLD file
    mask_file : Path
        Brain mask
    output_dir : Path
        Output directory for QC report
    threshold_fd : float
        Framewise displacement threshold for flagging (default: 0.5mm)
    
    Returns
    -------
    Path
        Path to HTML QC report
    """
    print(f"Generating motion QC report for {subject} {session}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Load motion parameters
    motion_params = np.loadtxt(motion_params_file)
    n_volumes = motion_params.shape[0]
    
    # Calculate metrics
    fd = calculate_framewise_displacement(motion_params)
    dvars = calculate_dvars(bold_file, mask_file)
    
    # Calculate summary statistics
    mean_fd = np.mean(fd)
    max_fd = np.max(fd)
    n_bad_volumes = np.sum(fd > threshold_fd)
    pct_bad_volumes = (n_bad_volumes / len(fd)) * 100
    
    mean_abs_displacement = np.mean(np.abs(motion_params[:, 3:]), axis=0)
    max_abs_displacement = np.max(np.abs(motion_params[:, 3:]), axis=0)
    
    mean_abs_rotation = np.mean(np.abs(motion_params[:, :3]), axis=0) * (180.0 / np.pi)  # Convert to degrees
    max_abs_rotation = np.max(np.abs(motion_params[:, :3]), axis=0) * (180.0 / np.pi)
    
    # =========================================================================
    # Generate Figures
    # =========================================================================
    
    # Figure 1: Motion parameters over time
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Rotations
    for i, label in enumerate(['X', 'Y', 'Z']):
        axes[0].plot(motion_params[:, i] * (180.0 / np.pi), label=f'Rotation {label}')
    axes[0].set_ylabel('Rotation (degrees)')
    axes[0].set_title('Rotational Motion Parameters')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Translations
    for i, label in enumerate(['X', 'Y', 'Z']):
        axes[1].plot(motion_params[:, i+3], label=f'Translation {label}')
    axes[1].set_xlabel('Volume')
    axes[1].set_ylabel('Translation (mm)')
    axes[1].set_title('Translational Motion Parameters')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    motion_params_fig = figures_dir / f'{subject}_{session}_motion_params.png'
    plt.savefig(motion_params_fig, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Framewise displacement and DVARS
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Framewise displacement
    volumes = np.arange(len(fd))
    axes[0].plot(volumes, fd, 'b-', linewidth=1)
    axes[0].axhline(y=threshold_fd, color='r', linestyle='--', label=f'Threshold ({threshold_fd}mm)')
    axes[0].fill_between(volumes, 0, fd, where=(fd > threshold_fd), 
                          color='red', alpha=0.3, label=f'Flagged volumes ({n_bad_volumes})')
    axes[0].set_ylabel('FD (mm)')
    axes[0].set_title(f'Framewise Displacement (Mean={mean_fd:.3f}mm, Max={max_fd:.3f}mm)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # DVARS
    axes[1].plot(np.arange(len(dvars)), dvars, 'g-', linewidth=1)
    axes[1].set_xlabel('Volume')
    axes[1].set_ylabel('DVARS')
    axes[1].set_title(f'DVARS (Mean={np.mean(dvars):.3f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fd_dvars_fig = figures_dir / f'{subject}_{session}_fd_dvars.png'
    plt.savefig(fd_dvars_fig, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Distribution of motion metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # FD distribution
    axes[0].hist(fd, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=mean_fd, color='blue', linestyle='--', linewidth=2, label=f'Mean={mean_fd:.3f}mm')
    axes[0].axvline(x=threshold_fd, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold_fd}mm')
    axes[0].set_xlabel('Framewise Displacement (mm)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('FD Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # DVARS distribution
    axes[1].hist(dvars, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=np.mean(dvars), color='green', linestyle='--', linewidth=2, 
                     label=f'Mean={np.mean(dvars):.3f}')
    axes[1].set_xlabel('DVARS')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('DVARS Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_fig = figures_dir / f'{subject}_{session}_motion_distributions.png'
    plt.savefig(dist_fig, dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Generate HTML Report
    # =========================================================================
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Motion QC Report - {subject} {session}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #666;
            margin-top: 30px;
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
        }}
        .metric-label {{
            font-weight: bold;
            color: #555;
        }}
        .metric-value {{
            color: #000;
            font-size: 1.1em;
        }}
        .warning {{
            color: #ff9800;
            font-weight: bold;
        }}
        .error {{
            color: #f44336;
            font-weight: bold;
        }}
        .good {{
            color: #4CAF50;
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
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Motion QC Report</h1>
        <p><strong>Subject:</strong> {subject}</p>
        <p><strong>Session:</strong> {session}</p>
        <p><strong>Number of volumes:</strong> {n_volumes}</p>
        
        <div class="summary">
            <h2>Summary Statistics</h2>
            
            <div class="metric">
                <span class="metric-label">Mean FD:</span>
                <span class="metric-value {'good' if mean_fd < threshold_fd else 'warning'}">{mean_fd:.3f} mm</span>
            </div>
            
            <div class="metric">
                <span class="metric-label">Max FD:</span>
                <span class="metric-value {'good' if max_fd < threshold_fd*2 else 'error'}">{max_fd:.3f} mm</span>
            </div>
            
            <div class="metric">
                <span class="metric-label">Bad volumes:</span>
                <span class="metric-value {'good' if pct_bad_volumes < 5 else 'warning' if pct_bad_volumes < 20 else 'error'}">
                    {n_bad_volumes} ({pct_bad_volumes:.1f}%)
                </span>
            </div>
            
            <div class="metric">
                <span class="metric-label">Mean DVARS:</span>
                <span class="metric-value">{np.mean(dvars):.3f}</span>
            </div>
        </div>
        
        <h2>Motion Parameters</h2>
        
        <table>
            <tr>
                <th>Parameter</th>
                <th>Mean (abs)</th>
                <th>Max (abs)</th>
            </tr>
            <tr>
                <td>Rotation X</td>
                <td>{mean_abs_rotation[0]:.3f}°</td>
                <td>{max_abs_rotation[0]:.3f}°</td>
            </tr>
            <tr>
                <td>Rotation Y</td>
                <td>{mean_abs_rotation[1]:.3f}°</td>
                <td>{max_abs_rotation[1]:.3f}°</td>
            </tr>
            <tr>
                <td>Rotation Z</td>
                <td>{mean_abs_rotation[2]:.3f}°</td>
                <td>{max_abs_rotation[2]:.3f}°</td>
            </tr>
            <tr>
                <td>Translation X</td>
                <td>{mean_abs_displacement[0]:.3f} mm</td>
                <td>{max_abs_displacement[0]:.3f} mm</td>
            </tr>
            <tr>
                <td>Translation Y</td>
                <td>{mean_abs_displacement[1]:.3f} mm</td>
                <td>{max_abs_displacement[1]:.3f} mm</td>
            </tr>
            <tr>
                <td>Translation Z</td>
                <td>{mean_abs_displacement[2]:.3f} mm</td>
                <td>{max_abs_displacement[2]:.3f} mm</td>
            </tr>
        </table>
        
        <h2>Motion Plots</h2>
        
        <h3>Motion Parameters Over Time</h3>
        <img src="figures/{motion_params_fig.name}" alt="Motion Parameters">
        
        <h3>Framewise Displacement and DVARS</h3>
        <img src="figures/{fd_dvars_fig.name}" alt="FD and DVARS">
        
        <h3>Distribution of Motion Metrics</h3>
        <img src="figures/{dist_fig.name}" alt="Motion Distributions">
        
        <h2>Quality Assessment</h2>
        <ul>
            <li><strong>Motion Quality:</strong> 
                {'<span class="good">GOOD</span> - Mean FD below threshold' if mean_fd < threshold_fd 
                 else '<span class="warning">FAIR</span> - Elevated motion detected' if mean_fd < threshold_fd * 2
                 else '<span class="error">POOR</span> - Excessive motion detected'}
            </li>
            <li><strong>Bad Volumes:</strong>
                {'<span class="good">GOOD</span> - Less than 5% flagged' if pct_bad_volumes < 5
                 else '<span class="warning">FAIR</span> - 5-20% flagged' if pct_bad_volumes < 20
                 else '<span class="error">POOR</span> - More than 20% flagged'}
            </li>
            <li><strong>Recommendation:</strong>
                {'Data quality appears good for analysis' if mean_fd < threshold_fd and pct_bad_volumes < 10
                 else 'Consider scrubbing high-motion volumes or using robust denoising methods'
                 if pct_bad_volumes < 30
                 else 'Consider excluding this subject due to excessive motion'}
            </li>
        </ul>
        
        <hr>
        <p style="text-align: center; color: #888; font-size: 0.9em;">
            Generated by Neurofaune functional preprocessing pipeline
        </p>
    </div>
</body>
</html>
"""
    
    # Save HTML report
    report_file = output_dir / f'{subject}_{session}_motion_qc.html'
    with open(report_file, 'w') as f:
        f.write(html_content)

    print(f"  Motion QC report saved: {report_file}")

    # Save metrics JSON for batch QC
    metrics = {
        'motion': {
            'n_volumes': int(n_volumes),
            'mean_fd': float(mean_fd),
            'max_fd': float(max_fd),
            'median_fd': float(np.median(fd)),
            'std_fd': float(np.std(fd)),
            'n_bad_volumes': int(n_bad_volumes),
            'pct_bad_volumes': float(pct_bad_volumes),
            'fd_threshold': float(threshold_fd),
            'mean_dvars': float(np.mean(dvars)),
            'max_dvars': float(np.max(dvars)),
            'std_dvars': float(np.std(dvars)),
        },
        'translation': {
            'mean_abs_x': float(mean_abs_displacement[0]),
            'mean_abs_y': float(mean_abs_displacement[1]),
            'mean_abs_z': float(mean_abs_displacement[2]),
            'max_abs_x': float(max_abs_displacement[0]),
            'max_abs_y': float(max_abs_displacement[1]),
            'max_abs_z': float(max_abs_displacement[2]),
        },
        'rotation': {
            'mean_abs_x_deg': float(mean_abs_rotation[0]),
            'mean_abs_y_deg': float(mean_abs_rotation[1]),
            'mean_abs_z_deg': float(mean_abs_rotation[2]),
            'max_abs_x_deg': float(max_abs_rotation[0]),
            'max_abs_y_deg': float(max_abs_rotation[1]),
            'max_abs_z_deg': float(max_abs_rotation[2]),
        }
    }

    metrics_file = output_dir / f'{subject}_{session}_func_qc_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  Metrics saved: {metrics_file}")

    return report_file
