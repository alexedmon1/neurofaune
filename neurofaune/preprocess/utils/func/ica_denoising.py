"""
Rodent-specific ICA denoising for fMRI data.

This module implements automated component classification based on:
1. Motion correlation
2. Spatial features (edge fraction, CSF overlap)
3. Temporal/frequency features

Unlike ICA-AROMA (which is human-specific), this approach is designed
for rodent brain anatomy and physiology.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import subprocess
import json
from scipy import stats, signal
from scipy.ndimage import binary_erosion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def run_melodic_ica(
    input_file: Path,
    output_dir: Path,
    brain_mask: Path,
    tr: float,
    n_components: int = 40,
    bg_image: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Run FSL MELODIC ICA decomposition.
    
    Parameters
    ----------
    input_file : Path
        Preprocessed BOLD timeseries (motion corrected, filtered)
    output_dir : Path
        Output directory for MELODIC results
    brain_mask : Path
        Brain mask
    tr : float
        Repetition time (seconds)
    n_components : int
        Number of ICA components (default: 40)
    bg_image : Path, optional
        Background image for visualization
    
    Returns
    -------
    dict
        Paths to MELODIC outputs
    """
    print(f"Running MELODIC ICA with {n_components} components...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'melodic',
        '-i', str(input_file),
        '-o', str(output_dir),
        '-m', str(brain_mask),
        f'--tr={tr}',  # FSL wants --tr=value format
        f'--dim={n_components}',
        '--nobet',  # Already masked
        '--report',  # Generate HTML report
        '-v'  # Verbose
    ]

    if bg_image is not None:
        cmd.extend([f'--bgimage={bg_image}'])
    
    subprocess.run(cmd, check=True)
    
    print(f"  MELODIC complete")
    
    # Return paths to key outputs
    return {
        'melodic_ic': output_dir / 'melodic_IC.nii.gz',  # Spatial maps (4D)
        'melodic_mix': output_dir / 'melodic_mix',  # Mixing matrix (timecourses)
        'melodic_ftmix': output_dir / 'melodic_FTmix',  # Frequency spectrum
        'report': output_dir / 'report.html'
    }


def calculate_edge_fraction(
    component_map: np.ndarray,
    brain_mask: np.ndarray,
    threshold: float = 2.0
) -> float:
    """
    Calculate fraction of component located at brain edges.
    
    Higher edge fraction suggests motion/susceptibility artifacts.
    
    Parameters
    ----------
    component_map : np.ndarray
        3D spatial map of ICA component (z-scored)
    brain_mask : np.ndarray
        Binary brain mask
    threshold : float
        Z-score threshold for considering voxel as "active"
    
    Returns
    -------
    float
        Edge fraction (0-1)
    """
    # Binarize component at threshold
    comp_binary = np.abs(component_map) > threshold
    
    # Erode brain mask to get interior
    brain_eroded = binary_erosion(brain_mask, iterations=2)
    
    # Edge is brain - eroded brain
    edge_mask = brain_mask & ~brain_eroded
    
    # Count voxels in component
    n_comp_voxels = np.sum(comp_binary & brain_mask)
    
    if n_comp_voxels == 0:
        return 0.0
    
    # Count component voxels at edge
    n_edge_voxels = np.sum(comp_binary & edge_mask)
    
    edge_fraction = n_edge_voxels / n_comp_voxels
    
    return edge_fraction


def calculate_csf_overlap(
    component_map: np.ndarray,
    csf_mask: np.ndarray,
    threshold: float = 2.0
) -> float:
    """
    Calculate overlap of component with CSF regions.
    
    High CSF overlap suggests physiological noise (pulsation).
    
    Parameters
    ----------
    component_map : np.ndarray
        3D spatial map of ICA component
    csf_mask : np.ndarray
        Binary or probabilistic CSF mask
    threshold : float
        Z-score threshold for component
    
    Returns
    -------
    float
        CSF overlap fraction (0-1)
    """
    # Binarize component
    comp_binary = np.abs(component_map) > threshold
    
    # Count overlap with CSF
    n_comp_voxels = np.sum(comp_binary)
    
    if n_comp_voxels == 0:
        return 0.0
    
    # If CSF mask is probabilistic, threshold it
    if csf_mask.dtype in [np.float32, np.float64]:
        csf_binary = csf_mask > 0.5
    else:
        csf_binary = csf_mask.astype(bool)
    
    n_csf_overlap = np.sum(comp_binary & csf_binary)
    
    csf_fraction = n_csf_overlap / n_comp_voxels
    
    return csf_fraction


def calculate_motion_correlation(
    component_timecourse: np.ndarray,
    motion_params: np.ndarray
) -> float:
    """
    Calculate maximum absolute correlation between component and motion.
    
    High correlation suggests motion artifact.
    
    Parameters
    ----------
    component_timecourse : np.ndarray
        ICA component timecourse (n_timepoints,)
    motion_params : np.ndarray
        Motion parameters (n_timepoints, 6)
        Columns: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    
    Returns
    -------
    float
        Maximum absolute correlation with any motion parameter
    """
    correlations = []
    
    for i in range(motion_params.shape[1]):
        r, _ = stats.pearsonr(component_timecourse, motion_params[:, i])
        correlations.append(abs(r))
    
    return max(correlations)


def calculate_frequency_content(
    component_timecourse: np.ndarray,
    tr: float,
    high_freq_threshold: float = 0.15
) -> float:
    """
    Calculate fraction of power in high frequencies.
    
    High-frequency dominated components are likely noise.
    
    Parameters
    ----------
    component_timecourse : np.ndarray
        ICA component timecourse
    tr : float
        Repetition time (seconds)
    high_freq_threshold : float
        Frequency threshold (Hz) for "high frequency"
        Default: 0.15 Hz (appropriate for rodents with fast TR)
    
    Returns
    -------
    float
        Fraction of power in high frequencies (0-1)
    """
    # Calculate power spectrum
    freqs, psd = signal.welch(
        component_timecourse,
        fs=1.0/tr,
        nperseg=min(len(component_timecourse), 128)
    )
    
    # Find frequencies above threshold
    high_freq_mask = freqs > high_freq_threshold
    
    # Calculate power fractions
    total_power = np.sum(psd)
    high_freq_power = np.sum(psd[high_freq_mask])
    
    if total_power == 0:
        return 0.0
    
    high_freq_fraction = high_freq_power / total_power
    
    return high_freq_fraction


def classify_ica_components(
    melodic_dir: Path,
    motion_params_file: Path,
    brain_mask_file: Path,
    tr: float,
    csf_mask_file: Optional[Path] = None,
    motion_threshold: float = 0.40,
    edge_threshold: float = 0.80,
    csf_threshold: float = 0.70,
    freq_threshold: float = 0.60,
    classification_mode: str = 'score'
) -> Dict[str, Any]:
    """
    Automatically classify ICA components as signal or noise.

    Two classification modes:

    1. 'threshold' (old, aggressive): Component is NOISE if ANY criterion is met
       - Motion correlation > motion_threshold
       - Edge fraction > edge_threshold
       - CSF overlap > csf_threshold (if CSF mask provided)
       - High-frequency power > freq_threshold

    2. 'score' (new, recommended): Component is scored on all features
       - Very high motion (>0.5) → automatically noise
       - Otherwise, score based on combination of features
       - Need score >= 2.0 to be classified as noise
       - Weights: motion=1.5, edge=1.0, csf=1.0, freq=0.8
    
    Parameters
    ----------
    melodic_dir : Path
        MELODIC output directory
    motion_params_file : Path
        Motion parameters file (.par format)
    brain_mask_file : Path
        Brain mask
    tr : float
        Repetition time (seconds)
    csf_mask_file : Path, optional
        CSF mask for CSF overlap calculation
    motion_threshold : float
        Threshold for motion correlation (default: 0.40)
    edge_threshold : float
        Threshold for edge fraction (default: 0.80)
    csf_threshold : float
        Threshold for CSF overlap (default: 0.70)
    freq_threshold : float
        Threshold for high-frequency power (default: 0.60)
    classification_mode : str
        Classification mode: 'threshold' or 'score' (default: 'score')
    
    Returns
    -------
    dict
        Classification results with component features and labels
    """
    print("Classifying ICA components...")
    
    # Load MELODIC outputs
    melodic_ic = nib.load(melodic_dir / 'melodic_IC.nii.gz')
    ic_data = melodic_ic.get_fdata()  # (x, y, z, n_components)
    n_components = ic_data.shape[3]
    
    # Load mixing matrix (timecourses)
    mix = np.loadtxt(melodic_dir / 'melodic_mix')  # (n_timepoints, n_components)
    
    # Load motion parameters
    motion = np.loadtxt(motion_params_file)  # (n_timepoints, 6)
    
    # Ensure motion and mix have same length
    min_len = min(len(motion), len(mix))
    motion = motion[:min_len]
    mix = mix[:min_len]
    
    # Load brain mask
    brain_mask = nib.load(brain_mask_file).get_fdata().astype(bool)
    
    # Load CSF mask if provided
    csf_mask = None
    if csf_mask_file is not None and csf_mask_file.exists():
        csf_mask = nib.load(csf_mask_file).get_fdata()
    
    # Calculate features for each component
    results = {
        'n_components': n_components,
        'components': []
    }
    
    for ic_idx in range(n_components):
        print(f"  Component {ic_idx + 1}/{n_components}...", end='\r')
        
        # Extract spatial map and timecourse
        spatial_map = ic_data[:, :, :, ic_idx]
        timecourse = mix[:, ic_idx]
        
        # Calculate features
        motion_corr = calculate_motion_correlation(timecourse, motion)
        edge_frac = calculate_edge_fraction(spatial_map, brain_mask)
        freq_frac = calculate_frequency_content(timecourse, tr)
        
        csf_frac = 0.0
        if csf_mask is not None:
            csf_frac = calculate_csf_overlap(spatial_map, csf_mask)

        # Classify component based on mode
        if classification_mode == 'threshold':
            # Old aggressive mode: ANY criterion triggers noise classification
            is_noise = (
                motion_corr > motion_threshold or
                edge_frac > edge_threshold or
                csf_frac > csf_threshold or
                freq_frac > freq_threshold
            )
            noise_score = None

        elif classification_mode == 'score':
            # New scoring mode: need multiple bad features to be noise

            # Automatic noise if very high motion
            if motion_corr > 0.5:
                is_noise = True
                noise_score = 999.0  # Extremely high score
            else:
                # Calculate weighted score
                noise_score = 0.0

                # Motion contribution (weight: 1.5)
                if motion_corr > motion_threshold:
                    noise_score += 1.5 * (motion_corr - motion_threshold) / (1.0 - motion_threshold)

                # Edge contribution (weight: 1.0)
                if edge_frac > edge_threshold:
                    noise_score += 1.0 * (edge_frac - edge_threshold) / (1.0 - edge_threshold)

                # CSF contribution (weight: 1.0)
                if csf_frac > csf_threshold:
                    noise_score += 1.0 * (csf_frac - csf_threshold) / (1.0 - csf_threshold)

                # Frequency contribution (weight: 0.8)
                if freq_frac > freq_threshold:
                    noise_score += 0.8 * (freq_frac - freq_threshold) / (1.0 - freq_threshold)

                # Classify as noise if score >= 2.0 (needs ~2 bad features)
                is_noise = noise_score >= 2.0

        else:
            raise ValueError(f"Unknown classification_mode: {classification_mode}")

        # Store results
        component_result = {
            'index': ic_idx,
            'label': 'noise' if is_noise else 'signal',
            'motion_correlation': motion_corr,
            'edge_fraction': edge_frac,
            'csf_fraction': csf_frac,
            'high_freq_power': freq_frac,
            'noise_score': noise_score,
            'classification_reason': []
        }

        # Record which criteria contributed to noise classification
        if is_noise:
            if classification_mode == 'score':
                component_result['classification_reason'].append(
                    f'noise_score={noise_score:.3f} >= 2.0'
                )
                if motion_corr > motion_threshold:
                    component_result['classification_reason'].append(
                        f'motion_corr={motion_corr:.3f} > {motion_threshold}'
                    )
                if edge_frac > edge_threshold:
                    component_result['classification_reason'].append(
                        f'edge_frac={edge_frac:.3f} > {edge_threshold}'
                    )
                if csf_frac > csf_threshold:
                    component_result['classification_reason'].append(
                        f'csf_frac={csf_frac:.3f} > {csf_threshold}'
                    )
                if freq_frac > freq_threshold:
                    component_result['classification_reason'].append(
                        f'freq_frac={freq_frac:.3f} > {freq_threshold}'
                    )
            else:
                # Threshold mode
                if motion_corr > motion_threshold:
                    component_result['classification_reason'].append(
                        f'motion_corr={motion_corr:.3f} > {motion_threshold}'
                    )
                if edge_frac > edge_threshold:
                    component_result['classification_reason'].append(
                        f'edge_frac={edge_frac:.3f} > {edge_threshold}'
                    )
                if csf_frac > csf_threshold:
                    component_result['classification_reason'].append(
                        f'csf_frac={csf_frac:.3f} > {csf_threshold}'
                    )
                if freq_frac > freq_threshold:
                    component_result['classification_reason'].append(
                        f'freq_frac={freq_frac:.3f} > {freq_threshold}'
                    )
        
        results['components'].append(component_result)
    
    print()  # New line after progress
    
    # Summary
    n_signal = sum(1 for c in results['components'] if c['label'] == 'signal')
    n_noise = sum(1 for c in results['components'] if c['label'] == 'noise')
    
    results['summary'] = {
        'n_signal': n_signal,
        'n_noise': n_noise,
        'signal_components': [c['index'] for c in results['components'] if c['label'] == 'signal'],
        'noise_components': [c['index'] for c in results['components'] if c['label'] == 'noise']
    }
    
    print(f"  Classification complete:")
    print(f"    Signal components: {n_signal}")
    print(f"    Noise components: {n_noise}")
    
    return results


def remove_noise_components(
    input_file: Path,
    output_file: Path,
    melodic_dir: Path,
    noise_components: List[int]
) -> Path:
    """
    Remove noise components from fMRI data using fsl_regfilt.
    
    Parameters
    ----------
    input_file : Path
        Input BOLD timeseries (should match what was input to MELODIC)
    output_file : Path
        Output denoised timeseries
    melodic_dir : Path
        MELODIC output directory
    noise_components : list of int
        Component indices to remove (0-indexed)
    
    Returns
    -------
    Path
        Path to denoised file
    """
    print(f"Removing {len(noise_components)} noise components...")
    
    if len(noise_components) == 0:
        print("  No noise components to remove, copying input to output")
        import shutil
        shutil.copy(str(input_file), str(output_file))
        return output_file
    
    # Convert to 1-indexed for FSL
    noise_indices_fsl = [str(idx + 1) for idx in noise_components]
    
    # Use fsl_regfilt to remove components
    cmd = [
        'fsl_regfilt',
        '-i', str(input_file),
        '-o', str(output_file),
        '-d', str(melodic_dir / 'melodic_mix'),
        '-f', ','.join(noise_indices_fsl)  # Components to remove
    ]
    
    subprocess.run(cmd, check=True)
    
    print(f"  Denoised data saved: {output_file}")
    
    return output_file


def generate_ica_denoising_qc(
    subject: str,
    session: str,
    classification_results: Dict[str, Any],
    melodic_dir: Path,
    output_dir: Path
) -> Path:
    """
    Generate QC report for ICA denoising.
    
    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier
    classification_results : dict
        Results from classify_ica_components()
    melodic_dir : Path
        MELODIC output directory
    output_dir : Path
        Output directory for QC report
    
    Returns
    -------
    Path
        Path to HTML QC report
    """
    print(f"Generating ICA denoising QC report...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    components = classification_results['components']
    summary = classification_results['summary']
    
    # =========================================================================
    # Figure 1: Feature distributions
    # =========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract features
    motion_corrs = [c['motion_correlation'] for c in components]
    edge_fracs = [c['edge_fraction'] for c in components]
    csf_fracs = [c['csf_fraction'] for c in components]
    freq_fracs = [c['high_freq_power'] for c in components]
    
    signal_indices = [c['index'] for c in components if c['label'] == 'signal']
    noise_indices = [c['index'] for c in components if c['label'] == 'noise']
    
    # Motion correlation
    axes[0, 0].scatter(signal_indices, [motion_corrs[i] for i in signal_indices],
                       c='green', label='Signal', alpha=0.6, s=50)
    axes[0, 0].scatter(noise_indices, [motion_corrs[i] for i in noise_indices],
                       c='red', label='Noise', alpha=0.6, s=50)
    axes[0, 0].axhline(y=0.35, color='orange', linestyle='--', label='Threshold')
    axes[0, 0].set_xlabel('Component Index')
    axes[0, 0].set_ylabel('Motion Correlation')
    axes[0, 0].set_title('Motion Correlation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Edge fraction
    axes[0, 1].scatter(signal_indices, [edge_fracs[i] for i in signal_indices],
                       c='green', label='Signal', alpha=0.6, s=50)
    axes[0, 1].scatter(noise_indices, [edge_fracs[i] for i in noise_indices],
                       c='red', label='Noise', alpha=0.6, s=50)
    axes[0, 1].axhline(y=0.50, color='orange', linestyle='--', label='Threshold')
    axes[0, 1].set_xlabel('Component Index')
    axes[0, 1].set_ylabel('Edge Fraction')
    axes[0, 1].set_title('Edge Fraction')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # CSF overlap
    axes[1, 0].scatter(signal_indices, [csf_fracs[i] for i in signal_indices],
                       c='green', label='Signal', alpha=0.6, s=50)
    axes[1, 0].scatter(noise_indices, [csf_fracs[i] for i in noise_indices],
                       c='red', label='Noise', alpha=0.6, s=50)
    axes[1, 0].axhline(y=0.50, color='orange', linestyle='--', label='Threshold')
    axes[1, 0].set_xlabel('Component Index')
    axes[1, 0].set_ylabel('CSF Overlap Fraction')
    axes[1, 0].set_title('CSF Overlap')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # High frequency power
    axes[1, 1].scatter(signal_indices, [freq_fracs[i] for i in signal_indices],
                       c='green', label='Signal', alpha=0.6, s=50)
    axes[1, 1].scatter(noise_indices, [freq_fracs[i] for i in noise_indices],
                       c='red', label='Noise', alpha=0.6, s=50)
    axes[1, 1].axhline(y=0.50, color='orange', linestyle='--', label='Threshold')
    axes[1, 1].set_xlabel('Component Index')
    axes[1, 1].set_ylabel('High Frequency Power Fraction')
    axes[1, 1].set_title('High Frequency Content')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    features_fig = figures_dir / f'{subject}_{session}_ica_features.png'
    plt.savefig(features_fig, dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Figure 2: Classification summary
    # =========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = ['Signal', 'Noise']
    counts = [summary['n_signal'], summary['n_noise']]
    colors = ['green', 'red']
    
    bars = ax.bar(labels, counts, color=colors, alpha=0.6, edgecolor='black')
    ax.set_ylabel('Number of Components')
    ax.set_title(f'ICA Component Classification (Total: {len(components)})')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    summary_fig = figures_dir / f'{subject}_{session}_ica_summary.png'
    plt.savefig(summary_fig, dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Generate HTML Report
    # =========================================================================
    
    # Build component table HTML
    component_rows = ""
    for comp in components:
        row_class = 'noise-row' if comp['label'] == 'noise' else 'signal-row'
        reasons = '<br>'.join(comp['classification_reason']) if comp['classification_reason'] else '-'
        
        component_rows += f"""
            <tr class="{row_class}">
                <td>{comp['index'] + 1}</td>
                <td><strong>{comp['label'].upper()}</strong></td>
                <td>{comp['motion_correlation']:.3f}</td>
                <td>{comp['edge_fraction']:.3f}</td>
                <td>{comp['csf_fraction']:.3f}</td>
                <td>{comp['high_freq_power']:.3f}</td>
                <td style="font-size: 0.85em;">{reasons}</td>
            </tr>
"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ICA Denoising QC - {subject} {session}</title>
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
            font-size: 0.9em;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        .signal-row {{
            background-color: #e8f5e9;
        }}
        .noise-row {{
            background-color: #ffebee;
        }}
        .info-box {{
            background-color: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ICA Denoising QC Report</h1>
        <p><strong>Subject:</strong> {subject}</p>
        <p><strong>Session:</strong> {session}</p>
        
        <div class="summary">
            <h2>Classification Summary</h2>
            
            <div class="metric">
                <span class="metric-label">Total Components:</span>
                <span class="metric-value">{len(components)}</span>
            </div>
            
            <div class="metric">
                <span class="metric-label">Signal Components:</span>
                <span class="metric-value" style="color: green;">{summary['n_signal']}</span>
            </div>
            
            <div class="metric">
                <span class="metric-label">Noise Components:</span>
                <span class="metric-value" style="color: red;">{summary['n_noise']}</span>
            </div>
            
            <div class="metric">
                <span class="metric-label">Signal Percentage:</span>
                <span class="metric-value">{summary['n_signal']/len(components)*100:.1f}%</span>
            </div>
        </div>
        
        <div class="info-box">
            <h3>Classification Criteria (Rodent-Specific)</h3>
            <p>Components are classified as <strong>NOISE</strong> if ANY of these criteria are met:</p>
            <ul>
                <li><strong>Motion Correlation > 0.35:</strong> High correlation with head motion parameters</li>
                <li><strong>Edge Fraction > 0.50:</strong> More than 50% of component voxels at brain edges</li>
                <li><strong>CSF Overlap > 0.50:</strong> More than 50% overlap with CSF regions (physiological noise)</li>
                <li><strong>High-Frequency Power > 0.50:</strong> More than 50% of spectral power above 0.15 Hz</li>
            </ul>
            <p><em>Note: These thresholds are optimized for rodent fMRI and differ from human-based ICA-AROMA.</em></p>
        </div>
        
        <h2>Classification Results</h2>
        <img src="figures/{summary_fig.name}" alt="Classification Summary">
        
        <h2>Component Features</h2>
        <img src="figures/{features_fig.name}" alt="Component Features">
        
        <h2>Detailed Component Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Component #</th>
                    <th>Classification</th>
                    <th>Motion Corr</th>
                    <th>Edge Frac</th>
                    <th>CSF Overlap</th>
                    <th>High Freq Power</th>
                    <th>Classification Reason</th>
                </tr>
            </thead>
            <tbody>
{component_rows}
            </tbody>
        </table>
        
        <h2>Signal Components (Retained)</h2>
        <p><strong>Component indices (1-indexed):</strong> {', '.join([str(i+1) for i in summary['signal_components']])}</p>
        
        <h2>Noise Components (Removed)</h2>
        <p><strong>Component indices (1-indexed):</strong> {', '.join([str(i+1) for i in summary['noise_components']])}</p>
        
        <h2>MELODIC Report</h2>
        <p>For detailed spatial maps and timecourses, see the <a href="{melodic_dir}/report.html">MELODIC HTML report</a>.</p>
        
        <hr>
        <p style="text-align: center; color: #888; font-size: 0.9em;">
            Generated by Neurofaune rodent-specific ICA denoising pipeline
        </p>
    </div>
</body>
</html>
"""
    
    # Save HTML report
    report_file = output_dir / f'{subject}_{session}_ica_denoising_qc.html'
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    # Save JSON with classification results
    json_file = output_dir / f'{subject}_{session}_ica_classification.json'
    with open(json_file, 'w') as f:
        json.dump(classification_results, f, indent=2)
    
    print(f"  ICA denoising QC report saved: {report_file}")
    print(f"  Classification results saved: {json_file}")
    
    return report_file
