"""
Registration QC for functional MRI preprocessing.

Generates quality control reports for fMRI registration to anatomical and atlas spaces.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn import plotting
import json


def generate_registration_qc(
    subject: str,
    session: str,
    bold_file: Path,
    t2w_file: Path,
    bold_in_t2w: Path,
    atlas_file: Optional[Path] = None,
    bold_in_atlas: Optional[Path] = None,
    output_dir: Path = None
) -> Path:
    """
    Generate registration QC report showing BOLD alignment to anatomical and atlas.

    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier
    bold_file : Path
        Mean BOLD in native space
    t2w_file : Path
        T2w anatomical reference
    bold_in_t2w : Path
        Registered BOLD in T2w space
    atlas_file : Path, optional
        SIGMA atlas template
    bold_in_atlas : Path, optional
        Registered BOLD in SIGMA atlas space
    output_dir : Path
        Output directory for QC report

    Returns
    -------
    Path
        Path to HTML QC report
    """
    print(f"\nGenerating registration QC for {subject} {session}...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Generate overlay plots
    print("  Creating registration overlay images...")

    # 1. BOLD → T2w registration
    bold_to_t2w_fig = figures_dir / f"{subject}_{session}_bold_to_t2w_overlay.png"
    create_overlay_plot(
        underlay=t2w_file,
        overlay=bold_in_t2w,
        output_file=bold_to_t2w_fig,
        title=f"BOLD (registered) overlaid on T2w - {subject} {session}"
    )

    # 2. Native BOLD for reference
    native_bold_fig = figures_dir / f"{subject}_{session}_bold_native.png"
    create_simple_plot(
        image_file=bold_file,
        output_file=native_bold_fig,
        title=f"Mean BOLD (native space) - {subject} {session}"
    )

    # 3. BOLD → SIGMA registration (if provided)
    if atlas_file and bold_in_atlas:
        bold_to_sigma_fig = figures_dir / f"{subject}_{session}_bold_to_sigma_overlay.png"
        create_overlay_plot(
            underlay=atlas_file,
            overlay=bold_in_atlas,
            output_file=bold_to_sigma_fig,
            title=f"BOLD (registered) overlaid on SIGMA - {subject} {session}"
        )
    else:
        bold_to_sigma_fig = None

    # 4. Anatomical reference
    t2w_fig = figures_dir / f"{subject}_{session}_t2w_reference.png"
    create_simple_plot(
        image_file=t2w_file,
        output_file=t2w_fig,
        title=f"T2w anatomical reference - {subject} {session}"
    )

    # Calculate registration quality metrics
    print("  Calculating registration quality metrics...")
    metrics = calculate_registration_metrics(
        fixed=t2w_file,
        moving=bold_in_t2w
    )

    if atlas_file and bold_in_atlas:
        atlas_metrics = calculate_registration_metrics(
            fixed=atlas_file,
            moving=bold_in_atlas
        )
        metrics['atlas_registration'] = atlas_metrics

    # Generate HTML report
    print("  Generating HTML report...")
    report_file = output_dir / f"{subject}_{session}_registration_qc.html"
    generate_html_report(
        subject=subject,
        session=session,
        metrics=metrics,
        figures={
            'native_bold': native_bold_fig.name,
            't2w_reference': t2w_fig.name,
            'bold_to_t2w': bold_to_t2w_fig.name,
            'bold_to_sigma': bold_to_sigma_fig.name if bold_to_sigma_fig else None
        },
        output_file=report_file
    )

    print(f"  ✓ Registration QC report: {report_file}")

    return report_file


def create_overlay_plot(
    underlay: Path,
    overlay: Path,
    output_file: Path,
    title: str,
    cmap: str = 'hot',
    alpha: float = 0.7
):
    """Create overlay visualization using nilearn."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Load images
    underlay_img = nib.load(underlay)
    overlay_img = nib.load(overlay)

    # Axial views
    for i, cut_coord in enumerate([-2, 0, 2]):
        display = plotting.plot_anat(
            underlay_img,
            cut_coords=[cut_coord],
            display_mode='z',
            axes=axes[0, i],
            title=f'Axial (z={cut_coord}mm)'
        )
        display.add_overlay(overlay_img, alpha=alpha, cmap=cmap)

    # Sagittal views
    for i, cut_coord in enumerate([-2, 0, 2]):
        display = plotting.plot_anat(
            underlay_img,
            cut_coords=[cut_coord],
            display_mode='x',
            axes=axes[1, i],
            title=f'Sagittal (x={cut_coord}mm)'
        )
        display.add_overlay(overlay_img, alpha=alpha, cmap=cmap)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def create_simple_plot(
    image_file: Path,
    output_file: Path,
    title: str
):
    """Create simple anatomical plot."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    img = nib.load(image_file)

    # Axial views
    for i, cut_coord in enumerate([-2, 0, 2]):
        plotting.plot_anat(
            img,
            cut_coords=[cut_coord],
            display_mode='z',
            axes=axes[0, i],
            title=f'Axial (z={cut_coord}mm)'
        )

    # Sagittal views
    for i, cut_coord in enumerate([-2, 0, 2]):
        plotting.plot_anat(
            img,
            cut_coords=[cut_coord],
            display_mode='x',
            axes=axes[1, i],
            title=f'Sagittal (x={cut_coord}mm)'
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_registration_metrics(
    fixed: Path,
    moving: Path
) -> dict:
    """
    Calculate registration quality metrics.

    Parameters
    ----------
    fixed : Path
        Fixed (reference) image
    moving : Path
        Moving (registered) image

    Returns
    -------
    dict
        Dictionary with quality metrics
    """
    # Load images
    fixed_img = nib.load(fixed)
    fixed_data = fixed_img.get_fdata()

    moving_img = nib.load(moving)
    moving_data = moving_img.get_fdata()

    # Ensure same shape
    if fixed_data.shape != moving_data.shape:
        print(f"    Warning: Shape mismatch - fixed: {fixed_data.shape}, moving: {moving_data.shape}")
        return {'error': 'shape_mismatch'}

    # Flatten and normalize
    fixed_flat = fixed_data.flatten()
    moving_flat = moving_data.flatten()

    # Remove zeros (background)
    mask = (fixed_flat > 0) & (moving_flat > 0)
    fixed_masked = fixed_flat[mask]
    moving_masked = moving_flat[mask]

    if len(fixed_masked) == 0:
        return {'error': 'no_overlap'}

    # Calculate correlation
    correlation = np.corrcoef(fixed_masked, moving_masked)[0, 1]

    # Calculate normalized mutual information (approximation)
    # Using histogram-based approach
    hist_2d, _, _ = np.histogram2d(
        fixed_masked,
        moving_masked,
        bins=50
    )

    # Normalize histogram
    hist_2d = hist_2d / np.sum(hist_2d)

    # Calculate marginal distributions
    px = np.sum(hist_2d, axis=1)
    py = np.sum(hist_2d, axis=0)

    # Calculate entropies (avoiding log(0))
    px = px[px > 0]
    py = py[py > 0]
    hist_2d_nonzero = hist_2d[hist_2d > 0]

    h_x = -np.sum(px * np.log(px))
    h_y = -np.sum(py * np.log(py))
    h_xy = -np.sum(hist_2d_nonzero * np.log(hist_2d_nonzero))

    # Normalized mutual information
    nmi = (h_x + h_y) / h_xy if h_xy > 0 else 0

    return {
        'correlation': float(correlation),
        'normalized_mutual_information': float(nmi),
        'n_voxels_overlap': int(len(fixed_masked))
    }


def generate_html_report(
    subject: str,
    session: str,
    metrics: dict,
    figures: dict,
    output_file: Path
):
    """Generate HTML registration QC report."""

    # Assess registration quality
    corr = metrics.get('correlation', 0)
    nmi = metrics.get('normalized_mutual_information', 0)

    if 'error' in metrics:
        quality = "ERROR"
        quality_class = "error"
    elif corr > 0.7 and nmi > 1.5:
        quality = "GOOD"
        quality_class = "good"
    elif corr > 0.5 and nmi > 1.2:
        quality = "FAIR"
        quality_class = "warning"
    else:
        quality = "POOR"
        quality_class = "error"

    # Get atlas metrics if available
    atlas_metrics = metrics.get('atlas_registration', {})
    atlas_corr = atlas_metrics.get('correlation', 0)
    atlas_nmi = atlas_metrics.get('normalized_mutual_information', 0)

    if atlas_corr > 0:
        if atlas_corr > 0.7 and atlas_nmi > 1.5:
            atlas_quality = "GOOD"
            atlas_quality_class = "good"
        elif atlas_corr > 0.5 and atlas_nmi > 1.2:
            atlas_quality = "FAIR"
            atlas_quality_class = "warning"
        else:
            atlas_quality = "POOR"
            atlas_quality_class = "error"
    else:
        atlas_quality = "N/A"
        atlas_quality_class = ""

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Registration QC - {subject} {session}</title>
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
        .good {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .warning {{
            color: #ff9800;
            font-weight: bold;
        }}
        .error {{
            color: #f44336;
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
        <h1>Registration QC Report</h1>
        <p><strong>Subject:</strong> {subject}</p>
        <p><strong>Session:</strong> {session}</p>

        <div class="summary">
            <h2>Registration Quality</h2>

            <div class="metric">
                <span class="metric-label">BOLD → T2w:</span>
                <span class="metric-value {quality_class}">{quality}</span>
            </div>

            <div class="metric">
                <span class="metric-label">Correlation:</span>
                <span class="metric-value">{corr:.3f}</span>
            </div>

            <div class="metric">
                <span class="metric-label">NMI:</span>
                <span class="metric-value">{nmi:.3f}</span>
            </div>

            {f'''
            <br>
            <div class="metric">
                <span class="metric-label">BOLD → SIGMA:</span>
                <span class="metric-value {atlas_quality_class}">{atlas_quality}</span>
            </div>

            <div class="metric">
                <span class="metric-label">Correlation:</span>
                <span class="metric-value">{atlas_corr:.3f}</span>
            </div>

            <div class="metric">
                <span class="metric-label">NMI:</span>
                <span class="metric-value">{atlas_nmi:.3f}</span>
            </div>
            ''' if atlas_corr > 0 else ''}
        </div>

        <h2>Native BOLD (mean)</h2>
        <img src="figures/{figures['native_bold']}" alt="Native BOLD">

        <h2>T2w Anatomical Reference</h2>
        <img src="figures/{figures['t2w_reference']}" alt="T2w Reference">

        <h2>BOLD → T2w Registration</h2>
        <img src="figures/{figures['bold_to_t2w']}" alt="BOLD to T2w">

        {f'''
        <h2>BOLD → SIGMA Atlas Registration</h2>
        <img src="figures/{figures['bold_to_sigma']}" alt="BOLD to SIGMA">
        ''' if figures['bold_to_sigma'] else ''}

        <h2>Quality Criteria</h2>
        <ul>
            <li><strong>Correlation > 0.7 and NMI > 1.5:</strong> GOOD registration</li>
            <li><strong>Correlation > 0.5 and NMI > 1.2:</strong> FAIR registration</li>
            <li><strong>Below thresholds:</strong> POOR registration - review manually</li>
        </ul>

        <hr>
        <p style="text-align: center; color: #888; font-size: 0.9em;">
            Generated by Neurofaune functional preprocessing pipeline
        </p>
    </div>
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html_content)
