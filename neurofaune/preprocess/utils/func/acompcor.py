"""
aCompCor (Anatomical Component-Based Noise Correction) for fMRI denoising.

This module implements aCompCor extraction from CSF and white matter regions
to remove physiological noise from functional data.

References
----------
Behzadi et al. (2007). A component based noise correction method (CompCor) for
BOLD and perfusion based fMRI. NeuroImage 37(1): 90-101.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from scipy.ndimage import binary_erosion
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def extract_acompcor_components(
    bold_file: Path,
    csf_mask: Path,
    wm_mask: Path,
    n_components: int = 5,
    variance_threshold: float = 0.5,
    erode_voxels: int = 1,
    brain_mask: Optional[Path] = None,
    drop_first_component: bool = False,
    output_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Extract aCompCor components from CSF and white matter regions.

    Parameters
    ----------
    bold_file : Path
        Preprocessed 4D BOLD timeseries (after motion correction, filtering)
    csf_mask : Path
        CSF probability map or binary mask
    wm_mask : Path
        White matter probability map or binary mask
    n_components : int
        Number of principal components to extract (default: 5)
    variance_threshold : float
        Minimum probability/intensity threshold for tissue masks (default: 0.5)
    erode_voxels : int
        Number of binary-erosion iterations used to strip partial-volume/edge
        voxels carrying GM/BOLD signal from the tissue masks (default: 1). Set
        to 0 to disable.

        When ``brain_mask`` is supplied (the intended path), the BRAIN mask is
        eroded by ``erode_voxels`` and each tissue mask is intersected with it,
        removing only the over-inclusive outer rim (e.g. animal skull-strip
        rims) while preserving thin interior WM/CSF structure. Eroding the thin
        tissue masks directly instead — the fallback when no brain mask is given
        — decimates them at rodent resolution (a 1-2 voxel shell nearly
        vanishes under one erosion), so a brain mask should almost always be
        passed. If erosion leaves fewer voxels than ``n_components`` for a
        tissue, that tissue keeps its un-eroded mask and a warning is printed.
    brain_mask : Path, optional
        Binary brain mask in the same space as the tissue masks. When given and
        ``erode_voxels > 0``, drives the erode-brain-then-intersect rim removal
        described above.
    drop_first_component : bool
        Discard the leading PC of each tissue and keep the next
        ``n_components`` instead (default: False, i.e. keep PC1..PCn).

        The leading PC of a large tissue mask is the global signal, not
        tissue-specific physiology. Measured on the cuprizone rat cohort,
        ``corr(PC1, brain-mean signal)`` is 0.980 for WM and 0.984 for CSF,
        while PC2 onward sit near zero (0.008, 0.099). A pipeline that also runs
        global signal regression therefore models the same nuisance three times,
        which costs degrees of freedom and makes the CSF and WM regressor sets
        look interchangeable when the informative components are not.

        This does NOT assume PC1 is always global: whenever ``brain_mask`` is
        given, the correlation between each dropped component and the brain-mean
        signal is returned in ``dropped_component_gs_corr`` so the assumption is
        checkable per session rather than taken on faith.
    output_file : Path, optional
        Output TSV file for aCompCor regressors

    Returns
    -------
    dict
        Dictionary containing:
        - 'components': numpy array of aCompCor components (n_timepoints x n_components)
        - 'explained_variance': variance explained by each component
        - 'csf_components': components from CSF only
        - 'wm_components': components from WM only
        - 'n_voxels_csf': number of CSF voxels used
        - 'n_voxels_wm': number of WM voxels used

    Examples
    --------
    >>> results = extract_acompcor_components(
    ...     bold_file=Path('bold_filtered.nii.gz'),
    ...     csf_mask=Path('label-CSF_probseg.nii.gz'),
    ...     wm_mask=Path('label-WM_probseg.nii.gz'),
    ...     n_components=5
    ... )
    """
    print(f"\nExtracting aCompCor components...")
    print(f"  BOLD: {bold_file.name}")
    print(f"  CSF mask: {csf_mask.name}")
    print(f"  WM mask: {wm_mask.name}")
    print(f"  n_components: {n_components}")

    # Load BOLD data
    bold_img = nib.load(bold_file)
    bold_data = bold_img.get_fdata()

    if len(bold_data.shape) != 4:
        raise ValueError(f"Expected 4D BOLD data, got shape {bold_data.shape}")

    n_timepoints = bold_data.shape[3]
    print(f"  n_timepoints: {n_timepoints}")

    # Load tissue masks
    csf_img = nib.load(csf_mask)
    csf_data = csf_img.get_fdata()

    wm_img = nib.load(wm_mask)
    wm_data = wm_img.get_fdata()

    # Resample masks to BOLD space if needed
    if csf_data.shape != bold_data.shape[:3]:
        print(f"  Warning: CSF mask shape {csf_data.shape} != BOLD shape {bold_data.shape[:3]}")
        print(f"  Attempting to use mask as-is (may need manual resampling)")

    if wm_data.shape != bold_data.shape[:3]:
        print(f"  Warning: WM mask shape {wm_data.shape} != BOLD shape {bold_data.shape[:3]}")
        print(f"  Attempting to use mask as-is (may need manual resampling)")

    # Threshold masks
    csf_binary = csf_data > variance_threshold
    wm_binary = wm_data > variance_threshold

    # Strip partial-volume/edge voxels so aCompCor samples tissue interiors
    # only (see erode_voxels docstring). Preferred path: erode the BRAIN mask
    # and intersect (rim removal, keeps thin interior WM/CSF). Fallback with no
    # brain mask: erode each tissue mask directly (decimates thin rodent masks).
    # Either way, keep a tissue's un-eroded mask if erosion drops it below the
    # component count.
    if erode_voxels > 0:
        eroded_brain = None
        if brain_mask is not None:
            brain_data = nib.load(brain_mask).get_fdata() > 0
            eroded_brain = binary_erosion(brain_data, iterations=erode_voxels)
            print(f"  Brain-mask erosion ({erode_voxels} iter): "
                  f"{int(np.sum(brain_data))} -> {int(np.sum(eroded_brain))} voxels")

        for label, binary in [('CSF', csf_binary), ('WM', wm_binary)]:
            n_orig = int(np.sum(binary))
            if eroded_brain is not None:
                candidate = binary & eroded_brain
            else:
                candidate = binary_erosion(binary, iterations=erode_voxels)
            n_new = int(np.sum(candidate))
            if n_new < n_components:
                print(f"  WARNING: {label} erosion ({erode_voxels} iter) left "
                      f"{n_new} < {n_components} voxels — using un-eroded mask")
                continue
            print(f"  {label} erosion ({erode_voxels} iter): {n_orig} -> {n_new} voxels")
            if label == 'CSF':
                csf_binary = candidate
            else:
                wm_binary = candidate

    n_voxels_csf = np.sum(csf_binary)
    n_voxels_wm = np.sum(wm_binary)

    print(f"  CSF voxels: {n_voxels_csf}")
    print(f"  WM voxels: {n_voxels_wm}")

    if n_voxels_csf < n_components:
        print(f"  WARNING: Not enough CSF voxels ({n_voxels_csf}) for {n_components} components!")
        print(f"  Reducing to {n_voxels_csf} components for CSF")
        n_components_csf = max(1, n_voxels_csf)
    else:
        n_components_csf = n_components

    if n_voxels_wm < n_components:
        print(f"  WARNING: Not enough WM voxels ({n_voxels_wm}) for {n_components} components!")
        print(f"  Reducing to {n_voxels_wm} components for WM")
        n_components_wm = max(1, n_voxels_wm)
    else:
        n_components_wm = n_components

    # Extract timeseries from CSF voxels
    print("  Extracting CSF timeseries...")
    csf_timeseries = bold_data[csf_binary, :]  # (n_voxels_csf, n_timepoints)

    # Demean and standardize
    csf_timeseries = csf_timeseries - np.mean(csf_timeseries, axis=1, keepdims=True)
    csf_std = np.std(csf_timeseries, axis=1, keepdims=True)
    csf_std[csf_std == 0] = 1  # Avoid division by zero
    csf_timeseries = csf_timeseries / csf_std

    # Fit one extra PC when the leading one is to be discarded, so the caller
    # still receives the number of regressors it asked for.
    n_skip = 1 if drop_first_component else 0

    # PCA on CSF
    print(f"  Running PCA on CSF ({n_components_csf} components)...")
    pca_csf = PCA(n_components=min(n_components_csf + n_skip, csf_timeseries.shape[0]))
    csf_components = pca_csf.fit_transform(csf_timeseries.T)  # (n_timepoints, n_components)

    print(f"    Explained variance: {pca_csf.explained_variance_ratio_.sum():.2%}")

    # Extract timeseries from WM voxels
    print("  Extracting WM timeseries...")
    wm_timeseries = bold_data[wm_binary, :]  # (n_voxels_wm, n_timepoints)

    # Demean and standardize
    wm_timeseries = wm_timeseries - np.mean(wm_timeseries, axis=1, keepdims=True)
    wm_std = np.std(wm_timeseries, axis=1, keepdims=True)
    wm_std[wm_std == 0] = 1  # Avoid division by zero
    wm_timeseries = wm_timeseries / wm_std

    # PCA on WM
    print(f"  Running PCA on WM ({n_components_wm} components)...")
    pca_wm = PCA(n_components=min(n_components_wm + n_skip, wm_timeseries.shape[0]))
    wm_components = pca_wm.fit_transform(wm_timeseries.T)  # (n_timepoints, n_components)

    print(f"    Explained variance: {pca_wm.explained_variance_ratio_.sum():.2%}")

    # Discard the leading (global-signal) PC of each tissue, recording how
    # global it actually was so the assumption is auditable per session.
    dropped_gs_corr: Dict[str, Optional[float]] = {}
    if drop_first_component:
        gs = None
        if brain_mask is not None:
            gs = bold_data[nib.load(brain_mask).get_fdata() > 0, :].mean(axis=0)
        for label, comps in (('CSF', csf_components), ('WM', wm_components)):
            if gs is not None and comps.shape[1] > 0 and np.std(comps[:, 0]) > 0:
                dropped_gs_corr[label] = float(
                    abs(np.corrcoef(comps[:, 0], gs)[0, 1]))
            else:
                dropped_gs_corr[label] = None
        csf_components = csf_components[:, 1:]
        wm_components = wm_components[:, 1:]
        n_components_csf = csf_components.shape[1]
        n_components_wm = wm_components.shape[1]
        shown = {k: (f"{v:.3f}" if v is not None else "n/a")
                 for k, v in dropped_gs_corr.items()}
        print(f"  Dropped the leading PC of each tissue "
              f"(|corr| with brain-mean signal: {shown})")

    # Combine CSF and WM components
    all_components = np.hstack([csf_components, wm_components])

    # Save to file if requested
    if output_file:
        print(f"  Saving aCompCor regressors to {output_file}")

        # Number columns by their TRUE PC index: with drop_first_component the
        # first retained regressor is PC2, and calling it comp_1 would erase
        # that from the record.
        headers = []
        headers.extend([f'csf_comp_{i+1+n_skip}' for i in range(n_components_csf)])
        headers.extend([f'wm_comp_{i+1+n_skip}' for i in range(n_components_wm)])

        # Save as TSV
        np.savetxt(
            output_file,
            all_components,
            delimiter='\t',
            header='\t'.join(headers),
            comments=''
        )

    results = {
        'components': all_components,
        'csf_components': csf_components,
        'wm_components': wm_components,
        'explained_variance_csf': pca_csf.explained_variance_ratio_,
        'explained_variance_wm': pca_wm.explained_variance_ratio_,
        'n_voxels_csf': int(n_voxels_csf),
        'n_voxels_wm': int(n_voxels_wm),
        'n_components_csf': int(n_components_csf),
        'n_components_wm': int(n_components_wm),
        'erode_voxels': int(erode_voxels),
        'drop_first_component': bool(drop_first_component),
        'dropped_component_gs_corr': dropped_gs_corr,
    }

    print(f"  ✓ Extracted {all_components.shape[1]} aCompCor components")

    return results


def generate_acompcor_qc(
    subject: str,
    session: str,
    acompcor_results: Dict[str, Any],
    output_dir: Path
) -> Path:
    """
    Generate QC report for aCompCor denoising.

    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier
    acompcor_results : dict
        Results from extract_acompcor_components()
    output_dir : Path
        Output directory for QC report

    Returns
    -------
    Path
        Path to HTML QC report
    """
    print(f"\nGenerating aCompCor QC report...")

    # Create figures directory
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot component timeseries
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # CSF components
    csf_comps = acompcor_results['csf_components']
    for i in range(csf_comps.shape[1]):
        axes[0].plot(csf_comps[:, i], label=f'CSF-{i+1}', alpha=0.7)
    axes[0].set_xlabel('Timepoint')
    axes[0].set_ylabel('Component value')
    axes[0].set_title('CSF aCompCor Components')
    axes[0].legend(ncol=5, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # WM components
    wm_comps = acompcor_results['wm_components']
    for i in range(wm_comps.shape[1]):
        axes[1].plot(wm_comps[:, i], label=f'WM-{i+1}', alpha=0.7)
    axes[1].set_xlabel('Timepoint')
    axes[1].set_ylabel('Component value')
    axes[1].set_title('White Matter aCompCor Components')
    axes[1].legend(ncol=5, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    timeseries_fig = figures_dir / f"{subject}_{session}_acompcor_timeseries.png"
    plt.savefig(timeseries_fig, dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Plot explained variance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # CSF variance
    var_csf = acompcor_results['explained_variance_csf']
    axes[0].bar(range(1, len(var_csf)+1), var_csf * 100)
    axes[0].set_xlabel('Component')
    axes[0].set_ylabel('Explained Variance (%)')
    axes[0].set_title(f'CSF Components (Total: {var_csf.sum()*100:.1f}%)')
    axes[0].grid(True, alpha=0.3)

    # WM variance
    var_wm = acompcor_results['explained_variance_wm']
    axes[1].bar(range(1, len(var_wm)+1), var_wm * 100)
    axes[1].set_xlabel('Component')
    axes[1].set_ylabel('Explained Variance (%)')
    axes[1].set_title(f'WM Components (Total: {var_wm.sum()*100:.1f}%)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    variance_fig = figures_dir / f"{subject}_{session}_acompcor_variance.png"
    plt.savefig(variance_fig, dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Plot correlation matrix
    all_comps = acompcor_results['components']
    corr_matrix = np.corrcoef(all_comps.T)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Correlation'}
    )

    # Labels
    labels = []
    labels.extend([f'CSF-{i+1}' for i in range(acompcor_results['n_components_csf'])])
    labels.extend([f'WM-{i+1}' for i in range(acompcor_results['n_components_wm'])])

    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    ax.set_title('aCompCor Component Correlation Matrix')

    plt.tight_layout()
    corr_fig = figures_dir / f"{subject}_{session}_acompcor_correlation.png"
    plt.savefig(corr_fig, dpi=150, bbox_inches='tight')
    plt.close()

    # Generate HTML report
    report_file = output_dir / f"{subject}_{session}_acompcor_qc.html"

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>aCompCor QC - {subject} {session}</title>
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
        <h1>aCompCor QC Report</h1>
        <p><strong>Subject:</strong> {subject}</p>
        <p><strong>Session:</strong> {session}</p>

        <div class="summary">
            <h2>Summary</h2>

            <div class="metric">
                <span class="metric-label">CSF Components:</span>
                <span class="metric-value">{acompcor_results['n_components_csf']}</span>
            </div>

            <div class="metric">
                <span class="metric-label">CSF Voxels:</span>
                <span class="metric-value">{acompcor_results['n_voxels_csf']}</span>
            </div>

            <div class="metric">
                <span class="metric-label">CSF Variance Explained:</span>
                <span class="metric-value">{acompcor_results['explained_variance_csf'].sum()*100:.1f}%</span>
            </div>

            <br>

            <div class="metric">
                <span class="metric-label">WM Components:</span>
                <span class="metric-value">{acompcor_results['n_components_wm']}</span>
            </div>

            <div class="metric">
                <span class="metric-label">WM Voxels:</span>
                <span class="metric-value">{acompcor_results['n_voxels_wm']}</span>
            </div>

            <div class="metric">
                <span class="metric-label">WM Variance Explained:</span>
                <span class="metric-value">{acompcor_results['explained_variance_wm'].sum()*100:.1f}%</span>
            </div>

            <br>

            <div class="metric">
                <span class="metric-label">Total Components:</span>
                <span class="metric-value">{acompcor_results['n_components_csf'] + acompcor_results['n_components_wm']}</span>
            </div>

            <div class="metric">
                <span class="metric-label">Mask Erosion (iter):</span>
                <span class="metric-value">{acompcor_results.get('erode_voxels', 0)}</span>
            </div>
        </div>

        <div class="info-box">
            <h3>About aCompCor</h3>
            <p>aCompCor (Anatomical Component-Based Noise Correction) extracts principal components from CSF and white matter regions to model physiological noise (cardiac, respiratory). These components can be used as nuisance regressors in GLM analysis.</p>
            <ul>
                <li><strong>CSF components:</strong> Capture pulsatile artifacts from cardiac and respiratory motion</li>
                <li><strong>WM components:</strong> Capture slow drifts and non-neural fluctuations</li>
                <li><strong>Usage:</strong> Include these regressors alongside motion parameters in GLM denoising</li>
            </ul>
        </div>

        <h2>Component Timeseries</h2>
        <img src="figures/{timeseries_fig.name}" alt="aCompCor Timeseries">

        <h2>Explained Variance</h2>
        <img src="figures/{variance_fig.name}" alt="Explained Variance">

        <h2>Component Correlations</h2>
        <p>Low correlations between components indicate they capture independent noise sources.</p>
        <img src="figures/{corr_fig.name}" alt="Component Correlations">

        <hr>
        <p style="text-align: center; color: #888; font-size: 0.9em;">
            Generated by Neurofaune functional preprocessing pipeline
        </p>
    </div>
</body>
</html>
"""

    with open(report_file, 'w') as f:
        f.write(html_content)

    print(f"  ✓ aCompCor QC report: {report_file}")

    return report_file
