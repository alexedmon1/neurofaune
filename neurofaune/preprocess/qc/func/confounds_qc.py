"""
Confounds QC for functional MRI preprocessing.

Generates quality control reports for confound regressors.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def generate_confounds_qc_report(
    subject: str,
    session: str,
    confounds_file: Path,
    output_dir: Path
) -> Path:
    """
    Generate QC report for confound regressors.
    
    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier
    confounds_file : Path
        Confounds TSV file
    output_dir : Path
        Output directory for QC report
    
    Returns
    -------
    Path
        Path to HTML QC report
    """
    print(f"Generating confounds QC report for {subject} {session}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Load confounds
    confounds = pd.read_csv(confounds_file, sep='\t')
    n_regressors = confounds.shape[1]
    n_timepoints = confounds.shape[0]
    
    # =========================================================================
    # Generate Figures
    # =========================================================================
    
    # Figure 1: Correlation matrix of confounds
    fig, ax = plt.subplots(figsize=(12, 10))
    
    corr_matrix = confounds.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Confound Regressor Correlation Matrix')
    
    plt.tight_layout()
    corr_fig = figures_dir / f'{subject}_{session}_confounds_correlation.png'
    plt.savefig(corr_fig, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Timeseries of primary confounds (first 6)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(confounds.columns[:6]):
        axes[i].plot(confounds[col], linewidth=1)
        axes[i].set_title(col)
        axes[i].set_xlabel('Volume')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    ts_fig = figures_dir / f'{subject}_{session}_confounds_timeseries.png'
    plt.savefig(ts_fig, dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Generate HTML Report
    # =========================================================================
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Confounds QC Report - {subject} {session}</title>
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
            padding: 8px;
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
        <h1>Confounds QC Report</h1>
        <p><strong>Subject:</strong> {subject}</p>
        <p><strong>Session:</strong> {session}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            
            <div class="metric">
                <span class="metric-label">Number of regressors:</span>
                <span class="metric-value">{n_regressors}</span>
            </div>
            
            <div class="metric">
                <span class="metric-label">Number of timepoints:</span>
                <span class="metric-value">{n_timepoints}</span>
            </div>
        </div>
        
        <h2>Confound Regressors</h2>
        <p>The following confound regressors were extracted:</p>
        <ul>
            <li><strong>6 motion parameters:</strong> 3 rotations (rad), 3 translations (mm)</li>
            <li><strong>6 temporal derivatives:</strong> First-order derivatives of motion parameters</li>
            <li><strong>12 squared terms:</strong> Squared motion parameters and their derivatives</li>
        </ul>
        
        <h3>Regressor Names</h3>
        <table>
            <tr>
                <th>#</th>
                <th>Regressor Name</th>
                <th>Mean</th>
                <th>Std Dev</th>
                <th>Min</th>
                <th>Max</th>
            </tr>
"""
    
    for i, col in enumerate(confounds.columns):
        mean_val = confounds[col].mean()
        std_val = confounds[col].std()
        min_val = confounds[col].min()
        max_val = confounds[col].max()
        
        html_content += f"""
            <tr>
                <td>{i+1}</td>
                <td><code>{col}</code></td>
                <td>{mean_val:.4f}</td>
                <td>{std_val:.4f}</td>
                <td>{min_val:.4f}</td>
                <td>{max_val:.4f}</td>
            </tr>
"""
    
    html_content += f"""
        </table>
        
        <h2>Visualizations</h2>
        
        <h3>Confound Correlation Matrix</h3>
        <p>Correlation between all confound regressors. High correlations (|r| > 0.9) may indicate multicollinearity.</p>
        <img src="figures/{corr_fig.name}" alt="Confound Correlation">
        
        <h3>Primary Confound Timeseries</h3>
        <p>Timeseries of the six primary motion parameters.</p>
        <img src="figures/{ts_fig.name}" alt="Confound Timeseries">
        
        <h2>Notes</h2>
        <ul>
            <li>These confound regressors can be used for GLM-based denoising</li>
            <li>Consider additional confounds such as CSF/WM signals if needed</li>
            <li>For task fMRI, include task regressors in addition to these confounds</li>
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
    report_file = output_dir / f'{subject}_{session}_confounds_qc.html'
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    print(f"  Confounds QC report saved: {report_file}")
    
    return report_file
