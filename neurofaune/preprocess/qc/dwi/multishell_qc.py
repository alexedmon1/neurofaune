"""
Multi-shell diffusion model QC.

Generates an HTML report with slice montages, histograms, and summary
statistics for DKI (MK, AK, RK, KFA) and NODDI (FICVF, ODI, FISO) metrics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def generate_multishell_qc_report(
    subject: str,
    session: str,
    dki_files: Optional[Dict[str, Path]],
    noddi_files: Optional[Dict[str, Path]],
    brain_mask: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate QC report for multi-shell diffusion model outputs.

    Parameters
    ----------
    subject, session : str
        BIDS identifiers.
    dki_files : dict or None
        ``{"MK": Path, "AK": Path, "RK": Path, "KFA": Path}``
    noddi_files : dict or None
        ``{"FICVF": Path, "ODI": Path, "FISO": Path}``
    brain_mask : Path
        Brain mask NIfTI.
    output_dir : Path
        QC output directory.

    Returns
    -------
    dict
        ``html_report``, ``metrics_file``, ``metrics``, ``figures`` paths.
    """
    print(f"\n{'=' * 80}")
    print(f"Multi-Shell QC Report: {subject} {session}")
    print(f"{'=' * 80}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    mask = nib.load(brain_mask).get_fdata() > 0
    metrics: Dict[str, float] = {}
    figure_paths: List[Path] = []

    # Collect all metric images
    all_metrics: Dict[str, tuple] = {}  # name -> (data, cmap, vmin, vmax, model)

    if dki_files:
        # Full DKI metrics (>= 15 dirs/shell)
        dki_spec = {
            "MK": ("hot", 0.0, 2.0),
            "AK": ("inferno", 0.0, 2.0),
            "RK": ("magma", 0.0, 2.0),
            "KFA": ("plasma", 0.0, 1.0),
            # MSDKI metrics (< 15 dirs/shell fallback)
            "MSK": ("hot", 0.0, 2.0),
            "MSD": ("viridis", 0.0, 0.003),
        }
        for name, (cmap, vmin, vmax) in dki_spec.items():
            p = dki_files.get(name)
            if p and p.exists():
                label = "MSDKI" if name in ("MSK", "MSD") else "DKI"
                all_metrics[name] = (nib.load(p).get_fdata(), cmap, vmin, vmax, label)

    if noddi_files:
        noddi_spec = {
            "FICVF": ("YlOrRd", 0.0, 1.0),
            "ODI": ("YlGnBu", 0.0, 1.0),
            "FISO": ("Blues", 0.0, 1.0),
        }
        for name, (cmap, vmin, vmax) in noddi_spec.items():
            p = noddi_files.get(name)
            if p and p.exists():
                all_metrics[name] = (nib.load(p).get_fdata(), cmap, vmin, vmax, "NODDI")

    if not all_metrics:
        print("  No metric images found; skipping QC.")
        return {"html_report": None, "metrics_file": None, "metrics": {}, "figures": []}

    # Statistics
    for name, (data, *_rest) in all_metrics.items():
        masked = data[mask]
        finite = masked[np.isfinite(masked)]
        if len(finite) == 0:
            continue
        metrics[f"{name}_mean"] = float(np.mean(finite))
        metrics[f"{name}_median"] = float(np.median(finite))
        metrics[f"{name}_std"] = float(np.std(finite))
        metrics[f"{name}_min"] = float(np.min(finite))
        metrics[f"{name}_max"] = float(np.max(finite))
        metrics[f"{name}_p25"] = float(np.percentile(finite, 25))
        metrics[f"{name}_p75"] = float(np.percentile(finite, 75))

    # Slice montages
    for name, (data, cmap, vmin, vmax, model) in all_metrics.items():
        fig_path = _plot_slice_montage(
            data, name, subject, session, figures_dir, cmap=cmap, vmin=vmin, vmax=vmax
        )
        figure_paths.append(fig_path)

    # Histograms
    hist_path = _plot_histograms(all_metrics, mask, subject, session, figures_dir)
    figure_paths.append(hist_path)

    # HTML report
    html_report = _create_html_report(subject, session, metrics, all_metrics, figure_paths, output_dir)

    # JSON metrics
    metrics_file = output_dir / f"{subject}_{session}_multishell_qc.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nQC report saved: {html_report}")
    print(f"Metrics saved:   {metrics_file}")

    return {
        "html_report": html_report,
        "metrics_file": metrics_file,
        "metrics": metrics,
        "figures": figure_paths,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_slice_montage(
    data: np.ndarray,
    metric_name: str,
    subject: str,
    session: str,
    output_dir: Path,
    cmap: str = "viridis",
    vmin: float = None,
    vmax: float = None,
    n_slices: int = 9,
) -> Path:
    z_dim = data.shape[2]
    slice_indices = np.linspace(0, z_dim - 1, n_slices, dtype=int)

    ncols = 3
    nrows = (n_slices + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for idx, sl in enumerate(slice_indices):
        im = axes[idx].imshow(np.rot90(data[:, :, sl]), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[idx].set_title(f"Slice {sl}")
        axes[idx].axis("off")
    for idx in range(len(slice_indices), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"{metric_name}: {subject} {session}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cbar = plt.colorbar(im, ax=list(axes), fraction=0.046, pad=0.04)
    cbar.set_label(metric_name)

    out = output_dir / f"multishell_{metric_name}_slices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def _plot_histograms(
    all_metrics: Dict[str, tuple],
    mask: np.ndarray,
    subject: str,
    session: str,
    output_dir: Path,
) -> Path:
    n = len(all_metrics)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for idx, (name, (data, cmap, vmin, vmax, model)) in enumerate(all_metrics.items()):
        masked = data[mask]
        finite = masked[np.isfinite(masked)]
        axes[idx].hist(finite, bins=50, alpha=0.7, edgecolor="black", color=plt.cm.get_cmap(cmap)(0.6))
        mean_val = np.mean(finite)
        axes[idx].axvline(mean_val, color="black", linestyle="--", label=f"Mean: {mean_val:.3f}")
        axes[idx].set_xlabel(name)
        axes[idx].set_ylabel("Frequency")
        axes[idx].set_title(f"{model} - {name}")
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"Multi-Shell Metric Distributions: {subject} {session}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = output_dir / "multishell_histograms.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _create_html_report(
    subject: str,
    session: str,
    metrics: Dict[str, float],
    all_metrics: Dict[str, tuple],
    figures: List[Path],
    output_dir: Path,
) -> Path:
    # Build stats tables
    tables_html = ""
    for name, (_, _cmap, _vmin, _vmax, model) in all_metrics.items():
        key = name
        if f"{key}_mean" not in metrics:
            continue
        tables_html += f"""
        <h3>{model} - {name}</h3>
        <table>
            <tr><th>Mean</th><th>Median</th><th>Std Dev</th><th>Range</th></tr>
            <tr>
                <td>{metrics[f'{key}_mean']:.4f}</td>
                <td>{metrics[f'{key}_median']:.4f}</td>
                <td>{metrics[f'{key}_std']:.4f}</td>
                <td>[{metrics[f'{key}_min']:.4f}, {metrics[f'{key}_max']:.4f}]</td>
            </tr>
        </table>
        """

    # Build figure tags
    figures_html = ""
    for fig_path in figures:
        if fig_path and fig_path.exists():
            rel = fig_path.name
            figures_html += f'<img src="figures/{rel}" alt="{fig_path.stem}">\n'

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Multi-Shell QC: {subject} {session}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 10px; }}
        th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Multi-Shell Diffusion QC Report</h1>
        <p>Subject: {subject} | Session: {session}</p>
    </div>
    <div class="section">
        <h2>Summary Statistics</h2>
        {tables_html}
    </div>
    <div class="section">
        <h2>Quality Control Figures</h2>
        {figures_html}
    </div>
</body>
</html>"""

    report_path = output_dir / f"{subject}_{session}_multishell_qc.html"
    with open(report_path, "w") as f:
        f.write(html)
    return report_path
