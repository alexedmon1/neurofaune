"""Quality control for single-voxel MR spectroscopy.

The three things that decide whether an SVS fit can be trusted are: was the
voxel where you meant it to be, is the spectrum good enough to fit (SNR and
linewidth), and did the model actually fit it (residual and CRLB). This module
reports all three, reusing the metrics ``fsl_mrs`` already computes rather than
recalculating them.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

#: Thresholds used to flag a session for review. Linewidth is the strongest
#: single indicator of a bad shim; CRLB above 20% is the conventional cutoff
#: for reporting an individual metabolite.
QC_THRESHOLDS = {
    'min_snr': 10.0,
    'max_fwhm_hz': 20.0,
    'max_crlb_percent': 20.0,
    'min_voxel_coverage': 0.9,
    # Share of the voxel that must fall inside the structure it was aimed at.
    # Not a tight bound -- a 7.5x2x2 mm box over a thin curved structure like
    # hippocampus reaches about 70% at best -- but low enough to sit well clear
    # of that and high enough to catch a voxel placed on the wrong structure,
    # which is what a wrong geometry convention produces.
    'min_target_overlap': 0.35,
}

#: Metabolites whose concentrations are usually reported, used for the
#: at-a-glance table. Others still appear in the full CSV.
KEY_METABOLITES = ['NAA', 'NAA+NAAG', 'Cr+PCr', 'GPC+PCh', 'Glu', 'Glu+Gln', 'Ins', 'Tau']


def _reference_metrics(quality: pd.DataFrame) -> Dict[str, float]:
    """SNR and linewidth of the reference peak (NAA, falling back to tCr)."""
    for name in ('NAA', 'Cr', 'PCr'):
        if name in quality.index:
            return {
                'reference_peak': name,
                'snr': float(quality.loc[name, 'SNR']),
                'fwhm_hz': float(quality.loc[name, 'FWHM']),
            }
    return {'reference_peak': 'none', 'snr': float('nan'), 'fwhm_hz': float('nan')}


def _crlb_percent(fit_dir: Path) -> pd.Series:
    """Per-metabolite CRLB as a percentage of the fitted concentration."""
    concentrations = pd.read_csv(fit_dir / 'concentrations.csv', header=[0, 1], index_col=0)
    raw = concentrations[('mean', 'raw')]
    sd = concentrations[('std', 'raw')]
    with np.errstate(divide='ignore', invalid='ignore'):
        return (100.0 * sd / raw).replace([np.inf, -np.inf], np.nan)


def plot_voxel_overlay(
    anat_image: Path,
    voxel_mask: Path,
    output_file: Path,
    subject: str,
    session: str,
) -> Optional[Path]:
    """Show the SVS voxel outlined on three anatomical slices.

    This is the check that the voxel is where it was meant to be. The Bruker
    geometry has to be mapped onto the image grid by hand (see
    :mod:`neurofaune.preprocess.utils.mrs.voxel_geometry`), so it is worth
    looking at rather than assuming.
    """
    anat = nib.load(str(anat_image)).get_fdata()
    mask = nib.load(str(voxel_mask)).get_fdata()

    slices = sorted({int(z) for z in np.argwhere(mask > 0.5)[:, 2]})
    if not slices:
        logger.warning("%s %s: voxel mask is empty, skipping overlay", subject, session)
        return None

    chosen = [slices[0], slices[len(slices) // 2], slices[-1]]
    fig, axes = plt.subplots(1, len(chosen), figsize=(4.5 * len(chosen), 4.8))
    axes = np.atleast_1d(axes)
    for axis, z in zip(axes, chosen):
        axis.imshow(anat[:, :, z].T, cmap='gray', origin='lower')
        axis.contour(mask[:, :, z].T, levels=[0.5], colors='red', linewidths=1.5)
        axis.set_title(f'slice {z}')
        axis.axis('off')
    fig.suptitle(f'{subject} {session} - SVS voxel placement')
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=110)
    plt.close(fig)
    return output_file


def plot_metabolite_crlb(
    crlb: pd.Series,
    output_file: Path,
    subject: str,
    session: str,
) -> Path:
    """Bar chart of per-metabolite CRLB, with the 20% reporting cutoff marked."""
    values = crlb.dropna().sort_values()
    fig, axis = plt.subplots(figsize=(9, max(3.0, 0.28 * len(values))))
    colours = ['tab:red' if v > QC_THRESHOLDS['max_crlb_percent'] else 'tab:blue'
               for v in values]
    axis.barh(values.index, values.values, color=colours)
    axis.axvline(QC_THRESHOLDS['max_crlb_percent'], color='k', ls='--', lw=1,
                 label=f"{QC_THRESHOLDS['max_crlb_percent']:.0f}% cutoff")
    axis.set_xlabel('CRLB (% of concentration)')
    axis.set_title(f'{subject} {session} - fit uncertainty')
    axis.legend()
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=110)
    plt.close(fig)
    return output_file


def plot_mm_envelope(
    envelope: pd.DataFrame,
    areas: pd.DataFrame,
    output_file: Path,
    subject: str,
    session: str,
) -> Path:
    """The MM fit: spline through the metabolite-free spectrum, bands shaded.

    Worth showing rather than reporting the areas alone. It was this plot that
    exposed the negative trough at 0.95-1.10 ppm, which the numbers alone hid
    and which set where the band edges had to go. Provisional bands are drawn
    in a different colour so they are not read as equal to MM09.

    Parameters
    ----------
    envelope : DataFrame
        A ``*_mm-envelope.csv``: ``ppm, signal, envelope, anchor``.
    areas : DataFrame
        A ``*_mm-areas.csv``.
    """
    from neurofaune.preprocess.utils.mrs.mm_quantify import DEFAULT_FLANKS

    fig, axis = plt.subplots(figsize=(9, 4))
    for low, high in DEFAULT_FLANKS:
        axis.axvspan(low, high, color='0.90', zorder=0)

    for row in areas.itertuples():
        inside = (envelope['ppm'] >= row.ppm_low) & (envelope['ppm'] <= row.ppm_high)
        axis.fill_between(envelope.loc[inside, 'ppm'], 0,
                          envelope.loc[inside, 'envelope'], zorder=1, alpha=0.30,
                          color='tab:orange' if row.provisional else 'tab:red',
                          label=f'{row.band} {row.area_per_tcr:.3f}/tCr'
                                + (' (provisional)' if row.provisional else ''))

    # The signal is drawn on the anchored scale so it and the spline share a zero.
    axis.plot(envelope['ppm'], envelope['signal'] - envelope['anchor'],
              lw=0.5, color='0.5', label='metabolite-free spectrum')
    axis.plot(envelope['ppm'], envelope['envelope'], lw=2, color='tab:blue',
              label='spline envelope')
    axis.axhline(0, lw=0.8, color='k')
    axis.invert_xaxis()
    axis.set_xlabel('ppm')
    axis.set_ylabel('signal (arbitrary units)')
    axis.set_title(f'{subject} {session} - macromolecule envelope '
                   '(grey = anchor flanks)')
    axis.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=110)
    plt.close(fig)
    return output_file


def _html_report(
    subject: str,
    session: str,
    metrics: Dict[str, Any],
    metadata: Dict[str, Any],
    concentrations: pd.DataFrame,
    crlb: pd.Series,
    figures: List[Path],
    fit_dir: Path,
    output_file: Path,
) -> Path:
    """Write a small self-contained HTML summary linking the fsl_mrs report."""
    def _flag(ok: bool) -> str:
        return ('<span style="color:#178a3a;font-weight:600">PASS</span>' if ok
                else '<span style="color:#c0392b;font-weight:600">REVIEW</span>')

    rows = []
    for name in KEY_METABOLITES:
        if name not in concentrations.index:
            continue
        rows.append(
            f"<tr><td>{name}</td>"
            f"<td>{concentrations.loc[name, ('mean', 'molarity')]:.2f}</td>"
            f"<td>{concentrations.loc[name, ('mean', 'internal')]:.3f}</td>"
            f"<td>{crlb.get(name, float('nan')):.1f}</td></tr>"
        )

    figure_html = '\n'.join(
        f'<img src="{Path(f).name}" style="max-width:100%;margin:12px 0">'
        for f in figures if f is not None
    )
    fractions = metadata.get('tissue_fractions', {})
    measured = 'measured from T2w' if fractions.get('measured') else 'assumed'

    placement_row = ''
    if 'target_overlap' in metrics:
        placement_row = (
            f"<tr><td>Voxel on target ({metrics.get('target_structure', '')})</td>"
            f"<td>{metrics['target_overlap'] * 100:.0f}%</td>"
            f"<td>&ge; {QC_THRESHOLDS['min_target_overlap'] * 100:.0f}%</td>"
            f"<td>{_flag(metrics['placement_pass'])}</td></tr>")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MRS QC - {subject} {session}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 60rem; }}
 table {{ border-collapse: collapse; margin: 1rem 0; }}
 th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
 th {{ background: #f2f2f2; }}
 .meta {{ color: #555; font-size: 0.9em; }}
</style></head><body>
<h1>MRS QC: {subject} {session}</h1>
<p class="meta">
 PRESS TE {metadata['echo_time_s'] * 1000:.0f} ms / TR {metadata['repetition_time_s']:.1f} s,
 voxel {' x '.join(f'{v:.1f}' for v in metadata['voxel_size_mm'])} mm
 ({metadata['voxel_volume_ul']:.1f} uL),
 {metadata['n_averages']} averages, {metadata['n_coils']} coils,
 read from <code>{metadata['source']}</code>.
</p>

<h2>Quality</h2>
<table>
 <tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th></tr>
 <tr><td>SNR ({metrics['reference_peak']})</td><td>{metrics['snr']:.1f}</td>
     <td>&ge; {QC_THRESHOLDS['min_snr']:.0f}</td><td>{_flag(metrics['snr_pass'])}</td></tr>
 <tr><td>Linewidth ({metrics['reference_peak']})</td><td>{metrics['fwhm_hz']:.1f} Hz</td>
     <td>&le; {QC_THRESHOLDS['max_fwhm_hz']:.0f} Hz</td><td>{_flag(metrics['fwhm_pass'])}</td></tr>
 <tr><td>Metabolites with CRLB &le; {QC_THRESHOLDS['max_crlb_percent']:.0f}%</td>
     <td>{metrics['n_metabolites_reliable']} / {metrics['n_metabolites']}</td>
     <td>-</td><td>-</td></tr>
 {placement_row}
 <tr><td>Voxel coverage by anatomical slab</td>
     <td>{metrics['voxel_coverage'] * 100:.0f}%</td>
     <td>&ge; {QC_THRESHOLDS['min_voxel_coverage'] * 100:.0f}%</td>
     <td>{_flag(metrics['coverage_pass'])}</td></tr>
</table>

<h2>Tissue composition ({measured})</h2>
<table>
 <tr><th>GM</th><th>WM</th><th>CSF</th></tr>
 <tr><td>{fractions.get('GM', float('nan')):.3f}</td>
     <td>{fractions.get('WM', float('nan')):.3f}</td>
     <td>{fractions.get('CSF', float('nan')):.3f}</td></tr>
</table>

<h2>Key metabolites</h2>
<table>
 <tr><th>Metabolite</th><th>mM (water-scaled)</th><th>/ (Cr+PCr)</th><th>CRLB %</th></tr>
 {''.join(rows)}
</table>

<h2>Figures</h2>
{figure_html}

<p><a href="{Path(fit_dir).resolve().as_uri()}/report.html">Full fsl_mrs fit report</a></p>
</body></html>
"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html)
    return output_file


def generate_mrs_qc_report(
    subject: str,
    session: str,
    fit_dir: Path,
    preproc_dir: Path,
    metadata: Dict[str, Any],
    qc_dir: Path,
    anat_image: Optional[Path] = None,
    voxel_mask: Optional[Path] = None,
    mm_areas: Optional[Path] = None,
    mm_envelope: Optional[Path] = None,
    mm_lineshape: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the per-session MRS QC report.

    Parameters
    ----------
    subject, session : str
    fit_dir : Path
        ``fsl_mrs`` output directory.
    preproc_dir : Path
        ``fsl_mrs_preproc`` output directory.
    metadata : dict
        The metadata block written by the workflow.
    qc_dir : Path
        Where the report and figures are written, e.g.
        ``{study}/mrs/qc/{subject}/{session}/``.
    anat_image, voxel_mask : Path, optional
        When both are given, a voxel-placement overlay is included.
    mm_areas, mm_envelope : Path, optional
        The ``*_mm-areas.csv`` / ``*_mm-envelope.csv`` written by
        :func:`~neurofaune.preprocess.workflows.mrs_preprocess.export_mm_areas`.
        When both are given, the macromolecule fit is plotted and the validated
        band areas are added to the metrics.

    Returns
    -------
    dict
        Metrics, pass/fail flags and the report path.
    """
    fit_dir = Path(fit_dir)
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = qc_dir / 'figures'

    quality = pd.read_csv(fit_dir / 'qc.csv', index_col=0)
    concentrations = pd.read_csv(fit_dir / 'concentrations.csv', header=[0, 1], index_col=0)
    crlb = _crlb_percent(fit_dir)

    metrics: Dict[str, Any] = _reference_metrics(quality)
    reliable = crlb.dropna() <= QC_THRESHOLDS['max_crlb_percent']
    coverage = float(metadata.get('tissue_fractions', {}).get('voxel_volume_ratio', 1.0))
    placement = metadata.get('placement') or {}
    if placement:
        metrics['target_structure'] = placement.get('target_structure', '')
        metrics['target_overlap'] = float(placement.get('overlap', 0.0))
        metrics['placement_pass'] = bool(
            metrics['target_overlap'] >= QC_THRESHOLDS['min_target_overlap'])

    metrics.update({
        'n_metabolites': int(len(crlb.dropna())),
        'n_metabolites_reliable': int(reliable.sum()),
        'voxel_coverage': coverage,
        'snr_pass': bool(metrics['snr'] >= QC_THRESHOLDS['min_snr']),
        'fwhm_pass': bool(metrics['fwhm_hz'] <= QC_THRESHOLDS['max_fwhm_hz']),
        'coverage_pass': bool(coverage >= QC_THRESHOLDS['min_voxel_coverage']),
    })
    metrics['overall_pass'] = bool(
        metrics['snr_pass'] and metrics['fwhm_pass'] and metrics['coverage_pass']
        and metrics.get('placement_pass', True)
    )

    figures: List[Optional[Path]] = []
    if anat_image and voxel_mask and Path(voxel_mask).exists():
        figures.append(plot_voxel_overlay(
            Path(anat_image), Path(voxel_mask),
            figures_dir / f'{subject}_{session}_voxel-placement.png', subject, session,
        ))
    figures.append(plot_metabolite_crlb(
        crlb, figures_dir / f'{subject}_{session}_crlb.png', subject, session,
    ))

    if mm_areas and mm_envelope and Path(mm_areas).exists() and Path(mm_envelope).exists():
        areas = pd.read_csv(mm_areas)
        figures.append(plot_mm_envelope(
            pd.read_csv(mm_envelope), areas,
            figures_dir / f'{subject}_{session}_mm-envelope.png', subject, session,
        ))
        # Only validated bands reach the metrics; the provisional ones stay in
        # the CSV and the figure, where their flag travels with them.
        for row in areas[~areas['provisional']].itertuples():
            metrics[f'mm_{row.band.lower()}_area'] = float(row.area)
            if np.isfinite(row.area_per_tcr):
                metrics[f'mm_{row.band.lower()}_per_tcr'] = float(row.area_per_tcr)

        # A NaN ratio beside a finite area means quantify_mm refused the
        # creatine reference. That is worth flagging rather than leaving as a
        # silent gap: such sessions fit, report plausible concentrations and
        # pass SNR and linewidth, so nothing else here would catch them.
        metrics['mm_reference_ok'] = bool(areas['area_per_tcr'].notna().any())

        # Lineshape fit, when the export carried the imaginary part. The phase
        # is worth surfacing per session: it is well determined (CV 1-5% across
        # baseline orders) and a session drifting far from the rest points at a
        # timing or phasing problem upstream.
        if mm_lineshape and Path(mm_lineshape).exists():
            line = pd.read_csv(mm_lineshape).iloc[0].to_dict()
            metrics['mm09_phase_deg'] = float(line['phase_deg'])
            metrics['mm09_lineshape_improvement'] = float(line['improvement'])
            if np.isfinite(line.get('area_absorption_per_tcr', float('nan'))):
                metrics['mm09_per_tcr_phase_corrected'] = float(
                    line['area_absorption_per_tcr'])
        if not metrics['mm_reference_ok']:
            logger.warning('%s %s: creatine reference rejected; MM areas are '
                           'reported unscaled', subject, session)

    # The fsl_mrs fit figure is the single most informative panel; copy it in
    # so the report is self-contained.
    fit_summary = fit_dir / 'fit_summary.png'
    if fit_summary.exists():
        import shutil

        copied = figures_dir / f'{subject}_{session}_fit.png'
        figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(fit_summary, copied)
        figures.append(copied)

    # Figures are referenced by name, so the HTML lives beside them.
    report = _html_report(
        subject, session, metrics, metadata, concentrations, crlb,
        [f for f in figures if f is not None], fit_dir,
        figures_dir / f'{subject}_{session}_mrs-qc.html',
    )

    metrics_file = qc_dir / f'{subject}_{session}_mrs-qc.json'
    with open(metrics_file, 'w') as handle:
        json.dump(metrics, handle, indent=2)

    logger.info(
        "MRS QC %s %s: SNR %.1f, FWHM %.1f Hz, %d/%d metabolites within CRLB -> %s",
        subject, session, metrics['snr'], metrics['fwhm_hz'],
        metrics['n_metabolites_reliable'], metrics['n_metabolites'],
        'PASS' if metrics['overall_pass'] else 'REVIEW',
    )

    return {'report': report, 'metrics': metrics, 'metrics_file': metrics_file}
