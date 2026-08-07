"""Single-voxel MR spectroscopy workflow (Bruker PRESS -> FSL-MRS).

The pipeline is:

1. Find the session's water-suppressed PRESS scan and convert it, plus its
   water reference, to NIfTI-MRS
   (:mod:`neurofaune.preprocess.utils.mrs.bruker_mrs`).
2. Optionally measure the GM/WM/CSF content of the voxel from the subject's
   own T2w segmentation
   (:mod:`neurofaune.preprocess.utils.mrs.voxel_geometry`).
3. Run ``fsl_mrs_preproc`` for coil combination, per-shot alignment, bad-average
   rejection, eddy-current correction and averaging.
4. Run ``fsl_mrs`` to fit the basis set and quantify.

Steps 3 and 4 shell out to FSL's own executables, the same pattern the other
neurofaune workflows use for BET/eddy/ANTs. FSL 6.0.7 bundles fsl_mrs 2.4.10
and spec2nii, so no separate conda environment is required; ``spectroscopy.fsl_bin`` in
the config points elsewhere if you want a different build.

Note on ``spec2nii``: it cannot read ParaVision 360.3 spectroscopy, which is
why step 1 uses neurofaune's own reader. See ``bruker_mrs`` for the details.
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from neurofaune.config import get_config_value
from neurofaune.preprocess.utils.mrs.bruker_mrs import (
    BrukerSVS,
    find_press_scans,
    read_bruker_svs,
    select_svs_scan,
    write_nifti_mrs,
)
from neurofaune.preprocess.utils.mrs.voxel_geometry import (
    compute_tissue_fractions,
    write_tissue_fraction_json,
)

logger = logging.getLogger(__name__)

#: Tissue fractions used when no subject segmentation is available. Roughly the
#: composition of a rodent hippocampal voxel; only affects the absolute
#: (water-scaled) concentrations, not the ratios to creatine.
DEFAULT_TISSUE_FRACTIONS = {'GM': 0.45, 'WM': 0.45, 'CSF': 0.10}

#: Metabolite pairs that are unresolvable at clinical field strengths and are
#: conventionally reported summed.
DEFAULT_COMBINE = [
    ['NAA', 'NAAG'],
    ['Cr', 'PCr'],
    ['GPC', 'PCh'],
    ['Glu', 'Gln'],
]

#: Internal concentration reference. Total creatine is the conventional choice.
DEFAULT_INTERNAL_REF = ['Cr', 'PCr']

#: Tried in order when the preferred internal reference fits to zero.
FALLBACK_INTERNAL_REFS = [['Cr', 'PCr'], ['NAA'], ['Ins']]


def find_fsl_binary(name: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Locate an FSL executable.

    Search order: the ``spectroscopy.fsl_bin`` config key, ``$FSLDIR/bin``, then ``PATH``.

    Raises
    ------
    FileNotFoundError
        If the executable cannot be found anywhere.
    """
    candidates: List[Path] = []

    configured = get_config_value(config or {}, 'spectroscopy.fsl_bin', default=None)
    if configured:
        candidates.append(Path(configured) / name)

    fsldir = os.environ.get('FSLDIR')
    if fsldir:
        candidates.append(Path(fsldir) / 'bin' / name)

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    found = shutil.which(name)
    if found:
        return found

    raise FileNotFoundError(
        f"Could not find '{name}'. It ships with FSL 6.0.7+; set $FSLDIR, or "
        f"point 'spectroscopy.fsl_bin' in the config at the directory containing it."
    )


def _run(command: List[str], description: str) -> subprocess.CompletedProcess:
    """Run a command, raising with its output on failure."""
    logger.info("%s: %s", description, ' '.join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {result.returncode}):\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return result


def convert_svs(
    session_dir: Path,
    output_dir: Path,
    subject: str,
    session: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Convert a session's PRESS acquisition to NIfTI-MRS.

    Returns None when the session has no water-suppressed PRESS scan.
    """
    scan_dir = select_svs_scan(session_dir)
    if scan_dir is None:
        logger.warning("%s %s: no water-suppressed PRESS scan", subject, session)
        return None

    prefer_raw = get_config_value(config or {}, 'spectroscopy.prefer_raw', default=True)
    svs = read_bruker_svs(scan_dir, prefer_raw=prefer_raw)
    if svs.source != 'rawdata':
        logger.warning(
            "%s %s: falling back to Bruker's pre-averaged reconstruction; "
            "coil combination, shot alignment and outlier rejection are "
            "unavailable for this session", subject, session,
        )

    outputs = write_nifti_mrs(svs, output_dir, f'{subject}_{session}')
    outputs['svs_data'] = svs
    outputs['scan_dir'] = scan_dir
    return outputs


def measure_tissue_fractions(
    svs: BrukerSVS,
    session_dir: Path,
    anat_scan: str,
    anat_image: Path,
    tissue_maps: Dict[str, Path],
    output_dir: Path,
    subject: str,
    session: str,
) -> Dict[str, Any]:
    """Measure the voxel's GM/WM/CSF content against the T2w segmentation."""
    return compute_tissue_fractions(
        svs=svs,
        anat_scan_dir=Path(session_dir) / anat_scan,
        anat_image=anat_image,
        gm_prob=tissue_maps['GM'],
        wm_prob=tissue_maps['WM'],
        csf_prob=tissue_maps['CSF'],
        mask_output=output_dir / f'{subject}_{session}_svs-mask.nii.gz',
    )


def run_fsl_mrs_preproc(
    metab_file: Path,
    wref_file: Path,
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    report: bool = True,
) -> Dict[str, Path]:
    """Run ``fsl_mrs_preproc``.

    The alignment defaults matter a lot here. With a small rodent voxel and a
    2 s TR, single averages are too noisy for FSL-MRS' per-shot alignment: on
    CPZ test data the default settings gave NAA SNR 6.0, while aligning in
    windows of 32 shots gave 28.7 with a narrower linewidth. So
    ``spectroscopy.align_window`` defaults to 32; set it to 0 to disable alignment
    entirely (SNR 28.5 on the same data, but no drift correction).
    """
    config = config or {}
    align_window = int(get_config_value(config, 'spectroscopy.align_window', default=32))
    remove_outliers = bool(get_config_value(config, 'spectroscopy.remove_outliers', default=True))
    remove_water = bool(get_config_value(config, 'spectroscopy.remove_water', default=False))

    command = [
        find_fsl_binary('fsl_mrs_preproc', config),
        '--data', str(metab_file),
        '--reference', str(wref_file),
        '--output', str(output_dir),
        '--overwrite',
    ]
    if align_window > 0:
        command += ['--align_window', str(align_window)]
    else:
        command.append('--noalign')
    if not remove_outliers:
        command.append('--noremoval')
    if remove_water:
        command.append('--remove-water')
    if report:
        command.append('--report')

    _run(command, 'fsl_mrs_preproc')
    return {'metab': output_dir / 'metab.nii.gz', 'wref': output_dir / 'wref.nii.gz'}


def run_fsl_mrs_fit(
    metab_file: Path,
    wref_file: Path,
    basis: Path,
    output_dir: Path,
    echo_time: float,
    repetition_time: float,
    tissue_fractions: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    report: bool = True,
) -> Dict[str, Any]:
    """Run ``fsl_mrs`` to fit and quantify the preprocessed spectrum.

    Two defaults are set for rodent data rather than left at FSL-MRS':

    ``--free_shift`` is on. Sessions vary by up to ~0.15 ppm in where the
    metabolite peaks land after preprocessing, and with the default tight shift
    prior a displaced NAA simply fits to zero (SNR 0 on a CPZ test session,
    against 16.3 with a free shift).

    Quantification falls back to a second internal reference. ``fsl_mrs``
    aborts outright if the reference metabolite fits to zero, which would throw
    away an otherwise usable session; when Cr+PCr collapses the fit is repeated
    against NAA and the substitution is recorded in the returned dict.

    Returns
    -------
    dict
        ``concentrations`` path and the ``internal_ref`` actually used.
    """
    config = config or {}
    fractions = tissue_fractions or DEFAULT_TISSUE_FRACTIONS
    ppm_low, ppm_high = get_config_value(config, 'spectroscopy.ppmlim', default=[0.2, 4.2])
    baseline = get_config_value(config, 'spectroscopy.baseline', default='poly,4')
    metab_groups = get_config_value(config, 'spectroscopy.metab_groups', default=['NAA'])
    combine = get_config_value(config, 'spectroscopy.combine', default=DEFAULT_COMBINE)
    free_shift = bool(get_config_value(config, 'spectroscopy.free_shift', default=True))
    internal_ref = get_config_value(
        config, 'spectroscopy.internal_ref', default=list(DEFAULT_INTERNAL_REF),
    )

    command = [
        find_fsl_binary('fsl_mrs', config),
        '--data', str(metab_file),
        '--basis', str(basis),
        '--output', str(output_dir),
        '--h2o', str(wref_file),
        # fsl_mrs takes the fractions in WM GM CSF order.
        '--tissue_frac', str(fractions['WM']), str(fractions['GM']), str(fractions['CSF']),
        '--TE', str(echo_time * 1000.0),
        '--TR', str(repetition_time),
        '--baseline', str(baseline),
        '--ppmlim', str(ppm_low), str(ppm_high),
        '--overwrite',
    ]
    if free_shift:
        command.append('--free_shift')
    if metab_groups:
        command += ['--metab_groups'] + [str(g) for g in metab_groups]
    for pair in combine:
        command += ['--combine'] + [str(m) for m in pair]
    if report:
        command.append('--report')

    references = [list(internal_ref)] + [
        ref for ref in FALLBACK_INTERNAL_REFS if ref != list(internal_ref)
    ]
    last_error: Optional[Exception] = None
    for attempt, reference in enumerate(references):
        try:
            _run(command + ['--internal_ref'] + reference, 'fsl_mrs')
        except RuntimeError as exc:
            if 'QuantificationError' not in str(exc):
                raise
            last_error = exc
            logger.warning(
                "fsl_mrs could not quantify against %s (it fitted to zero)%s",
                '+'.join(reference),
                f"; retrying against {'+'.join(references[attempt + 1])}"
                if attempt + 1 < len(references) else '',
            )
            continue
        return {'concentrations': output_dir / 'concentrations.csv',
                'internal_ref': reference}

    raise RuntimeError(
        f"fsl_mrs could not quantify against any of {references}. The spectrum "
        f"is probably unusable; check {output_dir.parent / 'preproc'}."
    ) from last_error


def _tidy_results(
    fit_dir: Path,
    subject: str,
    session: str,
    tissue_fractions: Dict[str, float],
    svs: BrukerSVS,
) -> pd.DataFrame:
    """Flatten the fsl_mrs outputs into one tidy row-per-metabolite table."""
    concentrations = pd.read_csv(fit_dir / 'concentrations.csv', header=[0, 1], index_col=0)
    quality = pd.read_csv(fit_dir / 'qc.csv', index_col=0)

    rows = []
    for metabolite in concentrations.index:
        row: Dict[str, Any] = {
            'subject': subject,
            'session': session,
            'metabolite': metabolite,
            'echo_time_s': svs.echo_time,
            'repetition_time_s': svs.repetition_time,
            'n_averages': svs.n_averages,
            'source': svs.source,
        }
        for scaling in ('raw', 'internal', 'molality', 'molarity'):
            if ('mean', scaling) in concentrations.columns:
                row[scaling] = float(concentrations.loc[metabolite, ('mean', scaling)])
                row[f'{scaling}_sd'] = float(concentrations.loc[metabolite, ('std', scaling)])
        if metabolite in quality.index:
            row['snr'] = float(quality.loc[metabolite, 'SNR'])
            row['fwhm_hz'] = float(quality.loc[metabolite, 'FWHM'])
        row.update({f'frac_{k}': v for k, v in tissue_fractions.items()
                    if k in ('GM', 'WM', 'CSF')})
        rows.append(row)

    return pd.DataFrame(rows)


def run_mrs_preprocessing(
    config: Dict[str, Any],
    subject: str,
    session: str,
    session_dir: Path,
    mrs_root: Path,
    basis: Optional[Path] = None,
    anat_scan: Optional[str] = None,
    anat_image: Optional[Path] = None,
    tissue_maps: Optional[Dict[str, Path]] = None,
    generate_qc: bool = True,
) -> Optional[Dict[str, Any]]:
    """Run the full SVS pipeline for one subject/session.

    Parameters
    ----------
    config : dict
        Loaded neurofaune configuration.
    subject, session : str
        BIDS-style identifiers, e.g. ``sub-CPZ01`` / ``ses-1``.
    session_dir : Path
        Raw Bruker session directory.
    mrs_root : Path
        Root of the spectroscopy output tree, e.g. ``{study}/mrs``. Spectroscopy
        is self-contained rather than living under ``derivatives/`` alongside
        the image modalities, because it never enters the BIDS tree in the
        first place -- spec2nii cannot convert it. Per-session outputs land in
        ``{mrs_root}/{subject}/{session}/`` and QC in
        ``{mrs_root}/qc/{subject}/{session}/``.
    basis : Path, optional
        FSL-MRS basis set directory. Defaults to ``spectroscopy.basis`` in the config.
    anat_scan : str, optional
        Scan number of the anatomical acquisition within ``session_dir``. When
        given together with ``anat_image`` and ``tissue_maps``, the voxel's
        tissue fractions are measured instead of assumed.
    anat_image : Path, optional
        The converted anatomical NIfTI.
    tissue_maps : dict, optional
        ``{'GM': Path, 'WM': Path, 'CSF': Path}`` probability maps.
    generate_qc : bool
        Write the QC report.

    Returns
    -------
    dict or None
        Paths and summary metrics, or None if the session has no SVS scan.
    """
    session_dir = Path(session_dir)
    mrs_root = Path(mrs_root)
    mrs_dir = mrs_root / subject / session
    mrs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MRS preprocessing: %s %s", subject, session)
    logger.info("=" * 70)

    converted = convert_svs(session_dir, mrs_dir, subject, session, config)
    if converted is None:
        return None
    svs: BrukerSVS = converted['svs_data']

    if converted['wref'] is None:
        raise RuntimeError(
            f"{subject} {session}: no water reference found in {converted['scan_dir']}; "
            f"coil combination and water scaling both need one"
        )

    # --- tissue fractions -------------------------------------------------
    fractions: Dict[str, Any] = dict(DEFAULT_TISSUE_FRACTIONS)
    fractions['measured'] = False
    if anat_scan and anat_image and tissue_maps:
        try:
            measured = measure_tissue_fractions(
                svs, session_dir, anat_scan, Path(anat_image), tissue_maps,
                mrs_dir, subject, session,
            )
            measured['measured'] = True
            fractions = measured
            logger.info(
                "Voxel tissue content: GM %.2f WM %.2f CSF %.2f (anatomical coverage %.0f%%)",
                fractions['GM'], fractions['WM'], fractions['CSF'],
                100 * fractions['voxel_volume_ratio'],
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.warning(
                "%s %s: could not measure tissue fractions (%s); using defaults "
                "%s", subject, session, exc, DEFAULT_TISSUE_FRACTIONS,
            )
    else:
        logger.info("No anatomical segmentation supplied; using default tissue fractions")

    write_tissue_fraction_json(fractions, mrs_dir / f'{subject}_{session}_tissue-frac.json')

    # --- preprocess and fit ----------------------------------------------
    preproc_dir = mrs_dir / 'preproc'
    preprocessed = run_fsl_mrs_preproc(
        converted['svs'], converted['wref'], preproc_dir, config,
    )

    basis_path = Path(basis) if basis else Path(get_config_value(config, 'spectroscopy.basis', default=''))
    if not basis_path or not basis_path.exists():
        raise FileNotFoundError(
            f"Basis set not found: {basis_path!s}. Set 'spectroscopy.basis' in the config "
            f"to an FSL-MRS basis directory matching the sequence "
            f"(PRESS, TE {svs.echo_time * 1000:.0f} ms, {svs.spectrometer_frequency:.0f} MHz)."
        )

    fit_dir = mrs_dir / 'fit'
    fit = run_fsl_mrs_fit(
        preprocessed['metab'], preprocessed['wref'], basis_path, fit_dir,
        echo_time=svs.echo_time, repetition_time=svs.repetition_time,
        tissue_fractions=fractions, config=config,
    )

    results = _tidy_results(fit_dir, subject, session, fractions, svs)
    results['internal_ref'] = '+'.join(fit['internal_ref'])
    summary_file = mrs_dir / f'{subject}_{session}_metabolites.csv'
    results.to_csv(summary_file, index=False)

    metadata = {
        'subject': subject,
        'session': session,
        'scan_dir': str(converted['scan_dir']),
        'source': svs.source,
        'n_averages': svs.n_averages,
        'n_coils': svs.n_coils,
        'echo_time_s': svs.echo_time,
        'repetition_time_s': svs.repetition_time,
        'spectrometer_frequency_mhz': svs.spectrometer_frequency,
        'voxel_size_mm': svs.voxel_size.tolist(),
        'voxel_position_mm': svs.voxel_position.tolist(),
        'voxel_volume_ul': float(np.prod(svs.voxel_size)),
        'basis': str(basis_path),
        'internal_ref': fit['internal_ref'],
        'tissue_fractions': {k: v for k, v in fractions.items() if k != 'mask'},
    }
    with open(mrs_dir / f'{subject}_{session}_mrs.json', 'w') as handle:
        json.dump(metadata, handle, indent=2, default=str)

    outputs = {
        'svs': converted['svs'],
        'wref': converted['wref'],
        'preproc_dir': preproc_dir,
        'fit_dir': fit_dir,
        'summary': summary_file,
        'metadata': metadata,
        'results': results,
        'voxel_mask': fractions.get('mask'),
    }

    if generate_qc:
        from neurofaune.preprocess.qc.mrs import generate_mrs_qc_report

        outputs['qc'] = generate_mrs_qc_report(
            subject=subject,
            session=session,
            fit_dir=fit_dir,
            preproc_dir=preproc_dir,
            metadata=metadata,
            qc_dir=mrs_root / 'qc' / subject / session,
            anat_image=Path(anat_image) if anat_image else None,
            voxel_mask=fractions.get('mask'),
        )

    return outputs


def inventory_mrs(session_dir: Path) -> List[Dict[str, Any]]:
    """List the PRESS scans in a session, for study set-up and triage."""
    return find_press_scans(Path(session_dir))
