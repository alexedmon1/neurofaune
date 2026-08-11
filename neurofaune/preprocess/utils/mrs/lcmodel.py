"""LCModel as an alternative fitter to ``fsl_mrs``.

Useful as an independent check rather than a replacement. The basis sets in
this study are natively LCModel ``.basis`` files (the FSL-MRS JSON directories
were converted from them), so the same basis, applied to the same preprocessed
FID by a different fitter, tests the whole chain -- including the Bruker
decoding that had to be reverse-engineered -- rather than just the fit.

LCModel also does its own referencing and phasing internally, which makes it a
second opinion on precisely the step that is fragile in ``fsl_mrs_preproc``.

Interface
---------
LCModel is a Fortran program driven by a namelist control file on stdin, and
reads its data as text ``.RAW`` files. Neither ``spec2nii`` nor ``fsl_mrs``
writes that format, so this module does.
"""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Default install location. The binary is normally at ``$HOME/.lcmodel/bin``
#: but this site keeps it on shared storage.
DEFAULT_LCMODEL_DIRS = (
    Path('/srv/.lcmodel/bin'),
    Path.home() / '.lcmodel' / 'bin',
)

#: Number of values per line in the RAW files written here.
RAW_FORMAT = '(2E15.6)'


def find_lcmodel(explicit: Optional[Path] = None) -> str:
    """Locate the ``lcmodel`` binary.

    Raises
    ------
    FileNotFoundError
        If it cannot be found.
    """
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
        candidates.append(Path(explicit) / 'lcmodel')
    candidates.extend(directory / 'lcmodel' for directory in DEFAULT_LCMODEL_DIRS)

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    found = shutil.which('lcmodel')
    if found:
        return found

    raise FileNotFoundError(
        "Could not find the 'lcmodel' binary. Set 'spectroscopy.lcmodel.bin' in "
        "the config to the directory containing it."
    )


def read_license(path: Optional[Path] = None) -> Tuple[int, str]:
    """Read the LCModel licence key and owner.

    LCModel refuses to run without a matching ``KEY`` and ``OWNER`` in the
    control file.

    Returns
    -------
    (key, owner)
    """
    candidates = [Path(path)] if path else [
        Path('/srv/.lcmodel/license'), Path.home() / '.lcmodel' / 'license',
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        text = candidate.read_text()
        key = re.search(r'key\s*=\s*(\d+)', text, re.IGNORECASE)
        owner = re.search(r"owner\s*=\s*'([^']*)'", text, re.IGNORECASE)
        if key and owner:
            return int(key.group(1)), owner.group(1)

    raise FileNotFoundError(
        "Could not read an LCModel licence (expected 'key = ...' and "
        "\"owner = '...'\"). Set 'spectroscopy.lcmodel.license' in the config."
    )


def write_raw(fid: np.ndarray, output_file: Path, identifier: str,
              scale: float = 1.0, conjugate: bool = True) -> Path:
    """Write a complex FID as an LCModel ``.RAW`` file.

    Parameters
    ----------
    fid : np.ndarray
        1-D complex FID, already preprocessed (coil-combined and averaged).
    output_file : Path
    identifier : str
        Written into the ``ID`` field, for traceability in LCModel's output.
    scale : float
        Multiplies the data. LCModel works in single precision, so very small
        or very large values are worth scaling into range.
    conjugate : bool
        LCModel takes the opposite FID convention to NIfTI-MRS, so data that
        fits correctly in fsl_mrs must be conjugated on the way out. Without
        it LCModel still runs and still produces a table, but the fit is
        meaningless -- on a test session it gave NAA a CRLB of 176% and most
        metabolites 999%, against 3-7% once conjugated. Since a wrong answer
        here looks like a plausible one, this defaults to on.

    Returns
    -------
    Path
    """
    fid = np.asarray(fid).ravel() * scale
    if conjugate:
        fid = np.conjugate(fid)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        ' $NMID',
        f" ID='{identifier[:60]}', FMTDAT='{RAW_FORMAT}'",
        ' TRAMP=1.0, VOLUME=1.0 $END',
    ]
    lines.extend(f'{value.real:15.6E}{value.imag:15.6E}' for value in fid)
    output_file.write_text('\n'.join(lines) + '\n')
    return output_file


def write_control(
    output_file: Path,
    raw_file: Path,
    basis_file: Path,
    output_prefix: Path,
    n_points: int,
    dwelltime: float,
    spectrometer_frequency: float,
    key: int,
    owner: str,
    h2o_file: Optional[Path] = None,
    ppm_range: Tuple[float, float] = (0.2, 4.2),
    title: str = '',
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write an LCModel control file.

    ``PPMEND``/``PPMST`` are LCModel's fit range, and are given as
    (downfield, upfield) -- i.e. ``PPMST`` is the larger number.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    settings: Dict[str, Any] = {
        'KEY(1)': key,
        'OWNER': f"'{owner}'",
        'TITLE': f"'{title[:100]}'",
        'NUNFIL': n_points,
        'DELTAT': f'{dwelltime:.8E}',
        'HZPPPM': f'{spectrometer_frequency:.6f}',
        'FILBAS': f"'{basis_file}'",
        'FILRAW': f"'{raw_file}'",
        'FILPS': f"'{output_prefix}.ps'",
        'LTABLE': 7,
        'FILTAB': f"'{output_prefix}.table'",
        'LCSV': 11,
        'FILCSV': f"'{output_prefix}.csv'",
        'LCOORD': 9,
        'FILCOO': f"'{output_prefix}.coord'",
        'PPMST': f'{max(ppm_range):.3f}',
        'PPMEND': f'{min(ppm_range):.3f}',
    }
    if h2o_file is not None:
        # DOWS turns on water scaling; WCONC/ATTH2O are left at LCModel's
        # defaults, so absolute values are directly comparable with fsl_mrs
        # only after the same tissue correction is applied.
        settings['FILH2O'] = f"'{h2o_file}'"
        settings['DOWS'] = 'T'
        settings['DOECC'] = 'F'  # eddy-current correction already applied
    if extra:
        settings.update(extra)

    lines = [' $LCMODL']
    lines.extend(f' {key_}={value}' for key_, value in settings.items())
    lines.append(' $END')
    output_file.write_text('\n'.join(lines) + '\n')
    return output_file


def parse_table(table_file: Path) -> pd.DataFrame:
    """Parse LCModel's ``.table`` output into a tidy frame.

    The concentration block looks like::

        Conc.  %SD   /Cr+PCr   Metabolite
        0.123  10%   0.456     NAA

    Returns
    -------
    pd.DataFrame
        Columns ``metabolite``, ``concentration``, ``crlb_percent``,
        ``ratio_to_cr``.
    """
    rows = []
    in_block = False
    for line in Path(table_file).read_text().splitlines():
        if 'Conc.' in line and 'Metabolite' in line:
            in_block = True
            continue
        if not in_block:
            continue
        if not line.strip():
            if rows:
                break
            continue
        # e.g. "  1.234E+00   12%    0.567  NAA"  (%SD may be "999%")
        match = re.match(
            r'\s*([\d.E+-]+)\s+(\d+)%\s+([\d.E+-]+)\s+(\S+)\s*$', line)
        if match is None:
            continue
        rows.append({
            'metabolite': match.group(4),
            'concentration': float(match.group(1)),
            'crlb_percent': float(match.group(2)),
            'ratio_to_cr': float(match.group(3)),
        })
    return pd.DataFrame(rows)


def run_lcmodel(
    control_file: Path,
    binary: Optional[str] = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Run LCModel, feeding it the control file on stdin.

    Raises
    ------
    RuntimeError
        If LCModel exits non-zero.
    """
    binary = binary or find_lcmodel()
    with open(control_file) as handle:
        result = subprocess.run(
            [binary], stdin=handle, capture_output=True, text=True, timeout=timeout,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"lcmodel failed (exit {result.returncode}) on {control_file}:\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return result


def read_preprocessed(path: Path) -> Tuple[np.ndarray, float, float]:
    """Read a preprocessed NIfTI-MRS file without importing fsl_mrs.

    Only the FID, dwell time and spectrometer frequency are needed, and those
    live in the NIfTI header plus its JSON header extension -- so nibabel is
    enough, and this stays runnable from neurofaune's own environment.

    Returns
    -------
    (fid, dwelltime, spectrometer_frequency)
    """
    import json

    import nibabel as nib

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    # Collapse any remaining higher dimensions; preprocessed data should
    # already be a single averaged, coil-combined spectrum.
    fid = data.reshape(data.shape[3], -1)
    if fid.shape[1] != 1:
        logger.warning("%s still has %d spectra; averaging them", path, fid.shape[1])
    fid = fid.mean(axis=1)

    dwelltime = float(image.header['pixdim'][4])
    frequency = 0.0
    for extension in image.header.extensions:
        if extension.get_code() != 44:
            continue
        hdr = json.loads(extension.get_content().decode('utf-8'))
        value = hdr.get('SpectrometerFrequency')
        frequency = float(value[0] if isinstance(value, list) else value)
    if not frequency:
        raise ValueError(f"No SpectrometerFrequency in the header extension of {path}")

    return fid, dwelltime, frequency


def fit_with_lcmodel(
    metab_file: Path,
    wref_file: Optional[Path],
    basis_file: Path,
    output_dir: Path,
    identifier: str,
    ppm_range: Tuple[float, float] = (0.2, 4.2),
    binary: Optional[str] = None,
    license_file: Optional[Path] = None,
    extra_control: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit a preprocessed NIfTI-MRS spectrum with LCModel.

    Parameters
    ----------
    metab_file : Path
        Preprocessed water-suppressed NIfTI-MRS, from either preprocessing
        chain -- it must already be coil-combined and averaged.
    wref_file : Path, optional
        Water reference, for water scaling.
    basis_file : Path
        An LCModel ``.basis`` file, not an FSL-MRS JSON directory.
    output_dir : Path
        Where the RAW, control and LCModel outputs are written.
    identifier : str
        Used for LCModel's ``TITLE`` and ``ID`` fields.
    ppm_range : tuple
        Fit range.
    binary, license_file : optional overrides.
    extra_control : dict, optional
        Additional control-file settings, merged last so they win.

    Returns
    -------
    dict
        ``results`` (DataFrame), plus the ``table``, ``csv``, ``control`` and
        ``raw`` paths.

    Raises
    ------
    RuntimeError
        If LCModel fails or produces no table.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fid, dwelltime, frequency = read_preprocessed(Path(metab_file))
    # LCModel is single precision; bring the data into a comfortable range.
    scale = 1.0 / np.abs(fid).max() * 1e4 if np.abs(fid).max() > 0 else 1.0
    raw_file = write_raw(fid, output_dir / 'metab.RAW', identifier, scale=scale)

    h2o_file = None
    if wref_file is not None:
        water, _, _ = read_preprocessed(Path(wref_file))
        h2o_file = write_raw(water, output_dir / 'wref.H2O', f'{identifier}_h2o',
                             scale=scale)

    key, owner = read_license(license_file)
    control_file = write_control(
        output_file=output_dir / 'lcmodel.control',
        raw_file=raw_file,
        basis_file=Path(basis_file),
        output_prefix=output_dir / 'lcmodel',
        n_points=fid.size,
        dwelltime=dwelltime,
        spectrometer_frequency=frequency,
        key=key,
        owner=owner,
        h2o_file=h2o_file,
        ppm_range=ppm_range,
        title=identifier,
        extra=extra_control,
    )

    run_lcmodel(control_file, binary=binary)

    table_file = output_dir / 'lcmodel.table'
    if not table_file.exists():
        raise RuntimeError(
            f"LCModel produced no table for {identifier}; see {output_dir}"
        )
    results = parse_table(table_file)

    return {
        'results': results,
        'table': table_file,
        'csv': output_dir / 'lcmodel.csv',
        'control': control_file,
        'raw': raw_file,
        'scale': scale,
    }
