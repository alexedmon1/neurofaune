"""Bruker single-voxel spectroscopy (PRESS) reader and NIfTI-MRS writer.

Why this exists
---------------
``spec2nii bruker`` cannot read ParaVision 360.3 SVS data. PV-360 no longer
writes the TopSpin-style ``fid`` that ``brukerapi`` expects; the only raw file
is ``rawdata.job0``, and ``brukerapi`` rejects it (``MissingProperty:
numpy_dtype``) because PV-360 also dropped ``GO_raw_data_format`` from
``acqp``. The ``2dseq`` and ``pv2tsdata`` routes fail too. So this module reads
the raw job file directly and writes NIfTI-MRS itself.

Data layout (established empirically and cross-checked against ``ACQ_jobs``,
``GRPDLY`` and Bruker's own reconstruction):

- ``rawdata.job0`` is little-endian ``int32``, real/imaginary interleaved,
  ordered ``(average, receive-channel, complex point)``. The channel axis is
  the inner one: with 2 coils the per-block amplitude alternates by ~20%
  block-to-block while the two halves of the file are indistinguishable.
- Every FID carries the digital-filter group delay ``GRPDLY`` (76.08 points
  here) before the echo top, so the raw FID must be advanced by that much
  before it means anything spectroscopically.
- The water reference lives in ``PVM_RefScan`` in the ``method`` file, shaped
  ``(channels, 2 * points)``, and carries the same group delay.

Keeping coils and averages separate (rather than using Bruker's pre-summed
``pdata/1/fid_proc.64``) is what lets ``fsl_mrs_preproc`` do SVD coil
combination, per-shot frequency/phase alignment and bad-average rejection.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neurofaune.preprocess.utils.mrs.bruker_params import read_jcampdx, read_scan_params

logger = logging.getLogger(__name__)

#: Bruker method string identifying single-voxel PRESS.
PRESS_METHOD = 'Bruker:PRESS'

#: Voxel-size scale factor matching the rest of neurofaune's Bruker handling
#: (rodent sub-mm voxels are scaled 10x for FSL/ANTs compatibility).
DEFAULT_SCALE_FACTOR = 10.0

#: NIfTI-MRS spec version stamped into ``intent_name``.
#:
#: FSL 6.0.7's bundled fsl_mrs gates on ``float(version) < 0.2``, so a file
#: declaring v0.10 or v0.11 is rejected as "NIFTI-MRS > V0.2 required" even
#: though its contents are readable. Declaring v0.9 -- the last version whose
#: string floats above that threshold, and the one FSL itself writes -- keeps
#: the output loadable by both old and new readers. The extension payload is
#: backwards compatible; later spec versions only added optional fields.
NIFTI_MRS_VERSION = (0, 9)

#: Chemical shift FSL-MRS assumes for the 1H carrier when it builds its ppm
#: axis. Bruker references this data to ``PVM_FrqRefPpm`` (4.7) instead, so the
#: converter re-references to match. See :func:`apply_ppm_reference_shift`.
FSL_MRS_REFERENCE_PPM = 4.65

#: Chemical shifts of the two singlets used to reference the spectrum.
TCR_PPM = 3.027
NAA_PPM = 2.008

#: Search windows for those peaks. Deliberately wide: the whole point is to
#: not depend on the spectrum already being well referenced.
TCR_SEARCH = (2.7, 3.4)
NAA_SEARCH = (1.7, 2.3)

#: How far the measured tCr-NAA separation may stray from its true value
#: before metabolite referencing is rejected as unreliable. Across 52 CPZ
#: sessions the measured separation was 1.0212 +/- 0.0010, so this is loose
#: enough to never trip on good data and tight enough to catch a misidentified
#: peak, which would otherwise bake a large error into the reference.
SEPARATION_TOLERANCE = 0.05


@dataclass
class BrukerSVS:
    """A single-voxel spectroscopy acquisition read from a Bruker scan.

    Attributes
    ----------
    metab : np.ndarray
        Water-suppressed data, complex, shaped ``(points, coils, averages)``.
    water_ref : np.ndarray or None
        Water-unsuppressed reference, complex, ``(points, coils, averages)``.
    dwelltime : float
        Spectral dwell time in seconds (``1 / PVM_SpecSWH``).
    spectrometer_frequency : float
        Transmit frequency in MHz.
    echo_time, repetition_time : float
        In seconds.
    nucleus : str
        e.g. ``'1H'``.
    voxel_size : np.ndarray
        Voxel dimensions in mm, unscaled, in the voxel's own frame.
    voxel_position : np.ndarray
        Voxel centre in magnet coordinates (mm).
    voxel_orientation : np.ndarray
        3x3 rotation from the voxel frame to magnet coordinates.
    source : str
        ``'rawdata'`` when read from ``rawdata.job0`` (coils and averages
        preserved) or ``'bruker_averaged'`` for the ``fid_proc.64`` fallback.
    ppm_correction : float
        Chemical-shift correction applied when re-referencing, in ppm.
    """

    metab: np.ndarray
    water_ref: Optional[np.ndarray]
    dwelltime: float
    spectrometer_frequency: float
    echo_time: float
    repetition_time: float
    nucleus: str
    voxel_size: np.ndarray
    voxel_position: np.ndarray
    voxel_orientation: np.ndarray
    source: str
    ppm_correction: float = 0.0
    scan_dir: Optional[Path] = None
    params: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n_points(self) -> int:
        return int(self.metab.shape[0])

    @property
    def n_coils(self) -> int:
        return int(self.metab.shape[1])

    @property
    def n_averages(self) -> int:
        return int(self.metab.shape[2])


# ---------------------------------------------------------------------------
# Scan discovery
# ---------------------------------------------------------------------------

def find_press_scans(session_dir: Path) -> List[Dict[str, Any]]:
    """List every ``Bruker:PRESS`` scan in a Bruker session directory.

    Parameters
    ----------
    session_dir : Path
        Raw Bruker session directory (the one holding numbered scan folders).

    Returns
    -------
    list of dict
        One record per PRESS scan with keys ``scan_dir``, ``scan_number``,
        ``protocol``, ``water_suppressed``, ``n_averages``, ``echo_time``,
        ``repetition_time`` and ``voxel_size``, sorted by scan number.
    """
    session_dir = Path(session_dir)
    records: List[Dict[str, Any]] = []

    for scan_dir in sorted(session_dir.iterdir()):
        if not scan_dir.is_dir() or not scan_dir.name.isdigit():
            continue
        method_file = scan_dir / 'method'
        if not method_file.exists():
            continue
        try:
            params = read_scan_params(scan_dir)
        except Exception as exc:  # unreadable scan should not kill discovery
            logger.warning("Could not read parameters for %s: %s", scan_dir, exc)
            continue
        if params.get('Method') != PRESS_METHOD:
            continue

        voxel_size = np.atleast_2d(np.asarray(params.get('PVM_VoxArrSize', [])))
        records.append({
            'scan_dir': scan_dir,
            'scan_number': int(scan_dir.name),
            # An aborted acquisition leaves its parameter files behind with no
            # data, so having a method file is not enough to be usable.
            'has_data': ((scan_dir / 'rawdata.job0').exists()
                         or (scan_dir / 'pdata' / '1' / 'fid_proc.64').exists()),
            'protocol': str(params.get('ACQ_scan_name', params.get('ACQ_protocol_name', ''))),
            'water_suppressed': str(params.get('PVM_WsOnOff', 'Off')).lower() == 'on',
            'suppression_mode': str(params.get('PVM_WsMode', 'NO_SUPPRESSION')),
            'n_averages': int(params.get('PVM_NAverages', 1)),
            'echo_time': float(params.get('PVM_EchoTime', 0.0)),
            'repetition_time': float(params.get('PVM_RepetitionTime', 0.0)),
            'voxel_size': voxel_size[0].tolist() if voxel_size.size else [],
        })

    records.sort(key=lambda r: r['scan_number'])
    return records


def select_svs_scan(session_dir: Path) -> Optional[Path]:
    """Pick the real SVS acquisition from a session's PRESS scans.

    A session typically holds several PRESS scans: unsuppressed single-average
    shim/prescan runs (``PRESS_ShimEPI``, ``PRESS_ShimHippo``) plus the actual
    water-suppressed acquisition. Selection therefore requires water
    suppression to be on, and breaks ties on the average count.

    Scans that hold no data are passed over: an acquisition aborted at the
    console still leaves its parameter files on disk, and picking one of those
    over an earlier complete scan would lose the session.

    Returns
    -------
    Path or None
        The chosen scan directory, or None when the session has no usable
        water-suppressed PRESS scan.
    """
    suppressed = [r for r in find_press_scans(session_dir) if r['water_suppressed']]
    candidates = [r for r in suppressed if r['has_data']]
    if not candidates:
        if suppressed:
            logger.warning(
                "%s: %d water-suppressed PRESS scan(s) but none hold data "
                "(aborted acquisition?)", session_dir, len(suppressed),
            )
        return None
    best = max(candidates, key=lambda r: (r['n_averages'], r['scan_number']))
    return best['scan_dir']


# ---------------------------------------------------------------------------
# Raw FID handling
# ---------------------------------------------------------------------------

def _read_group_delay(scan_dir: Path) -> float:
    """Return the digital-filter group delay in points (0.0 if not recorded).

    ``GRPDLY`` in ``acqp`` is ``-1`` (unset) on PV-360; the meaningful value is
    in the TopSpin-style ``acqus``/``acqu`` written alongside it.
    """
    for name in ('acqus', 'acqu'):
        candidate = scan_dir / name
        if not candidate.exists():
            continue
        try:
            value = float(read_jcampdx(candidate).get('GRPDLY', -1))
        except Exception:
            continue
        if value > 0:
            return value
    return 0.0


def _echo_top_index(fid: np.ndarray, search_limit: int) -> int:
    """Index of the echo top: the sample where |FID| peaks after the filter ramp."""
    profile = np.abs(fid.reshape(fid.shape[0], -1)).mean(axis=1)
    return int(profile[:search_limit].argmax())


def resolve_group_delay(fid: np.ndarray, group_delay: float) -> float:
    """Resolve the true delay in points from ``GRPDLY`` plus the data itself.

    ``GRPDLY`` gives the fractional part reliably but, on PV-360, its integer
    part is one point short: with ``GRPDLY = 76.083`` the echo top sits at
    sample 77, and advancing by 76.083 leaves it at sample 1. A residual
    one-point offset is a first-order phase across the whole spectrum, which
    shows up as dispersive (derivative-shaped) peaks in the fit.

    So take the integer part from where the echo top actually is and the
    fractional part from ``GRPDLY``. That is self-correcting across ParaVision
    versions rather than encoding one version's off-by-one.
    """
    if group_delay <= 0:
        return 0.0
    search_limit = min(fid.shape[0], int(2 * group_delay) + 16)
    integer_part = _echo_top_index(fid, search_limit)
    return integer_part + (group_delay - np.floor(group_delay))


def remove_group_delay(
    fid: np.ndarray,
    group_delay: float,
    resolve: bool = True,
) -> np.ndarray:
    """Advance FIDs by a (possibly fractional) digital-filter group delay.

    The shift is applied as a linear phase ramp in the frequency domain so the
    fractional part is handled exactly; the trailing points, into which the
    filter ramp wraps, are then discarded.

    Parameters
    ----------
    fid : np.ndarray
        Complex FIDs with the point axis first.
    group_delay : float
        Delay in points, as reported by ``GRPDLY``.
    resolve : bool
        Re-derive the integer part from the data -- see
        :func:`resolve_group_delay`. Pass False when ``group_delay`` has
        already been resolved, so that several arrays can be corrected by
        exactly the same amount and stay the same length.

    Returns
    -------
    np.ndarray
        FIDs shortened by the applied delay, rounded up.
    """
    delay = resolve_group_delay(fid, group_delay) if resolve else group_delay
    if delay <= 0:
        return fid

    n_points = fid.shape[0]
    freq = np.fft.fftfreq(n_points)
    ramp = np.exp(2j * np.pi * freq * delay)
    shape = [n_points] + [1] * (fid.ndim - 1)
    shifted = np.fft.ifft(np.fft.fft(fid, axis=0) * ramp.reshape(shape), axis=0)
    return shifted[: n_points - int(np.ceil(delay))]


def read_voxel_geometry(params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Voxel rotation and centre in magnet coordinates.

    ``PVM_VoxArrPosition`` is NOT in magnet coordinates: it is the centre
    expressed in the voxel's own rotated frame. Bruker states the magnet-frame
    position in the geometry object ``PVM_VoxelGeoCub`` instead, and the two
    are related by the voxel rotation -- ``R @ geocub_position`` reproduces
    ``PVM_VoxArrPosition`` exactly in all 52 cuprizone sessions.

    Using ``PVM_VoxArrPosition`` directly therefore misplaces the voxel by an
    amount proportional to how far it was rotated: nothing at 0 degrees, but
    1.7 mm at the 12 degrees of the most angled session here. That is why the
    error showed up first on the sessions whose voxel was tilted to follow the
    hippocampus.

    Returns
    -------
    (rotation, position)
        ``rotation`` rows are the voxel's own axes in magnet coordinates, in
        the same order as ``PVM_VoxArrSize``; ``position`` is the centre in mm.

    Raises
    ------
    KeyError
        If ``PVM_VoxelGeoCub`` is absent -- there is no safe fallback, since
        the alternative field is in the wrong frame.
    """
    text = params.get('PVM_VoxelGeoCub')
    if not text:
        raise KeyError(
            "PVM_VoxelGeoCub is missing; PVM_VoxArrPosition cannot be used in "
            "its place because it is expressed in the voxel's rotated frame"
        )
    values = [float(v) for v in
              re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', str(text).replace('\n', ' '))]
    if len(values) < 12:
        raise ValueError(f"Could not parse a rotation and position from "
                         f"PVM_VoxelGeoCub: {str(text)[:120]!r}")
    rotation = np.array(values[:9], dtype=float).reshape(3, 3)
    position = np.array(values[9:12], dtype=float)
    return rotation, position


def _deinterleave(raw: np.ndarray) -> np.ndarray:
    """Interleaved real/imaginary samples to complex."""
    return raw[0::2] + 1j * raw[1::2]


def _frequency_shift(fid: np.ndarray, delta_ppm: float,
                     spectrometer_frequency: float, dwelltime: float) -> np.ndarray:
    """Move every peak by ``delta_ppm`` along the chemical-shift axis."""
    if delta_ppm == 0.0:
        return fid
    # ppm axis convention, verified against Bruker's own reconstruction (NAA at
    # 2.01, tCr at 3.03): ppm = fftshift(fftfreq) / f0 + FSL_MRS_REFERENCE_PPM.
    # Under it, moving a peak up in ppm means moving it up in frequency.
    delta_hz = delta_ppm * spectrometer_frequency
    time = np.arange(fid.shape[0]) * dwelltime
    ramp = np.exp(2j * np.pi * delta_hz * time)
    return fid * ramp.reshape([-1] + [1] * (fid.ndim - 1))


def measure_water_ppm_offset(
    water_ref: np.ndarray,
    spectrometer_frequency: float,
    dwelltime: float,
    zero_fill: int = 8,
) -> float:
    """Chemical-shift offset of the water peak from the carrier, in ppm.

    The unsuppressed reference is a single dominant peak, so its position is
    unambiguous -- far more reliable than any parameter-derived assumption
    about where the carrier sits.
    """
    fid = water_ref.reshape(water_ref.shape[0], -1)
    # Combine coils by their first-point phase before locating the peak;
    # summing them raw would let them cancel.
    weights = np.conj(fid[0]) / np.abs(fid[0]).sum()
    combined = (fid * weights).sum(axis=1)

    n_points = combined.size * zero_fill
    spectrum = np.fft.fftshift(np.fft.fft(combined, n_points))
    frequency = np.fft.fftshift(np.fft.fftfreq(n_points, dwelltime))
    peak_hz = frequency[np.abs(spectrum).argmax()]
    return peak_hz / spectrometer_frequency


def _coil_combined_average(metab: np.ndarray, water_ref: Optional[np.ndarray]) -> np.ndarray:
    """A single averaged FID, good enough to locate peaks.

    Coils are combined on the water reference's first-point phase where one is
    available; summing them raw would let them cancel.
    """
    flat = metab.reshape(metab.shape[0], metab.shape[1], -1)
    if water_ref is not None:
        first = water_ref.reshape(water_ref.shape[0], water_ref.shape[1], -1)[0, :, 0]
    else:
        first = flat[0, :, 0]
    weights = np.conj(first) / np.abs(first).sum()
    return (flat * weights[None, :, None]).sum(axis=1).mean(axis=1)


def measure_metabolite_offset(
    metab: np.ndarray,
    water_ref: Optional[np.ndarray],
    spectrometer_frequency: float,
    dwelltime: float,
    zero_fill: int = 4,
    line_broadening: float = 4.0,
) -> Optional[float]:
    """Offset of total creatine from :data:`TCR_PPM`, in ppm.

    Referencing on a metabolite rather than on water is what actually matters
    downstream: ``fsl_mrs_preproc`` shifts and phases the spectrum on whatever
    is strongest in a hardcoded 2.9-3.1 ppm window, so tCr needs to be near
    3.027 before it runs, with margin on both sides.

    Picking a peak out of a wide window could bake in a large error if it
    picked the wrong one, so the result is cross-checked against NAA: the two
    singlets are a fixed 1.019 ppm apart. If the separation is wrong the peaks
    were misidentified and None is returned, leaving the caller on the
    water-based reference.

    Returns
    -------
    float or None
        ppm to add to move tCr onto :data:`TCR_PPM`, or None if the
        cross-check failed.
    """
    fid = _coil_combined_average(metab, water_ref)
    decay = np.exp(-np.arange(fid.size) * line_broadening * np.pi * dwelltime)
    n_points = fid.size * zero_fill
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(fid * decay, n_points)))
    ppm = (np.fft.fftshift(np.fft.fftfreq(n_points, dwelltime)) / spectrometer_frequency
           + FSL_MRS_REFERENCE_PPM)

    def peak(low: float, high: float) -> float:
        window = (ppm > low) & (ppm < high)
        return float(ppm[window][spectrum[window].argmax()])

    tcr, naa = peak(*TCR_SEARCH), peak(*NAA_SEARCH)
    if abs((tcr - naa) - (TCR_PPM - NAA_PPM)) > SEPARATION_TOLERANCE:
        logger.warning(
            "tCr/NAA separation %.3f ppm is not the expected %.3f; the peaks "
            "were probably misidentified, so metabolite referencing is skipped",
            tcr - naa, TCR_PPM - NAA_PPM,
        )
        return None
    return TCR_PPM - tcr


def apply_ppm_reference_shift(
    metab: np.ndarray,
    water_ref: Optional[np.ndarray],
    water_ppm: float,
    reference_ppm: float,
    spectrometer_frequency: float,
    dwelltime: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Re-reference the spectrum so water lands at its true chemical shift.

    Why this is needed, and why it is measured rather than assumed:
    ``fsl_mrs_preproc`` searches a hardcoded 2.9-3.1 ppm window for total
    creatine and shifts and phases the whole spectrum on whatever it finds
    there. Choline sits at 3.20 and tCr at 3.03, so the window only holds the
    intended peak if the ppm axis is right to within about 0.07 ppm. Bruker
    references the carrier to ``PVM_FrqRefPpm`` (4.7) while FSL-MRS assumes
    4.65, and sessions drift on top of that -- enough that the search
    intermittently locks onto choline instead, which displaces the spectrum by
    ~0.15 ppm and phases it upside down. Testing showed this is bistable: a
    fixed offset in either direction simply moves which sessions fail.

    Measuring the water peak from the unsuppressed reference removes the
    guesswork and centres tCr in the window with margin on both sides.

    Parameters
    ----------
    water_ppm : float
        True chemical shift of water, from ``PVM_FrqRefPpm`` (4.7).
    reference_ppm : float
        The shift FSL-MRS assigns to the carrier when it builds its ppm axis
        (4.65), which is what makes a correction necessary at all.

    Returns
    -------
    (metab, water_ref, applied_ppm)
        The shifted data and the correction that was applied.
    """
    if water_ref is None:
        return metab, water_ref, 0.0

    # FSL-MRS labels the carrier reference_ppm, so a peak at true shift d
    # currently reads (d - water_ppm + reference_ppm); correct that back, and
    # take out the session's measured water offset at the same time.
    offset = measure_water_ppm_offset(water_ref, spectrometer_frequency, dwelltime)
    correction = (water_ppm - reference_ppm) - offset
    return (
        _frequency_shift(metab, correction, spectrometer_frequency, dwelltime),
        _frequency_shift(water_ref, correction, spectrometer_frequency, dwelltime),
        correction,
    )


def _read_rawdata(scan_dir: Path, params: Dict[str, Any]) -> Optional[np.ndarray]:
    """Read ``rawdata.job0`` as ``(points, coils, averages)`` complex.

    Returns None (rather than raising) when the file is absent or its size
    doesn't match the parameters, so the caller can fall back to Bruker's own
    reconstruction.
    """
    raw_file = scan_dir / 'rawdata.job0'
    if not raw_file.exists():
        return None

    n_points = int(np.atleast_1d(params['PVM_SpecMatrix'])[0])
    n_coils = int(params.get('PVM_EncNReceivers', 1))
    n_averages = int(params.get('PVM_NAverages', 1))
    expected = n_points * n_coils * n_averages * 2

    raw = np.fromfile(raw_file, dtype='<i4').astype(np.float64)
    if raw.size != expected:
        logger.warning(
            "%s holds %d int32 values but parameters imply %d "
            "(points=%d coils=%d averages=%d); falling back to Bruker's "
            "reconstruction", raw_file, raw.size, expected,
            n_points, n_coils, n_averages,
        )
        return None

    # (average, coil, point) on disk -> (point, coil, average)
    data = _deinterleave(raw).reshape(n_averages, n_coils, n_points)
    return np.transpose(data, (2, 1, 0))


def _read_reference(scan_dir: Path, params: Dict[str, Any]) -> Optional[np.ndarray]:
    """Read the water reference as ``(points, coils, 1)`` complex.

    Prefers ``PVM_RefScan`` from the ``method`` file, which keeps the receive
    channels separate; falls back to the coil-combined ``fid_refscan.64``.
    """
    ref = params.get('PVM_RefScan')
    if isinstance(ref, np.ndarray) and ref.ndim == 2 and ref.shape[1] % 2 == 0:
        complex_ref = _deinterleave(ref.T)          # (points, coils)
        return complex_ref[:, :, np.newaxis]

    fallback = scan_dir / 'pdata' / '1' / 'fid_refscan.64'
    if fallback.exists():
        data = _deinterleave(np.fromfile(fallback, dtype='<f8'))
        return data[:, np.newaxis, np.newaxis]

    return None


def _read_bruker_averaged(scan_dir: Path) -> Optional[np.ndarray]:
    """Read Bruker's own reconstruction as ``(points, 1, 1)`` complex."""
    proc = scan_dir / 'pdata' / '1' / 'fid_proc.64'
    if not proc.exists():
        return None
    data = _deinterleave(np.fromfile(proc, dtype='<f8'))
    return data[:, np.newaxis, np.newaxis]


def read_bruker_svs(
    scan_dir: Path,
    prefer_raw: bool = True,
    reference_ppm: Optional[float] = FSL_MRS_REFERENCE_PPM,
) -> BrukerSVS:
    """Read a Bruker PRESS scan into a :class:`BrukerSVS`.

    Parameters
    ----------
    scan_dir : Path
        A single numbered Bruker scan directory.
    prefer_raw : bool
        Read ``rawdata.job0`` (coils and averages preserved) when available.
        Set False to force Bruker's pre-averaged reconstruction.
    reference_ppm : float, optional
        Re-reference the spectrum so the carrier corresponds to this chemical
        shift, matching the downstream tool's assumption. See
        :func:`apply_ppm_reference_shift`. Pass None to leave the Bruker
        referencing untouched.

    Returns
    -------
    BrukerSVS

    Raises
    ------
    ValueError
        If the scan is not PRESS, or no usable FID data can be read.
    """
    scan_dir = Path(scan_dir)
    params = read_scan_params(scan_dir)

    if params.get('Method') != PRESS_METHOD:
        raise ValueError(f"{scan_dir} is not a {PRESS_METHOD} scan "
                         f"(Method={params.get('Method')!r})")

    metab = _read_rawdata(scan_dir, params) if prefer_raw else None
    source = 'rawdata'
    if metab is None:
        metab = _read_bruker_averaged(scan_dir)
        source = 'bruker_averaged'
    if metab is None:
        raise ValueError(
            f"No usable FID data in {scan_dir}: neither rawdata.job0 nor "
            f"pdata/1/fid_proc.64 could be read"
        )

    water_ref = _read_reference(scan_dir, params)

    # Resolve the delay once and apply it to both arrays. It is a property of
    # the receive filter, so it is the same for the metabolite and reference
    # FIDs -- and resolving each separately can round to different integers,
    # leaving the two arrays one point apart in length, which later breaks
    # eddy-current correction with a broadcast error.
    group_delay = _read_group_delay(scan_dir)
    delay = resolve_group_delay(water_ref if water_ref is not None else metab, group_delay)
    metab = remove_group_delay(metab, delay, resolve=False)
    if water_ref is not None:
        water_ref = remove_group_delay(water_ref, delay, resolve=False)
        # Coil counts must agree for FSL-MRS to use the reference in coil
        # combination; drop to the common count if Bruker stored fewer.
        if water_ref.shape[1] != metab.shape[1] and water_ref.shape[1] == 1:
            water_ref = np.repeat(water_ref, metab.shape[1], axis=1)

    dwelltime = 1.0 / float(np.atleast_1d(params['PVM_SpecSWH'])[0])
    spectrometer_frequency = float(
        params.get('SFO1', np.atleast_1d(params['PVM_FrqWork'])[0])
    )

    ppm_correction = 0.0
    if reference_ppm is not None:
        # Coarse: put water at its true shift, from the unsuppressed reference.
        water_ppm = float(np.atleast_1d(params.get('PVM_FrqRefPpm', reference_ppm))[0])
        metab, water_ref, ppm_correction = apply_ppm_reference_shift(
            metab, water_ref, water_ppm, reference_ppm,
            spectrometer_frequency, dwelltime,
        )
        # Fine: land tCr on 3.027. Water referencing alone left a systematic
        # -0.088 ppm residual across 52 CPZ sessions, which put the worst
        # session's tCr 0.019 ppm from falling out of fsl_mrs_preproc's
        # 2.9-3.1 search window altogether.
        refinement = measure_metabolite_offset(
            metab, water_ref, spectrometer_frequency, dwelltime,
        )
        if refinement is not None:
            metab = _frequency_shift(metab, refinement, spectrometer_frequency, dwelltime)
            if water_ref is not None:
                water_ref = _frequency_shift(
                    water_ref, refinement, spectrometer_frequency, dwelltime)
            ppm_correction += refinement
        logger.debug("%s: applied %+.3f ppm reference correction", scan_dir, ppm_correction)

    voxel_size = np.atleast_2d(np.asarray(params['PVM_VoxArrSize'], dtype=float))[0]
    orientation, voxel_position = read_voxel_geometry(params)

    return BrukerSVS(
        metab=metab,
        water_ref=water_ref,
        dwelltime=dwelltime,
        spectrometer_frequency=spectrometer_frequency,
        echo_time=float(params['PVM_EchoTime']) / 1000.0,
        repetition_time=float(params['PVM_RepetitionTime']) / 1000.0,
        nucleus=str(params.get('PVM_Nucleus1', '1H')),
        voxel_size=voxel_size,
        voxel_position=voxel_position,
        voxel_orientation=orientation,
        source=source,
        ppm_correction=ppm_correction,
        scan_dir=scan_dir,
        params=params,
    )


# ---------------------------------------------------------------------------
# NIfTI-MRS output
# ---------------------------------------------------------------------------

def build_affine(svs: BrukerSVS, scale_factor: float = DEFAULT_SCALE_FACTOR) -> np.ndarray:
    """Build the NIfTI affine for the SVS voxel.

    Voxel dimensions and position are scaled by ``scale_factor`` to match the
    10x convention neurofaune applies to every other Bruker-derived image.
    """
    rotation = np.asarray(svs.voxel_orientation, dtype=float)
    affine = np.eye(4)
    affine[:3, :3] = rotation.T * (svs.voxel_size * scale_factor)
    affine[:3, 3] = svs.voxel_position * scale_factor
    return affine


#: Tags for the 5th and 6th NIfTI-MRS axes. Both the metabolite and reference
#: arrays are written as (points, coils, averages), keeping singleton axes so
#: fsl_mrs_preproc sees the same layout regardless of the reader path taken.
_DIM_TAGS = ('DIM_COIL', 'DIM_DYN')


def _build_hdr_ext(svs: BrukerSVS, n_dims: int, water_suppressed: bool):
    """Populate a NIfTI-MRS header extension for this acquisition."""
    from nifti_mrs.definitions import standard_defined
    from nifti_mrs.hdr_ext import Hdr_Ext

    hdr_ext = Hdr_Ext(svs.spectrometer_frequency, svs.nucleus, dimensions=n_dims)

    for dim_index, tag in enumerate(_DIM_TAGS):
        if 5 + dim_index <= n_dims:
            hdr_ext.set_dim_info(dim_index, tag)

    standard = {
        'EchoTime': svs.echo_time,
        'RepetitionTime': svs.repetition_time,
        'SpectralWidth': 1.0 / svs.dwelltime,
        'Manufacturer': 'Bruker',
        'WaterSuppressed': water_suppressed,
        'ConversionMethod': 'neurofaune.preprocess.utils.mrs.bruker_mrs',
    }
    for key, value in standard.items():
        if key in standard_defined:
            hdr_ext.set_standard_def(key, value)

    if str(svs.params.get('PVM_WsMode', '')) and water_suppressed:
        hdr_ext.set_standard_def('WaterSuppressionType', str(svs.params['PVM_WsMode']))

    hdr_ext.set_user_def(
        key='BrukerSource', value=svs.source,
        doc='neurofaune reader path: rawdata.job0 or Bruker-averaged fid_proc.64',
    )
    return hdr_ext


def write_nifti_mrs(
    svs: BrukerSVS,
    output_dir: Path,
    prefix: str,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    conjugate: bool = False,
) -> Dict[str, Path]:
    """Write a :class:`BrukerSVS` out as NIfTI-MRS files.

    Parameters
    ----------
    svs : BrukerSVS
        Data to write.
    output_dir : Path
        Directory to write into (created if needed).
    prefix : str
        Filename stem; ``_svs.nii.gz`` and ``_wref.nii.gz`` are appended.
    scale_factor : float
        Geometry scaling, see :func:`build_affine`.
    conjugate : bool
        Conjugate the FIDs before writing. Bruker's spectral sense already
        matches the NIfTI-MRS convention for this data, so this is off by
        default; it exists as an escape hatch for other sequences.

    Returns
    -------
    dict
        ``{'svs': Path, 'wref': Path or None}``.
    """
    from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    affine = build_affine(svs, scale_factor)

    def _write(data: np.ndarray, suffix: str, water_suppressed: bool) -> Path:
        if conjugate:
            data = np.conjugate(data)
        # NIfTI-MRS is (x, y, z, t, ...) with a singleton spatial voxel.
        payload = np.ascontiguousarray(data.reshape((1, 1, 1) + data.shape))
        hdr_ext = _build_hdr_ext(svs, payload.ndim, water_suppressed)
        image = gen_nifti_mrs_hdr_ext(
            payload, svs.dwelltime, hdr_ext, affine=affine, no_conj=True,
        )
        image.set_version_info(*NIFTI_MRS_VERSION)
        path = output_dir / f'{prefix}_{suffix}.nii.gz'
        image.save(path)
        return path

    outputs: Dict[str, Any] = {'svs': _write(svs.metab, 'svs', True)}
    outputs['wref'] = (
        _write(svs.water_ref, 'wref', False) if svs.water_ref is not None else None
    )
    return outputs


def convert_session(
    session_dir: Path,
    output_dir: Path,
    prefix: str,
    prefer_raw: bool = True,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
) -> Optional[Dict[str, Any]]:
    """Find, read and convert a session's SVS acquisition in one call.

    Returns None when the session contains no water-suppressed PRESS scan.
    """
    scan_dir = select_svs_scan(session_dir)
    if scan_dir is None:
        logger.warning("No water-suppressed PRESS scan found in %s", session_dir)
        return None

    svs = read_bruker_svs(scan_dir, prefer_raw=prefer_raw)
    outputs = write_nifti_mrs(svs, output_dir, prefix, scale_factor=scale_factor)
    outputs['svs_data'] = svs
    outputs['scan_dir'] = scan_dir
    return outputs
