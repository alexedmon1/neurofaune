"""Post-hoc macromolecule and lipid quantification by upfield area.

Why post hoc
------------
Every attempt to put macromolecules *into* the FSL-MRS fit failed, and failed
the same way (see :mod:`mm_envelope` and ``basis/README.md``): broad MM
components are collinear with the polynomial baseline and with the broad
metabolites, so adding them made the fit less identifiable, not more.
Metabolite estimates were driven toward zero and MM amplitudes diverged.
LCModel avoids that with priors on MM shift, width and concentration ratio that
FSL-MRS cannot express.

This takes the other route. It leaves the metabolite fit untouched -- so it
cannot destabilise anything -- and measures the MM/lipid signal afterwards,
from what the metabolite model did not explain.

The metabolite-free spectrum
----------------------------
``fit`` from FSL-MRS is metabolites *plus* baseline, so::

    metabolite model = fit - baseline
    metabolite-free  = data - (fit - baseline) = residual + baseline

Adding the baseline back is the point, not an oversight: with an MM-free basis,
the polynomial is where the macromolecule signal ends up, and discarding it
would discard most of what is being measured.

Upfield of ~1.8 ppm the metabolite basis is nearly empty -- lactate and alanine
are its only real occupants, and both are in the basis and therefore already
subtracted. What is left there is MM, mobile lipid, and noise. A least-squares
spline separates the broad envelope from the noise.

Why the flank anchor is needed
------------------------------
The metabolite-free spectrum sits on a pedestal of order -0.15 (arbitrary
units) that runs the whole width of the spectrum, present even with the
baseline switched off entirely. Integrating against absolute zero therefore
returned negative areas for the weaker bands and swung with baseline order
(mean CV 34%, several bands negative).

So the areas are measured against a straight line through MM-poor flank windows
either side of the envelope, not against zero. On four sessions across baseline
orders poly,2 through poly,5 that took MM09 from 34% to 7.6% CV, and made every
session's area positive.

The consequence is a stated convention: areas are relative to the 1.55-1.80 ppm
level, so MM17 is zero by construction and is not reported.

A trough that constrains the bands
----------------------------------
Plotting the envelope showed a systematic negative excursion at 0.95-1.10 ppm
in every session, immediately downfield of the MM09 peak. It is in the acquired
data, not introduced here: the raw spectrum averages -0.10 to -0.21 there
(about 3 standard errors below zero) while the fitted polynomial is smooth and
positive across the same window, and the metabolite model has almost no
amplitude there to over-subtract. A zero-order phase error was tested and ruled
out -- the rotation that best flattens the trough varies from 18 to 53 degrees
between sessions and removes only about a quarter of it, where a dispersion
artifact would rotate away almost entirely. The cause is unresolved.

Its practical consequence is that band edges cannot be placed conventionally:
the usual 0.70-1.10 MM09 window integrates straight through the trough and
subtracts real signal. The bands below stop at the zero-crossing instead.

What is measurable here, and what is not
----------------------------------------
On this study's data, tested across baseline orders:

===== ============ ======= ==========================================
band  mean /tCr    CV      verdict
===== ============ ======= ==========================================
MM09  0.56         5.2%    measurable -- better than NAA (9.1%)
MM12  0.14         44%     provisional -- reported, do not rely on it
MM14  -0.04        88%     not measurable; negative in 3 of 4 sessions
===== ============ ======= ==========================================

The number is an *area ratio*, not a concentration: no MM relaxation correction
is applied, and at TE 20 ms an unknown fraction of the MM signal has already
decayed. It also cannot separate macromolecules from instrumental baseline
roll, since the polynomial absorbed both and the anchor removes only the part
that is flat across the region. A metabolite-nulled acquisition remains the
correct solution; this measures the one resonance that is robust without one.

:func:`mm_stability` is the test behind the table, and it needs no reference to
another fitting package: if a band tracks the baseline order rather than the
data, it is reporting the polynomial.
"""

import logging
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Region the envelope is fitted over. The upper limit is the conventional
#: MM/metabolite split; the lower one stays clear of the fit-range edge, where
#: the polynomial baseline is least constrained.
DEFAULT_RANGE: Tuple[float, float] = (0.2, 1.8)

#: MM-poor windows used to define the zero level. Chosen as the widest windows
#: either side of the MM09/MM12 complex that carry little envelope amplitude.
DEFAULT_FLANKS: Tuple[Tuple[float, float], ...] = ((0.25, 0.60), (1.55, 1.80))

#: Bands reported. MM14 and MM17 are deliberately absent: MM17 defines the
#: upper anchor and is zero by construction, and MM14 failed the stability
#: test (CV 88%, negative in most sessions). Adding them back would produce
#: numbers that look like measurements and are not.
MM_BANDS: Dict[str, Tuple[float, float]] = {
    # MM09 / Lip09 methyl -- the reliable one. Stops at 0.95, not the
    # conventional 1.10: the preprocessed spectra carry a real negative trough
    # at 0.95-1.10 ppm (see the module docstring), and integrating through it
    # subtracted signal. Ending at the zero-crossing took the baseline-order CV
    # from 7.6% to 5.2% and the area from 0.40 to 0.56 /tCr.
    'MM09': (0.70, 0.95),
    # MM12 + Lip13 methylene, unresolved from each other -- provisional. Unlike
    # MM09 this is noise-limited rather than trough-limited: narrowing it makes
    # it worse (44% -> 60% -> 88% CV), so the conventional window is kept.
    'MM12': (1.10, 1.40),
}

#: Bands whose stability was not established on validation data. Reported with
#: a flag rather than silently, so they are not mistaken for MM09.
PROVISIONAL_BANDS = frozenset({'MM12'})

#: Spline knot spacing, in ppm. MM lines are ~0.05-0.10 ppm wide at 7 T, so
#: this follows the envelope while averaging many noise points per knot
#: interval. Knots sit on a regular grid rather than being chosen by a
#: smoothing criterion, so the result is deterministic.
DEFAULT_KNOT_SPACING = 0.10

#: Basis names summed to form the creatine reference.
TCR_COMPONENTS: Tuple[str, ...] = ('Cr', 'PCr')


def metabolite_free_spectrum(curves: pd.DataFrame) -> pd.DataFrame:
    """Recover the part of the spectrum the metabolite model did not explain.

    Parameters
    ----------
    curves : DataFrame
        A ``*_fit-curves.csv``: ``ppm, data, fit, baseline, residual``.

    Returns
    -------
    DataFrame
        ``ppm`` and ``signal``, sorted by ascending ppm.

    Raises
    ------
    ValueError
        If the curves are internally inconsistent. ``residual + baseline`` and
        ``data - fit + baseline`` are equal by construction, so a mismatch
        means the file came from the exporter version that wrote time-domain
        arrays into the baseline and residual columns. That data cannot be used
        here, and failing is better than returning a plausible wrong number.
    """
    for column in ('ppm', 'data', 'fit', 'baseline', 'residual'):
        if column not in curves:
            raise KeyError(f'fit curves lack a {column!r} column')

    signal = curves['residual'].to_numpy(float) + curves['baseline'].to_numpy(float)
    alternate = (curves['data'].to_numpy(float) - curves['fit'].to_numpy(float)
                 + curves['baseline'].to_numpy(float))
    scale = float(np.sqrt(np.mean(curves['data'].to_numpy(float) ** 2)))
    if not np.allclose(signal, alternate, atol=1e-6 * max(scale, 1e-12)):
        raise ValueError(
            'residual + baseline != data - fit + baseline; these fit curves are '
            'internally inconsistent (pre-fix export stored time-domain arrays '
            'in the baseline/residual columns). Re-export the fit.')

    out = pd.DataFrame({'ppm': curves['ppm'].to_numpy(float), 'signal': signal})
    return out.sort_values('ppm', ignore_index=True)


def fit_mm_spline(
    ppm: np.ndarray,
    signal: np.ndarray,
    ppm_range: Tuple[float, float] = DEFAULT_RANGE,
    knot_spacing: float = DEFAULT_KNOT_SPACING,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a least-squares cubic spline to the metabolite-free signal.

    Restricted to ``ppm_range``, so the spline describes the upfield MM region
    only and is never free to deform into the metabolite region.

    Returns
    -------
    (ppm, envelope)
        The samples inside the range and the spline evaluated on them.
    """
    from scipy.interpolate import LSQUnivariateSpline

    ppm = np.asarray(ppm, float)
    signal = np.asarray(signal, float)
    order = np.argsort(ppm)
    ppm, signal = ppm[order], signal[order]

    lo, hi = min(ppm_range), max(ppm_range)
    inside = (ppm >= lo) & (ppm <= hi)
    x, y = ppm[inside], signal[inside]
    if x.size < 16:
        raise ValueError(f'only {x.size} points between {lo} and {hi} ppm; '
                         'too few to fit an envelope')

    # Interior knots strictly inside the data range, as LSQUnivariateSpline
    # requires. Count scales with the width of the region, so the envelope has
    # the same flexibility per ppm however the range is set.
    n_interior = max(int(round((x[-1] - x[0]) / knot_spacing)) - 1, 0)
    if n_interior:
        knots = np.linspace(x[0], x[-1], n_interior + 2)[1:-1]
        envelope = LSQUnivariateSpline(x, y, knots, k=3)(x)
    else:
        envelope = np.poly1d(np.polyfit(x, y, 3))(x)
    return x, np.asarray(envelope, float)


def anchor_envelope(
    ppm: np.ndarray,
    envelope: np.ndarray,
    flanks: Sequence[Tuple[float, float]] = DEFAULT_FLANKS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reference the envelope to a straight line through the MM-poor flanks.

    Without this the areas are measured against absolute zero, which the
    spectrum does not provide: a pedestal of order -0.15 runs across the whole
    width, present even with the fit baseline switched off. See the module
    docstring for the measured effect.

    Returns
    -------
    (anchored, line)
        The envelope with the line removed, and the line itself.
    """
    ppm = np.asarray(ppm, float)
    envelope = np.asarray(envelope, float)
    mask = np.zeros(ppm.shape, bool)
    for lo, hi in flanks:
        mask |= (ppm >= min(lo, hi)) & (ppm <= max(lo, hi))
    if mask.sum() < 8:
        raise ValueError(f'only {int(mask.sum())} points in the anchor flanks '
                         f'{list(flanks)}; cannot define a zero level')
    line = np.polyval(np.polyfit(ppm[mask], envelope[mask], 1), ppm)
    return envelope - line, line


def integrate_bands(
    ppm: np.ndarray,
    envelope: np.ndarray,
    bands: Mapping[str, Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Area under the anchored envelope, per band.

    Areas are signed. A negative area is reported rather than clipped: it means
    the envelope fell below its own flank level there, which is information
    about the fit, and clipping it would turn that into a plausible-looking
    positive concentration.
    """
    bands = MM_BANDS if bands is None else bands
    ppm = np.asarray(ppm, float)
    envelope = np.asarray(envelope, float)

    def area(lo: float, hi: float) -> float:
        inside = (ppm >= lo) & (ppm <= hi)
        if inside.sum() < 2:
            return float('nan')
        return float(np.trapezoid(envelope[inside], ppm[inside]))

    return {name: area(lo, hi) for name, (lo, hi) in bands.items()}


def reference_area(
    metabolites: pd.DataFrame,
    components: Sequence[str] = TCR_COMPONENTS,
) -> float:
    """Area of the modelled creatine signal, for scaling the MM areas.

    Taken from the fitted metabolite curves rather than from a reported
    concentration, so numerator and denominator are areas of the same spectrum
    in the same arbitrary units and the ratio is scale-free.
    """
    present = [c for c in components if c in metabolites.columns]
    if not present:
        raise KeyError(f'none of {list(components)} in the metabolite curves; '
                       f'have {list(metabolites.columns)[:12]}')
    ppm = metabolites['ppm'].to_numpy(float)
    total = metabolites[present].to_numpy(float).sum(axis=1)
    order = np.argsort(ppm)
    return float(np.trapezoid(total[order], ppm[order]))


def quantify_mm(
    curves: pd.DataFrame,
    metabolites: Optional[pd.DataFrame] = None,
    ppm_range: Tuple[float, float] = DEFAULT_RANGE,
    knot_spacing: float = DEFAULT_KNOT_SPACING,
    flanks: Sequence[Tuple[float, float]] = DEFAULT_FLANKS,
    bands: Mapping[str, Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Measure the MM/lipid envelope from an exported FSL-MRS fit.

    Parameters
    ----------
    curves : DataFrame
        Contents of ``*_fit-curves.csv``.
    metabolites : DataFrame, optional
        Contents of ``*_fit-metabolites.csv``. Supplied, areas are also
        expressed relative to total creatine, which is what makes them
        comparable between sessions.

    Returns
    -------
    (summary, envelope)
        ``summary``: one row per band with ``band, area, area_per_tcr,
        ppm_low, ppm_high, provisional``.
        ``envelope``: ``ppm, signal, envelope, anchor`` for plotting the spline
        and its zero level over the metabolite-free spectrum.
    """
    free = metabolite_free_spectrum(curves)
    ppm, raw = fit_mm_spline(free['ppm'].to_numpy(), free['signal'].to_numpy(),
                             ppm_range=ppm_range, knot_spacing=knot_spacing)
    envelope, line = anchor_envelope(ppm, raw, flanks)
    areas = integrate_bands(ppm, envelope, bands)

    tcr = None
    if metabolites is not None:
        try:
            tcr = reference_area(metabolites)
        except KeyError as exc:
            logger.warning('no creatine reference for MM scaling: %s', exc)
    if tcr is not None and abs(tcr) < 1e-12:
        logger.warning('creatine reference area is ~0; MM ratios not computed')
        tcr = None

    limits = dict(MM_BANDS if bands is None else bands)
    summary = pd.DataFrame([
        {
            'band': name,
            'area': value,
            'area_per_tcr': value / tcr if tcr else float('nan'),
            'ppm_low': limits[name][0],
            'ppm_high': limits[name][1],
            'provisional': name in PROVISIONAL_BANDS,
        }
        for name, value in areas.items()
    ])

    inside = free['ppm'].isin(ppm)
    detail = pd.DataFrame({
        'ppm': ppm,
        'signal': free.loc[inside, 'signal'].to_numpy(float),
        'envelope': envelope,
        'anchor': line,
    })
    return summary, detail


def mm_stability(estimates: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    """Coefficient of variation of each MM band across baseline orders, in %.

    The test that matters here, and it needs no external reference. The MM area
    is measured partly *from* the polynomial baseline, so the obvious failure
    mode is that it reports the polynomial rather than the tissue -- in which
    case changing the baseline order changes the answer. A band that holds
    across orders is driven by signal the polynomial could not absorb.

    Parameters
    ----------
    estimates : mapping
        ``{baseline_label: {band: value}}`` for one session fitted at several
        baseline orders.
    """
    if not estimates:
        return {}
    bands = set.intersection(*(set(v) for v in estimates.values()))
    out: Dict[str, float] = {}
    for band in sorted(bands):
        values = np.array([estimates[k][band] for k in estimates], float)
        values = values[np.isfinite(values)]
        if values.size < 2 or abs(values.mean()) < 1e-12:
            continue
        out[band] = float(100.0 * values.std(ddof=1) / abs(values.mean()))
    return out
