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
level, which suppresses MM17 by construction.

That raised the obvious objection -- that MM14 and MM17 might be unreportable
because of where the anchor sits rather than because of the data. Four zero
references were compared on 87 sessions (both flanks sloped, both flat, upfield
flank only sloped, upfield flank only flat). Dropping the upper flank does free
MM17 to be non-zero, and it is still not measurable: negative in 26% of
sessions with a robust CV of 184%. MM14 is negative in 39-86% of sessions under
every variant. The objection was worth testing and the verdict survives it.

MM09 is essentially indifferent to the choice (median 0.566-0.577, never
negative) except under the upfield-sloped variant, which extrapolates a slope
across 1.2 ppm from a single 0.35 ppm window and degrades it as expected. The
default is kept because it gives MM09 the lowest between-session scatter and
MM12 its only tolerable one.

The trough, and what it is
--------------------------
Plotting the envelope showed a systematic negative excursion at 0.95-1.10 ppm
in every session, immediately downfield of the MM09 peak, deep enough that the
conventional 0.70-1.10 MM09 window integrates through it and subtracts real
signal. That is why the band below stops at 0.95, the zero-crossing.

It is **the dispersion lobe of the MM09 resonance itself**. Fitting the
metabolite-free spectrum over 0.60-1.35 ppm with a single complex Lorentzian of
free phase, plus a linear baseline, explains the positive peak and the negative
trough together 12-28% better (RMS) than the same model forced to pure
absorption. The fitted angle is negative in all 16 sessions tested, spanning
-37 to -71 degrees with a median near -59.

Three things it is not, each tested rather than assumed:

* Not metabolite over-subtraction. The metabolite model peaks at 0.23 there
  against a -0.76 trough, and the excursion is present in the raw data.
* Not eddy-current correction. Running the production chain with ``--no-ecc``
  leaves the trough in place, and makes one of three sessions twice as bad.
* Not a global zero-order phase. Rotating the whole spectrum trades the trough
  against the peak -- MM09 falls 20-46% for a flat trough -- because the offset
  is *differential*, between the MM and the metabolites, not global.

The mechanism is consistent with the first FID points. Macromolecule T2 is very
short, so MM signal lives in the first few samples while metabolites persist;
any residual sub-sample group delay or first-point error therefore imprints a
different effective phase on broad components than on narrow ones. The
converter's fractional group-delay correction is the obvious suspect, and that
remains a hypothesis -- the *observation* is what the 16-session fit supports.

The practical consequence is that these areas are absorption-mode projections
of a signal sitting about 59 degrees out of phase, so they capture roughly
cos(59 deg) ~ 0.5 of the MM09 amplitude. That is a systematic scale factor
shared by every session, so it biases the absolute value while leaving
between-session comparisons intact.

What is measurable here, and what is not
----------------------------------------
Measured on all 92 sessions of the full study, with baseline-order stability
from 4 sessions fitted at poly,2 through poly,5:

===== ========== ========= ======== ======== =================================
band  median     baseline  between  negative verdict
      /tCr       CV        CV
===== ========== ========= ======== ======== =================================
MM09  0.763      4.7%      19.5%    0%       measurable
MM12  0.290      44%       36.5%    0%       provisional -- do not rely on it
MM14  -0.020     88%       138%     78%      not measurable
MM17  0.006      --        92%      14%      not measurable
===== ========== ========= ======== ======== =================================

Two different quantities sit in that table and should not be conflated.
Baseline CV is stability of one session under a changed fit; between-session CV
is spread across animals, which mixes real biology with measurement error. For
scale, on the same sessions the metabolites give between-session CVs of 8.1%
(Glu+Gln), 8.8% (NAA+NAAG), 18.9% (GSH) and 24.3% (Tau). MM09 at 19.5% is
therefore noisier across animals than the strong singlets and sits alongside
GSH: usable, but needing larger groups than NAA to detect an effect.

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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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

#: Window integrated for the creatine reference: the tCr methyl singlet at
#: 3.027 ppm. Restricting to it is not cosmetic -- integrating the modelled
#: creatine across the whole fit range instead makes the reference
#: phase-sensitive rather than amplitude-sensitive, because the integral of a
#: spectrum over a wide range is dominated by the t=0 value of its FID. Five of
#: 92 sessions collapsed that way, one to a *negative* area, while their fitted
#: creatine amplitudes were entirely normal (Cr+PCr 0.201 against a 0.252
#: median) and their reported concentrations were fine.
#:
#: Measured over the same 92 sessions, the local window is better on every
#: count: the reference is never non-positive (min 0.131 against -0.016), the
#: MM09 ratio's between-session CV falls from 247% to 19.5%, its baseline-order
#: CV from 5.2% to 4.7%, and no session has to be excluded.
TCR_WINDOW: Tuple[float, float] = (2.90, 3.15)

#: Smallest share of the modelled signal *inside* TCR_WINDOW that creatine may
#: hold before the reference is refused. Across 92 sessions the share runs
#: 0.795 to 2.94 with a median of 0.998 -- creatine dominates that window, as
#: it should -- so this floor is a wide safety net that fires on none of them.
#: It exists to catch a genuinely collapsed fit, not to trim outliers.
MIN_REFERENCE_PURITY = 0.5



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
    window: Tuple[float, float] = TCR_WINDOW,
) -> float:
    """Area of the modelled creatine methyl singlet, for scaling the MM areas.

    Taken from the fitted metabolite curves rather than from a reported
    concentration, so numerator and denominator are areas of the same spectrum
    in the same arbitrary units and the ratio is scale-free.

    Integrated over ``window`` rather than the whole fit range. That matters:
    a wide-range integral of a real spectrum is dominated by the t=0 value of
    its FID and so reports phase rather than amplitude, which is how five of 92
    sessions produced a near-zero or negative reference from perfectly normal
    creatine amplitudes. See :data:`TCR_WINDOW`.
    """
    present = [c for c in components if c in metabolites.columns]
    if not present:
        raise KeyError(f'none of {list(components)} in the metabolite curves; '
                       f'have {list(metabolites.columns)[:12]}')
    ppm = metabolites['ppm'].to_numpy(float)
    total = metabolites[present].to_numpy(float).sum(axis=1)
    order = np.argsort(ppm)
    ppm, total = ppm[order], total[order]

    low, high = min(window), max(window)
    inside = (ppm >= low) & (ppm <= high)
    if inside.sum() < 2:
        raise ValueError(f'no modelled creatine between {low} and {high} ppm')
    return float(np.trapezoid(total[inside], ppm[inside]))


def check_reference(
    metabolites: pd.DataFrame,
    components: Sequence[str] = TCR_COMPONENTS,
    window: Tuple[float, float] = TCR_WINDOW,
    minimum_purity: float = MIN_REFERENCE_PURITY,
) -> Dict[str, Any]:
    """Is the creatine reference sound enough to divide by?

    A safety net rather than a filter. With the reference taken over the methyl
    singlet (see :data:`TCR_WINDOW`) none of this study's 92 sessions fail it.
    The check exists because the failure mode is silent when it does occur: a
    session whose modelled creatine collapses still fits, still reports
    plausible concentrations and still passes SNR, linewidth and placement QC,
    while any area *ratio* taken against it explodes. One session returned
    -11.7 /tCr from an entirely ordinary MM area of 0.18.

    ``purity`` is creatine's share of the modelled signal inside the window,
    which is self-normalising and so needs no cohort context. Observed range
    across 92 sessions: 0.795 to 2.94, median 0.998.

    Returns
    -------
    dict
        ``area``, ``window_total``, ``purity``, ``ok`` and ``reason``.
    """
    area = reference_area(metabolites, components, window)
    ppm = metabolites['ppm'].to_numpy(float)
    order = np.argsort(ppm)
    ppm_sorted = ppm[order]
    low, high = min(window), max(window)
    inside = (ppm_sorted >= low) & (ppm_sorted <= high)
    window_total = float(np.trapezoid(
        metabolites.drop(columns=['ppm']).to_numpy(float).sum(axis=1)[order][inside],
        ppm_sorted[inside]))
    purity = area / window_total if window_total else float('nan')

    if area <= 0:
        return {'area': area, 'window_total': window_total, 'purity': purity,
                'ok': False,
                'reason': f'creatine reference area is non-positive ({area:.4f}); '
                          'the fit put creatine to zero or into dispersion'}
    if not np.isfinite(purity) or purity < minimum_purity:
        return {'area': area, 'window_total': window_total, 'purity': purity,
                'ok': False,
                'reason': f'creatine is only {purity:.1%} of the modelled signal in '
                          f'{low}-{high} ppm, below the {minimum_purity:.0%} floor'}
    return {'area': area, 'window_total': window_total, 'purity': purity,
            'ok': True, 'reason': ''}


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

    # A bad reference must yield no ratio rather than a plausible-looking one:
    # the numerator is usually fine when the denominator collapses, so the
    # result looks like a measurement and is not. See check_reference.
    tcr = None
    if metabolites is not None:
        try:
            reference = check_reference(metabolites)
        except KeyError as exc:
            logger.warning('no creatine reference for MM scaling: %s', exc)
        else:
            if reference['ok']:
                tcr = reference['area']
            else:
                logger.warning('MM ratios not computed: %s', reference['reason'])

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
