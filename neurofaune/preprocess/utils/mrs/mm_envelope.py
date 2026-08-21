"""Macromolecule and lipid handling for FSL-MRS.

The problem
-----------
A short-TE brain spectrum carries a large macromolecule and mobile-lipid
contribution. An MM-free basis pushes it into the polynomial baseline, which
biases metabolites sitting on top of it and makes them sensitive to the
baseline order chosen.

Adding MM components naively makes things worse. Fitting a set of broad peaks
with free amplitudes against a flexible baseline is under-determined: the
components are collinear with each other, with the baseline, and with the
metabolites. Tested that way, metabolites were driven to zero and MM amplitudes
diverged.

The approach here
-----------------
Fit MM as a **single basis spectrum with fixed internal proportions and one
free amplitude**, rather than as many independently scaled components. That
turns an ill-posed problem into a one-parameter one, and the fixed shape cannot
deform to absorb metabolite signal.

It is also the assumption most defensible at an arbitrary echo time. The
absolute MM amplitude at TE 20 ms is unknown -- MM T2 is short and much has
already decayed -- but the components decay at broadly similar rates, so echo
time mostly rescales the envelope rather than reshaping it. The one thing left
free is precisely the one thing TE changes.

The shapes are the conventional MM/lipid resonances; the *relative* amplitudes
are prior knowledge, and the one free scale is what the data determines.

What this is not
----------------
It is not a measured MM basis. That requires a metabolite-nulled
(inversion-recovery) acquisition at the same field, echo time and preparation,
and remains the correct solution. This is a constrained-prior stand-in, and it
should be judged on whether it makes the metabolite estimates *more
identifiable* -- see :func:`baseline_sensitivity`, which measures that without
reference to any other fitting package.
"""

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Conventional MM/lipid resonances: (ppm, relative amplitude in protons).
#: Amplitudes are the standard parameterisation; only their *ratios* are used,
#: since the envelope carries a single free scale.
MM_RESONANCES: List[Tuple[float, float]] = [
    (0.90, 3.00),   # MM09, overlapping Lip09
    (1.21, 2.00),   # MM12
    (1.28, 2.00),   # Lip13, mobile lipid methylene
    (1.43, 2.00),   # MM14
    (1.67, 2.00),   # MM17
    (2.04, 1.33),   # Lip20 / MM20 methylene, under NAA and Glu
    (2.25, 0.67),   # MM/Lip
    (2.80, 0.87),   # Lip, allylic
    (3.00, 0.40),   # MM, under total creatine
]

#: Split point conventionally used to separate the MM-dominated upfield region
#: from the metabolite region. Note it does *not* isolate MM: about a quarter
#: of the envelope's amplitude lies above it, and that quarter is the part
#: under NAA and Glu.
UPFIELD_LIMIT = 1.8

#: Default envelope linewidths, in Hz, applied to every resonance. MM lines are
#: intrinsically broad; these are the free-ish knobs of the model and are
#: exposed rather than buried.
DEFAULT_GAMMA_HZ = 20.0
DEFAULT_SIGMA_HZ = 15.0

#: FWHM of a Gaussian in standard deviations.
_GAUSS_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def lorentzian_gamma(fwhm_hz: float) -> float:
    """FSL-MRS ``gamma`` for a Lorentzian of this FWHM (it damps by ``exp(-gamma t)``)."""
    return float(np.pi * fwhm_hz)


def gaussian_sigma(fwhm_hz: float) -> float:
    """FSL-MRS ``sigma`` for a Gaussian of this FWHM (it damps by ``exp(-sigma^2 t^2/2)``)."""
    return float(2.0 * np.pi * fwhm_hz / _GAUSS_FWHM)


def upfield_fraction(resonances: Sequence[Tuple[float, float]] = None,
                     limit: float = UPFIELD_LIMIT) -> float:
    """Share of the envelope's amplitude below ``limit`` ppm.

    Quantifies why an upfield-only treatment is incomplete: the remainder sits
    under the metabolites and is not addressed by integrating the upfield
    region.
    """
    resonances = list(MM_RESONANCES if resonances is None else resonances)
    total = sum(a for _, a in resonances)
    upfield = sum(a for ppm, a in resonances if ppm < limit)
    return float(upfield / total) if total else 0.0


def add_mm_envelope(
    basis,
    name: str = 'MMenv',
    gamma_hz: float = DEFAULT_GAMMA_HZ,
    sigma_hz: float = DEFAULT_SIGMA_HZ,
    resonances: Sequence[Tuple[float, float]] = None,
    conj: bool = False,
) -> str:
    """Add the MM envelope as one basis spectrum, in place.

    Parameters
    ----------
    basis : fsl_mrs.core.basis.Basis
        Loaded basis, modified in place.
    name : str
        Name for the component.
    gamma_hz, sigma_hz : float
        Lorentzian and Gaussian linewidths applied to every resonance.
    resonances : sequence of (ppm, amplitude), optional
        Defaults to :data:`MM_RESONANCES`.
    conj : bool
        Passed through to ``Basis.add_peak``.

    Returns
    -------
    str
        The name added.
    """
    resonances = list(MM_RESONANCES if resonances is None else resonances)
    basis.add_peak(
        [ppm for ppm, _ in resonances],
        [amp for _, amp in resonances],
        name,
        lorentzian_gamma(gamma_hz),
        gaussian_sigma(sigma_hz),
        conj=conj,
    )
    logger.info('Added MM envelope %r: %d resonances, gamma %.0f Hz, sigma %.0f Hz',
                name, len(resonances), gamma_hz, sigma_hz)
    return name


def baseline_sensitivity(estimates: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """How much each metabolite moves when only the baseline order changes.

    This is the criterion for whether an MM model helps, and it needs no
    external reference. A metabolite that is well separated from the baseline
    gives the same answer at any reasonable baseline order; one that is
    degenerate with it swings. Lower is better.

    Parameters
    ----------
    estimates : dict
        ``{baseline_label: {metabolite: value}}`` for the same session fitted
        at several baseline orders.

    Returns
    -------
    dict
        Per metabolite, the coefficient of variation across baseline orders,
        as a percentage.
    """
    metabolites = set.intersection(*(set(v) for v in estimates.values())) \
        if estimates else set()
    out: Dict[str, float] = {}
    for metabolite in sorted(metabolites):
        values = np.array([estimates[k][metabolite] for k in estimates], dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2 or np.abs(values.mean()) < 1e-9:
            continue
        out[metabolite] = float(100.0 * values.std(ddof=1) / np.abs(values.mean()))
    return out
