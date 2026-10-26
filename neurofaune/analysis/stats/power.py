"""The minimum detectable effect (MDE): the smallest difference a fitted design could see.

A test that found nothing is still a result, but only together with what it could
have found. DUET (OPEN.md #151) reads every finding's size against the MDE -- the
difference the design detects with 80% power at two-sided alpha 0.05 -- and the
tool that fitted the model has to supply it, because only the fit holds the
standard error and the degrees of freedom. So every tabular test row carries:

* ``se``  -- the standard error of the estimate, from the fitted model (covariates
  included), in the estimate's units;
* ``mde`` -- ``(t(1 - alpha/2, df) + t(power, df)) * se``, in the same units, a
  positive number; normal quantiles when df is missing or infinite;
* ``mde_percent`` -- ``100 * mde / |reference mean|``, the reference being the
  comparison group's mean on the estimate's scale; blank when there is no reference
  group (a slope, a one-sample mean) or when the reference mean is near zero;
* ``mde_note`` -- empty when ``mde`` is computed, else a reason starting
  ``"not computed: "``. An MDE is never left blank without one.

**Power 0.80 and alpha 0.05, nominal and uncorrected**, by the user's decision: the
MDE says what one test of this size and noise could detect, in the terms a reader
already weighs; a multiple-comparison correction would change the number by a
family that differs between analyses, and is left for later. They are recorded with
every table (the ``Power`` and ``Alpha`` qualifiers on the ``mde`` column,
docs/RESULTS_SPEC.md), so the number never travels without them.

**What is not computed.** A voxelwise map is corrected by randomise (TFCE / maximum-t
permutation), whose detectable effect depends on the spatial extent of the effect and
has no closed form; a map gets :data:`MAP_NOTE`, never a number. A cluster's effect is
measured on the voxels the test selected, so an MDE there would describe a selection,
not the design; cluster rows get :data:`CLUSTER_NOTE`. The same formulas and reference
values are implemented in source-analytics and neurovrai; a change here is a change to
that shared contract.
"""
from __future__ import annotations

import math

from scipy import stats

#: Power at which the MDE is defined (fixed; see the module docstring).
POWER = 0.80
#: Two-sided alpha at which the MDE is defined: nominal, no multiple-comparison correction.
ALPHA = 0.05
#: Every reason an MDE is not given starts with this.
NOT_COMPUTED = "not computed: "
MAP_NOTE = NOT_COMPUTED + "voxelwise map, corrected by randomise"
CLUSTER_NOTE = NOT_COMPUTED + "cluster selected by the test"


def mde(se: float, df: float | None = None, *, alpha: float = ALPHA, power: float = POWER) -> float:
    """The difference detected with ``power`` at two-sided ``alpha``, in ``se``'s units.

    ``(t.ppf(1 - alpha/2, df) + t.ppf(power, df)) * se``; with ``df`` None, NaN or
    infinite the normal quantiles are used. NaN when ``se`` is not a positive finite
    number or ``df`` is not positive -- the caller says why in ``mde_note``.

    >>> round(mde(1.0, 10), 6), round(mde(1.0, None), 6)
    (3.107197, 2.801585)
    """
    if se is None or not math.isfinite(se) or se <= 0:
        return math.nan
    if df is None or (isinstance(df, float) and math.isnan(df)) or math.isinf(df):
        q = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    elif df <= 0:
        return math.nan
    else:
        q = stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)
    return float(q * se)


def mde_percent(mde_value: float, reference_mean: float | None, *, near_zero: float = 0.0) -> float:
    """``100 * mde / |reference_mean|``; NaN without a reference or when it is near zero.

    ``near_zero`` is the magnitude at or below which the reference mean is treated as
    zero -- a percent of a mean that sits within the noise (a centred measure, a change
    score, a log ratio) is not a scale anyone reads. The randomise read-out passes the
    residual SD, so a percent is given only where the comparison group's mean stands
    clear of its own spread.
    """
    if reference_mean is None or not math.isfinite(mde_value) or not math.isfinite(reference_mean):
        return math.nan
    ref = abs(reference_mean)
    if ref == 0 or ref <= near_zero:
        return math.nan
    return float(100.0 * mde_value / ref)
