"""Test-retest reliability: ICC(2,1) across repeated measurements.

Two-way random effects, single measure, absolute agreement.

**Read the caveat before using this on a treatment study.** ICC(2,1) over
timepoints is a valid *test-retest reliability* only when no true change is
expected between them. Across a longitudinal treatment timecourse (cuprizone, for
instance) real biological change — and any session or acquisition artefact —
inflates the between-timepoint variance and drives this statistic toward zero. A
low value there means the brain changed, not that the measurement is unreliable.

For registration/segmentation reproducibility use spatial overlap instead (see
neurofaune's cross-sectional/longitudinal Dice QC), and model change over time with
a mixed-effects model rather than penalising it here.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_icc_2_1(matrix: np.ndarray) -> float:
    """ICC(2,1) for a subjects-by-measurements matrix.

    Parameters
    ----------
    matrix : ndarray
        Rows are subjects, columns are repeated measurements (e.g. sessions).
        Rows containing any NaN are dropped, since a two-way model needs a
        complete block.

    Returns
    -------
    float
        The coefficient, or NaN when fewer than two subjects or two measurements
        survive — under-determined rather than zero.
    """
    x = np.asarray(matrix, dtype=float)
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape if x.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return float("nan")

    grand = x.mean()
    ms_rows = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_err = (resid ** 2).sum() / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom) if denom > 0 else float("nan")
