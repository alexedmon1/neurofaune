"""The minimum detectable effect (DUET #151): the shared helper and the randomise read-out.

The reference values are the shared contract: neurofaune, source-analytics and
neurovrai each reproduce them, so a reader comparing their tables compares one number.
"""
import math

import numpy as np
import pytest

from neurofaune.analysis.stats import power
from neurofaune.analysis.stats import readout as ro


@pytest.mark.parametrize("se, df, expected", [
    (1.0, 10, 3.107197),
    (1.0, 26, 2.911195),
    (2.09, 24, 6.104376),
    (1.0, math.inf, 2.801585),
    (1.0, 1e9, 2.801585),
    (1.0, None, 2.801585),
    (0.5, 4, 1.858705),
])
def test_reference_values(se, df, expected):
    assert power.mde(se, df) == pytest.approx(expected, abs=5e-7)


def test_power_and_alpha_are_the_decided_ones():
    assert (power.POWER, power.ALPHA) == (0.80, 0.05)
    assert power.MAP_NOTE.startswith(power.NOT_COMPUTED) and power.CLUSTER_NOTE.startswith(power.NOT_COMPUTED)
    from neurofaune.results.spec import NOT_COMPUTED
    assert power.NOT_COMPUTED == NOT_COMPUTED               # the checker reads the same prefix


@pytest.mark.parametrize("se, df", [(0.0, 10), (-1.0, 10), (math.nan, 10), (1.0, 0)])
def test_no_mde_without_a_standard_error_or_degrees_of_freedom(se, df):
    assert math.isnan(power.mde(se, df))


def test_mde_percent_needs_a_reference_clear_of_zero():
    assert power.mde_percent(3.0, 60.0) == pytest.approx(5.0)
    assert power.mde_percent(3.0, -60.0) == pytest.approx(5.0)           # of the magnitude
    assert math.isnan(power.mde_percent(3.0, None))                      # no reference group
    assert math.isnan(power.mde_percent(3.0, 0.0))
    assert math.isnan(power.mde_percent(3.0, 0.5, near_zero=1.0))        # within the noise
    assert math.isnan(power.mde_percent(math.nan, 60.0))


# ---------------------------------------------------------------- read-out ---
def _two_groups(rng, n=8, shift=0.0, base=10.0):
    g = np.array([1] * n + [0] * n)
    X = np.column_stack([g, 1 - g]).astype(float)
    y = base + shift * g + rng.normal(0, 1, 2 * n)
    return X, y, g


def test_two_group_row_gives_se_mde_and_a_percent_of_the_minus_one_group():
    X, y, g = _two_groups(np.random.default_rng(3), shift=1.0)
    c = np.array([1.0, -1.0])
    eff = ro.contrast_effect(y, X, c, ro.describe_contrast(c, X, ["A", "B"]))
    a, b = y[g == 1], y[g == 0]
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    se = sp * np.sqrt(1 / 8 + 1 / 8)
    assert eff["se"] == pytest.approx(se)
    assert eff["mde"] == pytest.approx(power.mde(se, 14))
    assert eff["mde_percent"] == pytest.approx(100 * eff["mde"] / abs(b.mean()))     # B, weighted -1
    assert eff["mde_note"] == ""
    rev = ro.contrast_effect(y, X, -c, ro.describe_contrast(-c, X, ["A", "B"]))
    assert rev["mde"] == pytest.approx(eff["mde"])                                    # same design
    assert rev["mde_percent"] == pytest.approx(100 * eff["mde"] / abs(a.mean()))     # A is now -1


def test_covariates_are_in_the_standard_error():
    rng = np.random.default_rng(4)
    X, y, g = _two_groups(rng)
    cov = rng.normal(0, 1, len(y))
    y = y + 3 * cov
    Xc = np.column_stack([X, cov])
    c = np.array([1.0, -1.0, 0.0])
    with_cov = ro.contrast_effect(y, Xc, c, ro.describe_contrast(c, Xc, ["A", "B", "cov"]))
    without = ro.contrast_effect(y, X, c[:2], ro.describe_contrast(c[:2], X, ["A", "B"]))
    assert with_cov["se"] < without["se"]
    assert with_cov["mde"] == pytest.approx(power.mde(with_cov["se"], len(y) - 3))


def test_a_scaled_contrast_puts_the_reference_on_the_estimates_scale():
    X, y, _ = _two_groups(np.random.default_rng(5))
    full = ro.contrast_effect(y, X, np.array([1.0, -1.0]), ro.describe_contrast(np.array([1.0, -1.0]), X, "AB"))
    half = ro.contrast_effect(y, X, np.array([0.5, -0.5]), ro.describe_contrast(np.array([0.5, -0.5]), X, "AB"))
    assert half["mde"] == pytest.approx(full["mde"] / 2)
    assert half["mde_percent"] == pytest.approx(full["mde_percent"])


def test_no_percent_where_the_reference_mean_sits_in_the_noise():
    X, y, _ = _two_groups(np.random.default_rng(6), base=0.0)          # a change score around 0
    c = np.array([1.0, -1.0])
    eff = ro.contrast_effect(y, X, c, ro.describe_contrast(c, X, ["A", "B"]))
    assert eff["mde"] > 0 and math.isnan(eff["mde_percent"]) and eff["mde_note"] == ""


@pytest.mark.parametrize("kind", ["one-sample", "regression"])
def test_no_reference_group_no_percent(kind):
    rng = np.random.default_rng(7)
    n = 12
    if kind == "one-sample":
        X, c, cols = np.ones((n, 1)), np.array([1.0]), ["mean"]
    else:
        X = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
        c, cols = np.array([0.0, 1.0]), ["mean", "dose"]
    y = 5 + rng.normal(0, 1, n)
    eff = ro.contrast_effect(y, X, c, ro.describe_contrast(c, X, cols))
    assert eff["mde"] == pytest.approx(power.mde(eff["se"], n - X.shape[1]))
    assert math.isnan(eff["mde_percent"]) and eff["mde_note"] == ""


def test_a_design_with_no_residual_df_says_why():
    X = np.eye(3)
    c = np.array([1.0, -1.0, 0.0])
    eff = ro.contrast_effect(np.array([1.0, 2.0, 3.0]), X, c, ro.describe_contrast(c, X, "ABC"))
    assert math.isnan(eff["mde"]) and eff["mde_note"] == "not computed: no residual degrees of freedom"


def test_zero_residual_variance_says_why():
    X, _, g = _two_groups(np.random.default_rng(8))
    c = np.array([1.0, -1.0])
    eff = ro.contrast_effect(10.0 + g, X, c, ro.describe_contrast(c, X, ["A", "B"]))
    assert math.isnan(eff["mde"]) and eff["mde_note"] == "not computed: zero residual variance"
