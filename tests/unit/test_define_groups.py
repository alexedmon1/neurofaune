#!/usr/bin/env python3
"""
Unit tests for CovNet group definition and comparison construction.

These used to hardcode ``p30/p60/p90`` and the ``C/L/M/H`` dose scheme, so any
study naming its timepoints differently was silently reduced to nothing —
``define_groups`` dropped every row and the comparison helpers returned an empty
list. The tests pin both: the bpa-rat behaviour is unchanged, and a differently
named study now works instead of quietly producing no result.
"""

import pandas as pd
import pytest

from neurofaune.network.matrices import (
    cohorts_from_labels,
    cross_dose_timepoint_comparisons,
    cross_timepoint_comparisons,
    default_dose_comparisons,
    define_groups,
    dose_levels,
    parse_group_labels,
)

BPA_LABELS = [f"{c}_{d}" for c in ("p30", "p60", "p90") for d in ("C", "L", "M", "H")]
CUP_LABELS = ["week0_sham", "week0_cuprizone", "week6_sham", "week6_cuprizone"]


def frame(cohorts, doses=("C", "L"), sexes=("F", "M"), n=2):
    rows = []
    for cohort in cohorts:
        for dose in doses:
            for sex in sexes:
                for i in range(n):
                    rows.append({"subject": f"sub-{cohort}{dose}{sex}{i}",
                                 "cohort": cohort, "dose": dose, "sex": sex})
    return pd.DataFrame(rows)


# --- define_groups -----------------------------------------------------------

def test_pnd_dose_grouping_unchanged_for_bpa():
    groups = define_groups(frame(["p30", "p60", "p90"]), grouping="pnd_dose")
    assert sorted(groups) == ["p30_C", "p30_L", "p60_C", "p60_L", "p90_C", "p90_L"]
    assert all(len(g) == 4 for g in groups.values())


def test_unknown_cohort_still_excluded_by_default():
    groups = define_groups(frame(["p60", "unknown"]), grouping="pnd_dose")
    assert not any(label.startswith("unknown") for label in groups)


def test_arbitrary_cohort_names_are_kept():
    """The regression this fixes: cuprizone sessions were dropped in full."""
    groups = define_groups(frame(["week0", "week6"], doses=("sham", "cuprizone")),
                           grouping="pnd_dose")
    assert sorted(groups) == ["week0_cuprizone", "week0_sham",
                              "week6_cuprizone", "week6_sham"]


def test_explicit_cohorts_restrict_the_analysis():
    groups = define_groups(frame(["p30", "p60", "p90"]), cohorts=["p60", "p90"])
    assert {label.split("_")[0] for label in groups} == {"p60", "p90"}


def test_requesting_an_absent_cohort_warns(caplog):
    with caplog.at_level("WARNING"):
        define_groups(frame(["p60"]), cohorts=["p60", "p120"])
    assert "p120" in caplog.text


def test_empty_result_raises_instead_of_returning_nothing():
    """Silently returning zero groups is what made the old filter dangerous."""
    with pytest.raises(ValueError, match="No sessions left after cohort filtering"):
        define_groups(frame(["week0", "week6"]), cohorts=["p30", "p60", "p90"])


def test_dose_grouping_ignores_cohort():
    groups = define_groups(frame(["p30", "p60"]), grouping="dose")
    assert sorted(groups) == ["C", "L"]


def test_full_grouping_adds_sex():
    groups = define_groups(frame(["p60"]), grouping="full")
    assert sorted(groups) == ["p60_C_F", "p60_C_M", "p60_L_F", "p60_L_M"]


def test_unknown_grouping_rejected():
    with pytest.raises(ValueError, match="Unknown grouping"):
        define_groups(frame(["p60"]), grouping="batch")


# --- label parsing -----------------------------------------------------------

def test_label_split_falls_back_to_the_last_underscore():
    assert parse_group_labels(["p60_control"])["p60_control"] == ("p60", "control")


def test_known_cohorts_make_the_split_exact():
    """A dose name containing an underscore breaks the naive split."""
    labels = ["week_6_high_dose"]
    assert parse_group_labels(labels, cohorts=["week_6"]) == {
        "week_6_high_dose": ("week_6", "high_dose")
    }
    assert parse_group_labels(labels)["week_6_high_dose"] == ("week_6_high", "dose")


def test_cohort_order_overrides_sorted_order():
    labels = ["baseline_sham", "week12_sham", "week6_sham"]
    assert cohorts_from_labels(labels) == ["baseline", "week12", "week6"]
    order = ["baseline", "week6", "week12"]
    assert cohorts_from_labels(labels, cohort_order=order) == order


def test_control_level_detected_across_naming_conventions():
    assert dose_levels(BPA_LABELS) == ("C", ["L", "M", "H"])
    assert dose_levels(CUP_LABELS) == ("sham", ["cuprizone"])
    assert dose_levels([f"p60_{d}" for d in ("control", "low", "medium", "high")]) == (
        "control", ["low", "medium", "high"]
    )


def test_no_control_level_is_reported_not_guessed(caplog):
    labels = ["p60_alpha", "p60_beta"]
    assert dose_levels(labels)[0] is None
    with caplog.at_level("WARNING"):
        assert default_dose_comparisons(labels) == []
    assert "No control level" in caplog.text


# --- comparison construction -------------------------------------------------

def test_bpa_dose_comparisons_unchanged():
    assert default_dose_comparisons(BPA_LABELS) == [
        ("p30_L", "p30_C"), ("p30_M", "p30_C"), ("p30_H", "p30_C"),
        ("p60_L", "p60_C"), ("p60_M", "p60_C"), ("p60_H", "p60_C"),
        ("p90_L", "p90_C"), ("p90_M", "p90_C"), ("p90_H", "p90_C"),
    ]


def test_bpa_cross_timepoint_unchanged():
    pairs = cross_timepoint_comparisons(BPA_LABELS)
    assert len(pairs) == 12
    assert ("p30_C", "p60_C") in pairs and ("p60_H", "p90_H") in pairs


def test_bpa_cross_dose_timepoint_unchanged():
    pairs = cross_dose_timepoint_comparisons(BPA_LABELS)
    assert len(pairs) == 9
    assert ("p30_L", "p60_C") in pairs and ("p60_H", "p90_C") in pairs


def test_comparisons_work_for_a_two_group_study():
    assert default_dose_comparisons(CUP_LABELS) == [
        ("week0_cuprizone", "week0_sham"), ("week6_cuprizone", "week6_sham"),
    ]
    assert cross_timepoint_comparisons(CUP_LABELS) == [
        ("week0_sham", "week6_sham"), ("week0_cuprizone", "week6_cuprizone"),
    ]
    assert cross_dose_timepoint_comparisons(CUP_LABELS) == [
        ("week0_cuprizone", "week6_sham"),
    ]


def test_missing_control_group_skips_that_cohort():
    labels = ["p30_C", "p30_L", "p60_L"]
    assert default_dose_comparisons(labels) == [("p30_L", "p30_C")]
