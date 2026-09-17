#!/usr/bin/env python3
"""Unit tests for the group-comparison builders in `network/matrices.py`.

These arrived with the covnet work and are kept because main has no coverage for
them. Their companions -- tests for `define_groups`, `dose_levels`,
`parse_group_labels` and `cohorts_from_labels` -- were dropped in the rebase onto
the study-agnostic grouping API: the first is covered by
`test_define_groups_generic.py`, and the other three no longer exist.

What is pinned here is that the builders derive comparisons from the *labels* they
are given rather than from one study's design. A builder that quietly assumes four
dose levels and three timepoints works on the BPA study and silently returns the
wrong pairs, or none, on anything else.
"""

import pytest

from neurofaune.network.matrices import (
    cross_dose_timepoint_comparisons,
    cross_timepoint_comparisons,
    default_dose_comparisons,
)

BPA_LABELS = [f"{c}_{d}" for c in ("p30", "p60", "p90") for d in ("C", "L", "M", "H")]
CUP_LABELS = [f"{c}_{d}" for c in ("week0", "week6") for d in ("sham", "cuprizone")]


def test_bpa_dose_comparisons_unchanged():
    """The four-level/three-timepoint study the builders were written for."""
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
    """Two levels and two timepoints -- the cuprizone shape, not the BPA one."""
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
    """A cohort with no control level yields no pairs, rather than a wrong one."""
    labels = ["p30_C", "p30_L", "p60_L"]
    assert default_dose_comparisons(labels) == [("p30_L", "p30_C")]


def test_no_control_level_is_reported_not_guessed(caplog):
    """With no recognisable control, say so and return nothing -- never pick one."""
    labels = ["p60_alpha", "p60_beta"]
    with caplog.at_level("WARNING"):
        assert default_dose_comparisons(labels) == []
    assert "No control level" in caplog.text
