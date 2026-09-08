#!/usr/bin/env python3
"""
Unit tests for per-ROI distribution extraction.

The coverage rule is the reason these numbers are trustworthy, so most of these
pin it: out-of-slab voxels must not be averaged in, statistics must be NaN rather
than misleading when an ROI is barely covered, and the coverage fraction must be
reported even then so a caller can threshold afterwards.
"""

import numpy as np
import pandas as pd
import pytest

from neurofaune.network.roi_extraction import (
    DEFAULT_PERCENTILES,
    extract_roi_means,
    extract_roi_stats,
)

LABELS = pd.DataFrame({'Labels': [1, 2], 'roi_name': ['roi1', 'roi2']})


def _parcellation():
    parc = np.zeros((4, 4, 4), dtype=int)
    parc[0] = 1
    parc[1] = 2
    return parc


def test_mean_agrees_with_extract_roi_means():
    """Both read the same voxels; if they diverge the coverage rule has forked."""
    parc = _parcellation()
    img = np.zeros((4, 4, 4))
    img[0] = 0.3
    img[1] = 0.2
    stats = extract_roi_stats(img, parc, LABELS).set_index('roi_name')
    means = extract_roi_means(img, parc, LABELS)
    for roi, value in means.items():
        assert stats.loc[roi, 'mean'] == pytest.approx(value)


def test_spread_exposes_a_split_the_mean_hides():
    """Demyelination is patchy: percentiles move before the mean does."""
    parc = _parcellation()
    img = np.zeros((4, 4, 4))
    img[0, :2] = 0.1
    img[0, 2:] = 0.3      # roi1: bimodal, mean 0.2
    img[1] = 0.2          # roi2: uniform, mean 0.2
    stats = extract_roi_stats(img, parc, LABELS).set_index('roi_name')
    assert stats.loc['roi1', 'mean'] == pytest.approx(stats.loc['roi2', 'mean'])
    assert stats.loc['roi1', 'sd'] > stats.loc['roi2', 'sd']
    assert stats.loc['roi1', 'p5'] < stats.loc['roi2', 'p5']
    assert stats.loc['roi1', 'p95'] > stats.loc['roi2', 'p95']


def test_out_of_slab_zeros_are_excluded_not_averaged():
    """Averaging zeros in makes an ROI value a function of slab position."""
    parc = _parcellation()
    img = np.zeros((4, 4, 4))
    img[0, :2] = 0.4          # only half of roi1 was reached
    stats = extract_roi_stats(img, parc, LABELS).set_index('roi_name')
    assert stats.loc['roi1', 'mean'] == pytest.approx(0.4)
    assert stats.loc['roi1', 'coverage'] == pytest.approx(0.5)


def test_explicit_coverage_mask_beats_the_nonzero_fallback():
    """A genuine in-slab zero must count; the fallback cannot tell it apart."""
    parc = _parcellation()
    img = np.zeros((4, 4, 4))
    img[0, :2] = 0.4
    img[0, 2:] = 0.0          # genuinely zero, inside the slab
    mask = np.zeros((4, 4, 4), bool)
    mask[0] = True
    with_mask = extract_roi_stats(img, parc, LABELS, coverage_mask=mask)
    fallback = extract_roi_stats(img, parc, LABELS)
    assert with_mask.set_index('roi_name').loc['roi1', 'mean'] == pytest.approx(0.2)
    assert fallback.set_index('roi_name').loc['roi1', 'mean'] == pytest.approx(0.4)


def test_min_coverage_nans_the_statistics_but_keeps_the_coverage():
    """The reason must survive: a caller needs to see why a value is missing."""
    parc = _parcellation()
    img = np.zeros((4, 4, 4))
    img[0, :1] = 0.4          # 4 of roi1's 16 voxels
    row = extract_roi_stats(img, parc, LABELS, min_coverage=0.5
                            ).set_index('roi_name').loc['roi1']
    assert np.isnan(row['mean']) and np.isnan(row['median'])
    assert row['coverage'] == pytest.approx(0.25)
    assert row['n_covered'] == 4


def test_absent_label_is_nan_throughout():
    parc = np.zeros((4, 4, 4), dtype=int)
    parc[0] = 1               # label 2 never appears
    stats = extract_roi_stats(np.ones((4, 4, 4)), parc, LABELS).set_index('roi_name')
    assert np.isnan(stats.loc['roi2', 'mean'])
    assert np.isnan(stats.loc['roi2', 'coverage'])
    assert stats.loc['roi2', 'n_voxels'] == 0


def test_single_covered_voxel_has_no_sd():
    parc = _parcellation()
    img = np.zeros((4, 4, 4))
    img[0, 0, 0] = 0.5
    row = extract_roi_stats(img, parc, LABELS).set_index('roi_name').loc['roi1']
    assert row['mean'] == pytest.approx(0.5)
    assert np.isnan(row['sd'])


def test_requested_percentiles_appear_as_columns():
    parc = _parcellation()
    stats = extract_roi_stats(np.ones((4, 4, 4)), parc, LABELS, percentiles=(10, 90))
    assert {'p10', 'p90'} <= set(stats.columns)
    assert 'p5' not in stats.columns


def test_default_percentiles_are_present():
    stats = extract_roi_stats(np.ones((4, 4, 4)), _parcellation(), LABELS)
    assert all(f'p{int(q)}' in stats.columns for q in DEFAULT_PERCENTILES)


def test_nan_voxels_do_not_propagate():
    parc = _parcellation()
    img = np.full((4, 4, 4), 0.5)
    img[0, 0, 0] = np.nan
    row = extract_roi_stats(img, parc, LABELS).set_index('roi_name').loc['roi1']
    assert row['mean'] == pytest.approx(0.5)
