#!/usr/bin/env python3
"""
Unit tests for composite morphometry.

Covers the parts that decide whether a volume is meaningful:
- structure group rules resolving to label-id sets
- LUT text aliasing on top of load_parcellation's normalisation
- partial-volume arithmetic
- in-plane-only dilation (must never grow along the coarse axis)
- ICC(2,1)
"""

import numpy as np
import pandas as pd
import pytest

from neurofaune.analysis.stats.reliability import compute_icc_2_1
from neurofaune.network.morphometry import (
    coarse_axis,
    load_structure_groups,
    normalise_labels,
    resolve_labels,
    structure_volume,
    voxel_volume_mm3,
)

SPEC = {
    'normalize_lut_text': True,
    'aliases': {'Hippocampus Fomation': 'Hippocampus Formation'},
}

ROWS = [
    # Labels, Hemisphere, Matter,        Territories,    System,                 roi_name
    (11,   'L', 'Grey Matter',  'Cortex',        'Motor System',         'Motor_Cortex_L'),
    (12,   'R', 'Grey Matter',  'Cortex',        'Motor System',         'Motor_Cortex_R'),
    (71,   'L', 'Grey Matter',  'Cortex',        'Hippocampus Fomation', 'Cornu_Ammonis_1_L'),
    (31,   'L', 'Grey Matter',  'Cortex',        'Amygdala',             'Amygdala_L'),
    (891,  'L', 'White Matter', 'Fiber tract',   'Corpus Callosum',      'Corpus_Callosum_L'),
    (1131, 'L', 'White Matter', 'Diencephalon',  'Thalamus',             'Thalamus_L'),
    (1171, 'L', 'CSF',          'CSF',           'CSF',                  'Ventricular_System_L'),
]

COLUMNS = ['Labels', 'Hemisphere', 'Matter', 'Territories', 'System', 'roi_name']


@pytest.fixture
def labels_df():
    return normalise_labels(pd.DataFrame(ROWS, columns=COLUMNS), SPEC)


# --- group rules -------------------------------------------------------------

def test_aliases_apply_on_top_of_load_parcellation(labels_df):
    """load_parcellation fixes Olfactive/Olfactory; the group file fixes the rest."""
    assert 'Hippocampus Formation' in set(labels_df['System'])
    assert resolve_labels(labels_df, {'systems': ['Hippocampus Formation']}) == {71}


def test_selectors_intersect_by_default(labels_df):
    rule = {'territories': ['Cortex'], 'matter': ['Grey Matter']}
    assert resolve_labels(labels_df, rule) == {11, 12, 71, 31}


def test_combine_union_ors_selectors(labels_df):
    rule = {'territories': ['Fiber tract'], 'systems': ['Amygdala'], 'combine': 'union'}
    assert resolve_labels(labels_df, rule) == {891, 31}


def test_exclusions_run_after_selection(labels_df):
    """cortical_gm's real rule: Cortex minus what aseg calls subcortical."""
    rule = {'territories': ['Cortex'],
            'exclude_systems': ['Hippocampus Formation', 'Amygdala']}
    assert resolve_labels(labels_df, rule) == {11, 12}


def test_include_ids_bypasses_selectors(labels_df):
    """The thalamus proper is filed under White Matter, so it is named explicitly."""
    assert resolve_labels(labels_df, {'include_ids': [1131]}) == {1131}


def test_all_labels_takes_everything(labels_df):
    assert resolve_labels(labels_df, {'all_labels': True}) == {r[0] for r in ROWS}


def test_empty_rule_selects_nothing(labels_df):
    assert resolve_labels(labels_df, {}) == set()


def test_shipped_sigma_groups_all_resolve(labels_df):
    """Every structure in the shipped file must use selectors this engine knows."""
    spec = load_structure_groups()
    for name, rule in spec['structures'].items():
        resolve_labels(labels_df, rule)          # must not raise
        assert 'tissue' in rule, f"{name} has no tissue selector"


# --- partial-volume arithmetic ----------------------------------------------

def test_unweighted_volume_is_a_voxel_count():
    labels = np.array([[[1, 1], [2, 0]]], dtype=np.int32)
    assert structure_volume(labels, {1}, None, voxel_mm3=0.5) == (1.0, 2)


def test_weighting_integrates_the_posterior():
    labels = np.array([[[1, 1], [1, 0]]], dtype=np.int32)
    weight = np.array([[[0.25, 0.75], [1.0, 1.0]]], dtype=np.float32)
    volume, n = structure_volume(labels, {1}, weight, voxel_mm3=2.0)
    assert n == 3
    assert volume == pytest.approx((0.25 + 0.75 + 1.0) * 2.0)


def test_weighted_never_exceeds_unweighted():
    """A posterior is bounded by 1, so PV weighting can only shrink a volume."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=(6, 6, 6)).astype(np.int32)
    weight = rng.random(labels.shape).astype(np.float32)
    assert (structure_volume(labels, {1, 2}, weight, 0.1)[0]
            <= structure_volume(labels, {1, 2}, None, 0.1)[0])


def test_empty_label_set_is_zero_not_an_error():
    assert structure_volume(np.ones((2, 2, 2), np.int32), set(), None, 1.0) == (0.0, 0)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_dilation_never_grows_along_the_coarse_axis(axis):
    """One step along a 0.8 mm axis moves six times further than one in-plane."""
    labels = np.zeros((5, 5, 5), np.int32)
    labels[2, 2, 2] = 1
    _, n = structure_volume(labels, {1}, None, 1.0, dilate=1, plane_axis=axis)
    assert n == 9        # a 3x3 square in the two fine axes, one voxel deep


def test_coarse_axis_finds_the_thick_slice_direction():
    assert coarse_axis((1.25, 1.25, 8.0)) == 2
    assert coarse_axis((8.0, 1.25, 1.25)) == 0


def test_voxel_scale_undoes_header_scaling():
    """neurofaune stores rodent voxels x10; volumes must come out in true mm3."""
    class _Img:
        class header:
            @staticmethod
            def get_zooms():
                return (1.25, 1.25, 8.0)

    assert voxel_volume_mm3(_Img, 10.0) == pytest.approx(0.0125)
    assert voxel_volume_mm3(_Img, 1.0) == pytest.approx(12.5)


# --- reliability -------------------------------------------------------------

def test_icc_is_high_for_reproducible_measurements():
    rng = np.random.default_rng(0)
    base = rng.random((8, 1))
    assert compute_icc_2_1(base + rng.random((8, 3)) * 0.01) > 0.95


def test_icc_is_nan_when_under_determined():
    assert np.isnan(compute_icc_2_1(np.random.default_rng(0).random((1, 3))))


def test_icc_drops_incomplete_rows():
    """A two-way model needs a complete block, so NaN rows go rather than propagate."""
    x = np.array([[1.0, 1.1], [2.0, 2.1], [3.0, np.nan]])
    assert not np.isnan(compute_icc_2_1(x))
