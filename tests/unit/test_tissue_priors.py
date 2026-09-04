#!/usr/bin/env python3
"""
Unit tests for atlas-derived tissue priors.

The point of this module is that the atlas tissue maps must NOT be geometrically
resampled into subject space (that skips the registration). These tests pin the
alternative: reduce the maps to a per-label composition in atlas space, then paint
it onto the already-warped parcellation.
"""

import nibabel as nib
import numpy as np
import pytest

from neurofaune.atlas.tissue_priors import (
    TISSUE_ORDER,
    build_atlas_extent_mask,
    build_native_priors,
    compute_label_tissue_fractions,
    paint_priors,
)


@pytest.fixture
def atlas(tmp_path):
    """Two labels: 1 is pure WM, 2 is pure GM."""
    labels = np.zeros((4, 4, 4), np.int32)
    labels[0] = 1
    labels[1] = 2
    labels_path = tmp_path / 'labels.nii.gz'
    nib.save(nib.Nifti1Image(labels, np.eye(4)), str(labels_path))

    priors = {}
    for tissue in TISSUE_ORDER:
        arr = np.zeros((4, 4, 4), np.float32)
        if tissue == 'WM':
            arr[0] = 1.0
        elif tissue == 'GM':
            arr[1] = 1.0
        path = tmp_path / f'{tissue}.nii.gz'
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))
        priors[tissue] = path
    return labels_path, priors


def test_fractions_recover_each_label_composition(atlas):
    labels_path, priors = atlas
    fractions = compute_label_tissue_fractions(labels_path, priors)
    wm_index, gm_index = TISSUE_ORDER.index('WM'), TISSUE_ORDER.index('GM')
    assert fractions[1][wm_index] == pytest.approx(1.0)
    assert fractions[2][gm_index] == pytest.approx(1.0)


def test_fractions_sum_to_one(atlas):
    labels_path, priors = atlas
    for fraction in compute_label_tissue_fractions(labels_path, priors).values():
        assert fraction.sum() == pytest.approx(1.0)


def test_label_with_no_prior_support_falls_back_to_uniform(tmp_path):
    labels = np.ones((3, 3, 3), np.int32)
    labels_path = tmp_path / 'l.nii.gz'
    nib.save(nib.Nifti1Image(labels, np.eye(4)), str(labels_path))
    priors = {}
    for tissue in TISSUE_ORDER:
        path = tmp_path / f'{tissue}.nii.gz'
        nib.save(nib.Nifti1Image(np.zeros((3, 3, 3), np.float32), np.eye(4)), str(path))
        priors[tissue] = path
    fraction = compute_label_tissue_fractions(labels_path, priors)[1]
    assert fraction == pytest.approx(np.full(len(TISSUE_ORDER), 1 / len(TISSUE_ORDER)))


def test_mismatched_prior_grid_is_rejected(tmp_path):
    """A prior on a different grid means it was never in the atlas's space."""
    labels_path = tmp_path / 'l.nii.gz'
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.int32), np.eye(4)), str(labels_path))
    priors = {}
    for tissue in TISSUE_ORDER:
        path = tmp_path / f'{tissue}.nii.gz'
        nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), np.float32), np.eye(4)), str(path))
        priors[tissue] = path
    with pytest.raises(ValueError, match='same atlas space'):
        compute_label_tissue_fractions(labels_path, priors)


def test_missing_tissue_is_rejected(tmp_path):
    labels_path = tmp_path / 'l.nii.gz'
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.int32), np.eye(4)), str(labels_path))
    with pytest.raises(KeyError, match='GM'):
        compute_label_tissue_fractions(labels_path, {'CSF': labels_path, 'WM': labels_path})


def test_painting_places_composition_on_the_warped_parcellation():
    """The prior is aligned by construction: no transform is applied here."""
    dseg = np.array([[[1, 2]]], dtype=np.int32)
    fractions = {1: np.array([0.1, 0.2, 0.7], np.float32),
                 2: np.array([0.5, 0.5, 0.0], np.float32)}
    stack = paint_priors(dseg, fractions, n_tissues=3)
    assert [float(s[0, 0, 0]) for s in stack] == pytest.approx([0.1, 0.2, 0.7])
    assert [float(s[0, 0, 1]) for s in stack] == pytest.approx([0.5, 0.5, 0.0])


def test_painted_priors_sum_to_one_inside_labels():
    dseg = np.array([[[1, 0]]], dtype=np.int32)
    stack = paint_priors(dseg, {1: np.array([0.2, 0.3, 0.5], np.float32)}, 3)
    assert sum(float(s[0, 0, 0]) for s in stack) == pytest.approx(1.0)
    assert sum(float(s[0, 0, 1]) for s in stack) == pytest.approx(0.0)


def test_write_native_priors_numbering_matches_tissue_order(tmp_path):
    """ANTs returns posteriors in prior order, so the numbering IS the class map."""
    dseg_path = tmp_path / 'dseg.nii.gz'
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.int32), np.eye(4)), str(dseg_path))
    fractions = {1: np.array([0.1, 0.3, 0.6], np.float32)}
    pattern = build_native_priors(dseg_path, fractions, tmp_path / 'priors')
    assert pattern.endswith('prior_%02d.nii.gz')
    for index, expected in enumerate([0.1, 0.3, 0.6], start=1):
        arr = nib.load(str(tmp_path / 'priors' / f'prior_{index:02d}.nii.gz')).get_fdata()
        assert float(arr[0, 0, 0]) == pytest.approx(expected, abs=1e-6)


def test_atlas_extent_mask_is_the_labelled_brain(tmp_path):
    """The skull-strip mask is generous; its surplus is scored as CSF."""
    labels = np.zeros((4, 4, 4), np.int32)
    labels[1:3, 1:3, 1:3] = 5
    dseg_path = tmp_path / 'dseg.nii.gz'
    nib.save(nib.Nifti1Image(labels, np.eye(4)), str(dseg_path))
    out = build_atlas_extent_mask(dseg_path, tmp_path / 'mask.nii.gz')
    mask = nib.load(str(out)).get_fdata()
    assert set(np.unique(mask)) == {0.0, 1.0}
    assert mask.sum() == 8
