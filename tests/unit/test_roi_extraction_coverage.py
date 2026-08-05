"""ROI extraction: entity parsing and slab-coverage handling.

Both regressions these cover were silent — files that existed but were never
analysed, and ROI means that tracked slab coverage rather than tissue.
"""
import numpy as np
import pandas as pd
import pytest

from neurofaune.network.roi_extraction import (
    SIGMA_FUNC_RE,
    SIGMA_MAP_RE,
    extract_roi_means,
)
from neurofaune.utils.bids import space_entity


# --------------------------------------------------------------------------- #
# filename entity parsing
# --------------------------------------------------------------------------- #

FNAME_RE_CASES = [
    ("sub-1X_ses-1_space-SIGMA_FA.nii.gz", "sub-1X", "ses-1", "FA"),
    ("sub-1X_ses-1_space-SIGMA_MWF.nii.gz", "sub-1X", "ses-1", "MWF"),
    # Hyphenated entity VALUES: these are the multi-shell metrics, and a \w-only
    # metric group dropped every one of them silently.
    ("sub-1X_ses-1_space-SIGMA_model-DKI_MK.nii.gz", "sub-1X", "ses-1", "model-DKI_MK"),
    ("sub-1X_ses-1_space-SIGMA_model-NODDI_FICVF.nii.gz",
     "sub-1X", "ses-1", "model-NODDI_FICVF"),
    ("sub-Rat49_ses-p90_space-SIGMA_T2.nii.gz", "sub-Rat49", "ses-p90", "T2"),
]


@pytest.mark.parametrize("fname,sub,ses,metric", FNAME_RE_CASES)
def test_sigma_map_regex_parses_hyphenated_metrics(fname, sub, ses, metric):
    m = SIGMA_MAP_RE.match(fname)
    assert m is not None, f"{fname} did not parse"
    assert m.groups() == (sub, ses, metric)


@pytest.mark.parametrize("fname,metric", [
    ("sub-1X_ses-1_space-SIGMA_desc-fALFF_bold.nii.gz", "fALFF"),
    ("sub-1X_ses-1_space-SIGMA_desc-ReHozscore_bold.nii.gz", "ReHozscore"),
])
def test_sigma_func_regex_parses(fname, metric):
    m = SIGMA_FUNC_RE.match(fname)
    assert m is not None, f"{fname} did not parse"
    assert m.group(3) == metric


def test_space_entity_derives_from_source():
    assert space_entity("sub-1X_ses-1_space-SIGMA_desc-preproc_bold.nii.gz") == "space-SIGMA_"
    assert space_entity("sub-1X_ses-1_desc-preproc_bold.nii.gz") == ""
    assert space_entity("/a/b/sub-1X_ses-1_space-T2w_desc-preproc_bold.nii.gz") == "space-T2w_"


# --------------------------------------------------------------------------- #
# coverage-aware ROI means
# --------------------------------------------------------------------------- #

@pytest.fixture
def parcellation_and_labels():
    """Two ROIs of 100 voxels each, side by side."""
    par = np.zeros((10, 10, 2), dtype=int)
    par[..., 0] = 1
    par[..., 1] = 2
    labels = pd.DataFrame({'Labels': [1, 2], 'roi_name': ['roi_a', 'roi_b']})
    return par, labels


def test_out_of_slab_zeros_do_not_dilute_the_mean(parcellation_and_labels):
    """The bug: half a slab of zeros halved the ROI value."""
    par, labels = parcellation_and_labels
    img = np.zeros((10, 10, 2), dtype=np.float32)
    img[:, :, 0] = 60.0            # roi_a fully covered
    img[:5, :, 1] = 60.0           # roi_b only half covered, rest is out-of-slab 0

    means = extract_roi_means(img, par, labels)

    # Both ROIs are the same tissue; only coverage differs.
    assert means['roi_a'] == pytest.approx(60.0)
    assert means['roi_b'] == pytest.approx(60.0)


def test_coverage_is_reported(parcellation_and_labels):
    par, labels = parcellation_and_labels
    img = np.zeros((10, 10, 2), dtype=np.float32)
    img[:, :, 0] = 60.0
    img[:5, :, 1] = 60.0

    means, cov = extract_roi_means(img, par, labels, return_coverage=True)
    assert cov['roi_a'] == pytest.approx(1.0)
    assert cov['roi_b'] == pytest.approx(0.5)
    assert means['roi_a'] == pytest.approx(60.0)


def test_min_coverage_nans_out_poorly_covered_rois(parcellation_and_labels):
    par, labels = parcellation_and_labels
    img = np.zeros((10, 10, 2), dtype=np.float32)
    img[:, :, 0] = 60.0
    img[:2, :, 1] = 60.0           # 20% coverage

    means = extract_roi_means(img, par, labels, min_coverage=0.5)
    assert means['roi_a'] == pytest.approx(60.0)
    assert np.isnan(means['roi_b'])


def test_explicit_coverage_mask_keeps_genuine_zeros(parcellation_and_labels):
    """A real 0.0 inside the slab is data, not absence.

    MWF returns exact zeros where NNLS finds no short-T2 component. The nonzero
    fallback cannot tell those from out-of-slab voxels; an explicit mask can.
    """
    par, labels = parcellation_and_labels
    img = np.zeros((10, 10, 2), dtype=np.float32)
    img[:, :, 0] = 0.2
    img[:5, :, 0] = 0.0            # genuine in-slab zeros

    mask = np.zeros((10, 10, 2), dtype=bool)
    mask[..., 0] = True            # the whole of roi_a was acquired

    with_mask = extract_roi_means(img, par, labels, coverage_mask=mask)
    without = extract_roi_means(img, par, labels)

    # Mean over all 100 acquired voxels: half at 0.0, half at 0.2.
    assert with_mask['roi_a'] == pytest.approx(0.1)
    # The fallback drops the genuine zeros and overestimates.
    assert without['roi_a'] == pytest.approx(0.2)


def test_nonfinite_voxels_are_excluded(parcellation_and_labels):
    par, labels = parcellation_and_labels
    img = np.zeros((10, 10, 2), dtype=np.float32)
    img[:, :, 0] = 60.0
    img[0, 0, 0] = np.nan

    means, cov = extract_roi_means(img, par, labels, return_coverage=True)
    assert means['roi_a'] == pytest.approx(60.0)
    assert cov['roi_a'] == pytest.approx(0.99)


def test_uncovered_roi_is_nan_not_zero(parcellation_and_labels):
    """An ROI the slab never reached must be missing, never 0.0."""
    par, labels = parcellation_and_labels
    img = np.zeros((10, 10, 2), dtype=np.float32)
    img[:, :, 0] = 60.0            # roi_b entirely outside the slab

    means, cov = extract_roi_means(img, par, labels, return_coverage=True)
    assert np.isnan(means['roi_b'])
    assert cov['roi_b'] == pytest.approx(0.0)
