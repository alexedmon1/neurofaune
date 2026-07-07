"""Unit tests for cross-sectional / longitudinal registration-consistency QC.

Hermetic: the two measures are pure Dice over binary masks, so we write tiny
synthetic NIfTI masks to tmp_path — no real data, no ANTs.
"""
import numpy as np
import nibabel as nib
import pytest

from neurofaune.templates.consistency_qc import cross_sectional_dice, longitudinal_dice


def _mask(tmp_path, name, block):
    """Write a 6x6x6 volume with `block` (a slice tuple) set to 1; return its path."""
    arr = np.zeros((6, 6, 6), dtype=np.uint8)
    arr[block] = 1
    p = tmp_path / f"{name}.nii.gz"
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(p))
    return p


def test_cross_sectional_identical_masks_dice_one(tmp_path):
    full = np.s_[:, :, :]
    warped = {"sub-a": _mask(tmp_path, "a", full), "sub-b": _mask(tmp_path, "b", full)}
    res = cross_sectional_dice(warped, atlas_mask=_mask(tmp_path, "atlas", full))
    assert res["n_subjects"] == 2
    assert res["mean_pairwise"] == pytest.approx(1.0)
    assert res["mean_vs_atlas"] == pytest.approx(1.0)
    assert res["pairwise"][0][:2] == ("sub-a", "sub-b")


def test_cross_sectional_half_overlap(tmp_path):
    # two masks of equal size (108 vox) sharing half -> Dice = 2*54/(108+108)=0.5
    warped = {
        "sub-a": _mask(tmp_path, "a", np.s_[0:3, :, :]),
        "sub-b": _mask(tmp_path, "b", np.s_[0:3, :, :]),  # identical -> 1.0
    }
    res = cross_sectional_dice(warped)
    assert res["mean_pairwise"] == pytest.approx(1.0)
    assert res["vs_atlas"] == {}  # no atlas passed


def test_cross_sectional_known_partial(tmp_path):
    a = _mask(tmp_path, "a", np.s_[0:3, :, :])  # 108 vox
    b = _mask(tmp_path, "b", np.s_[2:5, :, :])  # 108 vox, overlap at x=2 -> 36 vox
    res = cross_sectional_dice({"sub-a": a, "sub-b": b})
    # Dice = 2*36/(108+108) = 0.3333
    assert res["pairwise"][0][2] == pytest.approx(2 * 36 / (108 + 108))


def test_longitudinal_pairs_and_values(tmp_path):
    warped = {
        "ses-1": _mask(tmp_path, "s1", np.s_[0:3, :, :]),
        "ses-2": _mask(tmp_path, "s2", np.s_[0:3, :, :]),   # identical to ses-1
        "ses-3": _mask(tmp_path, "s3", np.s_[3:6, :, :]),   # disjoint from ses-1/2
    }
    pairs = longitudinal_dice(warped)
    d = {f"{a}_{b}": v for a, b, v in pairs}
    assert set(d) == {"ses-1_ses-2", "ses-1_ses-3", "ses-2_ses-3"}
    assert d["ses-1_ses-2"] == pytest.approx(1.0)
    assert d["ses-1_ses-3"] == pytest.approx(0.0)
