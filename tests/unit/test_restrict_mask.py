"""Unit tests for ``restrict_mask_to`` — the post-eddy mask refinement.

Pins the property the whole design rests on: the refinement INTERSECTS, so it
can only ever remove. Replacing outright was measured on this cohort's DWI and
adds 46.5% more voxels at a third of brain-core intensity and 9x the rate of
degenerate FA — signal-free dorsal dropout the propagated structural mask covers
but the EPI cannot see.
"""
import nibabel as nib
import numpy as np
import pytest

from neurofaune.preprocess.utils.registration_utils import restrict_mask_to

SHAPE = (10, 10, 4)


def _save(arr, path):
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), np.eye(4)), path)
    return path


def _pair(tmp_path):
    base = np.zeros(SHAPE, bool)
    base[2:8, 2:8, 1:3] = True          # "brain" + some residue
    base[0:2, 0:2, 1:3] = True          # detached non-brain blob (muscle)

    limit = np.zeros(SHAPE, bool)
    limit[2:8, 2:8, 1:3] = True         # anatomical mask: brain only
    limit[8:10, 8:10, 1:3] = True       # ...plus territory the EPI cannot see

    return (_save(base, tmp_path / "base.nii.gz"),
            _save(limit, tmp_path / "limit.nii.gz"), base, limit)


def test_result_is_a_subset_of_the_base_mask(tmp_path):
    """The invariant: never add. Guards against reintroducing replace."""
    b, l, base, limit = _pair(tmp_path)
    out = tmp_path / "out.nii.gz"
    restrict_mask_to(b, l, out)
    got = nib.load(out).get_fdata() > 0

    assert not (got & ~base).any(), "refinement must never add voxels"
    assert np.array_equal(got, base & limit)


def test_removes_the_non_brain_blob_and_keeps_the_brain(tmp_path):
    b, l, base, limit = _pair(tmp_path)
    out = tmp_path / "out.nii.gz"
    info = restrict_mask_to(b, l, out)
    got = nib.load(out).get_fdata() > 0

    assert got[2:8, 2:8, 1:3].all()          # brain retained
    assert not got[0:2, 0:2, 1:3].any()      # detached blob stripped
    assert info["n_removed"] == 2 * 2 * 2
    assert info["n_after"] == info["n_before"] - info["n_removed"]
    assert 0 < info["fraction_removed"] < 1


def test_does_not_import_territory_the_base_mask_excludes(tmp_path):
    """The dropout region: limiting mask covers it, EPI mask does not."""
    b, l, base, limit = _pair(tmp_path)
    out = tmp_path / "out.nii.gz"
    restrict_mask_to(b, l, out)
    got = nib.load(out).get_fdata() > 0

    assert not got[8:10, 8:10, 1:3].any()


def test_in_place_overwrite_is_supported(tmp_path):
    b, l, base, limit = _pair(tmp_path)
    info = restrict_mask_to(b, l, b)          # out == base
    assert (nib.load(b).get_fdata() > 0).sum() == info["n_after"]


def test_identical_masks_remove_nothing(tmp_path):
    b, _, base, _ = _pair(tmp_path)
    out = tmp_path / "out.nii.gz"
    info = restrict_mask_to(b, b, out)
    assert info["n_removed"] == 0
    assert info["fraction_removed"] == 0.0


def test_shape_mismatch_raises(tmp_path):
    b, _, _, _ = _pair(tmp_path)
    bad = _save(np.ones((4, 4, 2), bool), tmp_path / "bad.nii.gz")
    with pytest.raises(ValueError, match="shape mismatch"):
        restrict_mask_to(b, bad, tmp_path / "out.nii.gz")
