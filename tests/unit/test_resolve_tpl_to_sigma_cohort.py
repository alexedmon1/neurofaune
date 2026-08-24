"""The cohort resolver that replaced eight open-coded transform lookups.

Every ``propagate_atlas_to_*`` used to build ``templates/anat/{cohort}/transforms``
itself and then name ``tpl-to-SIGMA_0GenericAffine.mat`` outright. That pair --
one study's directory layout plus one study's ANTs output prefix -- is what broke
twice on cuprizone, and it broke quietly: a missing transform makes the caller
skip, not raise.
"""
from pathlib import Path

import pytest

from neurofaune.templates.sigma_warp import (
    resolve_tpl_to_sigma,
    resolve_tpl_to_sigma_for_cohort,
)

CANON = ("tpl-to-SIGMA_0GenericAffine.mat",
         "tpl-to-SIGMA_1Warp.nii.gz",
         "tpl-to-SIGMA_1InverseWarp.nii.gz")
PREFIXED = ("tpl-CPZp60_to-SIGMA_0GenericAffine.mat",
            "tpl-CPZp60_to-SIGMA_1Warp.nii.gz",
            "tpl-CPZp60_to-SIGMA_1InverseWarp.nii.gz")


def _make(d: Path, names) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).touch()
    return d


def test_finds_the_legacy_timepoint_keyed_layout(tmp_path):
    _make(tmp_path / "anat" / "1" / "transforms", CANON)
    r = resolve_tpl_to_sigma_for_cohort(tmp_path, "1")
    assert r["found"]
    assert r["affine"].name == "tpl-to-SIGMA_0GenericAffine.mat"
    assert r["inverse_warp"].name == "tpl-to-SIGMA_1InverseWarp.nii.gz"


def test_finds_transforms_sitting_beside_the_template(tmp_path):
    """ANTs writes next to the moving image; the transforms/ subdir is a convention."""
    _make(tmp_path / "anat" / "1", CANON)
    assert resolve_tpl_to_sigma_for_cohort(tmp_path, "1")["found"]


def test_finds_the_study_prefixed_names(tmp_path):
    """The spelling ANTs actually produced on cuprizone, which broke the hardcoded lookup."""
    _make(tmp_path / "anat" / "2", PREFIXED)
    r = resolve_tpl_to_sigma_for_cohort(tmp_path, "2")
    assert r["affine"].name == "tpl-CPZp60_to-SIGMA_0GenericAffine.mat"
    assert r["warp"].name == "tpl-CPZp60_to-SIGMA_1Warp.nii.gz"
    assert r["inverse_warp"].name == "tpl-CPZp60_to-SIGMA_1InverseWarp.nii.gz"


def test_missing_is_reported_not_guessed(tmp_path):
    r = resolve_tpl_to_sigma_for_cohort(tmp_path, "1")
    assert r["found"] is False
    assert r["affine"] is None and r["inverse_warp"] is None
    # the searched list is what the caller puts in its error message
    assert [Path(d).name for d in r["searched"]] == ["transforms", "1"]


def test_does_not_borrow_another_timepoints_transform(tmp_path):
    """The dangerous failure is finding the WRONG warp, not finding none.

    A session warped with a different timepoint's transform produces numbers that
    look entirely plausible, so the resolver must not scan sibling cohorts.
    """
    _make(tmp_path / "anat" / "1" / "transforms", CANON)
    assert resolve_tpl_to_sigma_for_cohort(tmp_path, "3")["found"] is False


def test_affine_only_registration_yields_no_warps(tmp_path):
    _make(tmp_path / "anat" / "1" / "transforms", ("tpl-to-SIGMA_0GenericAffine.mat",))
    r = resolve_tpl_to_sigma_for_cohort(tmp_path, "1")
    assert r["found"] and r["warp"] is None and r["inverse_warp"] is None


def test_inverse_warp_is_paired_to_its_own_affine(tmp_path):
    """Two registrations in one directory must not be cross-paired."""
    d = _make(tmp_path / "anat" / "1" / "transforms", PREFIXED)
    (d / "tpl-CPZp120_to-SIGMA_1InverseWarp.nii.gz").touch()
    r = resolve_tpl_to_sigma_for_cohort(tmp_path, "1")
    assert r["inverse_warp"].name.startswith("tpl-CPZp60_")


@pytest.mark.parametrize("key", ["affine", "warp", "inverse_warp", "found", "searched"])
def test_result_shape_is_stable(tmp_path, key):
    """Callers index this dict directly, so the keys are part of the contract."""
    assert key in resolve_tpl_to_sigma(candidate_dirs=[tmp_path])
