"""The work-dir cache guard behind the BOLD registration reference.

Regression cover for a silent-staleness bug: work/ survives between runs, so an
"if not exists" guard reused the previous run's registration reference. A cohort
re-run that enabled functional.second_mask therefore produced correctly refined
brain masks while still registering against the old mask's reference image,
which retained 11-31% non-brain tissue.
"""
import os

import pytest

from neurofaune.preprocess.workflows.func_preprocess import cache_is_stale


def _write(p, mtime=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def test_missing_cache_is_stale(tmp_path):
    assert cache_is_stale(tmp_path / "nope.nii.gz", [])


def test_cache_newer_than_inputs_is_fresh(tmp_path):
    src = _write(tmp_path / "mask.nii.gz", mtime=1000)
    cache = _write(tmp_path / "ref.nii.gz", mtime=2000)
    assert not cache_is_stale(cache, [src])


def test_cache_older_than_any_input_is_stale(tmp_path):
    # the real case: the brain mask was rewritten by a later run
    bold = _write(tmp_path / "bold_mcf.nii.gz", mtime=1000)
    mask = _write(tmp_path / "mask.nii.gz", mtime=3000)
    cache = _write(tmp_path / "ref.nii.gz", mtime=2000)
    assert cache_is_stale(cache, [bold, mask])
    # and it is the mask alone that makes it stale
    assert not cache_is_stale(cache, [bold])


def test_missing_input_does_not_make_cache_stale(tmp_path):
    cache = _write(tmp_path / "ref.nii.gz", mtime=2000)
    assert not cache_is_stale(cache, [tmp_path / "absent.nii.gz"])


def test_no_inputs_means_fresh_when_present(tmp_path):
    cache = _write(tmp_path / "ref.nii.gz", mtime=2000)
    assert not cache_is_stale(cache, [])


@pytest.mark.parametrize("as_str", [True, False])
def test_accepts_str_and_path(tmp_path, as_str):
    src = _write(tmp_path / "mask.nii.gz", mtime=3000)
    cache = _write(tmp_path / "ref.nii.gz", mtime=2000)
    c = str(cache) if as_str else cache
    s = [str(src)] if as_str else [src]
    assert cache_is_stale(c, s)
