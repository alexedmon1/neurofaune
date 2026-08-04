"""Unit tests for SIGMA-space warping.

Group analysis globs ``space-SIGMA_*`` out of derivatives rather than warping
anything itself, so if preprocessing silently skips this step the analysis stage
finds nothing. These tests pin the two ways that happened in practice: a
transform resolver that only looked in one place, and outputs that were skipped
because a stale copy already existed.
"""
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurofaune.templates.sigma_warp import (
    DWI_SIGMA_METRICS,
    MSME_SIGMA_METRICS,
    build_metric_files,
    resolve_tpl_to_sigma,
    warp_maps_to_sigma,
)

SHAPE = (6, 6, 3)


def _nii(path, shape=SHAPE):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.ones(shape, np.float32), np.eye(4)), path)
    return path


# --------------------------------------------------------------- resolver ---

def test_resolver_finds_transforms_beside_the_template(tmp_path):
    tpl = tmp_path / "p60" / "tpl.nii.gz"
    _nii(tpl)
    tx = tmp_path / "p60" / "transforms"
    tx.mkdir()
    (tx / "tpl-to-SIGMA_0GenericAffine.mat").touch()
    (tx / "tpl-to-SIGMA_1Warp.nii.gz").touch()

    got = resolve_tpl_to_sigma(template_file=tpl)
    assert got["found"] and got["warp"] is not None


def test_resolver_finds_transforms_in_a_candidate_dir(tmp_path):
    """The real layout: transforms keyed by timepoint, not beside the template."""
    tpl = tmp_path / "p60" / "tpl.nii.gz"
    _nii(tpl)
    tx = tmp_path / "anat" / "1" / "transforms"
    tx.mkdir(parents=True)
    (tx / "tpl-to-SIGMA_0GenericAffine.mat").touch()

    got = resolve_tpl_to_sigma(template_file=tpl, candidate_dirs=[tx])
    assert got["found"]
    assert got["affine"].parent == tx
    assert got["warp"] is None          # affine-only registration


def test_resolver_reports_failure_and_what_it_tried(tmp_path):
    """Must not fail silently — that is how a cohort ends up with no outputs."""
    tpl = tmp_path / "p60" / "tpl.nii.gz"
    _nii(tpl)
    got = resolve_tpl_to_sigma(template_file=tpl,
                               candidate_dirs=[tmp_path / "nowhere"])
    assert not got["found"]
    assert got["affine"] is None
    assert len(got["searched"]) == 2      # candidate + beside-template


# ------------------------------------------------------------ metric sets ---

def test_dwi_metric_set_covers_tensor_kurtosis_and_noddi():
    """ODI/FICVF are the study's primary endpoints; they must be warped."""
    assert {"FA", "MD", "AD", "RD"} <= set(DWI_SIGMA_METRICS)
    assert {"MK", "AK", "RK", "KFA"} <= set(DWI_SIGMA_METRICS)
    assert {"ODI", "FICVF", "FISO"} <= set(DWI_SIGMA_METRICS)


def test_msme_metric_set(tmp_path):
    assert set(MSME_SIGMA_METRICS) == {"T2", "MWF", "IWF", "CSFF"}


def test_build_metric_files_skips_absent_maps(tmp_path):
    _nii(tmp_path / "sub-1X_ses-1_T2.nii.gz")
    _nii(tmp_path / "sub-1X_ses-1_MWF.nii.gz")
    got = build_metric_files(tmp_path, "sub-1X_ses-1", MSME_SIGMA_METRICS)
    assert set(got) == {"T2", "MWF"}


# ----------------------------------------------------------------- warping ---

@pytest.fixture
def warp_env(tmp_path, monkeypatch):
    """Stub antsApplyTransforms; record argv and emit the output file."""
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        out = cmd[cmd.index("-o") + 1]
        _nii(Path(out))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("neurofaune.templates.sigma_warp.subprocess.run", fake_run)
    sigma = _nii(tmp_path / "sigma.nii.gz")
    mov = tmp_path / "FA_to_template_0GenericAffine.mat"
    mov.touch()
    aff = tmp_path / "tpl-to-SIGMA_0GenericAffine.mat"
    aff.touch()
    wrp = _nii(tmp_path / "tpl-to-SIGMA_1Warp.nii.gz")
    return calls, sigma, mov, aff, wrp


def test_transform_order_is_reversed_for_ants(tmp_path, warp_env):
    """ANTs applies the LAST -t first, so modality->template must come last."""
    calls, sigma, mov, aff, wrp = warp_env
    fa = _nii(tmp_path / "in" / "sub-1X_ses-1_FA.nii.gz")
    warp_maps_to_sigma({"FA": fa}, mov, sigma, tmp_path / "out",
                       "sub-1X", "ses-1", aff, wrp)
    ts = [c for i, c in enumerate(calls[0]) if calls[0][i - 1] == "-t"]
    assert ts == [str(wrp), str(aff), str(mov)]


def test_output_naming_metric_and_bold_styles(tmp_path, warp_env):
    calls, sigma, mov, aff, wrp = warp_env
    src = _nii(tmp_path / "in" / "x.nii.gz")

    got = warp_maps_to_sigma({"FA": src}, mov, sigma, tmp_path / "o1",
                             "sub-1X", "ses-1", aff, wrp)
    assert got["FA"].name == "sub-1X_ses-1_space-SIGMA_FA.nii.gz"

    got = warp_maps_to_sigma({"bold": src}, mov, sigma, tmp_path / "o2",
                             "sub-1X", "ses-1", aff, wrp, suffix_style="bold")
    assert got["bold"].name == "sub-1X_ses-1_space-SIGMA_bold.nii.gz"

    got = warp_maps_to_sigma({"fALFF": src}, mov, sigma, tmp_path / "o3",
                             "sub-1X", "ses-1", aff, wrp, suffix_style="bold")
    # what roi_extraction globs for functional metrics
    assert got["fALFF"].name == "sub-1X_ses-1_space-SIGMA_desc-fALFF_bold.nii.gz"


def test_force_rewrites_stale_outputs(tmp_path, warp_env):
    """Default skip-if-exists silently keeps stale maps after a refit."""
    calls, sigma, mov, aff, wrp = warp_env
    src = _nii(tmp_path / "in" / "x.nii.gz")
    out = tmp_path / "out"
    warp_maps_to_sigma({"FA": src}, mov, sigma, out, "sub-1X", "ses-1", aff, wrp)
    assert len(calls) == 1

    warp_maps_to_sigma({"FA": src}, mov, sigma, out, "sub-1X", "ses-1", aff, wrp)
    assert len(calls) == 1, "second call should have skipped"

    warp_maps_to_sigma({"FA": src}, mov, sigma, out, "sub-1X", "ses-1", aff, wrp,
                       force=True)
    assert len(calls) == 2, "force must rewrite"


def test_affine_only_chain_omits_the_warp(tmp_path, warp_env):
    calls, sigma, mov, aff, _ = warp_env
    src = _nii(tmp_path / "in" / "x.nii.gz")
    warp_maps_to_sigma({"FA": src}, mov, sigma, tmp_path / "out",
                       "sub-1X", "ses-1", aff, None)
    ts = [c for i, c in enumerate(calls[0]) if calls[0][i - 1] == "-t"]
    assert ts == [str(aff), str(mov)]


def test_4d_input_gets_timeseries_flag(tmp_path, warp_env):
    """A 4D BOLD needs -e 3 or ANTs mangles it."""
    calls, sigma, mov, aff, wrp = warp_env
    bold = _nii(tmp_path / "in" / "bold.nii.gz", shape=SHAPE + (5,))
    warp_maps_to_sigma({"bold": bold}, mov, sigma, tmp_path / "out",
                       "sub-1X", "ses-1", aff, wrp, suffix_style="bold")
    assert "-e" in calls[0] and calls[0][calls[0].index("-e") + 1] == "3"


def test_missing_input_is_skipped_not_fatal(tmp_path, warp_env):
    calls, sigma, mov, aff, wrp = warp_env
    got = warp_maps_to_sigma({"FA": tmp_path / "nope.nii.gz"}, mov, sigma,
                             tmp_path / "out", "sub-1X", "ses-1", aff, wrp)
    assert got == {} and calls == []
