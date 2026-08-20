"""DKI and NODDI must reach SIGMA space in a normal run.

run_dwi_preprocessing warps to SIGMA before these models are fitted -- it prints
"not yet fitted, skipping" for the seven multishell maps and moves on -- and
nothing warped them afterwards. A cohort run therefore produced 4 of the 11
space-SIGMA maps, and the analysis stage, which reads space-SIGMA_* only, found
no kurtosis and no NODDI at all unless someone remembered to run
backfill_sigma_warps.py.
"""
import json

import numpy as np
import nibabel as nib
import pytest

from neurofaune.templates.sigma_warp import (
    DWI_SIGMA_METRICS,
    MULTISHELL_SIGMA_METRICS,
)
from neurofaune.preprocess.workflows.multishell_models import (
    _warp_multishell_to_sigma,
)

SHAPE = (5, 5, 3)


def _nii(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.ones(SHAPE, np.float32), np.eye(4)), str(p))
    return p


def test_multishell_set_is_exactly_the_post_tensor_metrics():
    """Derived from the full set, so the two cannot drift apart."""
    assert set(MULTISHELL_SIGMA_METRICS) == {
        "MK", "AK", "RK", "KFA", "ODI", "FICVF", "FISO"}
    # tensor metrics are warped earlier, by run_dwi_preprocessing
    assert not {"FA", "MD", "AD", "RD"} & set(MULTISHELL_SIGMA_METRICS)
    for name, pattern in MULTISHELL_SIGMA_METRICS.items():
        assert pattern == DWI_SIGMA_METRICS[name]


def _session(tmp_path, with_sidecar=True):
    deriv = tmp_path / "derivatives" / "sub-1X" / "ses-1" / "dwi"
    prefix = "sub-1X_ses-1"
    for pattern in MULTISHELL_SIGMA_METRICS.values():
        _nii(deriv / pattern.format(prefix=prefix))
    tpl = _nii(tmp_path / "templates" / "p60" / "tpl-CPZp60_T2w_template0.nii.gz")
    if with_sidecar:
        (deriv / f"{prefix}_FA_to_template_registration.json").write_text(json.dumps({
            "template_file": str(tpl),
            "affine_transform": str(tmp_path / "FA_to_template_0GenericAffine.mat"),
        }))
    return deriv


def test_multishell_maps_are_warped_after_fitting(tmp_path, monkeypatch):
    deriv = _session(tmp_path)
    seen = {}

    def fake_targets(config, **kw):
        return {"ready": True, "sigma_template": tmp_path / "sigma.nii.gz",
                "affine": tmp_path / "a.mat", "warp": None, "reason": ""}

    def fake_warp(metric_files, **kw):
        seen.update(metric_files)
        return {k: tmp_path / f"{k}.nii.gz" for k in metric_files}

    monkeypatch.setattr(
        "neurofaune.templates.sigma_warp.sigma_targets_from_config", fake_targets)
    monkeypatch.setattr(
        "neurofaune.templates.sigma_warp.warp_maps_to_sigma", fake_warp)

    out = _warp_multishell_to_sigma(
        deriv, tmp_path, "sub-1X", "ses-1", {"atlas": {}},
        {"dki": True, "noddi": True})

    assert set(seen) == set(MULTISHELL_SIGMA_METRICS), "a fitted map went unwarped"
    assert len(out) == 7


def test_missing_registration_sidecar_warns_and_does_not_raise(tmp_path, caplog):
    """Losing the maps must not also lose the session."""
    deriv = _session(tmp_path, with_sidecar=False)
    out = _warp_multishell_to_sigma(
        deriv, tmp_path, "sub-1X", "ses-1", {"atlas": {}},
        {"dki": True, "noddi": True})
    assert out == {}
    assert any("space-SIGMA" in r.message or "cannot place" in r.message
               for r in caplog.records)


def test_no_config_skips_rather_than_crashing(tmp_path):
    deriv = _session(tmp_path)
    assert _warp_multishell_to_sigma(
        deriv, tmp_path, "sub-1X", "ses-1", None, {"dki": True}) == {}


def test_nothing_fitted_means_nothing_to_warp(tmp_path):
    deriv = _session(tmp_path)
    assert _warp_multishell_to_sigma(
        deriv, tmp_path, "sub-1X", "ses-1", {"atlas": {}},
        {"dki": None, "noddi": None}) == {}
