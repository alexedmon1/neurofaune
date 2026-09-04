"""Tests for DWI→template registration options and the FA template builder.

Covers the cross-contrast trap: FA and T2w have an inverted white-matter
relationship, and ``antsRegistrationSyN.sh`` drives SyN with cross-correlation
and exposes no metric flag, so a deformable run must not go through it.
"""
from pathlib import Path

import numpy as np
import pytest

from neurofaune.preprocess.workflows.dwi_preprocess import (
    _build_syn_registration_cmd,
)
from neurofaune.templates.builder import build_dwi_template


# --- SyN command construction -----------------------------------------------

def test_syn_cmd_uses_antsregistration_not_the_syn_wrapper():
    """The wrapper cannot express a metric, so a deformable run must not use it."""
    cmd = _build_syn_registration_cmd(
        Path("/tpl.nii.gz"), Path("/fa.nii.gz"), Path("/out/pre_")
    )
    assert cmd[0] == "antsRegistration"
    assert not any("antsRegistrationSyN" in c for c in cmd)


def test_syn_cmd_defaults_to_mutual_information():
    """MI is valid across contrasts; it is the safe default for FA→T2w."""
    cmd = _build_syn_registration_cmd(
        Path("/tpl.nii.gz"), Path("/fa.nii.gz"), Path("/out/pre_")
    )
    syn_metric = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--metric"][-1]
    assert syn_metric.startswith("MI[")


def test_syn_cmd_cc_only_affects_the_deformable_stage():
    """CC is for within-modality use; the linear stages stay on MI regardless."""
    cmd = _build_syn_registration_cmd(
        Path("/tpl.nii.gz"), Path("/fa.nii.gz"), Path("/out/pre_"), metric="CC"
    )
    metrics = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--metric"]
    assert metrics[-1].startswith("CC[")
    assert all(m.startswith("MI[") for m in metrics[:-1])


def test_syn_cmd_has_rigid_affine_and_syn_stages():
    cmd = _build_syn_registration_cmd(
        Path("/tpl.nii.gz"), Path("/fa.nii.gz"), Path("/out/pre_")
    )
    transforms = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--transform"]
    assert [t.split("[")[0] for t in transforms] == ["Rigid", "Affine", "SyN"]


def test_syn_cmd_initialises_by_centre_of_mass():
    """Partial-coverage DWI sits at an arbitrary Z offset from a whole-brain template."""
    cmd = _build_syn_registration_cmd(
        Path("/tpl.nii.gz"), Path("/fa.nii.gz"), Path("/out/pre_")
    )
    init = cmd[cmd.index("--initial-moving-transform") + 1]
    assert init.endswith(",1]")


@pytest.mark.parametrize("bad", ["NCC", "mi2", "", "MSQ"])
def test_syn_cmd_rejects_unknown_metric(bad):
    with pytest.raises(ValueError, match="must be 'MI' or 'CC'"):
        _build_syn_registration_cmd(
            Path("/a"), Path("/b"), Path("/c"), metric=bad
        )


def test_register_fa_to_template_still_defaults_to_affine():
    """Default behaviour is unchanged; nonlinear is opt-in."""
    import inspect
    from neurofaune.preprocess.workflows.dwi_preprocess import (
        register_fa_to_template,
    )

    params = inspect.signature(register_fa_to_template).parameters
    assert params["transform_type"].default == "a"
    assert params["moving_file"].default is None
    assert params["metric"].default == "MI"


# --- FA template builder ----------------------------------------------------

def _make_fa(root: Path, subject: str, session: str) -> Path:
    d = root / subject / session / "dwi"
    d.mkdir(parents=True, exist_ok=True)
    import nibabel as nib

    f = d / f"{subject}_{session}_FA.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(f))
    return f


def test_build_dwi_template_requires_three_inputs(tmp_path):
    deriv = tmp_path / "derivatives"
    _make_fa(deriv, "sub-01", "ses-1")
    with pytest.raises(ValueError, match="at least 3"):
        build_dwi_template(deriv, tmp_path / "out", cohort="p60")


def test_build_dwi_template_honours_exclusions(tmp_path, monkeypatch):
    """QC exclusions matter more for a template than for one session: a blurred
    template degrades every subject registered to it."""
    deriv = tmp_path / "derivatives"
    for s in ("sub-01", "sub-02", "sub-03", "sub-04"):
        _make_fa(deriv, s, "ses-1")

    captured = {}

    def fake_build_template(input_files, output_prefix, **kw):
        captured["inputs"] = list(input_files)
        return {"template": Path(str(output_prefix) + "template0.nii.gz"),
                "work_dir": tmp_path}

    monkeypatch.setattr(
        "neurofaune.templates.builder.build_template", fake_build_template
    )
    res = build_dwi_template(
        deriv, tmp_path / "out", cohort="p60",
        exclude=[("sub-02", "ses-1")],
    )
    assert res["n_inputs"] == 3
    assert not any("sub-02" in str(f) for f in captured["inputs"])


def test_build_dwi_template_caps_by_even_stride(tmp_path, monkeypatch):
    """A cap must not select one end of an ordered cohort."""
    deriv = tmp_path / "derivatives"
    for i in range(10):
        _make_fa(deriv, f"sub-{i:02d}", "ses-1")

    monkeypatch.setattr(
        "neurofaune.templates.builder.build_template",
        lambda input_files, output_prefix, **kw: {
            "template": Path("t.nii.gz"), "work_dir": tmp_path
        },
    )
    res = build_dwi_template(
        deriv, tmp_path / "out", cohort="p60", max_subjects=4
    )
    assert res["n_inputs"] == 4
    names = [f.parent.parent.parent.name for f in res["inputs"]]
    # Spread across the cohort, not the first four.
    assert names != [f"sub-{i:02d}" for i in range(4)]
    assert len(set(names)) == 4
