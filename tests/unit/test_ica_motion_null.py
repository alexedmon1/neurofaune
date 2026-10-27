"""The other-animal null for the ICA motion criterion (functional.denoising.ica.motion_null).

A slow component correlates with an animal's slow motion drifts whether or not it is motion; a
fixed |r| > 0.5 then removes it. The null asks instead whether the correlation beats what other
animals' motion -- the same slowness, no temporal relation to this scan -- gives that component.
"""
import nibabel as nib
import numpy as np
import pytest

from neurofaune.preprocess.utils.func.ica_denoising import (
    MOTION_PARAMS_SUFFIX,
    classify_ica_components,
    load_other_animal_motion,
    motion_null_thresholds,
    write_motion_params,
)
from neurofaune.preprocess.workflows.func_preprocess import ica_motion_null_kwargs

N = 360


def _walks(rng, k):
    return np.cumsum(rng.standard_normal((N, k)), axis=0)


def _case(seed=1):
    """Own motion, 40 other animals' motion, and two components: an unrelated slow random walk
    (|r| with own motion 0.74 at this seed) and one that tracks a motion parameter."""
    rng = np.random.default_rng(seed)
    own = _walks(rng, 6)
    others = [_walks(rng, 6) for _ in range(40)]
    slow = _walks(rng, 1)[:, 0]
    motiony = own[:, 2] + 0.3 * rng.standard_normal(N) * own[:, 2].std()
    return own, others, np.column_stack([slow, motiony])


def _melodic_dir(tmp_path, mix, own):
    d = tmp_path / "melodic"
    d.mkdir()
    ic = np.random.default_rng(0).standard_normal((6, 6, 4, mix.shape[1])).astype(np.float32)
    nib.save(nib.Nifti1Image(ic, np.eye(4)), str(d / "melodic_IC.nii.gz"))
    np.savetxt(d / "melodic_mix", mix)
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((6, 6, 4), np.uint8), np.eye(4)), str(mask))
    par = tmp_path / "own.par"
    np.savetxt(par, own)
    return d, mask, par


def test_fixed_rule_removes_an_unrelated_slow_component_and_the_null_keeps_it(tmp_path):
    own, others, mix = _case()
    d, mask, par = _melodic_dir(tmp_path, mix, own)
    fixed = classify_ica_components(d, par, mask, tr=2.0)
    null = classify_ica_components(d, par, mask, tr=2.0, motion_null=others, motion_null_quantile=0.95)
    slow_fixed, motion_fixed = fixed["components"]
    slow_null, motion_null = null["components"]
    assert slow_fixed["motion_correlation"] > 0.5            # the precondition: a spurious |r| above 0.5
    assert slow_fixed["label"] == "noise"                    # ... which the fixed rule removes
    assert slow_null["label"] == "signal"                    # the null keeps it
    assert slow_null["motion_correlation"] < slow_null["motion_null_threshold"]
    assert motion_fixed["label"] == "noise" and motion_null["label"] == "noise"   # real motion goes either way
    assert "other-animal null" in " ".join(motion_null["classification_reason"])
    assert null["motion_null"]["sessions"] == 40 and null["motion_null"]["quantile"] == 0.95
    assert fixed["motion_null"] is None and slow_fixed["motion_null_threshold"] is None


def test_thresholds_are_per_component_quantiles_of_max_abs_r():
    own, others, mix = _case()
    thr = motion_null_thresholds(mix, others, 0.95)
    for k in range(mix.shape[1]):
        r = [np.abs(np.corrcoef(mix[:, k], m.T)[0, 1:]).max() for m in others]
        assert thr[k] == pytest.approx(np.quantile(r, 0.95), abs=1e-9)
    assert (motion_null_thresholds(mix, others, 0.5) <= thr).all()


def test_too_few_null_sessions_stop_the_classification(tmp_path):
    own, others, mix = _case()
    d, mask, par = _melodic_dir(tmp_path, mix, own)
    with pytest.raises(RuntimeError, match="motion null"):
        classify_ica_components(d, par, mask, tr=2.0, motion_null=others[:5], motion_null_min_sessions=20)


def test_motion_params_round_trip_and_column_check(tmp_path):
    par = tmp_path / "x.par"
    m = np.random.default_rng(2).standard_normal((N, 6))
    np.savetxt(par, m)
    out = write_motion_params(par, tmp_path / "d" / f"sub-1_ses-1{MOTION_PARAMS_SUFFIX}")
    assert out.read_text().splitlines()[0].split("\t") == ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    assert np.allclose(np.loadtxt(out, skiprows=1), m, rtol=1e-6, atol=1e-7)
    np.savetxt(par, m[:, :5])
    with pytest.raises(ValueError, match="expected 6"):
        write_motion_params(par, tmp_path / "bad.tsv")


def _derivatives(root, subjects, sessions=("ses-1", "ses-2")):
    rng = np.random.default_rng(3)
    root.mkdir(parents=True, exist_ok=True)
    for s in subjects:
        for e in sessions:
            par = root / "tmp.par"
            np.savetxt(par, rng.standard_normal((N, 6)))
            write_motion_params(par, root / s / e / "func" / f"{s}_{e}{MOTION_PARAMS_SUFFIX}")
    return root


def test_the_null_leaves_out_the_same_animal_and_needs_enough_sessions(tmp_path):
    root = _derivatives(tmp_path / "derivatives", [f"sub-{i}" for i in range(12)])
    null = load_other_animal_motion(root, "sub-3", min_sessions=20)
    assert len(null["motion"]) == 22 and not any(s.startswith("sub-3_") for s in null["sessions"])
    with pytest.raises(RuntimeError, match="first pass"):
        load_other_animal_motion(root, "sub-3", min_sessions=23)


def test_config_switch(tmp_path):
    _derivatives(tmp_path / "derivatives", [f"sub-{i}" for i in range(12)])
    assert ica_motion_null_kwargs({}, tmp_path, "sub-0") == {}
    assert ica_motion_null_kwargs({"motion_null": {"enabled": False}}, tmp_path, "sub-0") == {}
    kw = ica_motion_null_kwargs({"motion_null": {"enabled": True, "quantile": 0.9}}, tmp_path, "sub-0")
    assert len(kw["motion_null"]) == 22 and kw["motion_null_quantile"] == 0.9 and kw["motion_null_min_sessions"] == 20
    with pytest.raises(RuntimeError):
        ica_motion_null_kwargs({"motion_null": {"enabled": True, "min_sessions": 40}}, tmp_path, "sub-0")
