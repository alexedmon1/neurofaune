"""The study-space atlas is a rotation of the atlas, never a mirror.

Until 2026-10-06 setup_study_atlas reoriented SIGMA by a transpose and two flips --
three reflections -- so a study-space atlas named every hemisphere by the other's name
(the cuprizone study's F021). Synthetic atlas: RAS header, a left-hemisphere label at low
x, a superior marker at high z, an anterior marker at high y.
"""
import json

import nibabel as nib
import numpy as np
import pytest

from neurofaune.templates import slice_registration as sr

LABELS = "SIGMA_Rat_Brain_Atlases/SIGMA_Anatomical_Atlas/InVivo_Atlas/SIGMA_InVivo_Anatomical_Brain_Atlas.nii"


def _atlas(path, shape=(10, 14, 8)):
    a = np.zeros(shape, np.int16)
    a[1:3, 6:8, 3:5] = 1        # left hemisphere (low x in RAS)
    a[7:9, 6:8, 3:5] = 2        # right hemisphere
    a[4:6, 6:8, 7] = 3          # superior marker (top z)
    a[4:6, 13, 3:5] = 4         # anterior marker (front y)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(a, np.diag([1.0, 2.0, 3.0, 1.0])), str(path))
    return path


def _where(a, label, axis):
    return np.argwhere(a == label)[:, axis].mean()


def test_right_handed_codes():
    assert sr.is_right_handed("RAS") and sr.is_right_handed("RIA") and sr.is_right_handed("LPS")
    assert not sr.is_right_handed("LIA") and not sr.is_right_handed("LAS")


def test_a_mirror_target_is_refused():
    with pytest.raises(ValueError, match="mirror"):
        sr.reorientation("RAS", "LIA")


def test_reorienting_to_RIA_keeps_left_on_the_left(tmp_path):
    src = _atlas(tmp_path / "atlas.nii.gz")
    out, aff = sr.reorient_sigma_to_study(src, study_axes="RIA")
    assert out.shape == (10, 8, 14)                         # (x, z, y)
    assert np.allclose(np.diag(aff)[:3], [1.0, 3.0, 2.0])   # voxel sizes follow their axes
    assert _where(out, 1, 0) < _where(out, 2, 0)            # left below right along axis 0 (R)
    assert _where(out, 3, 1) < 1                            # superior at LOW axis 1 (axis 1 runs inferior)
    assert _where(out, 4, 2) > 12                           # anterior at HIGH axis 2


def test_the_old_reorientation_was_a_mirror_and_is_detected(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    (old / "atlas_metadata.json").write_text(json.dumps({"transformation": {
        "step1": "transpose(0, 2, 1) - swap Y and Z axes", "step2": "flip(axis=0) - flip X axis",
        "step3": "flip(axis=1) - flip Y axis"}}))
    assert sr.study_atlas_is_mirrored(old) is True
    assert sr.study_atlas_is_mirrored(tmp_path / "nothing") is None
    with pytest.raises(RuntimeError, match="mirroring"):
        sr.setup_study_atlas(tmp_path / "sigma", old)


def test_setup_records_a_rotation_and_the_declared_orientation(tmp_path):
    sigma = tmp_path / "sigma"
    _atlas(sigma / LABELS)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("atlas:\n  name: SIGMA\n")
    out = tmp_path / "study_space"
    sr.setup_study_atlas(sigma, out, config_path=cfg)
    meta = json.loads((out / "atlas_metadata.json").read_text())
    assert meta["study_axes"] == "RIA" and meta["source_axes"] == "RAS"
    assert meta["reorientation_determinant"] == 1
    assert sr.study_atlas_is_mirrored(out) is False
    a = np.asarray(nib.load(str(out / "SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz")).dataobj)
    assert _where(a, 1, 0) < _where(a, 2, 0)
    import yaml
    space = yaml.safe_load(cfg.read_text())["atlas"]["study_space"]
    assert space["axes"] == "RIA" and space["display_plane"] == "coronal"
    sr.setup_study_atlas(sigma, out)                          # a correct atlas may be rebuilt


def test_replacing_an_old_atlas_needs_saying_so(tmp_path):
    sigma = tmp_path / "sigma"
    _atlas(sigma / LABELS)
    old = tmp_path / "study_space"
    old.mkdir()
    (old / "atlas_metadata.json").write_text(json.dumps({"transformation": {
        "a": "transpose(0, 2, 1)", "b": "flip(axis=0)", "c": "flip(axis=1)"}}))
    sr.setup_study_atlas(sigma, old, replace_mirrored=True)
    assert sr.study_atlas_is_mirrored(old) is False
