"""Config-driven pieces of anat_preprocess that differ between species (hermetic)."""
import numpy as np

from neurofaune.preprocess.qc.anat import anat_qc
from neurofaune.preprocess.workflows import anat_preprocess as ap


def test_n4_bspline_distance_is_passed_only_when_set(monkeypatch, tmp_path):
    seen = {}

    class FakeN4:
        def __init__(self):
            self.inputs = type("I", (), {})()

        def run(self):
            seen.update(vars(self.inputs))

    monkeypatch.setattr(ap.ants, "N4BiasFieldCorrection", FakeN4)
    ap.bias_field_correction(tmp_path / "in.nii.gz", tmp_path / "out.nii.gz")
    assert "bspline_fitting_distance" not in seen          # rat: ANTs default
    seen.clear()
    ap.bias_field_correction(tmp_path / "in.nii.gz", tmp_path / "out.nii.gz", bspline_fitting_distance=100)
    assert seen["bspline_fitting_distance"] == 100.0


def test_stripping_qc_range_is_configurable():
    t2w = np.ones((10, 10, 10))
    mask = np.zeros_like(t2w, bool)
    mask[:, :, :1] = True                                  # brain = 10% of non-zero
    rat = anat_qc._calculate_skull_strip_metrics(t2w, t2w * mask, mask)
    mouse = anat_qc._calculate_skull_strip_metrics(t2w, t2w * mask, mask, (0.05, 0.30))
    assert rat["potential_over_stripping"] and not mouse["potential_over_stripping"]
