"""Unit test for content-based cut selection in func registration QC.

Pins the fix for blank BOLD->template overlays: in scaled rodent template space
the brain sits far from the world origin, so the old fixed cuts at [-2, 0, 2] mm
sliced empty space. _brain_cut_coords must return cuts through actual content.
"""
import numpy as np
import nibabel as nib

from neurofaune.preprocess.qc.func.registration_qc import _brain_cut_coords


def _offset_brain(tmp_path):
    # brain blob centered far from the world origin (like the scaled template)
    data = np.zeros((60, 60, 40), dtype=np.float32)
    data[25:35, 25:35, 18:26] = 100.0
    affine = np.diag([1.2, 1.2, 8.0, 1.0])
    affine[:3, 3] = [100.0, 100.0, 120.0]  # push origin far from brain
    img = nib.Nifti1Image(data, affine)
    return img


def test_cuts_follow_content_not_origin(tmp_path):
    img = _offset_brain(tmp_path)
    zc = _brain_cut_coords(img, 'z')
    xc = _brain_cut_coords(img, 'x')
    assert len(zc) == 3 and len(xc) == 3
    # cuts must land near the brain (world ~120mm in z, ~130mm in x), not the
    # origin — the old [-2, 0, 2] would be ~120mm away and render blank
    assert all(np.isfinite(zc)) and min(zc) > 50, f"z cuts near origin: {zc}"
    assert all(np.isfinite(xc)) and min(xc) > 50, f"x cuts near origin: {xc}"


def test_fallback_is_bounded(monkeypatch):
    # if find_cut_slices blows up, fall back to the legacy fixed cuts (not a crash)
    import neurofaune.preprocess.qc.func.registration_qc as rq
    import nilearn.plotting as p
    monkeypatch.setattr(p, "find_cut_slices", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    assert _brain_cut_coords(_offset_brain(None), 'z') == [-2, 0, 2]
