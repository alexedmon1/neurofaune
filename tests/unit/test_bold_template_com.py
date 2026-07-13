"""Pin the in-plane COM correction in BOLD->template registration.

The registration used to hard-code Tx=Ty=0 (assuming BOLD and template share an
in-plane origin); when the EPI and structural FOVs are not co-centred this left a
systematic in-plane offset. The fix aligns brain centroids in X/Y. This test pins
the translation the .mat gets written with (values + the ITK LPS X/Y sign), with
the NCC scan and antsApplyTransforms mocked out.
"""
import numpy as np
import nibabel as nib
import scipy.io as sio
import pytest
from scipy import ndimage

import neurofaune.preprocess.workflows.func_preprocess as fp


def _brain(shape, center, affine):
    d = np.zeros(shape, np.float32)
    cx, cy, cz = center
    d[cx - 3:cx + 3, cy - 3:cy + 3, cz - 2:cz + 2] = 100.0
    return nib.Nifti1Image(d, affine)


def _com_world(img):
    d = img.get_fdata()
    c = ndimage.center_of_mass(d > 0.1 * d.max())
    return (img.affine @ np.array([c[0], c[1], c[2], 1.0]))[:3]


def test_inplane_translation_from_com(tmp_path, monkeypatch):
    aff_bold = np.diag([4.0, 4.0, 6.0, 1.0])
    aff_tpl = np.diag([1.25, 1.25, 8.0, 1.0])
    bold = tmp_path / "bold.nii.gz"
    tpl = tmp_path / "tpl.nii.gz"
    nib.save(_brain((40, 40, 20), (30, 30, 10), aff_bold), bold)
    nib.save(_brain((256, 256, 41), (100, 90, 18), aff_tpl), tpl)

    monkeypatch.setattr(fp, "_find_z_offset_ncc",
                        lambda *a, **k: (None, {"z_offset_mm": 20.0}))
    monkeypatch.setattr(fp.subprocess, "run",
                        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": ""})())

    fp.register_bold_to_template(
        bold_ref_file=bold, template_file=tpl, output_dir=tmp_path,
        subject="sub-1A", session="ses-1", work_dir=tmp_path / "work",
    )

    mat = sio.loadmat(str(tmp_path / "transforms/sub-1A/ses-1/BOLD_to_template_0GenericAffine.mat"))
    tx, ty, tz = mat["AffineTransform_double_3_3"].ravel()[9:12]

    # in-plane COM offset (template - bold); the .mat carries the same sign in
    # X/Y (ITK LPS flips X/Y, so they match the world offset, unlike Z).
    bc, tc = _com_world(nib.load(str(bold))), _com_world(nib.load(str(tpl)))
    assert tx == pytest.approx(tc[0] - bc[0], abs=0.5)
    assert ty == pytest.approx(tc[1] - bc[1], abs=0.5)
    assert tz == pytest.approx(-20.0, abs=1e-6)   # Z keeps the -z_mm convention
    # in-plane offset must be non-trivial here (else the test proves nothing)
    assert abs(tc[0] - bc[0]) > 1.0
