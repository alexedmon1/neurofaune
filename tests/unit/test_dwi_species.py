"""Unit tests for the species-dependent pieces of DWI preprocessing (mouse phase 6c)."""
import json

import nibabel as nib
import numpy as np
import pytest
import yaml

from neurofaune.config import get_config_value, load_config
from neurofaune.preprocess.utils.dwi_utils import pad_slices_for_eddy
from neurofaune.preprocess.workflows.dwi_preprocess import (
    _pad_volume,
    _padded_affine,
    dwi_acqparams_row,
    register_dwi_via_anat,
    run_dwi_topup,
)

AFFINE = np.array([[1.5, 0, 0.25, -130.5], [0, 0.26, 7.9, -53.8], [0, -1.48, 1.4, 81.2], [0, 0, 0, 1]])


def nifti(path, data, affine=AFFINE, meta=None):
    nib.save(nib.Nifti1Image(np.asarray(data, np.float32), affine), str(path))
    if meta is not None:
        path.with_name(path.name.replace(".nii.gz", ".json")).write_text(json.dumps(meta))
    return path


class TestAcqparams:
    def test_config_is_the_default(self, tmp_path):
        dwi = nifti(tmp_path / "dwi.nii.gz", np.zeros((4, 4, 3, 2)))
        assert dwi_acqparams_row({}, dwi) == "0 -1 0 0.05"

    def test_sidecar(self, tmp_path):
        dwi = nifti(tmp_path / "dwi.nii.gz", np.zeros((4, 4, 3, 2)),
                    meta={"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.0253552})
        cfg = {"diffusion": {"eddy": {"acqparams_source": "sidecar"}}}
        assert dwi_acqparams_row(cfg, dwi) == "0 -1 0 0.025355"

    def test_sidecar_without_fields_is_an_error(self, tmp_path):
        dwi = nifti(tmp_path / "dwi.nii.gz", np.zeros((4, 4, 3, 2)), meta={})
        with pytest.raises(KeyError, match="readout"):
            dwi_acqparams_row({"diffusion": {"eddy": {"acqparams_source": "sidecar"}}}, dwi)

    def test_unknown_source(self, tmp_path):
        with pytest.raises(ValueError, match="acqparams_source"):
            dwi_acqparams_row({"diffusion": {"eddy": {"acqparams_source": "guess"}}}, tmp_path / "x.nii.gz")


def test_topup_padding_matches_eddy_padding(tmp_path):
    """The field must land on eddy's --imain grid: same values and affine as pad_slices_for_eddy."""
    data = np.random.default_rng(0).random((5, 6, 7, 2))
    src = nifti(tmp_path / "dwi.nii.gz", data)
    out, n = pad_slices_for_eddy(src, tmp_path / "pad.nii.gz", n_pad=2)
    eddy_img = nib.load(str(out))
    assert n == 7
    assert np.allclose(_pad_volume(data[..., 0].astype(np.float32), 2), eddy_img.get_fdata()[..., 0])
    assert np.allclose(_padded_affine(AFFINE, 2), eddy_img.affine)


class TestTopupInputs:
    meta = {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.025}

    def test_epi_on_another_grid_is_refused(self, tmp_path):
        dwi = nifti(tmp_path / "dwi.nii.gz", np.zeros((4, 4, 3, 2)), meta=self.meta)
        b0 = nifti(tmp_path / "b0.nii.gz", np.zeros((4, 4, 3)))
        fwd = nifti(tmp_path / "fwd.nii.gz", np.zeros((4, 4, 5)), meta=self.meta)
        rev = nifti(tmp_path / "rev.nii.gz", np.zeros((4, 4, 5)),
                    meta={"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.025})
        with pytest.raises(ValueError, match="DWI grid"):
            run_dwi_topup(dwi, b0, (fwd, rev), tmp_path / "w", n_pad=2)

    def test_dwi_must_match_one_of_the_pair(self, tmp_path):
        dwi = nifti(tmp_path / "dwi.nii.gz", np.zeros((4, 4, 3, 2)),
                    meta={"PhaseEncodingDirection": "i", "TotalReadoutTime": 0.025})
        b0 = nifti(tmp_path / "b0.nii.gz", np.zeros((4, 4, 3)))
        fwd = nifti(tmp_path / "fwd.nii.gz", np.zeros((4, 4, 3)), meta=self.meta)
        rev = nifti(tmp_path / "rev.nii.gz", np.zeros((4, 4, 3)),
                    meta={"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.025})
        with pytest.raises(ValueError, match="matches neither"):
            run_dwi_topup(dwi, b0, (fwd, rev), tmp_path / "w", n_pad=2)


def test_via_anat_registration_needs_the_t2w_template_transform(tmp_path):
    with pytest.raises(FileNotFoundError, match="T2w_to_template"):
        register_dwi_via_anat(tmp_path / "b0.nii.gz", tmp_path / "fa.nii.gz", tmp_path / "t.nii.gz",
                              tmp_path, "sub-A", "ses-1", tmp_path / "w")


def write_study(tmp_path, **study):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"study": {"name": "t", "code": "T", **study}}))
    return path


def test_mouse_profile_diffusion_settings(tmp_path):
    cfg = load_config(write_study(tmp_path, species="mouse"), validate=False)
    assert get_config_value(cfg, "diffusion.skull_strip.method") == "anat"
    assert get_config_value(cfg, "diffusion.eddy.acqparams_source") == "sidecar"
    assert get_config_value(cfg, "diffusion.registration.method") == "anat"


def test_rat_keeps_its_diffusion_behaviour(tmp_path):
    cfg = load_config(write_study(tmp_path, species="rat"), validate=False)
    assert get_config_value(cfg, "diffusion.skull_strip.method") == "atropos_bet"
    assert get_config_value(cfg, "diffusion.eddy.acqparams_source", default="config") == "config"
    assert get_config_value(cfg, "diffusion.registration.method", default="fa") == "fa"
