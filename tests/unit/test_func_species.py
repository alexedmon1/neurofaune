"""Unit tests for the species-dependent pieces of functional preprocessing.

Mouse phase 6b (FAC600 pilot): in-plane erosion, the anat-guided BOLD mask, the
atlas tissue source for aCompCor / ICA, the NCC voxel-frame guard, and the
atlas WM/CSF masks.
"""
import json

import nibabel as nib
import numpy as np
import pytest
import yaml

from neurofaune.config import get_config_value, load_config
from neurofaune.preprocess.utils.func.ica_denoising import calculate_edge_fraction
from neurofaune.preprocess.utils.func.masks import erode, erosion_structure
from neurofaune.preprocess.utils.registration_utils import (
    find_z_offset_ncc,
    require_shared_voxel_frame,
    voxel_world_axes,
)
from neurofaune.preprocess.workflows.func_preprocess import (
    anat_guided_func_mask,
    anat_tissue_maps,
)


def slab(shape=(12, 12, 3)):
    m = np.zeros(shape, bool)
    m[2:10, 2:10, :] = True       # brain fills every slice of a 3-slice slab
    return m


class TestErosion:
    def test_3d_erosion_empties_the_slab_end_slices(self):
        out = erode(slab(), 1, inplane=False)
        assert out[:, :, 0].sum() == 0 and out[:, :, 2].sum() == 0
        assert out[:, :, 1].sum() > 0

    def test_inplane_erosion_keeps_every_slice(self):
        out = erode(slab(), 1)                                   # in-plane is the default
        assert out.sum(axis=(0, 1)).tolist() == [36, 36, 36]     # 8x8 -> 6x6 per slice

    def test_structure_has_no_through_slice_neighbours_inplane(self):
        s = erosion_structure(inplane=True)
        assert not s[:, :, 0].any() and not s[:, :, 2].any()
        assert s[:, :, 1].sum() == 5

    def test_zero_iterations_is_identity(self):
        assert np.array_equal(erode(slab(), 0), slab())

    def test_ica_edge_fraction_inplane(self):
        comp = np.zeros((12, 12, 3))
        comp[5:7, 5:7, 0] = 5.0        # interior of the first slice
        brain = slab()
        assert calculate_edge_fraction(comp, brain, inplane=False) == pytest.approx(1.0)  # 3D: all "edge"
        assert calculate_edge_fraction(comp, brain) == 0.0


def affine_axial_like():
    """Voxel axes -> world x, z, y, as the FAC600 mouse axial EPI."""
    a = np.zeros((4, 4))
    a[0, 0], a[2, 1], a[1, 2], a[3, 3] = 2.7, -2.6, 10.0, 1.0
    return a


class TestVoxelFrameGuard:
    def test_axes(self):
        assert voxel_world_axes(affine_axial_like()).tolist() == [0, 2, 1]
        assert voxel_world_axes(np.eye(4)).tolist() == [0, 1, 2]

    def test_shared_frame_passes(self):
        img = nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.diag([2, 2, 5, 1.0]))
        require_shared_voxel_frame(img, nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), np.eye(4)))

    def test_permuted_frame_raises_in_ncc_search(self, tmp_path):
        moving = nib.Nifti1Image(np.ones((10, 10, 4), np.float32), affine_axial_like())
        fixed = nib.Nifti1Image(np.ones((20, 20, 20), np.float32), np.eye(4))
        with pytest.raises(ValueError, match="T2w_to_template"):
            find_z_offset_ncc(moving, fixed, tmp_path)


class TestTissueSource:
    def test_segmentation_is_the_default(self, tmp_path):
        maps = anat_tissue_maps({}, tmp_path, "sub-A", "ses-1")
        assert maps["WM"].name == "sub-A_ses-1_label-WM_probseg.nii.gz"
        assert maps["CSF"].parent == tmp_path / "derivatives/sub-A/ses-1/anat"

    def test_atlas_source(self, tmp_path):
        cfg = {"functional": {"denoising": {"acompcor": {"tissue_source": "atlas"}}}}
        maps = anat_tissue_maps(cfg, tmp_path, "sub-A", "ses-1")
        assert maps["CSF"].name == "sub-A_ses-1_label-CSF_desc-atlas_probseg.nii.gz"

    def test_unknown_source_is_an_error(self, tmp_path):
        cfg = {"functional": {"denoising": {"acompcor": {"tissue_source": "fast"}}}}
        with pytest.raises(ValueError, match="tissue_source"):
            anat_tissue_maps(cfg, tmp_path, "sub-A", "ses-1")


def test_anat_guided_mask_needs_the_anat_outputs(tmp_path):
    func_dir = tmp_path / "derivatives/sub-A/ses-1/func"
    func_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="anat_preprocess"):
        anat_guided_func_mask({}, "sub-A", "ses-1", func_dir, tmp_path / "r.nii.gz",
                              tmp_path / "raw.nii.gz", tmp_path / "b.nii.gz",
                              tmp_path / "m.nii.gz", tmp_path)


def test_atlas_tissue_to_anat_needs_the_template_registration(tmp_path):
    from neurofaune.preprocess.workflows.anat_atlas import atlas_tissue_to_anat
    with pytest.raises(FileNotFoundError, match="register_anat_to_atlas_template"):
        atlas_tissue_to_anat({}, "sub-A", "ses-1", tmp_path)


def test_add_tissue_masks(tmp_path):
    from neurofaune.atlas.antwerp import add_tissue_masks
    bg = tmp_path / "bg"
    bg.mkdir()
    (bg / "metadata.json").write_text(json.dumps({"name": "toy", "orientation": "asr",
                                                  "resolution": [25, 25, 25]}))
    (bg / "structures.csv").write_text(
        "id,name,acronym,structure_id_path,parent_structure_id\n"
        "997,root,root,/997/,\n"
        "1009,fiber tracts,fiber tracts,/997/1009/,997.0\n"
        "776,corpus callosum,cc,/997/1009/776/,1009.0\n"
        "73,ventricular systems,VS,/997/73/,997.0\n"
        "81,lateral ventricle,VL,/997/73/81/,73.0\n"
        "315,Isocortex,Isocortex,/997/315/,997.0\n")
    out = tmp_path / "atlas"
    ann = np.zeros((6, 6, 6), np.uint32)
    ann[0:2] = 315
    ann[2:4] = 776            # descendant of fiber tracts
    ann[4, :, :] = 81         # descendant of VS
    for d in (out, out / "hub100um"):
        d.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(ann, np.diag([0.1, 0.1, 0.1, 1.0])), str(d / "ccfv3_annotation.nii.gz"))
    info = add_tissue_masks(out, bg)
    wm = np.asanyarray(nib.load(str(out / "wm_mask.nii.gz")).dataobj)
    csf = np.asanyarray(nib.load(str(out / "hub100um/csf_mask.nii.gz")).dataobj)
    assert wm.sum() == 72 and wm[2:4].all() and not wm[0:2].any()
    assert csf.sum() == 36 and csf[4].all()
    assert info["native_wm_mm3"] == pytest.approx(0.072)
    scaled = nib.load(str(out / "scaled10/wm_mask.nii.gz"))
    assert scaled.header.get_zooms()[:3] == pytest.approx((1.0, 1.0, 1.0))


def write_study(tmp_path, **study):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"study": {"name": "t", "code": "T", **study}}))
    return path


def test_mouse_profile_functional_settings(tmp_path):
    cfg = load_config(write_study(tmp_path, species="mouse"), validate=False)
    assert get_config_value(cfg, "functional.skull_strip_method") == "anat"
    assert get_config_value(cfg, "functional.skull_strip_adaptive.mask_erode") == 0
    assert get_config_value(cfg, "functional.motion_qc.head_radius_mm") == pytest.approx(3.2)
    assert get_config_value(cfg, "functional.denoising.acompcor.tissue_source") == "atlas"
    assert get_config_value(cfg, "atlas.study_space.wm_prob").endswith("scaled10/wm_mask.nii.gz")
    # the rat-tuned adaptive-BET settings underneath are inherited, not removed
    assert get_config_value(cfg, "functional.skull_strip_adaptive.target_ratio") == 0.15


def test_in_plane_erosion_is_the_default_for_every_species(tmp_path):
    for species in ("rat", "mouse"):
        cfg = load_config(write_study(tmp_path, species=species), validate=False)
        assert get_config_value(cfg, "functional.erode_inplane") is True


def test_rat_keeps_its_functional_behaviour(tmp_path):
    cfg = load_config(write_study(tmp_path, species="rat"), validate=False)
    assert get_config_value(cfg, "functional.skull_strip_method") == "adaptive"
    assert get_config_value(cfg, "functional.motion_qc.head_radius_mm", default=5.0) == 5.0
    assert get_config_value(cfg, "functional.denoising.acompcor.tissue_source",
                            default="segmentation") == "segmentation"
