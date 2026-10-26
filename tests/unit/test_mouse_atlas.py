"""Unit tests for BrainGlobe I/O and the Antwerp hub parcellation rebuild."""

import nibabel as nib
import numpy as np
import pytest

from neurofaune.atlas.antwerp import hub_grid, parcellate, scale_header
from neurofaune.atlas.brainglobe import dice, orientation_affine


class TestOrientation:
    def test_asr_axes_point_posterior_inferior_left(self):
        a = orientation_affine("asr", (25, 25, 25), (10, 10, 10))
        assert nib.aff2axcodes(a) == ("P", "I", "L")
        assert np.linalg.norm(a[:3, :3], axis=0) == pytest.approx([0.025] * 3)

    def test_volume_is_centred(self):
        a = orientation_affine("asr", (25, 25, 25), (11, 21, 31))
        centre = a @ np.array([5, 10, 15, 1.0])
        assert centre[:3] == pytest.approx([0, 0, 0])

    def test_bad_orientation(self):
        with pytest.raises(ValueError):
            orientation_affine("aar", (1, 1, 1), (2, 2, 2))


def ras_volume(shape=(20, 30, 10), vox=0.1):
    return np.diag([vox, vox, vox, 1.0])


class TestParcellate:
    def _annotation(self):
        ann = np.zeros((20, 30, 10), np.uint32)
        ann[2:18, 5:25, 2:8] = 100           # one bilateral structure
        ann[2:18, 10:20, 2:4] = 200          # "hippocampus", y 10..19
        return nib.Nifti1Image(ann, ras_volume())

    def test_hemispheres_split_at_the_centre_of_mass(self):
        mapping = {"rois": {
            "0": {"id": 0, "name": "Background"},
            "1": {"id": 1, "name": "S_L", "hemisphere": "L", "allen_structure_ids": [100]},
            "2": {"id": 2, "name": "S_R", "hemisphere": "R", "allen_structure_ids": [100]},
        }}
        img, m = parcellate(self._annotation(), mapping)
        d = np.asanyarray(img.dataobj)
        assert (d[:10] == 1).any() and not (d[:10] == 2).any()   # low x = left in RAS
        assert (d[10:] == 2).any() and not (d[10:] == 1).any()
        assert m["hemisphere_split_x_mm"] == pytest.approx(0.95, abs=0.05)

    def test_hippocampus_ant_is_rostral(self):
        mapping = {"rois": {
            "7": {"id": 7, "name": "Hippocampus_Ant_L", "hemisphere": "B", "allen_structure_ids": [200]},
            "8": {"id": 8, "name": "Hippocampus_Post_L", "hemisphere": "B", "allen_structure_ids": [200]},
        }}
        img, m = parcellate(self._annotation(), mapping, caudal_hippocampus_fraction=0.3)
        d = np.asanyarray(img.dataobj)
        ys_ant = np.nonzero(d == 7)[1]
        ys_post = np.nonzero(d == 8)[1]
        assert ys_ant.min() > ys_post.max()                       # Ant = higher y = rostral
        frac = (d == 8).sum() / ((d == 7).sum() + (d == 8).sum())
        assert frac == pytest.approx(0.3, abs=0.06)
        assert "swapped" in m["hippocampus_split"]["naming"]


def test_hub_grid_covers_native_fov():
    native = nib.Nifti1Image(np.zeros((64, 256, 50)), np.diag([0.203, 0.08, 0.2, 1.0]))
    hub = hub_grid(native, 0.1)
    assert np.allclose(np.diag(hub.affine)[:3], 0.1)
    far = hub.affine @ np.r_[np.array(hub.shape) - 1, 1]
    assert (far[:3] >= np.array([63 * 0.203, 255 * 0.08, 49 * 0.2]) - 1e-6).all()


def test_scale_header_inflates_affine_only():
    img = nib.Nifti1Image(np.ones((2, 2, 2)), np.diag([0.1, 0.1, 0.1, 1.0]))
    s = scale_header(img, 10.0)
    assert s.header.get_zooms() == pytest.approx((1.0, 1.0, 1.0))
    assert np.array_equal(np.asanyarray(s.dataobj), np.ones((2, 2, 2)))


def test_dice():
    a = np.zeros(10, bool); a[:4] = True
    b = np.zeros(10, bool); b[2:6] = True
    assert dice(a, b) == pytest.approx(0.5)
    assert dice(np.zeros(3, bool), np.zeros(3, bool)) is None


def test_warp_labels_preserves_ids_beyond_float32(tmp_path):
    import shutil
    if shutil.which("antsApplyTransforms") is None:
        pytest.skip("ANTs not installed")
    from neurofaune.atlas.label_transfer import warp_labels
    data = np.zeros((8, 8, 8), np.uint32)
    data[2:6, 2:6, 2:6] = 614454277          # CCFv3 layer-subdivision ID
    data[2:6, 2:6, 6:8] = 526157193
    src = tmp_path / "lab.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(src))
    out = warp_labels(src, src, ["identity"], tmp_path / "out.nii.gz", n_threads=1)
    got = np.asanyarray(nib.load(str(out)).dataobj)
    assert set(np.unique(got)) == {0, 614454277, 526157193}
    assert np.array_equal(got, data)


def test_compare_parcellations_reports_shift_in_world_mm():
    from neurofaune.atlas.compare import compare_parcellations
    aff = np.diag([0.1, 0.2, 0.1, 1.0])
    a = np.zeros((20, 20, 20), np.uint16); a[5:10, 5:10, 5:10] = 3
    b = np.zeros_like(a); b[5:10, 7:12, 5:10] = 3          # moved +2 voxels in y = +0.4 mm rostral
    rows = compare_parcellations(nib.Nifti1Image(a, aff), nib.Nifti1Image(b, aff), {3: "X"})
    r = rows[0]
    assert r["name"] == "X" and r["shift_A_mm"] == pytest.approx(0.4)
    assert r["shift_R_mm"] == pytest.approx(0) and r["dice"] == pytest.approx(0.6)
    assert r["mean_boundary_distance_mm"] > 0


def test_fit_frame_recovers_bregma_and_axes():
    from neurofaune.atlas.stereotaxic import fit_frame, world_to_stereo
    shape = (40, 50, 30)
    affine = np.diag([0.05, 0.05, 0.05, 1.0])
    affine[:3, 3] = [-1.0, -1.25, -0.75]
    ijk = np.indices(shape).reshape(3, -1).T
    world = (affine @ np.column_stack([ijk, np.ones(len(ijk))]).T).T[:, :3].reshape(*shape, 3)
    bregma = np.array([0.0, 0.3, 0.9])                 # above the "brain"
    ml = world[..., 0] - bregma[0]
    ap = world[..., 1] - bregma[1]
    dv = -(world[..., 2] - bregma[2])                    # + ventral
    brain = (np.abs(ml) < 0.8) & (np.abs(ap) < 1.0) & (dv > 0.2)
    x = np.where(brain, np.abs(ml), 0); y = np.where(brain, ap, 0); z = np.where(brain, dv, 0)
    hem = np.where(brain & (ml > 0), 1, 0)
    fr = fit_frame((x, y, z), hem, affine)
    assert fr["bregma_world_mm"] == pytest.approx(bregma, abs=1e-6)
    assert fr["ml_axis"] == pytest.approx([1, 0, 0], abs=1e-6)
    assert fr["dv_axis"] == pytest.approx([0, 0, -1], abs=1e-6)
    assert world_to_stereo(fr, [bregma + [0, -1, 0]])[0] == pytest.approx([0, -1, 0], abs=1e-6)
    assert fr["residual_mm"]["rms"] < 1e-9


def test_rabies_roles_and_component_description():
    from neurofaune.atlas.functional import describe_components, rabies_role
    assert rabies_role(5) == "bold" and rabies_role(0) == "confound" and rabies_role(1) == "unassigned"
    lab = np.zeros((20, 10, 10), np.uint16)
    lab[2:10, 2:8, 2:8] = 1          # left parcel
    lab[10:18, 2:8, 2:8] = 2         # right parcel
    ic = np.zeros(lab.shape + (2,), np.float32)
    ic[3:9, 3:7, 3:7, 0] = 5.0       # left only
    ic[11:17, 3:7, 3:7, 0] = 0.0
    ic[3:9, 3:7, 3:7, 1] = 5.0       # bilateral, mirrored
    ic[11:17, 3:7, 3:7, 1] = 5.0
    rows = describe_components(nib.Nifti1Image(ic, np.eye(4)), nib.Nifti1Image(lab, np.eye(4)), {1: "L", 2: "R"})
    assert rows[0]["top_parcels"].startswith("L (100%)")
    assert rows[1]["lr_symmetry_r"] == pytest.approx(1.0)     # mirrored bilateral network
    assert rows[0]["lr_symmetry_r"] < 0                        # one-sided network


def test_cortical_projection_takes_the_dorsal_surface_and_a_labelled_parcel():
    from neurofaune.atlas.electrodes import cortical_projection
    shape = (21, 21, 12)
    aff = np.diag([0.1, 0.1, 0.1, 1.0]); aff[:3, 3] = [-1.0, -1.0, 0.0]
    ijk = np.indices(shape)
    world = np.einsum("ab,bijk->aijk", aff[:3, :3], ijk) + aff[:3, 3:4, None, None]
    brain = np.zeros(shape, bool); brain[2:19, 2:19, 1:9] = True
    ml = np.where(brain, world[0], 0.0); ap = np.where(brain, world[1], 0.0)
    lab = np.zeros(shape, np.uint16); lab[2:19, 2:19, 1:8] = 5          # top brain layer unlabelled
    rows = cortical_projection([{"label": "E1", "ml": 0.3, "ap": -0.2}],
                               nib.Nifti1Image(ml, aff), nib.Nifti1Image(ap, aff), brain,
                               nib.Nifti1Image(lab, aff), {5: "Cortex"}, tolerance_mm=0.05)
    r = rows[0]
    assert r["found"] and r["z_mm"] == pytest.approx(0.8)               # most dorsal brain voxel
    assert r["parcel"] == "Cortex" and r["parcel_depth_mm"] == pytest.approx(0.1)
    off = cortical_projection([{"label": "E2", "ml": 5.0, "ap": 0.0}], nib.Nifti1Image(ml, aff),
                              nib.Nifti1Image(ap, aff), brain, tolerance_mm=0.05)
    assert off[0]["found"] is False


def test_cement_footprint_finds_the_dark_cap():
    from neurofaune.atlas.electrodes import cement_footprint
    t2 = np.full((40, 40, 30), 100.0)
    brain = np.zeros(t2.shape, bool); brain[5:35, 5:35, 2:15] = True
    t2[10:30, 8:28, 18:24] = 1.0                                         # cement band 0.6-1.35 mm above the brain
    cap = cement_footprint(t2, brain, np.diag([0.15, 0.15, 0.15, 1.0]), noise_threshold=10.0)
    assert cap["found"] and cap["area_mm2"] == pytest.approx(20 * 20 * 0.0225, rel=0.2)
    assert cap["centroid_mm"][0] == pytest.approx(19.5 * 0.15, abs=0.1)
