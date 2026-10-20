"""Unit tests for 3D Bruker affines and the subject -> animal-RAS rotation."""

import nibabel as nib
import numpy as np
import pytest

from neurofaune.preprocess.utils.mrs.bruker_affine import affine_from_visu
from neurofaune.utils.bids import BidsConfig
from neurofaune.utils.bruker_geometry import (
    SUBJECT_TO_ANATOMICAL,
    anatomical_affine,
    subject_to_anatomical,
)


def write_visu(scan_dir, *, size, extent, orientation, positions,
               subject_type="Quadruped", subject_position="Head_Supine", thickness=0.6):
    rotation = np.asarray(orientation, dtype=float)
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    n = len(positions)
    proc = scan_dir / "pdata" / "1"
    proc.mkdir(parents=True, exist_ok=True)
    lines = [
        "##TITLE=synthetic",
        f"##$VisuCoreSize=( {len(size)} )\n" + " ".join(map(str, size)),
        f"##$VisuCoreExtent=( {len(extent)} )\n" + " ".join(map(str, extent)),
        f"##$VisuCoreFrameThickness=( 1 )\n{thickness}",
        f"##$VisuSubjectType={subject_type}",
        f"##$VisuSubjectPosition={subject_position}",
        f"##$VisuCoreOrientation=( {n}, 9 )",
        " ".join(" ".join(f"{v:.10g}" for v in rotation.ravel()) for _ in range(n)),
        f"##$VisuCorePosition=( {n}, 3 )",
        " ".join(" ".join(f"{v:.10g}" for v in p) for p in positions),
        "##END=",
    ]
    (proc / "visu_pars").write_text("\n".join(lines) + "\n")
    return scan_dir


@pytest.fixture
def scan3d(tmp_path):
    # A 3D slab like the mouse pilot's sagittal RARE: 100 x 160 x 100 at 0.15 mm.
    d = tmp_path / "7"
    return write_visu(d, size=(100, 160, 100), extent=(15, 24, 15),
                      orientation=np.eye(3), positions=[(-7.5, -12.0, -7.5)], thickness=15)


@pytest.fixture
def axial(tmp_path):
    # 2D stack, slices stepping along subject +z (the bore / rostral axis).
    d = tmp_path / "21"
    positions = [(-12.0, -12.0, -6.0 + 1.0 * k) for k in range(13)]
    return write_visu(d, size=(90, 90), extent=(24, 24), orientation=np.eye(3), positions=positions)


class TestAffine3D:
    def test_third_axis_spacing_is_extent_over_size_not_slab_thickness(self, scan3d):
        # VisuCoreFrameThickness is the whole 15 mm slab for a 3D scan.
        spacing = np.linalg.norm(affine_from_visu(scan3d)[:3, :3], axis=0)
        assert spacing == pytest.approx([0.15, 0.15, 0.15])

    def test_3d_affine_is_on_voxel_centres(self, scan3d):
        affine = affine_from_visu(scan3d)
        assert affine[:3, 3] == pytest.approx([-7.5 + 0.075, -12.0 + 0.075, -7.5 + 0.075])
        centre = affine @ np.array([49.5, 79.5, 49.5, 1.0])
        assert centre[:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


class TestSubjectToAnatomical:
    def test_calibrated_rotation_is_proper(self):
        # A mirror here would silently swap hemispheres.
        for rotation in SUBJECT_TO_ANATOMICAL.values():
            assert np.linalg.det(rotation) == pytest.approx(1.0)
            assert rotation @ rotation.T == pytest.approx(np.eye(3))

    def test_quadruped_head_supine_axes(self, axial):
        # Subject +z (head) is rostral = anatomical A; subject -y (back) is
        # dorsal = anatomical S; subject +x stays R.
        rotation = subject_to_anatomical(axial)
        assert rotation @ [0, 0, 1] == pytest.approx([0, 1, 0])
        assert rotation @ [0, -1, 0] == pytest.approx([0, 0, 1])
        assert rotation @ [1, 0, 0] == pytest.approx([1, 0, 0])

    def test_uncalibrated_position_is_refused(self, tmp_path):
        d = write_visu(tmp_path / "9", size=(64, 64), extent=(24, 24), orientation=np.eye(3),
                       positions=[(0, 0, 0)], subject_position="Head_Prone")
        with pytest.raises(KeyError, match="no calibrated"):
            subject_to_anatomical(d)

    def test_axial_stack_is_coronal_in_the_animal(self, axial):
        # Bruker 'axial' (perpendicular to the bore) is the animal's coronal
        # plane: slices run rostro-caudally. Voxel axes R, I, A -- the same
        # STUDY_NATIVE_AXES neurofaune traced for axial rat acquisitions.
        affine = anatomical_affine(axial)
        assert nib.aff2axcodes(affine) == ("R", "I", "A")

    def test_voxel_scale_scales_the_whole_affine(self, axial):
        one, ten = anatomical_affine(axial, 1.0), anatomical_affine(axial, 10.0)
        assert ten[:3, :] == pytest.approx(10.0 * one[:3, :])
        assert np.linalg.norm(ten[:3, :3], axis=0) == pytest.approx([2.6667, 2.6667, 10.0], abs=1e-3)


def test_bids_config_affine_mode():
    roots = {"raw_root": "/r", "bids_root": "/b"}
    assert BidsConfig.from_config({"bids": roots}).affine == "identity"
    assert BidsConfig.from_config({"bids": {**roots, "affine": "scanner"}}).affine == "scanner"
    with pytest.raises(ValueError, match="bids.affine"):
        BidsConfig.from_config({"bids": {**roots, "affine": "oblique"}})
