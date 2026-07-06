"""Unit tests for aCompCor tissue-mask erosion (``extract_acompcor_components``).

Pins the erosion behaviour so over-inclusive tissue masks (e.g. animal
skull-strip rims) can't silently feed partial-volume GM/BOLD edge voxels into
aCompCor. Erosion pulls the thresholded WM/CSF masks off the tissue boundary
before timeseries extraction; it is gate-able (``erode_voxels=0``) and falls
back per-tissue when it would decimate a small mask below the component count.
"""
import numpy as np
import nibabel as nib
import pytest

from neurofaune.preprocess.utils.func.acompcor import extract_acompcor_components

AFFINE = np.eye(4)
SHAPE = (20, 20, 20)
N_TP = 30


def _write_nifti(path, data):
    nib.save(nib.Nifti1Image(data.astype(np.float32), AFFINE), str(path))
    return path


def _bold(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(100.0, 5.0, size=SHAPE + (N_TP,))
    return _write_nifti(tmp_path / "bold.nii.gz", data)


def _block_mask(tmp_path, name, lo, hi):
    """Solid cuboid probability mask (value 1.0 inside, 0 outside)."""
    data = np.zeros(SHAPE)
    data[lo:hi, lo:hi, lo:hi] = 1.0
    return _write_nifti(tmp_path / f"{name}.nii.gz", data)


def test_erosion_shrinks_masks(tmp_path):
    bold = _bold(tmp_path)
    csf = _block_mask(tmp_path, "csf", 4, 14)   # 10^3 = 1000 voxels
    wm = _block_mask(tmp_path, "wm", 4, 16)      # 12^3 = 1728 voxels

    eroded = extract_acompcor_components(bold, csf, wm, n_components=5, erode_voxels=1)
    uneroded = extract_acompcor_components(bold, csf, wm, n_components=5, erode_voxels=0)

    assert eroded["n_voxels_csf"] < uneroded["n_voxels_csf"]
    assert eroded["n_voxels_wm"] < uneroded["n_voxels_wm"]
    # one-voxel erosion of a solid cuboid removes exactly the outer shell
    assert eroded["n_voxels_csf"] == 8 ** 3
    assert eroded["n_voxels_wm"] == 10 ** 3
    assert eroded["erode_voxels"] == 1
    assert uneroded["erode_voxels"] == 0


def test_erode_zero_is_noop(tmp_path):
    bold = _bold(tmp_path)
    csf = _block_mask(tmp_path, "csf", 4, 14)
    wm = _block_mask(tmp_path, "wm", 4, 16)
    res = extract_acompcor_components(bold, csf, wm, n_components=5, erode_voxels=0)
    assert res["n_voxels_csf"] == 10 ** 3
    assert res["n_voxels_wm"] == 12 ** 3


def test_erosion_falls_back_when_it_would_empty_a_small_mask(tmp_path):
    # A thin 3-voxel-thick slab: eroding it away would leave < n_components
    # voxels, so the un-eroded mask must be kept for that tissue.
    bold = _bold(tmp_path)
    thin = np.zeros(SHAPE)
    thin[8:11, 8:11, 5] = 1.0   # single-slice 3x3 patch = 9 voxels
    csf = _write_nifti(tmp_path / "csf.nii.gz", thin)
    wm = _block_mask(tmp_path, "wm", 4, 16)

    res = extract_acompcor_components(bold, csf, wm, n_components=5, erode_voxels=1)
    # erosion of a single-slice patch -> 0 voxels < 5 components -> fall back
    assert res["n_voxels_csf"] == 9
    # WM is a fat cuboid, erosion still applies there
    assert res["n_voxels_wm"] == 10 ** 3


def test_default_erodes(tmp_path):
    # Default (no erode_voxels arg) must erode — the whole point of the fix.
    bold = _bold(tmp_path)
    csf = _block_mask(tmp_path, "csf", 4, 14)
    wm = _block_mask(tmp_path, "wm", 4, 16)
    res = extract_acompcor_components(bold, csf, wm, n_components=5)
    assert res["erode_voxels"] == 1
    assert res["n_voxels_csf"] == 8 ** 3


def test_brain_intersect_preserves_interior_tissue(tmp_path):
    # The intended path: erode the BRAIN mask and intersect. Interior tissue
    # (well inside the brain) must survive fully — unlike per-tissue erosion,
    # which shrinks it. This is the fix for thin rodent masks vanishing.
    bold = _bold(tmp_path)
    brain = _block_mask(tmp_path, "brain", 2, 18)   # 16^3 brain
    wm = _block_mask(tmp_path, "wm", 5, 15)          # 10^3, fully interior
    csf = _block_mask(tmp_path, "csf", 5, 15)

    res = extract_acompcor_components(
        bold, csf, wm, n_components=5, erode_voxels=1, brain_mask=brain
    )
    # eroded brain = [3:17]^3 still fully contains the [5:15]^3 tissue blocks
    assert res["n_voxels_wm"] == 10 ** 3
    assert res["n_voxels_csf"] == 10 ** 3
    # ...whereas per-tissue erosion of the same block would shrink it to 8^3
    per_tissue = extract_acompcor_components(
        bold, csf, wm, n_components=5, erode_voxels=1, brain_mask=None
    )
    assert per_tissue["n_voxels_wm"] == 8 ** 3
    assert res["n_voxels_wm"] > per_tissue["n_voxels_wm"]


def test_brain_intersect_strips_rim(tmp_path):
    # Tissue voxels sitting on the brain surface (the over-inclusive rim) must
    # be removed by the eroded-brain intersection; interior voxels kept.
    bold = _bold(tmp_path)
    brain_data = np.zeros(SHAPE)
    brain_data[2:18, 2:18, 2:18] = 1.0
    brain = _write_nifti(tmp_path / "brain.nii.gz", brain_data)

    wm_data = np.zeros(SHAPE)
    wm_data[5:15, 5:15, 5:15] = 1.0   # 1000 interior voxels
    wm_data[2, 5:15, 5:15] = 1.0      # 100 voxels on the brain face (rim)
    wm = _write_nifti(tmp_path / "wm.nii.gz", wm_data)
    csf = _block_mask(tmp_path, "csf", 5, 15)

    res = extract_acompcor_components(
        bold, csf, wm, n_components=5, erode_voxels=1, brain_mask=brain
    )
    # rim voxels at x=2 fall outside eroded brain ([3:17]) -> dropped; interior kept
    assert res["n_voxels_wm"] == 1000
