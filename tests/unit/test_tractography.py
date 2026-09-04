"""Unit tests for the tractography module.

Synthetic data only — no external tools, no study data. The MRtrix- and
FSL-dependent entry points are exercised only through their guard logic.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurofaune.tractography.adequacy import (
    InadequateDataError,
    assess_tractography_adequacy,
    max_feasible_lmax,
    sh_coefficients,
)
from neurofaune.tractography.connectome import compute_node_coverage
from neurofaune.tractography.fivett import build_5tt_from_probseg
from neurofaune.tractography.fsl import (
    ball_and_sticks_parameters,
    max_supported_fibres,
)


# --- helpers ----------------------------------------------------------------

def _write_scheme(tmp_path: Path, shells: dict, name: str = "dwi") -> tuple:
    """Write bval/bvec for ``{bvalue: n_directions}`` plus ``{0: n_b0}``.

    Directions are spread over the sphere by golden-angle spiral so they are
    genuinely distinct, which the adequacy check verifies.
    """
    bvals, bvecs = [], []
    for b, n in shells.items():
        if b == 0:
            bvals += [0] * n
            bvecs += [[0.0, 0.0, 0.0]] * n
            continue
        idx = np.arange(n) + 0.5
        phi = np.arccos(1 - idx / n)          # hemisphere is enough (antipodal)
        theta = np.pi * (1 + 5**0.5) * idx
        for p, t in zip(phi, theta):
            bvecs.append(
                [float(np.cos(t) * np.sin(p)), float(np.sin(t) * np.sin(p)),
                 float(np.cos(p))]
            )
            bvals.append(b)

    bval_file = tmp_path / f"{name}.bval"
    bvec_file = tmp_path / f"{name}.bvec"
    np.savetxt(bval_file, np.array(bvals)[None, :], fmt="%d")
    np.savetxt(bvec_file, np.array(bvecs).T, fmt="%.6f")
    return bval_file, bvec_file


def _write_image(path: Path, data: np.ndarray, zooms=(2.5, 2.5, 5.0)) -> Path:
    affine = np.diag([*zooms, 1.0])
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    img.header.set_zooms(zooms[: data.ndim] if data.ndim <= 3 else (*zooms, 1.0))
    nib.save(img, str(path))
    return path


# --- spherical harmonics ----------------------------------------------------

@pytest.mark.parametrize(
    "lmax,expected", [(0, 1), (2, 6), (4, 15), (6, 28), (8, 45)]
)
def test_sh_coefficients(lmax, expected):
    assert sh_coefficients(lmax) == expected


@pytest.mark.parametrize(
    "n_dirs,expected", [(6, 2), (14, 2), (15, 4), (27, 4), (28, 6), (45, 8), (90, 8)]
)
def test_max_feasible_lmax(n_dirs, expected):
    # 90 directions supports lmax=10 (66 coeffs) mathematically, but the
    # helper reports the true ceiling; callers cap it.
    result = max_feasible_lmax(n_dirs)
    assert sh_coefficients(result) <= n_dirs
    assert sh_coefficients(result + 2) > n_dirs
    if n_dirs <= 45:
        assert result == expected


# --- adequacy ---------------------------------------------------------------

def test_six_direction_acquisition_is_blocked(tmp_path):
    """A 6-direction scheme carries only tensor-grade angular information."""
    bval, bvec = _write_scheme(tmp_path, {0: 1, 860: 6})
    result = assess_tractography_adequacy(bval, bvec, require_csd=True)

    assert not result.feasible
    assert result.recommended_model == "none"
    assert result.max_lmax_per_shell == 2
    assert any("lmax=2" in b for b in result.blockers)
    with pytest.raises(InadequateDataError):
        result.raise_if_infeasible()


def test_six_direction_allowed_when_csd_not_required(tmp_path):
    bval, bvec = _write_scheme(tmp_path, {0: 1, 860: 6})
    result = assess_tractography_adequacy(bval, bvec, require_csd=False)

    assert result.feasible
    assert result.recommended_model == "tensor"
    assert any("tensor" in w for w in result.warnings)


def test_multishell_hardi_recommends_msmt(tmp_path):
    """3 shells x 30 directions is the cuprizone-style acquisition."""
    bval, bvec = _write_scheme(tmp_path, {0: 5, 1000: 30, 2000: 30, 3000: 30})
    result = assess_tractography_adequacy(bval, bvec)

    assert result.feasible
    assert result.recommended_model == "msmt_csd"
    assert result.n_shells == 3
    assert result.total_dw_directions == 90
    # Pooling shells is the point: 30 dirs alone caps at lmax=6, but MSMT fits
    # against all 90 and supports lmax=8.
    assert result.max_lmax_per_shell == 6
    assert result.wm_lmax == 8
    assert not result.blockers


def test_single_shell_hardi_recommends_csd(tmp_path):
    bval, bvec = _write_scheme(tmp_path, {0: 3, 1000: 45})
    result = assess_tractography_adequacy(bval, bvec)

    assert result.feasible
    assert result.recommended_model == "csd"
    assert any("single shell" in w for w in result.warnings)


def test_repeated_directions_are_not_credited(tmp_path):
    """Duplicate directions average; they do not add angular resolution."""
    bvals = [0] + [1000] * 60
    base = np.random.default_rng(0).normal(size=(30, 3))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    bvecs = [[0.0, 0.0, 0.0]] + [v.tolist() for v in np.vstack([base, base])]

    bval_file = tmp_path / "r.bval"
    bvec_file = tmp_path / "r.bvec"
    np.savetxt(bval_file, np.array(bvals)[None, :], fmt="%d")
    np.savetxt(bvec_file, np.array(bvecs).T, fmt="%.6f")

    result = assess_tractography_adequacy(bval_file, bvec_file)
    assert result.total_dw_directions == 30      # not 60
    assert any("unique directions" in w for w in result.warnings)


def test_geometry_warnings(tmp_path):
    """Anisotropy and a thin slab are flagged but do not block."""
    bval, bvec = _write_scheme(tmp_path, {0: 5, 1000: 30, 2000: 30, 3000: 30})
    # 0.2 x 0.2 x 0.8 mm real, 11 slices -> 8.8 mm, at voxel_scale=10
    dwi = _write_image(
        tmp_path / "dwi.nii.gz", np.zeros((16, 16, 11, 95)), zooms=(2.0, 2.0, 8.0)
    )
    result = assess_tractography_adequacy(bval, bvec, dwi, voxel_scale=10.0)

    assert result.anisotropy_ratio == pytest.approx(4.0)
    assert result.coverage_mm == pytest.approx(8.8)
    assert any("anisotropic" in w for w in result.warnings)
    assert any("slab covers" in w for w in result.warnings)
    assert result.feasible          # geometry alone never blocks


def test_missing_b0_blocks(tmp_path):
    bval, bvec = _write_scheme(tmp_path, {1000: 30, 2000: 30})
    result = assess_tractography_adequacy(bval, bvec)
    assert not result.feasible
    assert any("b=0" in b for b in result.blockers)


# --- ball and sticks --------------------------------------------------------

@pytest.mark.parametrize("n,expected", [(1, 5), (2, 8), (3, 11)])
def test_ball_and_sticks_parameters(n, expected):
    assert ball_and_sticks_parameters(n) == expected


def test_max_supported_fibres_rejects_underdetermined():
    """7 volumes cannot support 2 sticks (8 parameters)."""
    assert max_supported_fibres(7) == 0
    assert max_supported_fibres(15) == 1
    assert max_supported_fibres(95) == 3


# --- 5TT --------------------------------------------------------------------

def test_build_5tt_partition_of_unity(tmp_path):
    rng = np.random.default_rng(1)
    shape = (8, 8, 6)
    gm, wm, csf = (rng.random(shape).astype(np.float32) for _ in range(3))
    paths = [
        _write_image(tmp_path / f"{n}.nii.gz", d)
        for n, d in (("gm", gm), ("wm", wm), ("csf", csf))
    ]
    mask = np.zeros(shape)
    mask[2:6, 2:6, 1:5] = 1
    mask_p = _write_image(tmp_path / "mask.nii.gz", mask)

    out = build_5tt_from_probseg(*paths, tmp_path / "5tt.nii.gz", brain_mask=mask_p)
    data = nib.load(str(out)).get_fdata()

    assert data.shape == (*shape, 5)
    total = data.sum(axis=-1)
    assert np.allclose(total[mask > 0], 1.0, atol=1e-5)
    assert np.allclose(total[mask == 0], 0.0)
    assert np.allclose(data[..., 4], 0.0)          # no pathology volume
    # Without an atlas, all grey matter is cortical.
    assert np.allclose(data[..., 1], 0.0)


def test_build_5tt_rejects_shape_mismatch(tmp_path):
    a = _write_image(tmp_path / "gm.nii.gz", np.ones((4, 4, 4)))
    b = _write_image(tmp_path / "wm.nii.gz", np.ones((4, 4, 4)))
    c = _write_image(tmp_path / "csf.nii.gz", np.ones((5, 5, 5)))
    with pytest.raises(ValueError, match="differ in shape"):
        build_5tt_from_probseg(a, b, c, tmp_path / "5tt.nii.gz")


# --- coverage ---------------------------------------------------------------

def test_compute_node_coverage(tmp_path):
    """Coverage is the fraction of each node inside the DWI field of view."""
    par = np.zeros((10, 10, 4), dtype=np.int32)
    par[0:4, :, :] = 1        # fully inside the FOV below
    par[4:8, :, :] = 2        # half inside
    par[8:10, :, :] = 3       # fully outside
    fov = np.zeros((10, 10, 4))
    fov[0:6, :, :] = 1

    p = _write_image(tmp_path / "par.nii.gz", par)
    f = _write_image(tmp_path / "fov.nii.gz", fov)
    df = compute_node_coverage(p, f).set_index("node")

    assert df.loc[1, "coverage"] == pytest.approx(1.0)
    assert df.loc[2, "coverage"] == pytest.approx(0.5)
    assert df.loc[3, "coverage"] == pytest.approx(0.0)


def test_compute_node_coverage_rejects_mismatched_grids(tmp_path):
    p = _write_image(tmp_path / "par.nii.gz", np.ones((4, 4, 4)))
    f = _write_image(tmp_path / "fov.nii.gz", np.ones((5, 5, 5)))
    with pytest.raises(ValueError, match="same grid"):
        compute_node_coverage(p, f)
