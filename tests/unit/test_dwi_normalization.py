"""Unit tests for DWI intensity normalization (``normalize_dwi_intensity``).

Pins the transform to a PURE MULTIPLICATIVE rescale so the historical failure
cannot return. The old version clipped to [p_min, p_max] and subtracted p_min.
Because p_max is a percentile pooled over all volumes, and b0 volumes are both
much brighter than diffusion-weighted ones and a small minority of the volumes,
p_max landed *below* the b0 brain signal — so the clip truncated S0 across ~90%
of brain voxels, flattening every decay curve and driving the fitted kurtosis
negative in ~40% of voxels.

The invariant that matters for diffusion is that S(b)/S0 survives untouched:
normalization may rescale, but it may not deform the signal.
"""
import nibabel as nib
import numpy as np
import pytest

from neurofaune.preprocess.utils.dwi_utils import normalize_dwi_intensity

SHAPE = (12, 12, 4)


def _multishell(tmp_path, n_b0=5, n_dw=30, s0=12000.0, seed=0):
    """A 4D phantom shaped like a real multi-shell acquisition.

    Few bright b0 volumes + many dimmer diffusion-weighted volumes, which is
    exactly the geometry that made the pooled percentile clip destroy S0.
    """
    rng = np.random.default_rng(seed)
    vols = [np.full(SHAPE, s0) + rng.normal(0, 20, SHAPE) for _ in range(n_b0)]
    for shell in (1000.0, 2000.0, 3000.0):
        atten = np.exp(-shell * 0.7e-3)
        vols += [np.full(SHAPE, s0 * atten) + rng.normal(0, 20, SHAPE)
                 for _ in range(n_dw)]
    data = np.stack(vols, axis=-1)
    path = tmp_path / "dwi.nii.gz"
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)
    return path, data


def test_normalization_is_a_pure_scale(tmp_path):
    src, data = _multishell(tmp_path)
    out, params = normalize_dwi_intensity(src, tmp_path / "norm.nii.gz")
    got = nib.load(out).get_fdata()

    # One scalar multiply, nothing else: no clip, no floor, no offset.
    assert np.allclose(got, data * params["scale_factor"], rtol=1e-5)


def test_b0_signal_is_not_truncated(tmp_path):
    """The regression itself: the brightest (b0) voxels must survive."""
    src, data = _multishell(tmp_path)
    out, params = normalize_dwi_intensity(src, tmp_path / "norm.nii.gz")
    got = nib.load(out).get_fdata()

    b0_in, b0_out = data[..., :5], got[..., :5]
    # b0 keeps its spread — a clip would collapse it onto a constant ceiling.
    assert b0_out.std() == pytest.approx(b0_in.std() * params["scale_factor"], rel=1e-3)
    # and nothing piles up at the target ceiling
    assert (b0_out >= params["target_max"] * 0.999).mean() < 0.5


def test_diffusion_ratios_are_invariant(tmp_path):
    """S(b)/S0 must be untouched — every DTI/DKI/NODDI metric depends on it."""
    src, data = _multishell(tmp_path)
    out, _ = normalize_dwi_intensity(src, tmp_path / "norm.nii.gz")
    got = nib.load(out).get_fdata()

    before = data[..., 5:] / data[..., :1]
    after = got[..., 5:] / got[..., :1]
    assert np.allclose(before, after, rtol=1e-5)


def test_rescales_wildly_different_input_scales_to_same_range(tmp_path):
    """The function's actual purpose: consistent range for BET across recon scales."""
    ranges = []
    for i, slope in enumerate((1.0, 1e-4, 3e3)):
        src, _ = _multishell(tmp_path, seed=i)
        data = nib.load(src).get_fdata() * slope
        scaled = tmp_path / f"scaled{i}.nii.gz"
        nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), scaled)

        out, params = normalize_dwi_intensity(scaled, tmp_path / f"n{i}.nii.gz")
        ranges.append(nib.load(out).get_fdata().max())
        assert params["was_normalized"]

    # All three land at a comparable magnitude regardless of input scaling.
    assert max(ranges) / min(ranges) < 1.05


def test_brain_extraction_copy_does_clip(tmp_path):
    """The masking helper is the ONE place clipping is wanted."""
    from neurofaune.preprocess.utils.dwi_utils import normalize_for_brain_extraction

    src, data = _multishell(tmp_path)
    out, params = normalize_for_brain_extraction(src, tmp_path / "strip.nii.gz")
    got = nib.load(out).get_fdata()

    assert got.max() == pytest.approx(params["target_max"], rel=1e-6)
    assert params["masking_only"] is True
    # explicitly NOT a pure scale — that's the point
    assert not np.allclose(got, data * params["scale_factor"], rtol=1e-3)


def test_brain_extraction_reference_saturates_the_b0(tmp_path):
    """Percentiles from the full 4D must saturate the b0 into one plateau.

    Regression: taking percentiles from the b0 alone leaves only its top 0.5%
    at the ceiling, the bright plateau never forms, atropos picks a small hot
    subset and the mask collapses (observed: 46k brain -> 794 voxels).
    """
    from neurofaune.preprocess.utils.dwi_utils import normalize_for_brain_extraction

    # Geometry matters here: the brain is a small BRIGHT minority of the volume
    # (~7% in this cohort). Pooled over 95 volumes, every b0 brain voxel then
    # falls inside the top 0.5%, so p_max lands below the whole brain and it
    # saturates. Taken from the b0 alone, p99.5 sits near the brain's own max
    # and almost nothing saturates.
    shape = (10, 10, 4)
    rng = np.random.default_rng(3)
    brain = np.zeros(shape, bool)
    brain[3:7, 3:7, 1:3] = True                      # 32/400 voxels = 8%
    s0 = np.where(brain, rng.normal(20_000, 2_000, shape),
                  rng.normal(1_500, 50, shape))
    vols = [s0 + rng.normal(0, 50, shape) for _ in range(5)]
    for shell in (1018.0, 2021.0, 3025.0):
        dw = np.where(brain, s0 * np.exp(-shell * 0.7e-3), s0)
        vols += [dw + rng.normal(0, 50, shape) for _ in range(30)]
    data = np.stack(vols, axis=-1)
    src = tmp_path / "dwi4d.nii.gz"
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), src)
    b0 = tmp_path / "b0.nii.gz"
    nib.save(nib.Nifti1Image(data[..., 0].astype(np.float32), np.eye(4)), b0)

    pooled = nib.load(normalize_for_brain_extraction(
        b0, tmp_path / "pooled.nii.gz", reference_file=src)[0]).get_fdata()
    alone = nib.load(normalize_for_brain_extraction(
        b0, tmp_path / "alone.nii.gz")[0]).get_fdata()

    # measured over BRAIN voxels — those are the ones that must form the plateau
    at_ceiling = lambda a: (a[brain] >= a.max() * 0.999).mean()
    assert at_ceiling(pooled) > 0.9, "pooled reference must saturate the brain"
    assert at_ceiling(alone) < 0.1, "b0-only reference leaves no plateau"


def test_all_zero_input_is_passed_through(tmp_path):
    src = tmp_path / "zeros.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(SHAPE + (4,), dtype=np.float32), np.eye(4)), src)
    out, params = normalize_dwi_intensity(src, tmp_path / "norm.nii.gz")
    assert params["was_normalized"] is False
    assert not np.any(nib.load(out).get_fdata())
