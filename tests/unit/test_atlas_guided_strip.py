#!/usr/bin/env python3
"""
Unit tests for atlas-guided brain mask refinement.

The properties worth pinning are the ones that make this safe to run over a cohort:
the union must never add a voxel without image evidence, the boundary must never
grow along the anisotropic axis, and the QC gate must name what failed -- because
the union makes the mask follow the registration, so a misregistration yields a
mask that looks plausible in isolation.
"""

import numpy as np
import pytest

from neurofaune.preprocess.utils.atlas_guided_strip import (
    DEFAULT_QC,
    compute_brain_mask_qc,
    refine_boundary,
    segment_brain_atlas_guided,
)


def _scene(shape=(24, 24, 12)):
    """A bright block of 'brain' plus a detached bright 'bulb' the seed misses."""
    raw = np.full(shape, 10.0, np.float32)          # background
    parcellation = np.zeros(shape, np.int32)
    raw[6:16, 6:16, 3:9] = 100.0                    # main brain
    parcellation[6:16, 6:16, 3:9] = 1
    raw[16:20, 9:13, 4:8] = 95.0                    # bulb: bright, atlas says brain
    parcellation[16:20, 9:13, 4:8] = 2
    seed = np.zeros(shape, bool)
    seed[6:16, 6:16, 3:9] = True                    # seed covers brain, not bulb
    return raw, seed, parcellation


def test_union_recovers_tissue_the_seed_missed():
    """The bulb is bright and the atlas claims it, so it must come back."""
    raw, seed, parcellation = _scene()
    bulb = parcellation == 2
    assert not (seed & bulb).any()
    out = segment_brain_atlas_guided(raw, seed, parcellation)
    assert (out & bulb).sum() / bulb.sum() > 0.9


def test_union_never_adds_a_voxel_without_signal():
    """Atlas says brain but the image is background -- it must stay out."""
    raw, seed, parcellation = _scene()
    raw[16:20, 9:13, 4:8] = 10.0        # bulb region is now background intensity
    out = segment_brain_atlas_guided(raw, seed, parcellation)
    region = parcellation == 2
    assert (out & region).sum() / region.sum() < 0.1


def test_tissue_factor_controls_how_much_is_added():
    raw, seed, parcellation = _scene()
    raw[16:20, 9:13, 4:8] = 25.0        # 2.5x background
    lenient = segment_brain_atlas_guided(raw, seed, parcellation, tissue_factor=2.0)
    strict = segment_brain_atlas_guided(raw, seed, parcellation, tissue_factor=5.0)
    assert lenient.sum() > strict.sum()


def test_result_contains_the_seed_interior():
    raw, seed, parcellation = _scene()
    out = segment_brain_atlas_guided(raw, seed, parcellation)
    from scipy import ndimage as ndi
    assert out[ndi.binary_erosion(seed, iterations=2)].all()


def test_boundary_never_grows_along_the_anisotropic_axis():
    """One step out of plane moves six times further than one step within it."""
    raw, seed, parcellation = _scene()
    reach = np.ones_like(seed)
    out = refine_boundary(raw, seed, reach, anisotropy_axis=2)
    seed_z = set(np.argwhere(seed)[:, 2])
    out_z = set(np.argwhere(out)[:, 2])
    assert out_z <= seed_z


def test_connected_anatomy_yields_one_component():
    """A contiguous brain must not come out fragmented."""
    from scipy import ndimage as ndi

    raw, seed, parcellation = _scene()
    out = segment_brain_atlas_guided(raw, seed, parcellation)
    _, n = ndi.label(out)
    assert n == 1


# --- QC gate -----------------------------------------------------------------

def _mask_of_volume(parcellation, n_voxels):
    out = np.zeros(parcellation.shape, bool)
    out.ravel()[:n_voxels] = True
    return out


def test_qc_passes_a_good_mask():
    _, seed, parcellation = _scene()
    qc = compute_brain_mask_qc(seed | (parcellation > 0), parcellation, voxel_mm3=2.5,
                               thresholds={'volume_mm3': (1000.0, 3000.0)})
    assert qc['passed'] and not qc['failures']
    assert qc['atlas_coverage'] == pytest.approx(1.0)


def test_qc_flags_an_implausible_volume_and_says_so():
    _, _, parcellation = _scene()
    qc = compute_brain_mask_qc(parcellation > 0, parcellation, voxel_mm3=100.0)
    assert not qc['passed']
    assert any('volume' in f for f in qc['failures'])


def test_qc_flags_incomplete_atlas_coverage():
    """The failure the whole exercise exists to catch: brain left outside."""
    _, seed, parcellation = _scene()
    half = seed.copy(); half[11:] = False
    qc = compute_brain_mask_qc(half, parcellation, voxel_mm3=2.5,
                               thresholds={'volume_mm3': (0.0, 1e9)})
    assert not qc['passed']
    assert any('coverage' in f for f in qc['failures'])
    assert qc['atlas_coverage'] < 0.95


def test_qc_flags_non_brain_inclusion():
    _, _, parcellation = _scene()
    bloated = np.ones(parcellation.shape, bool)
    qc = compute_brain_mask_qc(bloated, parcellation, voxel_mm3=2.5,
                               thresholds={'volume_mm3': (0.0, 1e9)})
    assert not qc['passed']
    assert any('non-brain' in f for f in qc['failures'])


def test_qc_reports_every_failure_not_just_the_first():
    _, _, parcellation = _scene()
    qc = compute_brain_mask_qc(np.ones(parcellation.shape, bool), parcellation,
                               voxel_mm3=100.0)
    assert len(qc['failures']) >= 2


def test_default_thresholds_span_a_plausible_rat_brain():
    lo, hi = DEFAULT_QC['volume_mm3']
    assert lo < 2200 < hi          # measured refined masks land ~2100-2400 mm3
