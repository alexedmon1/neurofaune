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
from scipy import ndimage

from neurofaune.preprocess.utils.atlas_guided_strip import (
    DEFAULT_QC,
    compute_brain_mask_qc,
    refine_boundary,
    refine_iterative,
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
    half = seed.copy()
    half[11:] = False
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


# --- registration-independent gates ------------------------------------------

def test_parcellation_relative_gates_are_blind_to_misregistration():
    """The failure this module can cause: mask and parcellation move together.

    Both `atlas_coverage` and `non_brain_mm3` are measured against the
    parcellation, so a registration error that moves both leaves them untouched.
    This test exists so that blind spot is never mistaken for a clean bill.
    """
    raw, seed, parcellation = _scene()
    mask = segment_brain_atlas_guided(raw, seed, parcellation)
    good = compute_brain_mask_qc(mask, parcellation, voxel_mm3=2.5,
                                 thresholds={'volume_mm3': (0.0, 1e9)})
    shifted = compute_brain_mask_qc(np.roll(mask, 4, axis=0),
                                    np.roll(parcellation, 4, axis=0), voxel_mm3=2.5,
                                    thresholds={'volume_mm3': (0.0, 1e9)})
    assert shifted['atlas_coverage'] == pytest.approx(good['atlas_coverage'])
    assert shifted['non_brain_mm3'] == pytest.approx(good['non_brain_mm3'])


def test_tissue_fraction_catches_a_mask_sitting_on_air():
    """Registration-independent: it asks the image, not the parcellation."""
    raw, seed, parcellation = _scene()
    mask = segment_brain_atlas_guided(raw, seed, parcellation)
    onto_air = np.roll(mask, 8, axis=0)
    qc = compute_brain_mask_qc(onto_air, np.roll(parcellation, 8, axis=0),
                               voxel_mm3=2.5, raw=raw,
                               thresholds={'volume_mm3': (0.0, 1e9)})
    assert qc['tissue_fraction'] < 0.9
    assert any('tissue fraction' in f for f in qc['failures'])


def test_dice_with_initial_catches_a_relocated_brain():
    raw, seed, parcellation = _scene()
    mask = segment_brain_atlas_guided(raw, seed, parcellation)
    qc = compute_brain_mask_qc(np.roll(mask, 8, axis=0), parcellation, voxel_mm3=2.5,
                               initial_mask=mask,
                               thresholds={'volume_mm3': (0.0, 1e9),
                                           'atlas_coverage': 0.0})
    assert qc['dice_with_initial'] < 0.7
    assert any('moved rather than the boundary' in f for f in qc['failures'])


def test_optional_inputs_are_genuinely_optional():
    """Callers without the raw image still get the parcellation-relative checks."""
    raw, seed, parcellation = _scene()
    mask = segment_brain_atlas_guided(raw, seed, parcellation)
    qc = compute_brain_mask_qc(mask, parcellation, voxel_mm3=2.5,
                               thresholds={'volume_mm3': (0.0, 1e9)})
    assert np.isnan(qc['tissue_fraction']) and np.isnan(qc['dice_with_initial'])
    assert qc['passed']


def test_dice_threshold_sits_below_the_observed_range():
    """0.70 is a catastrophe detector, not a quality score.

    Across 92 real sessions Dice-vs-initial spanned 0.780-0.923 with no Tukey
    outliers, and it correlates -0.98 with how much non-brain the ORIGINAL mask
    carried -- i.e. it measures how bad the input was, not how good the output is
    (correlation with the refined tissue fraction is -0.01). The threshold must
    therefore sit below the whole observed range.
    """
    assert DEFAULT_QC['dice_with_initial'] < 0.78


# --- iterative refinement ----------------------------------------------------
# The loop's contract is that it must NOT simply run to convergence: it registers
# each pass to the previously stripped image, so a clipped mask teaches the next
# pass to clip further and the sequence contracts onto a shrunken brain. What is
# pinned here is that it stops on coverage against a FIXED reference, that it keeps
# the best-covering pass rather than the last, and that over-inclusion cannot win.

def _register_stub(parcellation, shrink=0.0):
    """`register(mask)` returning the atlas guide eroded by `shrink` per call.

    `shrink=0` is a stable registration; a positive value mimics the failure mode
    where each pass registers to the previously stripped brain and the guide
    follows the clipping inward.
    """
    state = {'calls': 0}

    def register(mask):
        state['calls'] += 1
        guide = parcellation > 0
        for _ in range(int(round(shrink * state['calls']))):
            guide = ndimage.binary_erosion(guide)
        return guide, np.where(guide, parcellation, 0)

    register.state = state
    return register


def test_stops_as_soon_as_coverage_reaches_the_target():
    """A pass that covers the reference well enough ends the loop immediately."""
    raw, seed, parcellation = _scene()
    register = _register_stub(parcellation)
    mask, history = refine_iterative(raw, seed, register, voxel_mm3=1.0,
                                     reference_parcellation=parcellation,
                                     iterations=5, target_coverage=0.99)
    assert len(history) == 1
    assert register.state['calls'] == 1
    assert history[0]['atlas_coverage'] >= 0.99
    assert history[0]['selected']
    assert mask.any()


def test_keeps_the_best_covering_pass_not_the_last():
    """When later passes contract, the earlier better-covering mask is returned."""
    raw, seed, parcellation = _scene()
    mask, history = refine_iterative(raw, seed, _register_stub(parcellation, shrink=1.0),
                                     voxel_mm3=1.0, reference_parcellation=parcellation,
                                     iterations=3, target_coverage=1.01)
    coverages = [h['atlas_coverage'] for h in history]
    assert coverages[-1] < coverages[0], 'stub should contract'
    selected = [h for h in history if h['selected']]
    assert len(selected) == 1
    assert selected[0]['iteration'] == 1
    assert mask.sum() == pytest.approx(history[0]['volume_mm3'])


def test_convergence_alone_does_not_stop_the_loop():
    """A settled volume is recorded but must not end the loop -- it can settle wrong.

    The stub returns the identical mask every pass, so |dVolume| is 0 from pass 2
    on. With the coverage target unreachable the loop must still run to its cap.
    """
    raw, seed, parcellation = _scene()
    _, history = refine_iterative(raw, seed, _register_stub(parcellation),
                                  voxel_mm3=1.0, reference_parcellation=parcellation,
                                  iterations=3, target_coverage=1.01,
                                  tolerance_mm3=1.0)
    assert len(history) == 3, 'a settled volume must not stop the loop'
    assert history[-1]['settled']


def test_coverage_is_scored_against_the_fixed_reference():
    """Scoring against the pass's own parcellation would hide the contraction.

    The stub shrinks guide and parcellation together, so coverage measured
    self-relatively stays put while coverage against the original falls. Only the
    latter can drive the stop rule.
    """
    raw, seed, parcellation = _scene()
    _, history = refine_iterative(raw, seed, _register_stub(parcellation, shrink=1.0),
                                  voxel_mm3=1.0, reference_parcellation=parcellation,
                                  iterations=2, target_coverage=1.01)
    assert history[1]['atlas_coverage'] < history[0]['atlas_coverage']


def test_an_over_inclusive_pass_cannot_win_on_coverage(monkeypatch):
    """Coverage rises with mask size, so the gates are what stop a bloated mask.

    Pass 1 swallows the whole image -- 100% coverage, and the highest possible --
    but fails the non-brain gate. Pass 2 is a tight correct mask. The tight one
    must be returned, otherwise coverage is selecting on its own.
    """
    raw, seed, parcellation = _scene()
    tight = segment_brain_atlas_guided(raw, parcellation > 0, parcellation)
    masks = [np.ones_like(seed), tight]
    monkeypatch.setattr(
        'neurofaune.preprocess.utils.atlas_guided_strip.segment_brain_atlas_guided',
        lambda *a, **k: masks.pop(0))

    mask, history = refine_iterative(raw, seed, lambda m: (m, parcellation),
                                     voxel_mm3=1.0, reference_parcellation=parcellation,
                                     iterations=2, target_coverage=0.99)
    assert history[0]['atlas_coverage'] == pytest.approx(1.0)
    assert not history[0]['gates_passed'], 'a whole-image mask must fail the gates'
    assert history[1]['selected']
    assert mask.sum() == tight.sum()


def test_falls_back_to_best_coverage_when_no_pass_clears_the_gates():
    """Never return nothing: if every pass fails the gates, keep the best anyway."""
    raw, seed, parcellation = _scene()
    _, history = refine_iterative(
        raw, seed, _register_stub(parcellation), voxel_mm3=1.0,
        reference_parcellation=parcellation, iterations=2, target_coverage=1.01,
        qc_thresholds={'non_brain_mm3': -1.0})
    assert not any(h['gates_passed'] for h in history)
    assert sum(h['selected'] for h in history) == 1


def test_history_describes_the_returned_mask():
    """The selected entry has to match the mask actually handed back."""
    raw, seed, parcellation = _scene()
    mask, history = refine_iterative(raw, seed, _register_stub(parcellation, shrink=1.0),
                                     voxel_mm3=0.5, reference_parcellation=parcellation,
                                     iterations=3, target_coverage=1.01)
    selected = next(h for h in history if h['selected'])
    assert selected['volume_mm3'] == pytest.approx(mask.sum() * 0.5)


def test_deltas_are_consistent_with_the_volume_series():
    """`delta_mm3` must be the step-to-step difference, including the first step."""
    raw, seed, parcellation = _scene()
    start = seed.sum() * 1.0
    _, history = refine_iterative(raw, seed, _register_stub(parcellation, shrink=1.0),
                                  voxel_mm3=1.0, reference_parcellation=parcellation,
                                  iterations=3, target_coverage=1.01)
    volumes = [start] + [h['volume_mm3'] for h in history]
    for i, entry in enumerate(history):
        assert entry['delta_mm3'] == pytest.approx(volumes[i + 1] - volumes[i])


def test_pass_one_uses_the_prior_transform_and_does_not_register():
    """The loop must start from the registration already on disk, not a fresh one.

    A one-shot registration of the atlas onto a mask-stripped subject is worse than
    the original chain through the study template -- 95.1% coverage against 98.7%
    on sub-2Z/ses-1 -- so starting the loop with `register` makes pass 1 worse than
    doing nothing.
    """
    raw, seed, parcellation = _scene()
    register = _register_stub(parcellation)
    _, history = refine_iterative(raw, seed, register, voxel_mm3=1.0,
                                  reference_parcellation=parcellation,
                                  reference_guide=parcellation > 0,
                                  iterations=1, target_coverage=1.01)
    assert register.state['calls'] == 0
    assert not history[0]['registered']


def test_only_later_passes_register():
    """Passes 2+ are the ones that may re-register; pass 1 never does."""
    raw, seed, parcellation = _scene()
    register = _register_stub(parcellation)
    _, history = refine_iterative(raw, seed, register, voxel_mm3=1.0,
                                  reference_parcellation=parcellation,
                                  reference_guide=parcellation > 0,
                                  iterations=3, target_coverage=1.01)
    assert [h['registered'] for h in history] == [False, True, True]
    assert register.state['calls'] == 2


def test_one_pass_matches_the_plain_single_pass_refinement():
    """`iterations=1` with a prior guide must equal calling the refinement directly.

    This is what makes the loop a strict superset of the single pass: it can add
    passes, but it cannot change the answer the single pass would have given.
    """
    raw, seed, parcellation = _scene()
    direct = segment_brain_atlas_guided(raw, seed, parcellation)
    looped, _ = refine_iterative(raw, seed, _register_stub(parcellation), voxel_mm3=1.0,
                                 reference_parcellation=parcellation,
                                 reference_guide=seed, iterations=1,
                                 target_coverage=1.01)
    assert np.array_equal(direct, looped)
