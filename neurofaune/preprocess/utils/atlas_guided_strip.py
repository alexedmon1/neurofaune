"""Atlas-guided refinement of a brain mask (second pass).

Rodent skull-stripping fails in two directions at once, and on this cohort it fails
in both: measured over 92 sessions the mask carries a median 779 mm3 of non-brain
(skull, muscle, extra-axial CSF at ~3x background) while simultaneously clipping a
median 27.8% of the olfactory bulb -- tissue sitting at ~8x background that the
strip threw away. 70 of 92 sessions lose more than a quarter of the bulb, and
because the amount lost varies between animals (SD 6.8 percentage points) it adds
between-animal noise rather than a constant bias.

This runs as a SECOND PASS, after an initial strip and the registration it enables:

  initial strip -> register to atlas -> segment_brain_atlas_guided() -> (optionally re-register)

Three steps, each doing only what it is reliable at:

1. **Seed from the existing transform.** Warp the atlas brain mask into subject
   space through the registration already computed. This is reproducible and fixes
   the over-inclusion outright (779 -> ~140 mm3). What it cannot do is recover the
   bulb, because that registration was computed on the already-stripped image, so
   the transform never saw the clipped tissue.

   A fresh full-head-to-subject registration was tried instead and rejected: repeat
   runs of the identical command returned 2243, 3956, 3986 and 2394 mm3 -- it settles
   either on the brain or on a scaled-up fit to the wider subject FOV. Fixing the
   random seed did not make it deterministic, and of the metrics tried only `mattes`
   reliably found the right basin, still with ~200 mm3 of run-to-run jitter.

2. **Refine the boundary on the raw, unstripped image.** Seeds come from where the
   sources agree; only the disputed band is settled by intensity, which avoids the
   global-threshold instability (an Otsu threshold landed above the brain median on
   some subjects because bright fat and muscle skew the histogram).

3. **Union with the atlas extent where there is tissue.** The walker cannot reach
   the bulb -- an intensity discontinuity separates it from the main brain, and
   sweeping the diffusion parameter from 30 to 200 moves retention only 75.0 to
   75.6%. So voxels the atlas calls brain AND that carry real signal are added back.
   Requiring signal is what keeps this from being "trust the atlas blindly".

Measured on six sessions spanning all four cohorts: atlas coverage 98.4-99.0%
(against 97.0-97.8% for the current mask, which only reaches that by being
over-inclusive), olfactory bulb 96.2-97.7% (against 58.5-87.2%), non-brain
114-135 mm3 (against 427-1199). Between-animal spread in volume falls from 923 mm3
to 177, and in bulb retention from 28.7 points to 1.5.

**Known consequence.** Step 3 makes the mask partly atlas-determined, so it can no
longer serve as an independent check on the registration: if the registration is
wrong somewhere the mask will follow it there. That is what `compute_brain_mask_qc` is for.
"""

import logging

import numpy as np
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

# Never dilate along the through-plane axis: rodent anatomicals are strongly
# anisotropic (0.125 x 0.125 x 0.8 mm here), so one step out of plane moves the
# boundary six times further than one step within it.
INPLANE = np.ones((3, 3, 3), bool)
INPLANE[:, :, ::2] = False

# Named refinement methods, selected by `<modality>.skull_strip.refine.method`.
# These are NOT peers of the first-pass methods (atropos/bet/ants): those run
# before any registration exists, whereas a refinement needs one. A refinement
# therefore always runs as a second stage, after the initial strip and the
# registration it enables.
REFINE_METHODS = ('atlas_iterative', 'none')
DEFAULT_METHOD = 'atlas_iterative'

# Stop on the volume increment, not on overlap. At ~2100 mm3 a 16 mm3 change is
# Dice 0.991 against the previous pass, which reads as converged long before it is.
DEFAULT_TOLERANCE_MM3 = 25.0
DEFAULT_TARGET_COVERAGE = 0.99
DEFAULT_ITERATIONS = 1

DEFAULT_QC = {
    # Physiological plausibility. Widened from an initial 1700-2600 after the first
    # cohort run: the observed refined range was 2127-2624 mm3, so the old ceiling
    # flagged a visually correct mask on a technicality.
    'volume_mm3': (1700.0, 2700.0),
    'atlas_coverage': 0.95,           # fraction of the parcellation inside the mask
    'non_brain_mm3': 400.0,           # tissue well outside the parcellation
    # Independent of the registration -- see the module docstring.
    'tissue_fraction': 0.90,          # masked voxels carrying signal, not air
    'dice_with_initial': 0.70,        # a refinement moves a boundary, not a brain
}


def _inplane_structure(anisotropy_axis: int) -> np.ndarray:
    structure = np.ones((3, 3, 3), bool)
    structure[(slice(None),) * anisotropy_axis + (slice(None, None, 2),)] = False
    return structure


def refine_boundary(
    raw: np.ndarray,
    seed: np.ndarray,
    reach: np.ndarray,
    erode: int = 2,
    band: int = 3,
    beta: float = 90.0,
    anisotropy_axis: int = 2,
) -> np.ndarray:
    """Settle the mask boundary on raw intensity, seeded from confident regions.

    `seed` is the region confidently inside the brain; `reach` bounds where the
    boundary is allowed to move. Voxels well outside `reach` seed the background.
    Only the band between them is decided, so no global threshold is needed.
    """
    from skimage.segmentation import random_walker

    structure = _inplane_structure(anisotropy_axis)
    inner = ndi.binary_erosion(seed, structure=structure, iterations=erode)
    outer = ~ndi.binary_dilation(reach, structure=structure, iterations=band)
    if not inner.any() or not outer.any():
        logger.warning('Refinement skipped: seeds are empty after erosion/dilation')
        return seed

    markers = np.zeros(raw.shape, np.uint8)
    markers[inner] = 1
    markers[outer] = 2

    # Solve only around the boundary; everything else is already settled.
    idx = np.argwhere(ndi.binary_dilation(reach, structure=structure, iterations=band + 2))
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = np.minimum(idx.max(0) + 3, raw.shape)
    window = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))

    x = raw[window].astype(np.float64)
    x = (x - x.min()) / max(1e-9, np.ptp(x))

    out = np.zeros(raw.shape, bool)
    out[window] = random_walker(x, markers[window], beta=beta, mode='cg_j') == 1
    out = ndi.binary_fill_holes(out)
    labelled, n = ndi.label(out)
    if n:
        out = labelled == (np.bincount(labelled.ravel())[1:].argmax() + 1)
    return out


def segment_brain_atlas_guided(
    raw: np.ndarray,
    atlas_mask: np.ndarray,
    parcellation: np.ndarray,
    tissue_factor: float = 2.0,
    anisotropy_axis: int = 2,
    **refine_kwargs,
) -> np.ndarray:
    """Refined brain mask from the raw image, an atlas-derived seed and the labels.

    Parameters
    ----------
    raw : ndarray
        The UNSTRIPPED anatomical. Using the stripped image here would be circular
        -- the tissue to recover is exactly what the strip removed.
    atlas_mask : ndarray
        The atlas brain mask warped into subject space through the existing
        registration. Provides a reliable, well-placed seed.
    parcellation : ndarray
        The atlas labels in subject space, non-zero inside the brain. Bounds where
        the boundary may grow and supplies the union in step 3.
    tissue_factor : float
        A voxel counts as tissue above this multiple of the background median.
        Guards the union: the atlas alone never adds a voxel.
    """
    atlas_extent = parcellation > 0
    background = np.median(raw[~atlas_extent & ~atlas_mask])
    if not np.isfinite(background) or background <= 0:
        background = max(1e-6, float(np.percentile(raw, 5)))

    refined = refine_boundary(raw, seed=atlas_mask,
                              reach=atlas_mask | atlas_extent,
                              anisotropy_axis=anisotropy_axis, **refine_kwargs)
    tissue = raw > tissue_factor * background
    final = ndi.binary_fill_holes(refined | (atlas_extent & tissue))

    logger.info('Refined mask: %d voxels (seed %d, refined %d, +atlas-union %d)',
                int(final.sum()), int(atlas_mask.sum()), int(refined.sum()),
                int(final.sum() - refined.sum()))
    return final


def compute_brain_mask_qc(
    mask: np.ndarray,
    parcellation: np.ndarray,
    voxel_mm3: float,
    raw: np.ndarray | None = None,
    initial_mask: np.ndarray | None = None,
    thresholds: dict | None = None,
    anisotropy_axis: int = 2,
    tissue_factor: float = 2.0,
) -> dict:
    """Check a refined mask and say which gate failed.

    Pass `raw` and `initial_mask` whenever they are available. Without them only
    the parcellation-relative checks run, and those cannot see a misregistration --
    they are computed against a parcellation that a bad registration would have
    moved along with the mask. Shifting both together by 40 voxels leaves
    `atlas_coverage` and `non_brain_mm3` exactly unchanged.

    `tissue_fraction` is the share of masked voxels carrying signal above
    background: a mask sitting partly on air fails it wherever the parcellation
    went. `dice_with_initial` compares against the pre-existing mask -- a
    refinement moves a boundary, so a low value means the brain relocated.
    """
    thresholds = {**DEFAULT_QC, **(thresholds or {})}
    atlas_extent = parcellation > 0
    generous = ndi.binary_dilation(atlas_extent,
                                   structure=_inplane_structure(anisotropy_axis),
                                   iterations=2)

    volume = float(mask.sum() * voxel_mm3)
    coverage = float((atlas_extent & mask).sum() / max(1, atlas_extent.sum()))
    non_brain = float((mask & ~generous).sum() * voxel_mm3)
    lo, hi = thresholds['volume_mm3']

    failures = []
    if not lo <= volume <= hi:
        failures.append(f'volume {volume:.0f} mm3 outside {lo:.0f}-{hi:.0f}')
    if coverage < thresholds['atlas_coverage']:
        failures.append(f'atlas coverage {coverage:.1%} below '
                        f"{thresholds['atlas_coverage']:.0%}")
    if non_brain > thresholds['non_brain_mm3']:
        failures.append(f'non-brain {non_brain:.0f} mm3 above '
                        f"{thresholds['non_brain_mm3']:.0f}")

    result = {'volume_mm3': volume, 'atlas_coverage': coverage,
              'non_brain_mm3': non_brain,
              'tissue_fraction': float('nan'), 'dice_with_initial': float('nan')}

    if raw is not None and mask.any():
        outside = raw[~mask & ~atlas_extent]
        background = float(np.median(outside)) if outside.size else 0.0
        if not np.isfinite(background) or background <= 0:
            background = max(1e-6, float(np.percentile(raw, 5)))
        fraction = float((raw[mask] > tissue_factor * background).mean())
        result['tissue_fraction'] = fraction
        if fraction < thresholds['tissue_fraction']:
            failures.append(f'tissue fraction {fraction:.1%} below '
                            f"{thresholds['tissue_fraction']:.0%} -- the mask "
                            'covers voxels with no signal')

    if initial_mask is not None:
        initial = initial_mask > 0
        total = mask.sum() + initial.sum()
        dice = float(2 * (mask & initial).sum() / total) if total else float('nan')
        result['dice_with_initial'] = dice
        if np.isfinite(dice) and dice < thresholds['dice_with_initial']:
            failures.append(f'Dice with the initial mask {dice:.2f} below '
                            f"{thresholds['dice_with_initial']:.2f} -- the brain "
                            'moved rather than the boundary')

    result['passed'] = not failures
    result['failures'] = failures
    return result


def refine_iterative(
    raw: np.ndarray,
    initial_mask: np.ndarray,
    register: "callable",
    voxel_mm3: float,
    reference_parcellation: np.ndarray,
    reference_guide: np.ndarray | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
    tolerance_mm3: float = DEFAULT_TOLERANCE_MM3,
    anisotropy_axis: int = 2,
    qc_thresholds: dict | None = None,
    **refine_kwargs,
) -> tuple:
    """Alternate registration and refinement, keeping the best-covering pass.

    `register(mask) -> (atlas_mask, parcellation)` re-registers the atlas to the
    subject stripped with `mask`, returning the atlas brain mask and parcellation
    warped into subject space. Supplied by the caller so this stays independent of
    which registration backend and which atlas are in use.

    **Default `iterations=1`: measured over 16 sessions, re-registering never once
    helped.** Every pass beyond the first lowered atlas coverage, without exception
    -- pass 1 > pass 2 > pass 3 on 16 of 16 -- and the best-covering pass was pass 1
    every time, for a median gain of exactly 0.0000. The worst case, `sub-4C/ses-1`,
    ran 98.50% -> 90.95% -> 89.83%. Each pass registers to the *previously stripped*
    image, so anything the last mask clipped is missing from the next pass's target
    and the guide follows the clipping inward: a contraction that settles on the
    wrong fixed point. A volume-convergence rule cannot see this -- it called all 16
    converged, and on `sub-10C/ses-1` reported |dVolume| = 1 mm3 while
    olfactory-bulb retention fell from 97.6% to 75.9%.

    The loop is kept, defaulted off, because with the coverage rule below it is safe
    by construction: extra passes can only be *selected* if they cover better, so
    raising `iterations` cannot return a worse mask than pass 1. If a cohort ever
    arrives whose initial transform chain is poor enough for re-registration to win,
    the machinery is here and the history will show it.

    **Pass 1 re-uses the registration you already have.** `reference_guide` is the
    atlas brain mask warped through the original transform chain -- the one that
    produced `reference_parcellation`. Pass 1 refines against it and performs no
    registration at all, so the loop is a strict superset of the single-pass
    refinement and cannot do worse than it. Only passes 2+ call `register`. This is
    not a shortcut: the original chain goes through the study template and is better
    conditioned than a one-shot SyN of the atlas onto a mask-stripped subject, which
    is what `register` can offer. On `sub-2Z/ses-1` refining against the chain
    reaches 98.7% coverage while a fresh registration on the same session reaches
    95.1%. Omit `reference_guide` only if no prior guide exists, in which case pass 1
    registers like any other.

    So the loop stops on **atlas coverage against a fixed reference parcellation**
    and returns the pass that covered it best. Two conditions make that sound:

    1. `reference_parcellation` must be the parcellation from the *original*
       registration, held fixed across passes -- never the one `register` just
       returned. Coverage is a ratio against the parcellation, and a parcellation
       that shrinks along with the mask keeps that ratio high while the brain is
       being clipped. Scored against its own re-registered parcellation the
       `sub-10C` run above still looks fine; scored against the fixed one it drops
       2.1 points, which is the signal.
    2. Coverage rises monotonically with mask size -- a mask covering the whole
       image scores 100% -- so it cannot select on its own. The best pass is chosen
       among those that still pass the `non_brain_mm3` and `tissue_fraction` gates,
       which is what penalises over-inclusion. If no pass clears them, the best
       coverage is returned and a warning says nothing qualified.

    `tolerance_mm3` no longer ends the loop; it only marks a pass `settled` in the
    history, so a caller can tell "stopped because it stopped moving" apart from
    "stopped because coverage was good enough".

    **`iterations` defaults to 1, so by default nothing re-registers.** Over 16
    sessions spanning the full range of initial mask quality, coverage against the
    fixed reference fell at every single session and pass 1 was selected 16/16 --
    30 registrations that changed no output. The degradation was worst where the
    initial mask was worst (sub-4C/ses-1, 98.5% -> 91.0% over two passes), since a
    one-shot SyN onto a badly stripped subject is furthest from the chain it
    replaces. The loop is kept because the best-of-pass selection is what makes
    raising `iterations` safe to try on new data, not because iterating pays here.

    Returns
    -------
    (mask, history)
        `history` is one dict per iteration with `volume_mm3`, `delta_mm3`,
        `dice_with_previous`, `atlas_coverage`, `non_brain_mm3`,
        `tissue_fraction`, `gates_passed`, `settled`, `registered` and `selected`,
        so a caller can see which pass was kept and why the rest were not.
    """
    thresholds = {**DEFAULT_QC, **(qc_thresholds or {})}
    mask = initial_mask > 0
    previous_volume = float(mask.sum() * voxel_mm3)
    history: list[dict] = []
    candidates: list[tuple] = []

    for step in range(1, int(iterations) + 1):
        if step == 1 and reference_guide is not None:
            atlas_mask, parcellation = reference_guide > 0, reference_parcellation
        else:
            atlas_mask, parcellation = register(mask)
        nxt = segment_brain_atlas_guided(raw, atlas_mask, parcellation,
                                         anisotropy_axis=anisotropy_axis,
                                         **refine_kwargs)
        volume = float(nxt.sum() * voxel_mm3)
        total = nxt.sum() + mask.sum()
        dice = float(2 * (nxt & mask).sum() / total) if total else float('nan')
        delta = volume - previous_volume

        # Scored against the FIXED reference, not `parcellation` -- see above.
        qc = compute_brain_mask_qc(nxt, reference_parcellation, voxel_mm3, raw=raw,
                                   anisotropy_axis=anisotropy_axis,
                                   thresholds=thresholds)
        gates_passed = bool(
            qc['non_brain_mm3'] <= thresholds['non_brain_mm3']
            and (not np.isfinite(qc['tissue_fraction'])
                 or qc['tissue_fraction'] >= thresholds['tissue_fraction']))

        entry = {'iteration': step,
                 'registered': not (step == 1 and reference_guide is not None),
                 'volume_mm3': volume, 'delta_mm3': delta,
                 'dice_with_previous': dice,
                 'atlas_coverage': qc['atlas_coverage'],
                 'non_brain_mm3': qc['non_brain_mm3'],
                 'tissue_fraction': qc['tissue_fraction'],
                 'gates_passed': gates_passed,
                 'settled': abs(delta) < tolerance_mm3,
                 'selected': False}
        history.append(entry)
        candidates.append((nxt, entry))
        logger.info('  iteration %d (%s): %.0f mm3 (%+.0f), coverage %.1f%%, '
                    'tissue %.1f%%, non-brain %.0f mm3%s',
                    step, 'prior transform' if not entry['registered'] else 're-registered',
                    volume, delta, 100 * qc['atlas_coverage'],
                    100 * qc['tissue_fraction'], qc['non_brain_mm3'],
                    '' if gates_passed else '  [gates FAILED]')

        mask, previous_volume = nxt, volume
        if gates_passed and qc['atlas_coverage'] >= target_coverage:
            logger.info('  coverage %.1f%% reached the %.0f%% target; stopping',
                        100 * qc['atlas_coverage'], 100 * target_coverage)
            break
    else:
        if iterations > 1:
            logger.info('  ran all %d passes without reaching %.0f%% coverage',
                        iterations, 100 * target_coverage)
        else:
            # The default, and not a failure: coverage short of target at
            # iterations=1 only means no further pass was allowed, and further
            # passes do not help anyway.
            logger.debug('  single pass, coverage %.1f%% (target %.0f%%)',
                         100 * history[-1]['atlas_coverage'], 100 * target_coverage)

    eligible = [c for c in candidates if c[1]['gates_passed']]
    if not eligible:
        logger.warning('  no pass cleared the non-brain/tissue gates; keeping the '
                       'best coverage anyway')
        eligible = candidates
    best_mask, best_entry = max(eligible, key=lambda c: c[1]['atlas_coverage'])
    best_entry['selected'] = True
    if best_entry['iteration'] != history[-1]['iteration']:
        logger.info('  kept pass %d (coverage %.1f%%) over the final pass %d '
                    '(%.1f%%) -- later passes contracted',
                    best_entry['iteration'], 100 * best_entry['atlas_coverage'],
                    history[-1]['iteration'], 100 * history[-1]['atlas_coverage'])

    return best_mask, history
