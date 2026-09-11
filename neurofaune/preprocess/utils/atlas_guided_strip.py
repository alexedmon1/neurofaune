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

DEFAULT_QC = {
    'volume_mm3': (1700.0, 2600.0),   # plausible rat brain incl. surface CSF
    'atlas_coverage': 0.95,           # fraction of the parcellation inside the mask
    'non_brain_mm3': 400.0,           # tissue well outside the parcellation
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
    thresholds: dict | None = None,
    anisotropy_axis: int = 2,
) -> dict:
    """Check a refined mask and say which gate failed.

    Necessary because the union step makes the mask follow the registration: a
    misregistration produces a mask that looks plausible in isolation. These gates
    catch it by comparing against the parcellation and against physiology.
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

    return {'volume_mm3': volume, 'atlas_coverage': coverage,
            'non_brain_mm3': non_brain, 'passed': not failures,
            'failures': failures}
