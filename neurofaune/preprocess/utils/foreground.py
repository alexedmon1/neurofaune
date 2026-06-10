"""
Shared foreground / background-noise estimation for skull stripping.

Every Atropos-based skull-strip path needs an *initial foreground mask* to seed
the segmentation. The historical approach — keep voxels above the 5th percentile
of the non-zero intensities — silently assumes the background is ≈ 0. On
reconstructions with a **non-zero Rician noise floor** (e.g. some Bruker
ParaVision recons) the 5th percentile keeps ~95% of the image, Atropos clusters
the whole head, and the result is a whole-head "brain" mask that the auto-QC does
not flag (see gh #12).

This module replaces that assumption with an explicit noise-floor estimate. It
reuses the **validated** MSME air-noise estimator: the lowest-intensity air
voxels are Rayleigh-distributed magnitude noise, so for the noise mean ``m``::

    sigma = m / sqrt(pi/2)        sigma**2 = m**2 * (2/pi)

The foreground is then everything brighter than ``mu + k*sigma``, which is robust
to a non-zero floor and to a bright tail (skull, muscle, fat).

Two entry points:

- :func:`estimate_noise_floor` — ``(img, mask=None) -> NoiseFloor``. With a mask
  it reproduces the MSME background estimator bit-for-bit (so the MSME workflow
  can delegate to it). Without a mask it uses the lowest-quartile of the whole
  image as the air proxy.
- :func:`foreground_mask` — ``(img, k=...) -> ndarray[uint8]`` seed mask for
  Atropos.
"""

from typing import NamedTuple, Optional

import numpy as np


class NoiseFloor(NamedTuple):
    """Background magnitude-noise estimate (Rayleigh model).

    Attributes
    ----------
    mean : float
        Mean intensity of the selected air/noise voxels (``m`` above). This is
        the floor the foreground threshold is measured from.
    sigma : float
        Gaussian-equivalent noise std, ``sqrt(mean**2 * 2/pi)``.
    sigma2 : float
        Gaussian-equivalent noise variance, ``mean**2 * 2/pi``. Kept as a field
        (rather than ``sigma**2``) so callers that need the variance get the
        exact value the MSME workflow has always used.
    n_noise_voxels : int
        Number of voxels used for the estimate.
    used_fallback : bool
        True if the masked path fell back to the lowest-``fallback_quantile`` of
        all voxels because the background had too few voxels (mask covers nearly
        the whole image).
    """

    mean: float
    sigma: float
    sigma2: float
    n_noise_voxels: int
    used_fallback: bool


def estimate_noise_floor(
    img_data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    bg_quantile: float = 0.25,
    fallback_quantile: float = 0.10,
    min_bg_voxels: int = 100,
) -> NoiseFloor:
    """Estimate the background magnitude-noise floor via a Rayleigh model.

    The lowest-intensity air voxels are pure Rayleigh-distributed magnitude
    noise. Taking their mean ``m`` gives a scale-aware noise estimate::

        sigma**2 = m**2 * (2/pi)

    Parameters
    ----------
    img_data : np.ndarray
        Image intensities (any shape). Not modified.
    mask : np.ndarray, optional
        Boolean/binary brain mask. If given, noise is estimated from the
        background **outside** the mask (the MSME strategy). If ``None``, the
        whole image is used and the lowest ``bg_quantile`` of the non-zero
        voxels serves as the air proxy — the right choice *before* any mask
        exists (the skull-strip seeding use case).
    bg_quantile : float
        Lower quantile of the candidate-noise voxels kept as the pure noise
        floor (default 0.25 — the MSME value). The brighter tail of the
        background (skull/muscle/fat) is discarded.
    fallback_quantile : float
        Masked path only: if the background has < ``min_bg_voxels`` voxels (mask
        covers nearly the whole image), fall back to the lowest
        ``fallback_quantile`` of *all* non-zero voxels (default 0.10).
    min_bg_voxels : int
        Threshold for triggering the masked fallback (default 100).

    Returns
    -------
    NoiseFloor
    """
    arr = np.asarray(img_data)

    used_fallback = False
    if mask is not None:
        background = arr[~np.asarray(mask).astype(bool)]
        background = background[background > 0]  # exclude exact zeros
        if len(background) < min_bg_voxels:
            # Mask covers nearly the entire image — the "background" is
            # unreliable. Fall back to the lowest fraction of ALL non-zero
            # voxels as a noise proxy (matches MSME's degenerate-mask path).
            used_fallback = True
            candidates = arr[arr > 0]
            cut = np.percentile(candidates, fallback_quantile * 100.0)
            noise_voxels = candidates[candidates <= cut]
        else:
            cut = np.percentile(background, bg_quantile * 100.0)
            noise_voxels = background[background <= cut]
    else:
        # No mask yet: the whole-image lowest quartile is the air/noise floor.
        candidates = arr[arr > 0]
        if len(candidates) == 0:
            raise ValueError("Image has no non-zero voxels for noise estimation")
        cut = np.percentile(candidates, bg_quantile * 100.0)
        noise_voxels = candidates[candidates <= cut]

    if len(noise_voxels) == 0:
        raise ValueError("No noise voxels selected for floor estimation")

    # Rayleigh magnitude noise: mean = sigma * sqrt(pi/2).
    # Compute sigma2 the way the MSME workflow always has, so a caller that
    # delegates here gets bit-identical numbers.
    noise_mean = float(np.mean(noise_voxels))
    sigma2 = noise_mean ** 2 * (2.0 / np.pi)
    sigma = float(np.sqrt(sigma2))

    return NoiseFloor(
        mean=noise_mean,
        sigma=sigma,
        sigma2=sigma2,
        n_noise_voxels=int(len(noise_voxels)),
        used_fallback=used_fallback,
    )


def foreground_mask(
    img_data: np.ndarray,
    k: float = 4.0,
    floor: Optional[NoiseFloor] = None,
    bg_quantile: float = 0.25,
) -> np.ndarray:
    """Initial foreground mask robust to a non-zero background noise floor.

    Keeps voxels brighter than ``mu + k*sigma``, where ``(mu, sigma)`` is the
    Rayleigh background estimate from :func:`estimate_noise_floor`. This replaces
    the "above the 5th percentile" seed that assumed background ≈ 0.

    Parameters
    ----------
    img_data : np.ndarray
        Image intensities.
    k : float
        Sigma multiplier above the floor (default 4.0). This is the one real
        tunable — validate it visually on the target contrast before a cohort
        run; typical range 3–5.
    floor : NoiseFloor, optional
        Precomputed noise floor. If ``None`` it is estimated from ``img_data``
        with no mask.
    bg_quantile : float
        Passed through to :func:`estimate_noise_floor` when ``floor`` is None.

    Returns
    -------
    np.ndarray
        uint8 foreground mask, same shape as ``img_data``.
    """
    arr = np.asarray(img_data)
    if floor is None:
        floor = estimate_noise_floor(arr, mask=None, bg_quantile=bg_quantile)

    threshold = floor.mean + k * floor.sigma
    return (arr > threshold).astype(np.uint8)
