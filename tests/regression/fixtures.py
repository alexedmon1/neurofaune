"""Canonical synthetic inputs for regression tests.

This is the single well every regression test and every candidate draws from,
so candidates are comparable. Inputs are defined by a generator + fixed seed
(matching the seeded-synthetic style of the unit tests), so only the *golden
output* needs to be committed — the input regenerates identically.

Keep these tiny (the contract must be committable) and deterministic. Add a
new generator here when a new data *shape* is needed; reuse existing ones
across tests wherever possible.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

# A single project-wide default seed keeps "the canonical input" unambiguous.
DEFAULT_SEED = 20260526


def roi_subject_matrix(n_subjects: int = 24, n_rois: int = 20, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Subjects × ROIs feature matrix (connectivity / correlation inputs)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_subjects, n_rois))


def correlation_vector(n: int = 100, seed: int = 7) -> np.ndarray:
    """A vector of correlation values in (-1, 1) (e.g. for Fisher-z)."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.uniform(-0.95, 0.95, size=n), -0.999, 0.999)


def connectome(n: int = 20, density: float = 0.2, seed: int = 7) -> np.ndarray:
    """Symmetric weighted adjacency matrix, zero diagonal (graph metrics)."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.0, 1.0, size=(n, n))
    w = np.triu(w, k=1)
    mask = rng.uniform(0.0, 1.0, size=(n, n)) < density
    w = w * np.triu(mask, k=1)
    a = w + w.T
    return a


def tmap_volume(shape: tuple[int, ...] = (8, 8, 8), seed: int = 42) -> np.ndarray:
    """Small synthetic t-statistic volume (voxel-based analysis inputs)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 2.0).astype(np.float32)


def tmap_nifti(shape: tuple[int, ...] = (8, 8, 8), seed: int = 42) -> nib.Nifti1Image:
    """``tmap_volume`` wrapped as a NIfTI with an identity affine."""
    return nib.Nifti1Image(tmap_volume(shape, seed), np.eye(4))
