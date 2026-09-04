"""Data-adequacy assessment for tractography.

Tractography degrades silently. A 6-direction acquisition will happily run
through ``bedpostx`` and ``probtrackx2`` and emit a full connectivity matrix
that is entirely model and coverage artifact, with nothing in the output to
say so. This module makes that failure loud and *upstream* of any compute.

The central fact this encodes: a q-space sampling of ``N`` directions supports
a spherical harmonic expansion of order ``lmax`` only when the number of even
SH coefficients ``(lmax+1)(lmax+2)/2`` does not exceed ``N``. Six directions
give exactly six coefficients — ``lmax=2`` — which carries precisely the
angular information of a diffusion tensor and nothing more. No amount of
downstream modelling recovers crossing fibres from data that never sampled
them, so the honest response is to refuse rather than to produce a matrix.

Used by every entry point in :mod:`neurofaune.tractography` as a precondition;
also useful standalone to triage a cohort before committing to a pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np

from neurofaune.preprocess.utils.dwi_utils import round_bvals_to_shells

logger = logging.getLogger(__name__)


# --- Model requirements -----------------------------------------------------

MIN_DIRS_TENSOR = 6
"""Directions needed to determine the six independent tensor components."""

MIN_DIRS_CSD = 28
"""Directions for a usable single-shell FOD (lmax=6). Below this, an FOD
carries no angular information the tensor does not already carry."""

MIN_SHELLS_MSMT = 2
"""Non-zero shells needed to separate tissue responses in MSMT-CSD. With one
shell the three-tissue decomposition is degenerate."""

MIN_B0 = 1
"""Minimum b=0 volumes. MSMT response estimation additionally needs b0 to
anchor the CSF/GM compartments; more is better."""


def sh_coefficients(lmax: int) -> int:
    """Number of even-order spherical harmonic coefficients up to ``lmax``."""
    return (lmax + 1) * (lmax + 2) // 2


def max_feasible_lmax(n_directions: int) -> int:
    """Largest even ``lmax`` whose SH basis is determined by ``n_directions``.

    Returns 0 if not even a single coefficient is supported.
    """
    lmax = 0
    while sh_coefficients(lmax + 2) <= n_directions:
        lmax += 2
    return lmax


@dataclass
class TractographyAdequacy:
    """Verdict on whether an acquisition can support tractography.

    Attributes
    ----------
    feasible : bool
        False when ``blockers`` is non-empty. Entry points refuse on False.
    recommended_model : str
        ``"msmt_csd"``, ``"csd"``, ``"tensor"`` or ``"none"``.
    shells : list of (b-value, n_volumes)
        Detected shells including b=0, ascending.
    n_shells : int
        Non-zero shells.
    min_dirs_per_shell : int
        Smallest direction count across non-zero shells.
    total_dw_directions : int
        All non-b0 volumes. MSMT-CSD fits the WM FOD against *all* shells
        jointly, so this — not the per-shell count — bounds the WM lmax.
    max_lmax_per_shell, wm_lmax : int
        Feasible SH order for a single shell, and the order recommended for
        the WM FOD under the recommended model.
    voxel_size_mm, anisotropy_ratio, coverage_mm : geometry, in real mm
        (i.e. already divided by ``voxel_scale``).
    warnings : list of str
        Degrades quality; does not block.
    blockers : list of str
        Makes the result uninterpretable; blocks.
    """

    feasible: bool
    recommended_model: str
    shells: List[Tuple[int, int]]
    n_shells: int
    min_dirs_per_shell: int
    total_dw_directions: int
    n_b0: int
    max_lmax_per_shell: int
    wm_lmax: int
    voxel_size_mm: Optional[Tuple[float, float, float]] = None
    anisotropy_ratio: Optional[float] = None
    coverage_mm: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable multi-line report."""
        shell_str = ", ".join(
            f"b={b}:{n}" for b, n in self.shells
        )
        lines = [
            f"Tractography adequacy: {'FEASIBLE' if self.feasible else 'NOT FEASIBLE'}",
            f"  recommended model : {self.recommended_model}",
            f"  shells            : {shell_str}",
            f"  DW directions     : {self.total_dw_directions} "
            f"(min {self.min_dirs_per_shell}/shell), b0 x{self.n_b0}",
            f"  feasible lmax     : {self.max_lmax_per_shell} per shell, "
            f"WM FOD lmax={self.wm_lmax}",
        ]
        if self.voxel_size_mm is not None:
            vx, vy, vz = self.voxel_size_mm
            lines.append(
                f"  voxel size        : {vx:.3f} x {vy:.3f} x {vz:.3f} mm "
                f"(anisotropy {self.anisotropy_ratio:.1f}:1)"
            )
        if self.coverage_mm is not None:
            lines.append(f"  slice coverage    : {self.coverage_mm:.1f} mm")
        for w in self.warnings:
            lines.append(f"  [warn]    {w}")
        for b in self.blockers:
            lines.append(f"  [BLOCKER] {b}")
        return "\n".join(lines)

    def raise_if_infeasible(self) -> None:
        """Raise :class:`InadequateDataError` when tractography would be invalid."""
        if not self.feasible:
            raise InadequateDataError(
                "Acquisition cannot support tractography.\n" + self.summary()
            )


class InadequateDataError(RuntimeError):
    """Raised when an acquisition cannot support the requested model."""


def assess_tractography_adequacy(
    bval_file: Path,
    bvec_file: Optional[Path] = None,
    dwi_file: Optional[Path] = None,
    voxel_scale: float = 10.0,
    b0_threshold: float = 50.0,
    require_csd: bool = True,
) -> TractographyAdequacy:
    """Assess whether a DWI acquisition can support tractography.

    Parameters
    ----------
    bval_file : Path
        FSL-format b-values. The minimum needed for a verdict.
    bvec_file : Path, optional
        FSL-format b-vectors. When given, directions are counted as
        *antipodally unique* rather than as volumes, so a scheme that repeats
        directions is not credited with angular resolution it does not have.
    dwi_file : Path, optional
        Preprocessed DWI. When given, voxel geometry and slab coverage are
        reported and checked.
    voxel_scale : float
        Header-to-real-mm divisor. Rodent data in this package is scaled 10x
        for FSL/ANTs compatibility (``bids.voxel_scale``), so headers read in
        tenths of a millimetre.
    b0_threshold : float
        b-values below this count as b=0.
    require_csd : bool
        When True (default) a tensor-only acquisition is a blocker, because
        tensor tractography cannot resolve crossing fibres and its connectomes
        are not interpretable. Set False to allow it with a loud warning.

    Returns
    -------
    TractographyAdequacy
    """
    bvals_raw = np.loadtxt(bval_file)
    if bvals_raw.ndim > 1:
        bvals_raw = bvals_raw.flatten()
    bvals = round_bvals_to_shells(bvals_raw, b0_threshold=b0_threshold)

    n_b0 = int(np.sum(bvals < b0_threshold))
    non_zero = np.unique(bvals[bvals >= b0_threshold])

    shells: List[Tuple[int, int]] = []
    if n_b0:
        shells.append((0, n_b0))
    for bv in sorted(non_zero):
        shells.append((int(bv), int(np.sum(bvals == bv))))

    warnings: List[str] = []
    blockers: List[str] = []

    # Direction counts. Prefer antipodally-unique directions when bvecs are
    # available — a scheme that samples the same axis twice gains averaging,
    # not angular resolution, and crediting it with the latter would inflate
    # the feasible lmax.
    dirs_per_shell = {int(bv): int(np.sum(bvals == bv)) for bv in non_zero}
    if bvec_file is not None and Path(bvec_file).exists():
        bvecs = np.loadtxt(bvec_file)
        if bvecs.shape[0] != 3:
            bvecs = bvecs.T
        for bv in non_zero:
            sel = bvecs[:, bvals == bv].T
            uniq = {
                tuple(np.round(v * np.sign(v[np.argmax(np.abs(v))]), 3))
                for v in sel
                if np.linalg.norm(v) > 0
            }
            if len(uniq) < dirs_per_shell[int(bv)]:
                warnings.append(
                    f"shell b={int(bv)} has {dirs_per_shell[int(bv)]} volumes but only "
                    f"{len(uniq)} unique directions (repeats average, not resolve)"
                )
            dirs_per_shell[int(bv)] = len(uniq)

    n_shells = len(non_zero)
    min_dirs = min(dirs_per_shell.values()) if dirs_per_shell else 0
    total_dw = sum(dirs_per_shell.values())

    max_lmax_shell = max_feasible_lmax(min_dirs)

    # --- Model selection ----------------------------------------------------
    # MSMT pools every shell into one FOD fit, so the WM lmax is bounded by the
    # total DW sampling; single-shell CSD is bounded by that one shell.
    if n_shells >= MIN_SHELLS_MSMT and n_b0 >= MIN_B0 and min_dirs >= MIN_DIRS_CSD:
        model = "msmt_csd"
        wm_lmax = min(max_feasible_lmax(total_dw), 8)
    elif min_dirs >= MIN_DIRS_CSD:
        model = "csd"
        wm_lmax = min(max_lmax_shell, 8)
    elif min_dirs >= MIN_DIRS_TENSOR:
        model = "tensor"
        wm_lmax = max_lmax_shell
    else:
        model = "none"
        wm_lmax = 0

    # --- Blockers -----------------------------------------------------------
    if min_dirs < MIN_DIRS_TENSOR:
        blockers.append(
            f"only {min_dirs} directions in the sparsest shell; "
            f"{MIN_DIRS_TENSOR} are needed even to determine a tensor"
        )
    elif model == "tensor":
        msg = (
            f"{min_dirs} directions supports lmax={max_lmax_shell} only, which is "
            f"angularly equivalent to a diffusion tensor; crossing fibres are "
            f"unrecoverable and connectome edges would be model artifact "
            f"(need >={MIN_DIRS_CSD} directions for a meaningful FOD)"
        )
        (blockers if require_csd else warnings).append(msg)

    if n_b0 < MIN_B0:
        blockers.append("no b=0 volume; response estimation cannot be anchored")

    # --- Warnings -----------------------------------------------------------
    if n_shells < MIN_SHELLS_MSMT and model == "csd":
        warnings.append(
            "single shell: tissue responses cannot be separated (no MSMT-CSD); "
            "consider MRtrix3Tissue ss3t_csd if GM/CSF partial volume matters"
        )
    if n_b0 < 3 and model == "msmt_csd":
        warnings.append(
            f"only {n_b0} b=0 volume(s); MSMT tissue separation is better "
            "conditioned with >=3"
        )

    voxel_mm = aniso = coverage = None
    if dwi_file is not None and Path(dwi_file).exists():
        hdr = nib.load(str(dwi_file)).header
        zooms = [float(z) / voxel_scale for z in hdr.get_zooms()[:3]]
        shape = hdr.get_data_shape()
        voxel_mm = (zooms[0], zooms[1], zooms[2])
        aniso = max(zooms) / min(zooms)
        coverage = shape[2] * zooms[2]

        if aniso > 3.0:
            warnings.append(
                f"voxels are {aniso:.1f}:1 anisotropic; through-plane streamline "
                "orientation is poorly constrained"
            )
        if coverage < 15.0:
            warnings.append(
                f"slab covers {coverage:.1f} mm; streamlines terminate at slab "
                "edges for geometric rather than anatomical reasons, so any "
                "connectome is slab-restricted and nodes must be coverage-filtered"
            )

    feasible = not blockers
    result = TractographyAdequacy(
        feasible=feasible,
        recommended_model=model if feasible else "none",
        shells=shells,
        n_shells=n_shells,
        min_dirs_per_shell=min_dirs,
        total_dw_directions=total_dw,
        n_b0=n_b0,
        max_lmax_per_shell=max_lmax_shell,
        wm_lmax=wm_lmax,
        voxel_size_mm=voxel_mm,
        anisotropy_ratio=aniso,
        coverage_mm=coverage,
        warnings=warnings,
        blockers=blockers,
    )
    logger.info(result.summary())
    return result
