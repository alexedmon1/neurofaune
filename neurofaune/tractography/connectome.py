"""Structural connectome construction from a SIFT2-weighted tractogram.

Produces a node-by-node connectivity matrix in the same wide-CSV form the rest
of ``neurofaune.network`` consumes, so the existing graph-theory, CovNet and
NBS machinery applies unchanged.

**Coverage filtering is not optional here.** Partial-coverage acquisitions —
the norm for rodent DWI — put many atlas regions wholly or partly outside the
field of view. A node lying outside the slab still receives an edge weight of
zero rather than a missing value, and zeros are not neutral: they shift group
means, and if coverage differs systematically between groups they manufacture
an effect indistinguishable from a real one. This module therefore measures
per-node coverage against the DWI brain mask, drops nodes below a threshold,
and records what it dropped alongside the matrix.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from neurofaune.tractography.layout import resolve_output_dir, work_dir
from neurofaune.tractography.mrtrix import _run, require_mrtrix

logger = logging.getLogger(__name__)


def compute_node_coverage(
    parcellation: Path,
    fov_mask: Path,
) -> pd.DataFrame:
    """Fraction of each parcellation node lying inside the DWI field of view.

    .. important::
       Both inputs must be in the **same space, and that space should be the
       native anatomical one** — that is, warp the *DWI* FOV mask up into
       anatomical space, rather than warping the atlas down into DWI space and
       measuring there.

       Measuring in DWI space does not work. Once the atlas has been resampled
       onto the DWI grid it has already been clipped to the slab, so a region
       half outside the field of view appears as a whole (if smaller) region
       whose every voxel lies inside the brain mask, and scores ~1.0. On this
       study that mistake found 1 under-covered node where there are 32.
       Comparing node *volumes* across the two grids does not fix it either,
       because the coarser DWI grid loses volume for small structures and the
       resulting ratio conflates that loss with genuine slab clipping.

    Parameters
    ----------
    parcellation : Path
        Integer-labelled atlas in native anatomical space.
    fov_mask : Path
        DWI field-of-view (brain) mask warped into that same space.

    Returns
    -------
    pandas.DataFrame
        Columns ``node``, ``n_voxels_total``, ``n_voxels_covered``, ``coverage``.
    """
    par = nib.load(str(parcellation)).get_fdata().astype(np.int32)
    mask = nib.load(str(fov_mask)).get_fdata() > 0
    if par.shape != mask.shape:
        raise ValueError(
            f"parcellation {par.shape} and FOV mask {mask.shape} differ; both "
            "must be on the same grid (preferably the anatomical one)"
        )

    nodes = np.unique(par)
    nodes = nodes[nodes > 0]
    rows = []
    for n in nodes:
        sel = par == n
        total = int(sel.sum())
        covered = int((sel & mask).sum())
        rows.append(
            {
                "node": int(n),
                "n_voxels_total": total,
                "n_voxels_covered": covered,
                "coverage": covered / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _node_names(
    labels_csv: Optional[Path],
    nodes: List[int],
) -> Dict[int, str]:
    """Map integer labels to atlas region names, falling back to ``node-<id>``."""
    names = {n: f"node-{n}" for n in nodes}
    if labels_csv is None or not Path(labels_csv).exists():
        return names
    labels = pd.read_csv(labels_csv)
    name_col = next(
        (c for c in ("Region of interest", "ROI", "Name") if c in labels.columns),
        None,
    )
    if name_col is None or "Labels" not in labels.columns:
        logger.warning("labels CSV lacks expected columns; using numeric names")
        return names
    for _, row in labels.iterrows():
        try:
            lid = int(row["Labels"])
        except (ValueError, TypeError):
            continue
        if lid in names:
            # Match the column convention used elsewhere in neurofaune, where
            # dots in atlas names become underscores.
            names[lid] = str(row[name_col]).strip().replace(".", "_")
    return names


def build_connectome(
    tractogram: Path,
    parcellation: Path,
    output_dir: Path,
    subject: str,
    session: str,
    weights: Optional[Path] = None,
    coverage_parcellation: Optional[Path] = None,
    coverage_fov_mask: Optional[Path] = None,
    labels_csv: Optional[Path] = None,
    min_coverage: float = 0.5,
    assignment_radius_mm: float = 0.4,
    voxel_scale: float = 10.0,
    symmetric: bool = True,
    scale_invnodevol: bool = False,
    config: Optional[dict] = None,
    force: bool = False,
    nthreads: Optional[int] = None,
    derive_layout: bool = True,
) -> Dict[str, Path]:
    """Build a structural connectivity matrix from a tractogram.

    Parameters
    ----------
    tractogram : Path
        ``.tck`` from :func:`~neurofaune.tractography.tractogram.run_tractography`.
    parcellation : Path
        Integer atlas **already resampled into DWI space** (nearest neighbour).
    output_dir : Path
        The **study root**; the layout below it is derived. Pass
        ``derive_layout=False`` to treat this as a literal destination.
    subject, session : str
        BIDS identifiers for naming.
    weights : Path, optional
        SIFT2 per-streamline weights. Strongly recommended — without them
        edges are raw streamline counts, which are biased by tract length and
        geometry in ways that do not cancel across groups.
    coverage_parcellation, coverage_fov_mask : Path, optional
        The atlas in **native anatomical space** and the DWI field-of-view mask
        warped into that same space. Supply both to enable coverage filtering.
        These are deliberately not the DWI-space parcellation and brain mask:
        see :func:`compute_node_coverage` for why measuring in DWI space
        silently under-detects clipped nodes. When omitted, no filtering is
        applied and a warning is emitted.
    labels_csv : Path, optional
        SIGMA labels CSV, used to name nodes.
    min_coverage : float
        Nodes whose voxels lie inside the FOV at less than this fraction are
        dropped. Set to 0 to keep everything (not advised for slab data).
    assignment_radius_mm : float
        Radial search distance, in **real** mm, for assigning a streamline
        endpoint to a node.
    voxel_scale : float
        Header-to-real-mm divisor (``bids.voxel_scale``).
    symmetric, scale_invnodevol : bool
        Passed to ``tck2connectome``. ``scale_invnodevol`` divides each edge by
        the summed node volume, which partly compensates for unequal region
        sizes.

    Returns
    -------
    dict
        ``matrix`` (CSV, named rows/columns), ``matrix_raw`` (MRtrix CSV),
        ``coverage`` (per-node CSV), ``assignments``, ``metadata`` (JSON).
    """
    bin_dir = require_mrtrix(config)
    session_out = resolve_output_dir(output_dir, subject, session, derive_layout)
    scratch_out = (
        work_dir(output_dir, subject, session) if derive_layout else session_out
    )
    session_out.mkdir(parents=True, exist_ok=True)
    scratch_out.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject}_{session}"

    # The unlabelled matrix and the per-streamline endpoint assignments are
    # intermediates: the labelled, coverage-filtered CSV is the product.
    out = {
        "matrix": session_out / f"{prefix}_desc-connectome_relmat.csv",
        "matrix_raw": scratch_out / f"{prefix}_desc-connectomeRaw_relmat.csv",
        "coverage": session_out / f"{prefix}_desc-nodeCoverage.csv",
        "assignments": scratch_out / f"{prefix}_desc-assignments.txt",
        "metadata": session_out / f"{prefix}_desc-connectome.json",
    }
    if out["matrix"].exists() and not force:
        logger.info("connectome already exists for %s %s", subject, session)
        return out

    logger.info("=" * 70)
    logger.info("Connectome: %s %s", subject, session)
    logger.info("=" * 70)

    cmd: List[str] = [
        "tck2connectome", str(tractogram), str(parcellation), str(out["matrix_raw"]),
        "-assignment_radial_search", str(assignment_radius_mm * voxel_scale),
        "-out_assignments", str(out["assignments"]),
    ]
    if weights is not None and Path(weights).exists():
        cmd += ["-tck_weights_in", str(weights)]
    else:
        logger.warning(
            "no SIFT2 weights supplied: edges will be raw streamline counts, "
            "which carry length and geometry bias"
        )
    if symmetric:
        cmd += ["-symmetric"]
    if scale_invnodevol:
        cmd += ["-scale_invnodevol"]
    cmd += ["-zero_diagonal", "-force"]

    _run(cmd, bin_dir, "tck2connectome", nthreads)

    matrix = np.loadtxt(out["matrix_raw"], delimiter=",", ndmin=2)
    par_img = nib.load(str(parcellation)).get_fdata().astype(np.int32)
    present = sorted(int(n) for n in np.unique(par_img) if n > 0)

    # tck2connectome emits a matrix indexed 1..max_label, so rows exist for
    # labels absent from this subject's parcellation. Index by position.
    max_label = matrix.shape[0]
    all_nodes = list(range(1, max_label + 1))

    dropped: List[int] = []
    keep_mask = np.zeros(max_label, dtype=bool)
    # Labels absent from this subject's parcellation are not nodes at all;
    # tck2connectome pads the matrix out to the largest label id.
    for i, n in enumerate(all_nodes):
        keep_mask[i] = n in set(present)

    coverage_filtered = False
    if (
        coverage_parcellation is not None
        and coverage_fov_mask is not None
        and Path(coverage_parcellation).exists()
        and Path(coverage_fov_mask).exists()
    ):
        coverage_df = compute_node_coverage(coverage_parcellation, coverage_fov_mask)
        coverage_df.to_csv(out["coverage"], index=False)
        cov_map = dict(zip(coverage_df["node"], coverage_df["coverage"]))
        for i, n in enumerate(all_nodes):
            if not keep_mask[i]:
                continue
            if cov_map.get(n, 0.0) < min_coverage:
                keep_mask[i] = False
                dropped.append(n)
        coverage_filtered = True
        logger.info(
            "  coverage filter: dropped %d of %d present nodes below %.0f%% FOV "
            "coverage; %d retained",
            len(dropped), len(present), 100 * min_coverage, int(keep_mask.sum()),
        )
    else:
        logger.warning(
            "no native-space parcellation + FOV mask supplied: skipping coverage "
            "filtering. On partial-coverage data this leaves clipped and "
            "out-of-FOV nodes in the matrix as structural zeros, which bias "
            "group means rather than reading as missing."
        )

    kept_nodes = [n for n, k in zip(all_nodes, keep_mask) if k]
    names = _node_names(labels_csv, kept_nodes)
    sub = matrix[np.ix_(keep_mask, keep_mask)]

    df = pd.DataFrame(
        sub,
        index=[names[n] for n in kept_nodes],
        columns=[names[n] for n in kept_nodes],
    )
    df.to_csv(out["matrix"])

    density = float((sub > 0).sum()) / max(sub.size - sub.shape[0], 1)
    metadata = {
        "subject": subject,
        "session": session,
        "n_nodes_in_parcellation": len(present),
        "n_nodes_kept": len(kept_nodes),
        "n_nodes_dropped_coverage": len(dropped),
        "dropped_nodes": dropped,
        "coverage_filtered": coverage_filtered,
        "min_coverage": min_coverage if coverage_filtered else None,
        "sift2_weighted": weights is not None and Path(weights).exists(),
        "symmetric": symmetric,
        "scale_invnodevol": scale_invnodevol,
        "assignment_radius_mm": assignment_radius_mm,
        "edge_density": density,
    }
    out["metadata"].write_text(json.dumps(metadata, indent=2))
    logger.info(
        "  matrix %dx%d, edge density %.3f -> %s",
        len(kept_nodes), len(kept_nodes), density, out["matrix"].name,
    )
    return out
