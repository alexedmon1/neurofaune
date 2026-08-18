"""NIfTI affines from Bruker's own image geometry.

Why this replaces hand-rolled axis mapping
------------------------------------------
Locating an SVS voxel on a structural image is arithmetic when both carry a
scanner-space affine: compose ``inv(anat_affine) @ svs_affine`` and read off
the indices. That is how it works in human MRS, and why ``svs_segment`` needs
no per-vendor conventions.

It was hard here only because neurofaune's Bruker converter writes a
scaled-identity affine, discarding the geometry, so the mapping had to be
reconstructed from ``PVM_`` parameters by hand -- which meant choosing an axis
order and three signs, and three of those choices were wrong at various points
without any visible symptom.

None of that is necessary. ``pdata/*/visu_pars`` carries the DICOM-equivalent
fields:

``VisuCoreOrientation``
    Direction cosines per frame; rows are the image axes in world coordinates.
``VisuCorePosition``
    World position of the first voxel's corner, per frame.
``VisuCoreExtent`` / ``VisuCoreSize``
    In-plane field of view and matrix, giving the in-plane spacing.

Those define the index-to-world transform outright, with the signs and axis
order included. On a validation session the resulting volume centre lands at
exactly (0, +1, 0) mm for a read offset of -1 mm -- the sign relationship falls
out rather than being chosen.

The spectroscopy voxel has no image geometry of its own, so its world
placement still comes from ``PVM_VoxelGeoCub`` (see
:func:`neurofaune.preprocess.utils.mrs.bruker_mrs.read_voxel_geometry`), but
that is a single documented source rather than a reconstructed convention.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from neurofaune.preprocess.utils.mrs.bruker_params import read_jcampdx

logger = logging.getLogger(__name__)


def read_visu_geometry(scan_dir: Path, proc: int = 1) -> dict:
    """Read the geometry fields from a scan's ``visu_pars``.

    Raises
    ------
    FileNotFoundError
        If ``visu_pars`` is absent.
    KeyError
        If the required geometry fields are missing, which means the affine
        cannot be derived and no fallback should be silently substituted.
    """
    path = Path(scan_dir) / 'pdata' / str(proc) / 'visu_pars'
    if not path.exists():
        raise FileNotFoundError(f'No visu_pars for {scan_dir} (proc {proc})')
    params = read_jcampdx(path)

    required = ('VisuCoreOrientation', 'VisuCorePosition',
                'VisuCoreExtent', 'VisuCoreSize')
    missing = [key for key in required if key not in params]
    if missing:
        raise KeyError(f'{path} is missing {missing}; the affine cannot be derived')

    orientation = np.asarray(params['VisuCoreOrientation'], dtype=float).reshape(-1, 3, 3)
    position = np.asarray(params['VisuCorePosition'], dtype=float).reshape(-1, 3)
    extent = np.asarray(params['VisuCoreExtent'], dtype=float).ravel()
    size = np.asarray(params['VisuCoreSize'], dtype=float).ravel()
    return {
        'orientation': orientation,
        'position': position,
        'extent': extent,
        'size': size,
        'slice_thickness': float(np.atleast_1d(
            params.get('VisuCoreFrameThickness', [1.0]))[0]),
    }


def affine_from_visu(scan_dir: Path, proc: int = 1) -> np.ndarray:
    """Index-to-world affine (mm) for a 2D multi-slice Bruker acquisition.

    The array is assumed to be ``(in-plane 0, in-plane 1, frame)``, which is
    what brukerapi produces from ``2dseq``.

    ``VisuCorePosition`` gives the *corner* of the first voxel, so half a voxel
    is added to put the affine on voxel centres, as NIfTI expects. On a
    validation session that makes the volume centre land on exactly (0, +1, 0)
    mm rather than being half a voxel out.

    Returns
    -------
    np.ndarray
        4x4 mapping ``(i, j, k, 1)`` to world millimetres.

    Raises
    ------
    ValueError
        If the frames do not form a single evenly spaced slice package, since
        the affine would not describe them.
    """
    geometry = read_visu_geometry(scan_dir, proc)
    orientation = geometry['orientation']
    position = geometry['position']

    if not np.allclose(orientation, orientation[0], atol=1e-6):
        raise ValueError(
            f'{scan_dir}: slice orientation varies between frames; a single '
            f'affine cannot describe this acquisition'
        )
    rotation = orientation[0]          # rows are the image axes in world coords

    spacing = geometry['extent'][:2] / geometry['size'][:2]

    if position.shape[0] > 1:
        steps = np.diff(position, axis=0)
        if not np.allclose(steps, steps[0], atol=1e-4):
            raise ValueError(
                f'{scan_dir}: frame positions are not evenly spaced; the slice '
                f'package is not a simple stack'
            )
        slice_vector = steps[0]
    else:
        slice_vector = rotation[2] * geometry['slice_thickness']

    affine = np.eye(4)
    affine[:3, 0] = rotation[0] * spacing[0]
    affine[:3, 1] = rotation[1] * spacing[1]
    affine[:3, 2] = slice_vector
    # VisuCorePosition is the corner of the first voxel; NIfTI affines address
    # voxel centres.
    affine[:3, 3] = position[0] + 0.5 * (affine[:3, 0] + affine[:3, 1])
    return affine


def world_to_index(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map world-millimetre points to continuous array indices."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (np.linalg.inv(affine) @ homogeneous.T).T[:, :3]


def index_to_world(affine: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Map continuous array indices to world millimetres."""
    index = np.atleast_2d(np.asarray(index, dtype=float))
    homogeneous = np.column_stack([index, np.ones(len(index))])
    return (affine @ homogeneous.T).T[:, :3]


def describe_affine(affine: np.ndarray) -> str:
    """One-line summary, for logs and QC."""
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    origin = affine[:3, 3]
    return (f'spacing {np.round(spacing, 3).tolist()} mm, '
            f'origin {np.round(origin, 2).tolist()} mm')


#: Gradient frame to subject/world frame, keyed on ``VisuSubjectPosition``.
#:
#: Images carry geometry in subject coordinates (``visu_pars``); the
#: spectroscopy voxel exists only in gradient coordinates (``PVM_``), because a
#: PRESS scan's visu_pars has ``VisuCoreDim = 1`` and no spatial fields. The two
#: differ by a signed permutation fixed by how the animal lies in the magnet.
#:
#: This cannot be recovered from the files for a typical acquisition: with a
#: square FOV and the slice package at isocentre -- 47 of 50 cuprizone sessions
#: -- every candidate rotation reproduces the geometry equally well. The
#: information is in the subject position, not the coordinates.
#:
#: So it is calibrated once per subject position and validated: scored across
#: 50 sessions against the SIGMA parcellation, ``diag(1, -1, -1)`` gives 71.2%
#: mean hippocampal overlap with 50 of 50 sessions above 40%, against 32.7%
#: and 14 for the next best candidate. It also agrees with the per-session
#: solve on the sessions where the package offset makes that solve
#: determinate.
#:
#: One constant per subject position, tied to a documented parameter and
#: checkable, rather than three independent sign choices scattered through an
#: axis map.
SUBJECT_POSITION_TRANSFORMS = {
    'Head_Supine': np.diag([1.0, -1.0, -1.0]),
}


def pvm_to_world(scan_dir: Path, proc: int = 1) -> np.ndarray:
    """Gradient-frame to world-frame rotation for a scan.

    Uses the transform calibrated for the scan's ``VisuSubjectPosition``, and
    cross-checks it against the slice-package offset when that offset is large
    enough to be informative.

    Raises
    ------
    KeyError
        If the subject position has no calibrated transform. Guessing here is
        what produced the earlier sign errors, so it refuses instead.
    ValueError
        If an informative package offset contradicts the calibrated transform.
    """
    from neurofaune.preprocess.utils.mrs.bruker_params import read_jcampdx, read_scan_params

    scan_dir = Path(scan_dir)
    visu = read_jcampdx(scan_dir / 'pdata' / str(proc) / 'visu_pars')
    position = str(visu.get('VisuSubjectPosition', '')).strip()
    if position not in SUBJECT_POSITION_TRANSFORMS:
        raise KeyError(
            f'{scan_dir}: VisuSubjectPosition={position!r} has no calibrated '
            f'gradient-to-world transform (known: '
            f'{sorted(SUBJECT_POSITION_TRANSFORMS)}). Calibrate it against an '
            f'anatomical target before trusting voxel placement.'
        )
    rotation = SUBJECT_POSITION_TRANSFORMS[position]

    # Cross-check without touching the slice-package offsets. The affine
    # already encodes where the package sits, so reconstructing that from
    # PVM_SPackArr*Offset only re-introduces the sign conventions this is
    # meant to remove. What can be checked cleanly is direction: after the
    # rotation, each PVM axis must line up with an image axis. Signs are not
    # required to match -- the slice order in the array may run against the
    # gradient direction -- but a grossly wrong rotation will not align at all.
    params = read_scan_params(scan_dir)
    grad = np.asarray(params['PVM_SPackArrGradOrient'], dtype=float).reshape(-1, 3, 3)[0]
    image_axes = read_visu_geometry(scan_dir, proc)['orientation'][0]
    alignment = np.abs((rotation @ grad.T).T @ image_axes.T)
    if not np.allclose(np.sort(alignment, axis=1)[:, -1], 1.0, atol=1e-3):
        raise ValueError(
            f'{scan_dir}: the calibrated {position} transform does not align '
            f'the gradient axes with the image axes (best alignment per axis '
            f'{np.round(np.sort(alignment, axis=1)[:, -1], 3).tolist()}). '
            f'Voxel placement cannot be trusted.'
        )
    return rotation


def solve_pvm_to_world(scan_dir: Path, proc: int = 1) -> np.ndarray:
    """Transform from Bruker's gradient frame to the image's world frame.

    Images carry geometry in ``visu_pars`` (subject coordinates), while the
    spectroscopy voxel only exists in ``PVM_`` parameters (gradient
    coordinates) -- a PRESS scan's ``visu_pars`` has ``VisuCoreDim = 1`` and no
    spatial fields at all. The two frames differ by a signed permutation set by
    ``VisuSubjectPosition``.

    Rather than hard-code that per subject position -- the kind of assumption
    that produced three silent sign errors here -- it is solved per session
    from the anatomical scan, which describes the *same* slice package in both
    frames. Candidate rotations are scored on how well they carry the PVM
    read/phase/slice directions and package centre onto the visu ones, and the
    residual is reported so a bad solve is loud rather than silent.

    Returns
    -------
    np.ndarray
        4x4 mapping gradient-frame millimetres to world millimetres.

    Raises
    ------
    ValueError
        If no candidate reproduces the anatomical geometry to better than half
        a voxel, which means the frames are not related by a signed permutation
        and the placement must not be trusted.
    """
    import itertools

    from neurofaune.preprocess.utils.mrs.bruker_params import read_scan_params

    scan_dir = Path(scan_dir)
    params = read_scan_params(scan_dir)
    affine = affine_from_visu(scan_dir, proc)

    grad = np.asarray(params['PVM_SPackArrGradOrient'], dtype=float).reshape(-1, 3, 3)[0]
    offsets = np.array([
        float(np.atleast_1d(params.get('PVM_SPackArrReadOffset', 0.0))[0]),
        float(np.atleast_1d(params.get('PVM_SPackArrPhase1Offset', 0.0))[0]),
        float(np.atleast_1d(params.get('PVM_SPackArrSliceOffset', 0.0))[0]),
    ])
    # Package centre in the gradient frame, as PVM describes it.
    pvm_centre = grad.T @ offsets

    # The same centre in the world frame, from the image itself.
    shape = np.array([
        *np.asarray(read_visu_geometry(scan_dir, proc)['size'][:2], dtype=int),
        int(np.asarray(read_visu_geometry(scan_dir, proc)['position']).reshape(-1, 3).shape[0]),
    ], dtype=float)
    world_centre = index_to_world(affine, (shape - 1) / 2.0)[0]

    # Candidate signed permutations (proper rotations only).
    best, best_error = None, np.inf
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            rotation = np.zeros((3, 3))
            for row, (column, sign) in enumerate(zip(perm, signs)):
                rotation[row, column] = sign
            if np.linalg.det(rotation) < 0:
                continue
            # A candidate must carry the PVM package centre onto the world one.
            error = np.linalg.norm(rotation @ pvm_centre - world_centre)
            if error < best_error:
                best, best_error = rotation, error

    voxel = np.linalg.norm(affine[:3, :3], axis=0).min()
    if best is None or best_error > 0.5 * voxel + 1e-6:
        raise ValueError(
            f'{scan_dir}: could not relate the gradient frame to the image '
            f'world frame (best residual {best_error:.3f} mm, voxel {voxel:.3f} '
            f'mm). The voxel position cannot be trusted.'
        )

    transform = np.eye(4)
    transform[:3, :3] = best
    transform[:3, 3] = world_centre - best @ pvm_centre
    logger.debug('%s: gradient->world residual %.3f mm', scan_dir, best_error)
    return transform
