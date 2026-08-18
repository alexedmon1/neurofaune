"""Locate the SVS voxel on the anatomical image and measure its tissue content.

Absolute (water-scaled) metabolite concentrations depend on how much of the
spectroscopy voxel is grey matter, white matter and CSF, because the three
compartments hold different amounts of water with different relaxation times.
Assuming fixed fractions is common but adds subject-to-subject bias, so this
module measures them from the subject's own T2w segmentation.

``svs_segment`` (FSL-MRS' own tool) cannot be used here for two reasons: it
wants ``fsl_anat`` output, which does not apply to rodent brains, and it
locates the voxel through the NIfTI affines. neurofaune's Bruker converter
writes images with a scaled-identity affine, so the anatomical NIfTI carries no
scanner-frame geometry at all. The voxel is therefore placed from the Bruker
geometry parameters of both scans, which share the magnet frame.

Bruker geometry conventions used here
-------------------------------------
``PVM_SPackArrGradOrient`` is a 3x3 whose rows are the read, phase and slice
unit vectors expressed in magnet coordinates. The slice-package centre in
magnet coordinates is the read/phase/slice offsets projected back through it,
with the slice offset negated -- see :func:`read_anat_geometry`.

For the spectroscopy voxel, do NOT use ``PVM_VoxArrPosition``: it is the centre
in the voxel's own rotated frame, not in magnet coordinates. The magnet-frame
position and rotation come from the geometry object ``PVM_VoxelGeoCub``
instead; see :func:`neurofaune.preprocess.utils.mrs.bruker_mrs.read_voxel_geometry`.

Every sign in this module was established by scoring candidates against the
SIGMA parcellation warped into subject space across 50 sessions, not by
inspecting one session. Three separate sign errors survived visual checks here
because each is invisible on sessions whose corresponding offset is near zero,
and most sessions are.
"""

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import nibabel as nib
import numpy as np

from neurofaune.preprocess.utils.mrs.bruker_mrs import BrukerSVS
from neurofaune.preprocess.utils.mrs.bruker_params import read_scan_params

logger = logging.getLogger(__name__)


#: Verified mapping from Bruker read/phase/slice coordinates to the axes of the
#: NIfTI array brukerapi produces, keyed by ``RECO_transposition``.
#:
#: Bruker's reconstruction can transpose the in-plane axes, and it records that
#: in ``RECO_transposition`` in ``pdata/1/reco``. Each entry gives, per array
#: axis, which rps component it takes and with what sign.
#:
#: The ``1`` mapping was established by regressing the position of each
#: session's whole-brain PRESS shim box against the centre of mass of that
#: session's brain mask, over 47 cuprizone sessions -- the shim box is placed
#: on the brain by the operator, so the two must track. All three axes were
#: scored against every (component, sign) candidate; phase/+1, read/-1 and
#: slice/-1 won their axes.
#:
#: An earlier version had slice/+1, taken from a single session by eye. That
#: session's slice offset was 0.26 mm, so the sign was unobservable in it, and
#: the error only showed on sessions where the voxel was moved further from
#: centre -- displacing them by twice their offset along the rostrocaudal axis.
#: A one-session visual check cannot establish this; it needs the spread of
#: prescriptions across a study.
#:
#: The ``0`` case is the straightforward reading of the same convention and is
#: NOT verified against data. Any other value is rejected rather than guessed
#: at: a misplaced voxel yields plausible-looking but wrong tissue fractions.
_RECO_AXIS_MAP = {
    0: ((0, 1.0), (1, 1.0), (2, 1.0)),    # array (read, phase, slice)
    1: ((1, 1.0), (0, -1.0), (2, -1.0)),  # array (phase, -read, -slice)
}


@dataclass
class AnatGeometry:
    """Slice-package geometry of a 2D multi-slice anatomical acquisition.

    Attributes
    ----------
    grad_orient : np.ndarray
        3x3; rows are the read, phase and slice directions in magnet coords.
    centre : np.ndarray
        Slice-package centre in magnet coordinates (mm).
    fov : np.ndarray
        In-plane field of view (read, phase) in mm.
    matrix : np.ndarray
        In-plane matrix (read, phase).
    n_slices : int
        Number of slices.
    slice_distance : float
        Centre-to-centre slice spacing in mm.
    transposition : int
        ``RECO_transposition``; selects the rps-to-array axis mapping.
    """

    grad_orient: np.ndarray
    centre: np.ndarray
    fov: np.ndarray
    matrix: np.ndarray
    n_slices: int
    slice_distance: float
    transposition: int = 0

    @property
    def axis_map(self) -> Tuple[Tuple[int, float], ...]:
        """Per array axis, the (rps component, sign) it is built from."""
        if self.transposition not in _RECO_AXIS_MAP:
            raise ValueError(
                f"Unsupported RECO_transposition={self.transposition}; the "
                f"read/phase/slice to array-axis mapping is only verified for "
                f"{sorted(_RECO_AXIS_MAP)}"
            )
        return _RECO_AXIS_MAP[self.transposition]

    @property
    def rps_extent(self) -> np.ndarray:
        """Field of view along (read, phase, slice) in mm."""
        return np.array([
            self.fov[0], self.fov[1], self.n_slices * self.slice_distance,
        ])

    @property
    def rps_samples(self) -> np.ndarray:
        """Sample count along (read, phase, slice)."""
        return np.array([self.matrix[0], self.matrix[1], self.n_slices], dtype=float)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Array shape, in the axis order brukerapi produces."""
        return tuple(int(self.rps_samples[component]) for component, _ in self.axis_map)

    @property
    def spacing(self) -> np.ndarray:
        """Voxel spacing along the array axes in mm."""
        rps_spacing = self.rps_extent / self.rps_samples
        return np.array([rps_spacing[component] for component, _ in self.axis_map])


def read_anat_geometry(scan_dir: Path) -> AnatGeometry:
    """Read the slice-package geometry of a Bruker anatomical scan."""
    params = read_scan_params(scan_dir)

    grad_orient = np.asarray(params['PVM_SPackArrGradOrient'], dtype=float).reshape(-1, 3, 3)[0]
    # The slice offset carries the opposite sign to the in-plane ones: the
    # slice index runs against the direction PVM_SPackArrSliceOffset is
    # measured in. Measured against the SIGMA parcellation over 50 sessions,
    # negating it takes mean hippocampal overlap from 57.4% to 70.8% and the
    # count of sessions above 40% from 39 to 49; negating the in-plane offsets
    # too makes things worse, breaking the two sessions that have a read
    # offset. Only 15 of 50 sessions here have a non-zero slice offset, which
    # is why this stayed hidden -- the other 35 are unaffected by definition.
    offsets = np.array([
        float(np.atleast_1d(params.get('PVM_SPackArrReadOffset', 0.0))[0]),
        float(np.atleast_1d(params.get('PVM_SPackArrPhase1Offset', 0.0))[0]),
        -float(np.atleast_1d(params.get('PVM_SPackArrSliceOffset', 0.0))[0]),
    ])
    # Offsets are along read/phase/slice; project back into magnet coordinates.
    centre = grad_orient.T @ offsets

    fov = np.asarray(params['PVM_Fov'], dtype=float)[:2]
    matrix = np.asarray(params['PVM_Matrix'], dtype=float)[:2]
    n_slices = int(np.atleast_1d(params['PVM_SPackArrNSlices'])[0])
    slice_distance = float(np.atleast_1d(
        params.get('PVM_SPackArrSliceDistance', params.get('PVM_SliceThick', 1.0))
    )[0])

    return AnatGeometry(
        grad_orient=grad_orient,
        centre=centre,
        fov=fov,
        matrix=matrix,
        n_slices=n_slices,
        slice_distance=slice_distance,
        transposition=_read_transposition(scan_dir),
    )


def _read_transposition(scan_dir: Path) -> int:
    """Read ``RECO_transposition`` from ``pdata/1/reco`` (0 when absent)."""
    from neurofaune.preprocess.utils.mrs.bruker_params import read_jcampdx

    reco_file = Path(scan_dir) / 'pdata' / '1' / 'reco'
    if not reco_file.exists():
        logger.warning("%s not found; assuming RECO_transposition=0", reco_file)
        return 0
    value = read_jcampdx(reco_file).get('RECO_transposition', 0)
    return int(np.atleast_1d(value).ravel()[0])


def magnet_to_index(geometry: AnatGeometry, points: np.ndarray) -> np.ndarray:
    """Convert magnet-frame points (mm) to continuous array indices.

    Parameters
    ----------
    geometry : AnatGeometry
    points : np.ndarray
        ``(n, 3)`` magnet coordinates in mm.

    Returns
    -------
    np.ndarray
        ``(n, 3)`` continuous indices into the anatomical array.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    rps = (geometry.grad_orient @ (points - geometry.centre).T).T
    fractional = rps / geometry.rps_extent  # -0.5 .. +0.5 across the FOV

    shape = np.array(geometry.shape, dtype=float)
    index = np.empty_like(fractional)
    for axis, (component, sign) in enumerate(geometry.axis_map):
        index[:, axis] = (
            sign * fractional[:, component] * shape[axis] + shape[axis] / 2.0 - 0.5
        )
    return index


def index_to_magnet(geometry: AnatGeometry, index: np.ndarray) -> np.ndarray:
    """Inverse of :func:`magnet_to_index`."""
    index = np.atleast_2d(np.asarray(index, dtype=float))
    shape = np.array(geometry.shape, dtype=float)

    fractional = np.empty_like(index)
    for axis, (component, sign) in enumerate(geometry.axis_map):
        fractional[:, component] = (
            (index[:, axis] - shape[axis] / 2.0 + 0.5) / shape[axis] / sign
        )
    rps = fractional * geometry.rps_extent
    return rps @ geometry.grad_orient + geometry.centre


def voxel_corners(svs: BrukerSVS) -> np.ndarray:
    """The eight corners of the SVS voxel in magnet coordinates (mm)."""
    half = svs.voxel_size / 2.0
    signs = np.array([[i, j, k]
                      for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
    # Rows of the orientation matrix are the voxel's own axes in magnet coords.
    return svs.voxel_position + (signs * half) @ svs.voxel_orientation


def make_voxel_mask(
    svs: BrukerSVS,
    anat_scan_dir: Optional[Path],
    anat_image: Path,
    output_file: Optional[Path] = None,
    supersample: int = 3,
    geometry: Optional[AnatGeometry] = None,
) -> Tuple[np.ndarray, Optional[Path]]:
    """Rasterise the SVS voxel onto the anatomical image grid.

    The mask is anti-aliased: each anatomical voxel is subdivided
    ``supersample**3`` times and the returned value is the fraction of those
    sub-points falling inside the spectroscopy voxel. That matters because the
    anatomical slices (0.8 mm) are coarse relative to the SVS voxel, so a
    binary mask would quantise the tissue fractions noticeably.

    Parameters
    ----------
    svs : BrukerSVS
        The spectroscopy acquisition, carrying the voxel geometry.
    anat_scan_dir : Path or None
        Bruker scan directory of the anatomical image, read for its geometry.
        May be None when ``geometry`` is supplied directly.
    anat_image : Path
        The converted anatomical NIfTI, used for its grid and header.
    output_file : Path, optional
        Where to write the mask. Not written when None.
    supersample : int
        Sub-samples per axis per anatomical voxel.
    geometry : AnatGeometry, optional
        Pre-read geometry, used instead of reading ``anat_scan_dir``.

    Returns
    -------
    (mask, path)
        The fractional mask array and the file it was written to (or None).

    Raises
    ------
    ValueError
        If the anatomical NIfTI's shape disagrees with its Bruker geometry,
        which would mean the two are not the same acquisition.
    """
    if geometry is None:
        geometry = read_anat_geometry(anat_scan_dir)
    anat = nib.load(str(anat_image))
    shape = anat.shape[:3]

    if tuple(shape) != geometry.shape:
        raise ValueError(
            f"{anat_image} has shape {tuple(shape)} but {anat_scan_dir} describes "
            f"{geometry.shape}; they are not the same acquisition"
        )

    # Restrict the rasterisation to the voxel's index-space bounding box; the
    # SVS voxel is a tiny part of a 256x256x41 volume and sub-sampling the
    # whole grid would be wasteful.
    corners = magnet_to_index(geometry, voxel_corners(svs))
    lower = np.maximum(np.floor(corners.min(axis=0)).astype(int) - 1, 0)
    upper = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 2, shape)

    mask = np.zeros(shape, dtype=np.float64)
    if np.any(lower >= upper):
        logger.warning("SVS voxel falls entirely outside %s", anat_image)
        return mask, _save_mask(mask, anat, output_file)

    # Sub-sample offsets within one anatomical voxel, in index units.
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    sub = np.stack(np.meshgrid(offsets, offsets, offsets, indexing='ij'), axis=-1)
    sub = sub.reshape(-1, 3)

    axes = [np.arange(lo, hi, dtype=float) for lo, hi in zip(lower, upper)]
    block_shape = tuple(len(a) for a in axes)
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)

    half = svs.voxel_size / 2.0
    counts = np.zeros(grid.shape[0], dtype=np.float64)
    for offset in sub:
        magnet = index_to_magnet(geometry, grid + offset)
        local = (magnet - svs.voxel_position) @ svs.voxel_orientation.T
        counts += np.all(np.abs(local) <= half, axis=1)

    block = (counts / sub.shape[0]).reshape(block_shape)
    mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] = block

    return mask, _save_mask(mask, anat, output_file)


def _save_mask(mask: np.ndarray, anat, output_file: Optional[Path]) -> Optional[Path]:
    """Write the fractional mask alongside the anatomical image."""
    if output_file is None:
        return None
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.float32), anat.affine, anat.header), path)
    return path


def tissue_fractions(
    mask: np.ndarray,
    gm_prob: Path,
    wm_prob: Path,
    csf_prob: Path,
) -> Dict[str, float]:
    """Mask-weighted GM/WM/CSF fractions inside the SVS voxel.

    Parameters
    ----------
    mask : np.ndarray
        Fractional voxel mask on the anatomical grid.
    gm_prob, wm_prob, csf_prob : Path
        Tissue probability maps written by
        :func:`neurofaune.preprocess.workflows.anat_preprocess.segment_brain_tissue`.

    Returns
    -------
    dict
        ``{'GM': .., 'WM': .., 'CSF': ..}``, normalised to sum to 1, plus
        ``'voxel_volume_ratio'`` -- the share of the SVS voxel that the
        anatomical slab actually covers. A low value means the voxel extends
        beyond the anatomical slices and the fractions are extrapolated.

    Raises
    ------
    ValueError
        If the mask is empty, or the tissue maps disagree with its shape.
    """
    weights = np.asarray(mask, dtype=float)
    if weights.sum() <= 0:
        raise ValueError(
            "SVS voxel does not intersect the anatomical image; check that the "
            "spectroscopy and anatomical scans come from the same session"
        )

    fractions: Dict[str, float] = {}
    for label, path in (('GM', gm_prob), ('WM', wm_prob), ('CSF', csf_prob)):
        data = nib.load(str(path)).get_fdata()
        if data.shape != weights.shape:
            raise ValueError(
                f"{path} has shape {data.shape}, expected {weights.shape}"
            )
        fractions[label] = float((data * weights).sum())

    total = sum(fractions.values())
    if total <= 0:
        raise ValueError(
            "SVS voxel contains no segmented tissue; the anatomical brain mask "
            "may not cover the voxel location"
        )

    covered = total / weights.sum()
    normalised = {label: value / total for label, value in fractions.items()}
    normalised['voxel_volume_ratio'] = float(covered)
    return normalised


def compute_tissue_fractions(
    svs: BrukerSVS,
    anat_scan_dir: Path,
    anat_image: Path,
    gm_prob: Path,
    wm_prob: Path,
    csf_prob: Path,
    mask_output: Optional[Path] = None,
) -> Dict[str, Any]:
    """Rasterise the voxel and measure its tissue content in one call."""
    mask, mask_path = make_voxel_mask(svs, anat_scan_dir, anat_image, mask_output)
    fractions = tissue_fractions(mask, gm_prob, wm_prob, csf_prob)
    fractions['mask'] = mask_path
    return fractions


def write_tissue_fraction_json(fractions: Dict[str, Any], output_file: Path) -> Path:
    """Write tissue fractions in the JSON layout ``fsl_mrs --tissue_frac`` reads."""
    import json

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: float(fractions[key]) for key in ('GM', 'WM', 'CSF')}
    with open(output_file, 'w') as handle:
        json.dump(payload, handle, indent=2)
    return output_file
