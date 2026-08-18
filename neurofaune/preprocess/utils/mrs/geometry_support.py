"""What of the Bruker geometry convention has actually been validated.

Voxel localisation is reconstructed by hand from Bruker parameters, because the
converter writes images with a scaled-identity affine and there is no
scanner-frame geometry in the NIfTI to compose against. Every axis assignment
and sign in that reconstruction is a convention, and three of them were wrong
in the first implementation.

All three failed the same way. The error is proportional to some offset that is
zero on most sessions, so the majority looked correct and the minority looked
like operator error:

======================  ===========================  ==========================
Sign error              Invisible when               Found by
======================  ===========================  ==========================
``PVM_VoxArrPosition``  the voxel is not rotated     a rotated session looking
in the voxel's own                                   tilted and off-centre
rotated frame

Slice axis direction    the voxel sits at            36 of 52 sessions being
                        isocentre                    displaced >2 slices

``PVM_SPackArr``        the slice package is not     6 sessions with the only
``SliceOffset`` sign    offset (41 of 53 here)       non-zero offsets missing
                                                     the target
======================  ===========================  ==========================

A visual check on one session cannot establish any of these, and did not. What
worked was scoring candidate conventions against an anatomical target across a
whole study. So this module does two things: it refuses to silently extrapolate
to acquisition geometry that has not been validated, and it provides the
target-overlap measurement that makes a misplacement visible in QC rather than
plausible-looking.

Validated against 53 cuprizone sessions (2D axial RARE, A_P read, single slice
package, ``RECO_transposition = 1``). Anything else is unproven, which is not
the same as wrong -- it means check it before trusting it, using
:func:`neurofaune.preprocess.utils.mrs.voxel_geometry.target_overlap`.
"""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

#: Acquisition geometry the localisation has been validated against.
VALIDATED = {
    'slice_orientation': {'axial'},
    'spatial_dimension': {'2D'},
    'n_slice_packages': {1},
    'reco_transposition': {1},
}

#: Offsets that were exercised, and so are known to be handled correctly.
#: Phase offsets are essentially untested: 52 of 53 sessions had none.
VALIDATED_OFFSETS = {
    'read': 'non-zero on 2 of 53 sessions (-1 and -2 mm)',
    'slice': 'non-zero on 12 of 53 sessions (-1 to +2.5 mm)',
    'phase': 'non-zero on 1 of 53 sessions; effectively untested',
}


def check_geometry_support(params: Dict[str, Any], scan: str = '') -> List[str]:
    """List the ways an acquisition departs from the validated geometry.

    Returns a warning per departure rather than raising, because an untested
    combination is not necessarily wrong -- it means the placement needs
    checking against anatomy before the tissue fractions are trusted. The
    caller decides how loud to be; :func:`read_anat_geometry` logs them.

    Parameters
    ----------
    params : dict
        Merged ``method``/``acqp`` parameters for the anatomical scan.
    scan : str
        Identifier used in the messages.

    Returns
    -------
    list of str
        Empty when the acquisition matches what has been validated.
    """
    problems: List[str] = []
    prefix = f'{scan}: ' if scan else ''

    def first(key, default=None):
        value = params.get(key, default)
        if isinstance(value, (list, np.ndarray)):
            flat = np.atleast_1d(value).ravel()
            return flat[0] if flat.size else default
        return value

    orientation = str(first('PVM_SPackArrSliceOrient', '')).strip().lower()
    if orientation and orientation not in VALIDATED['slice_orientation']:
        problems.append(
            f"{prefix}slice orientation is {orientation!r}; voxel localisation "
            f"has only been validated for "
            f"{sorted(VALIDATED['slice_orientation'])}. The read/phase/slice to "
            f"array-axis mapping may differ -- verify against anatomy."
        )

    dimension = str(first('PVM_SpatDimEnum', '')).strip()
    if dimension and dimension not in VALIDATED['spatial_dimension']:
        problems.append(
            f"{prefix}acquisition is {dimension}, not 2D multi-slice; the slice "
            f"package geometry used here does not describe it."
        )

    packages = first('PVM_NSPacks', 1)
    if packages is not None and int(packages) not in VALIDATED['n_slice_packages']:
        problems.append(
            f"{prefix}{int(packages)} slice packages; only the first is read, "
            f"so the voxel may be placed against the wrong one."
        )

    phase_offset = first('PVM_SPackArrPhase1Offset', 0.0)
    if phase_offset is not None and abs(float(phase_offset)) > 1e-6:
        problems.append(
            f"{prefix}phase offset is {float(phase_offset):+.2f} mm. Its sign is "
            f"effectively untested ({VALIDATED_OFFSETS['phase']}), and the "
            f"analogous slice-offset sign was wrong -- check the placement."
        )

    return problems


def check_reco_transposition(transposition: int, scan: str = '') -> List[str]:
    """Warn when the reconstruction transposition has not been validated."""
    if int(transposition) in VALIDATED['reco_transposition']:
        return []
    prefix = f'{scan}: ' if scan else ''
    return [
        f'{prefix}RECO_transposition={transposition}; only '
        f'{sorted(VALIDATED["reco_transposition"])} has been validated against '
        f'anatomy. The axis mapping for this value is a straightforward reading '
        f'of the convention but is unproven.'
    ]


def log_geometry_support(params: Dict[str, Any], transposition: int,
                         scan: str = '') -> List[str]:
    """Emit warnings for every unvalidated aspect of an acquisition."""
    problems = check_geometry_support(params, scan) + \
        check_reco_transposition(transposition, scan)
    for problem in problems:
        logger.warning('%s', problem)
    return problems
