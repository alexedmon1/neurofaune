"""Animal-anatomical NIfTI affines from Bruker scanner geometry.

neurofaune's converter historically wrote a scaled-identity affine. That is
workable when every scan in a study shares one slice orientation (the axial
rat protocols), because the study-space atlas is rotated once to match it. It
is not workable for a session that mixes orientations -- the mouse EEG pilot
has a sagittal 3D RARE, axial and coronal EPI, and a 3D field map -- because
identity affines place each of those in its own unrelated frame.

``visu_pars`` gives each scan's geometry in Bruker's *subject* frame
(:func:`~neurofaune.preprocess.utils.mrs.bruker_affine.affine_from_visu`). For a
quadruped that frame describes the animal as if it were a standing human: the
head is "superior" and the back is "posterior". So it is a fixed rotation away
from the animal's own anatomical RAS (Right, Anterior = rostral, Superior =
dorsal), keyed on how the animal was registered at the console.

Calibration
-----------
``('Quadruped', 'Head_Supine')`` was read off the anatomy of the mouse pilot
(FAC600_Test_Mouse1, 3D T2 RARE): the olfactory bulb lies toward subject +z
and the spinal cord toward -z, so rostral is +z. The cochleae, skull base,
tongue and jaw lie toward +y and the cortical dome toward -y, so dorsal is -y.
The subject frame is right-handed and so is RAS, so those two axes fix the
third: animal Right is subject +x. Sagittal, axial and coronal scans from that
session then overlay to about one voxel.

Left/right cannot be checked from a symmetric brain. It follows here from
handedness, given the two axes that can be checked. That holds whether or not
the console position matched how the animal lay, because the observed axes
already include any such mismatch -- but only for sessions that share this
calibration. Other positions are refused until calibrated the same way.
"""

from pathlib import Path
from typing import Tuple

import numpy as np

from neurofaune.preprocess.utils.mrs.bruker_affine import affine_from_visu
from neurofaune.preprocess.utils.mrs.bruker_params import read_jcampdx

#: Subject frame -> animal anatomical RAS, keyed on
#: ``(VisuSubjectType, VisuSubjectPosition)``. Rows are the RAS axes expressed
#: in subject coordinates: R = +x, A (rostral) = +z, S (dorsal) = -y.
SUBJECT_TO_ANATOMICAL = {
    ("Quadruped", "Head_Supine"): np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]),
}


def subject_position(scan_dir: Path, proc: int = 1) -> Tuple[str, str]:
    """``(VisuSubjectType, VisuSubjectPosition)`` for a scan."""
    visu = read_jcampdx(Path(scan_dir) / "pdata" / str(proc) / "visu_pars")
    return (str(visu.get("VisuSubjectType", "")).strip(),
            str(visu.get("VisuSubjectPosition", "")).strip())


def subject_to_anatomical(scan_dir: Path, proc: int = 1) -> np.ndarray:
    """Rotation from Bruker's subject frame to animal anatomical RAS.

    Raises
    ------
    KeyError
        If the scan's subject type/position has no calibrated rotation.
        Guessing would risk a left/right swap with no visible symptom.
    """
    key = subject_position(scan_dir, proc)
    if key not in SUBJECT_TO_ANATOMICAL:
        raise KeyError(
            f"{scan_dir}: (VisuSubjectType, VisuSubjectPosition)={key} has no "
            f"calibrated subject->anatomical rotation (known: "
            f"{sorted(SUBJECT_TO_ANATOMICAL)}). Calibrate it from anatomy "
            f"(rostral and dorsal directions) before converting with "
            f"bids.affine='scanner'."
        )
    return SUBJECT_TO_ANATOMICAL[key]


def anatomical_affine(scan_dir: Path, voxel_scale: float = 1.0, proc: int = 1) -> np.ndarray:
    """Index -> animal-RAS affine, with world coordinates multiplied by ``voxel_scale``.

    The whole affine is scaled (translation included), so every scan of a
    session stays in one consistent, inflated world frame and still overlays.
    """
    rotation = np.eye(4)
    rotation[:3, :3] = subject_to_anatomical(scan_dir, proc)
    scale = np.diag([voxel_scale, voxel_scale, voxel_scale, 1.0])
    return scale @ rotation @ affine_from_visu(scan_dir, proc)
