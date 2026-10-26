"""EPI readout geometry from Bruker parameters: what topup and fieldmap correction need.

Distortion correction needs three things per EPI: the voxel axis that phase
encoding runs along, its polarity, and the readout duration. These follow the
BIDS definitions:

``EffectiveEchoSpacing``
    ``PVM_EpiEchoSpacing / (PVM_EpiNShots * phase-encode acceleration)``, in
    seconds. Segmented EPI interleaves shots in k-space, so the effective
    spacing between adjacent phase-encode lines is the per-shot echo spacing
    divided by the number of shots.
``TotalReadoutTime``
    ``EffectiveEchoSpacing * (N_PE - 1)``, where ``N_PE`` is the
    *reconstructed* phase-encode matrix (``VisuCoreSize`` along that axis,
    after partial-Fourier/zero filling). This is what FSL's topup ``datain``
    expects.
``PhaseEncodingDirection``
    ``i``/``j``/``k`` plus an optional ``-``, in voxel axes of the written
    NIfTI. The axis is where ``VisuAcqGradEncoding`` says ``phase_enc``. It is
    *not* always ``j``: a coronal Bruker EPI with read ``H_F`` encodes phase
    along image axis 0.

Polarity calibration
--------------------
The sign was calibrated on the mouse pilot (FAC600, 7 T): topup was run on
each reverse-phase-encode pair and its field compared with the post-shim B0 map
(``ADJ_B0MAP_DistCorr``, in Hz), inside an eroded brain mask. With the scan whose
``PVM_EpiBlipOddGrad`` is positive labelled ``+``, every pair came out
anti-correlated with the map: GE-EPI axial (PE ``j``) r = -0.71, slope -0.91;
GE-EPI coronal (PE ``i``) r = -0.70, slope -0.92; SE-EPI axial r = -0.42. Hence
a positive blip traverses the phase-encode axis toward its *negative* end. The
slope of ~0.9 for both GE pairs also confirms ``TotalReadoutTime`` above.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from neurofaune.preprocess.utils.mrs.bruker_params import read_jcampdx


def _scalar(value: Any, index: int = 0) -> Optional[float]:
    if value is None:
        return None
    arr = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    return float(arr[min(index, arr.size - 1)]) if arr.size else None


def _grad_encoding(visu: Dict[str, Any]):
    enc = visu.get("VisuAcqGradEncoding")
    if enc is None:
        return None
    if isinstance(enc, str):
        enc = enc.replace("<", " ").replace(">", " ").split()
    return [str(e).strip() for e in enc]


def phase_encoding_axis(visu: Dict[str, Any]) -> int:
    """Voxel axis (0/1) carrying phase encoding; 1 when visu_pars does not say."""
    enc = _grad_encoding(visu)
    if enc and "phase_enc" in enc:
        return enc.index("phase_enc")
    return 1


def epi_readout(scan_dir: Path, proc: int = 1) -> Optional[Dict[str, Any]]:
    """BIDS readout fields for a Bruker EPI scan, or None if it is not EPI.

    Blips-off reference scans (``PVM_EpiBlipsOff = Yes``) get no
    ``PhaseEncodingDirection``: they are not phase encoded along that axis.
    """
    scan_dir = Path(scan_dir)
    method = read_jcampdx(scan_dir / "method")
    if "PVM_EpiEchoSpacing" not in method:
        return None
    visu = read_jcampdx(scan_dir / "pdata" / str(proc) / "visu_pars")

    axis = phase_encoding_axis(visu)
    n_pe = int(_scalar(visu.get("VisuCoreSize"), axis))
    shots = _scalar(method.get("PVM_EpiNShots")) or 1.0
    ppi = _scalar(method.get("PVM_EncPpi"), 1) or 1.0
    eff_esp = _scalar(method["PVM_EpiEchoSpacing"]) / 1000.0 / (shots * ppi)

    out: Dict[str, Any] = {
        "EffectiveEchoSpacing": eff_esp,
        "TotalReadoutTime": eff_esp * (n_pe - 1),
        "PhaseEncodingAxis": "ijk"[axis],
    }
    blip = _scalar(method.get("PVM_EpiBlipOddGrad"))
    blips_off = str(method.get("PVM_EpiBlipsOff", "No")).strip() == "Yes"
    if blip and not blips_off:
        out["PhaseEncodingDirection"] = "ijk"[axis] + ("-" if blip > 0 else "")
    return out


def displacement_voxels(field_hz, effective_echo_spacing: float, n_pe: int):
    """EPI displacement along phase encoding, in voxels, for an off-resonance field.

    One voxel of displacement per ``1 / (EffectiveEchoSpacing * N_PE)`` Hz --
    the phase-encode bandwidth per pixel.
    """
    return np.asarray(field_hz, dtype=float) * effective_echo_spacing * n_pe


#: Main methylene (-CH2-) resonance of fat / petrolatum relative to water, ppm.
FAT_WATER_PPM = 3.4
#: Proton gyromagnetic ratio / 2pi, Hz per tesla.
GAMMA_HZ_PER_T = 42.577478e6


def chemical_shift_hz(field_strength_t: float, ppm: float = FAT_WATER_PPM) -> float:
    """Fat-water frequency offset in Hz at a given field strength."""
    return ppm * 1e-6 * GAMMA_HZ_PER_T * field_strength_t
