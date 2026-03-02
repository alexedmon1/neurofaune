"""
VBM preparation utilities.

Core functions for warping tissue probability maps to SIGMA space,
computing Jacobian determinants for volume modulation, and smoothing.
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def warp_tissue_to_sigma(
    tissue_map: Path,
    reference: Path,
    transforms: List[Path],
    output_path: Path,
    interpolation: str = 'Linear',
) -> Path:
    """
    Warp a native-space tissue probability map to SIGMA space.

    Uses antsApplyTransforms with the full T2w -> template -> SIGMA
    transform chain.

    Parameters
    ----------
    tissue_map : Path
        Native-space tissue probability map (e.g. label-GM_probseg.nii.gz).
    reference : Path
        SIGMA template image defining output geometry.
    transforms : list of Path
        Transform files in ANTs order (applied last-to-first):
        [tpl_to_sigma_1Warp, tpl_to_sigma_0GenericAffine,
         T2w_to_template_1Warp, T2w_to_template_0GenericAffine]
    output_path : Path
        Where to save the warped tissue map.
    interpolation : str
        Interpolation method (default: Linear).

    Returns
    -------
    Path
        Path to the warped tissue map.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(tissue_map),
        '-r', str(reference),
        '-o', str(output_path),
        '-n', interpolation,
    ]
    for t in transforms:
        cmd.extend(['-t', str(t)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"antsApplyTransforms failed for {tissue_map.name}:\n{result.stderr}"
        )

    return output_path


def compose_warp_field(
    reference: Path,
    transforms: List[Path],
    output_path: Path,
) -> Path:
    """
    Compose multiple transforms into a single displacement field.

    Uses antsApplyTransforms with output-data-type displacement field
    to produce a single warp that can be used for Jacobian computation.

    Parameters
    ----------
    reference : Path
        Reference image defining output geometry.
    transforms : list of Path
        Transform files in ANTs order.
    output_path : Path
        Where to save the composed warp field.

    Returns
    -------
    Path
        Path to the composed displacement field.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-r', str(reference),
        '-o', f'[{output_path},1]',
    ]
    for t in transforms:
        cmd.extend(['-t', str(t)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"antsApplyTransforms (compose) failed:\n{result.stderr}"
        )

    return output_path


def compute_jacobian(
    warp_field: Path,
    output_path: Path,
    use_log: bool = False,
) -> Path:
    """
    Compute Jacobian determinant from a displacement field.

    Uses ANTs CreateJacobianDeterminantImage. Values > 1 indicate
    local expansion, < 1 indicate contraction.

    Parameters
    ----------
    warp_field : Path
        Displacement field (from compose_warp_field or registration).
    output_path : Path
        Where to save the Jacobian determinant image.
    use_log : bool
        If True, compute log-Jacobian (default: False).

    Returns
    -------
    Path
        Path to the Jacobian determinant image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'CreateJacobianDeterminantImage',
        '3',
        str(warp_field),
        str(output_path),
        '1' if use_log else '0',  # doLogJacobian
        '0',  # useGeometric (0 = arithmetic)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"CreateJacobianDeterminantImage failed:\n{result.stderr}"
        )

    return output_path


def modulate_tissue(
    tissue_sigma: Path,
    jacobian: Path,
    output_path: Path,
) -> Path:
    """
    Multiply tissue probability by Jacobian determinant for volume preservation.

    Modulated GM/WM maps reflect local tissue *volume* rather than
    concentration, accounting for the expansion/contraction introduced
    by spatial normalization.

    Parameters
    ----------
    tissue_sigma : Path
        Warped tissue probability map in SIGMA space.
    jacobian : Path
        Jacobian determinant image (same geometry as tissue_sigma).
    output_path : Path
        Where to save the modulated map.

    Returns
    -------
    Path
        Path to the modulated tissue map.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tissue_img = nib.load(tissue_sigma)
    tissue_data = tissue_img.get_fdata(dtype=np.float32)

    jacobian_data = nib.load(jacobian).get_fdata(dtype=np.float32)

    # Use absolute value of Jacobian (determinant should be positive
    # for well-behaved warps, but ensure robustness)
    modulated = tissue_data * np.abs(jacobian_data)

    nib.save(
        nib.Nifti1Image(modulated, tissue_img.affine, tissue_img.header),
        output_path,
    )

    return output_path


def smooth_volume(
    input_path: Path,
    output_path: Path,
    fwhm_mm: float = 3.0,
) -> Path:
    """
    Gaussian smooth a 3D volume using fslmaths.

    Parameters
    ----------
    input_path : Path
        Input NIfTI volume.
    output_path : Path
        Where to save the smoothed volume.
    fwhm_mm : float
        Full-width at half-maximum in mm (default: 3.0).
        Converted to sigma via sigma = FWHM / 2.3548.

    Returns
    -------
    Path
        Path to the smoothed volume.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sigma = fwhm_mm / 2.3548

    cmd = [
        'fslmaths',
        str(input_path),
        '-kernel', 'gauss', f'{sigma:.4f}',
        '-fmean',
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"fslmaths smooth failed:\n{result.stderr}")

    return output_path
