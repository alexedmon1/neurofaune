"""
Skull stripping utilities for functional MRI data.

Supports multiple skull stripping methods optimized for rodent brains.
"""

from pathlib import Path
import subprocess
import shutil
from typing import Dict, Optional


def brain_extraction_bet(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    frac: float = 0.3
) -> Dict[str, Path]:
    """
    Brain extraction using FSL BET.

    Parameters
    ----------
    input_file : Path
        Input image (3D or 4D)
    output_file : Path
        Output brain image
    mask_file : Path
        Output binary brain mask
    frac : float, optional
        Fractional intensity threshold (0.0-1.0), default 0.3

    Returns
    -------
    dict
        Dictionary with 'brain' and 'mask' file paths
    """
    print(f"Extracting brain using FSL BET (frac={frac})...")

    # Use BET with functional flag
    cmd = [
        "bet",
        str(input_file),
        str(output_file.with_suffix('')),  # BET adds .nii.gz
        "-f", str(frac),
        "-R",  # Robust brain center estimation
        "-m",  # Create binary mask
        "-n"   # Don't generate segmented brain surface
    ]

    subprocess.run(cmd, check=True)

    # BET creates mask with _mask suffix
    bet_mask = output_file.with_suffix('').with_suffix('').with_name(
        output_file.stem.replace('.nii', '') + '_mask.nii.gz'
    )

    # Rename mask to expected name
    if bet_mask.exists() and bet_mask != mask_file:
        shutil.move(str(bet_mask), str(mask_file))

    print(f"  Brain extraction complete")
    print(f"  Brain: {output_file}")
    print(f"  Mask: {mask_file}")

    return {
        'brain': output_file,
        'mask': mask_file
    }


def brain_extraction_bet4animal(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    frac: float = 0.7,
    center: tuple = (40, 25, 5),
    radius: int = 125,
    scale: tuple = (1, 1, 1.5),
    width: float = 2.5
) -> Dict[str, Path]:
    """
    Brain extraction using bet4animal (rodent-optimized).

    Parameters from old successful pipeline:
    - frac=0.7 (higher threshold for rodent brains)
    - center=(40, 25, 5) in voxels
    - radius=125 voxels
    - scale=(1, 1, 1.5) for anisotropic voxels
    - width=2.5

    Parameters
    ----------
    input_file : Path
        Input image (3D or 4D)
    output_file : Path
        Output brain image
    mask_file : Path
        Output binary brain mask
    frac : float, optional
        Fractional intensity threshold, default 0.7
    center : tuple, optional
        Brain center coordinates (x, y, z) in voxels, default (40, 25, 5)
    radius : int, optional
        Brain radius in voxels, default 125
    scale : tuple, optional
        Voxel scaling factors (x, y, z), default (1, 1, 1.5)
    width : float, optional
        Smoothness parameter, default 2.5

    Returns
    -------
    dict
        Dictionary with 'brain' and 'mask' file paths
    """
    print(f"Extracting brain using bet4animal (frac={frac})...")
    print(f"  Parameters: center={center}, radius={radius}, scale={scale}, width={width}")

    # Build bet4animal command
    cmd = [
        "bet4animal",
        str(input_file),
        str(output_file.with_suffix('')),  # bet4animal adds .nii.gz
        "-f", str(frac),
        "-c", f"{center[0]} {center[1]} {center[2]}",
        "-r", str(radius),
        "-x", f"{scale[0]},{scale[1]},{scale[2]}",
        "-w", str(width),
        "-m"  # Create binary mask
    ]

    subprocess.run(cmd, check=True)

    # bet4animal creates mask with _mask suffix
    bet_mask = output_file.with_suffix('').with_suffix('').with_name(
        output_file.stem.replace('.nii', '') + '_mask.nii.gz'
    )

    # Rename mask to expected name
    if bet_mask.exists() and bet_mask != mask_file:
        shutil.move(str(bet_mask), str(mask_file))

    print(f"  Brain extraction complete")
    print(f"  Brain: {output_file}")
    print(f"  Mask: {mask_file}")

    return {
        'brain': output_file,
        'mask': mask_file
    }


def brain_extraction(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    method: str = "bet",
    **kwargs
) -> Dict[str, Path]:
    """
    Brain extraction with automatic method selection.

    Parameters
    ----------
    input_file : Path
        Input image (3D or 4D)
    output_file : Path
        Output brain image
    mask_file : Path
        Output binary brain mask
    method : str, optional
        Skull stripping method: 'bet' or 'bet4animal', default 'bet'
    **kwargs
        Additional parameters passed to the specific method

    Returns
    -------
    dict
        Dictionary with 'brain' and 'mask' file paths

    Raises
    ------
    ValueError
        If method is not recognized
    """
    if method == "bet":
        return brain_extraction_bet(input_file, output_file, mask_file, **kwargs)
    elif method == "bet4animal":
        return brain_extraction_bet4animal(input_file, output_file, mask_file, **kwargs)
    else:
        raise ValueError(f"Unknown skull stripping method: {method}. Use 'bet' or 'bet4animal'")
