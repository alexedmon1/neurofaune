"""
Functional fMRI preprocessing workflow.

This module provides a complete preprocessing pipeline for resting-state and
task-based fMRI data, including motion correction, denoising, and confound regression.

NOTE: Registration to study-specific template and SIGMA atlas will be added
separately in template building/registration modules.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import subprocess
import json
import shutil

from neurofaune.preprocess.utils.validation import validate_image, print_validation_results
from neurofaune.preprocess.utils.orientation import (
    match_orientation_to_reference,
    save_orientation_metadata,
    print_orientation_info
)
from neurofaune.utils.transforms import TransformRegistry
from neurofaune.preprocess.qc.func import (
    generate_motion_qc_report,
    generate_confounds_qc_report,
)
from neurofaune.preprocess.qc import get_subject_qc_dir
from neurofaune.preprocess.utils.func.ica_denoising import (
    run_melodic_ica,
    classify_ica_components,
    remove_noise_components,
    generate_ica_denoising_qc
)
from neurofaune.preprocess.utils.skull_strip import skull_strip
from neurofaune.preprocess.utils.func.acompcor import (
    extract_acompcor_components,
    generate_acompcor_qc
)
from neurofaune.preprocess.utils.func.slice_timing import (
    run_slice_timing_correction,
    detect_slice_order
)

from typing import Union


def _find_z_offset_ncc(
    bold_img: nib.Nifti1Image,
    t2w_img: nib.Nifti1Image,
    work_dir: Path,
    z_range: Optional[Tuple[int, int]] = None
) -> Tuple[Path, Dict]:
    """
    Find optimal Z offset for partial-coverage BOLD to full T2w registration.

    Scans Z translations and computes normalized cross-correlation between
    resampled BOLD slices and corresponding T2w slices. Writes an ITK-format
    initial transform encoding the optimal Z translation for use with
    antsRegistration --initial-moving-transform.

    Parameters
    ----------
    bold_img : Nifti1Image
        BOLD reference volume
    t2w_img : Nifti1Image
        T2w volume
    work_dir : Path
        Working directory
    z_range : tuple of (int, int), optional
        (min_slice, max_slice) to constrain the Z search. If None, searches
        all valid positions. Use this to avoid spurious NCC peaks when the
        expected slab position is known (e.g., hippocampal MSME at slices
        14-28 in template space).

    Returns
    -------
    tuple of (Path, dict)
        Path to initial transform .mat file, and dict with offset info
    """
    from scipy.ndimage import zoom as scipy_zoom

    bold_data = bold_img.get_fdata()
    t2w_data = t2w_img.get_fdata()
    bold_zooms = bold_img.header.get_zooms()
    t2w_zooms = t2w_img.header.get_zooms()

    # Resample BOLD in-plane to T2w resolution
    scale_xy = float(bold_zooms[0]) / float(t2w_zooms[0])
    bold_resampled = scipy_zoom(bold_data, (scale_xy, scale_xy, 1), order=1)

    # Z scale: how many T2w slices per BOLD slice
    z_scale = float(bold_zooms[2]) / float(t2w_zooms[2])
    n_bold_in_t2w = int(round(bold_data.shape[2] * z_scale))

    # Determine Z search range
    max_offset = t2w_data.shape[2] - n_bold_in_t2w
    if z_range is not None:
        z_start = max(0, z_range[0])
        z_end = min(max_offset, z_range[1]) + 1
        print(f"  Z search range: slices {z_start}-{min(max_offset, z_range[1])} "
              f"(constrained from full range 0-{max_offset})")
    else:
        z_start = 0
        z_end = max(max_offset, 1)

    # Scan Z offsets
    best_ncc = -1
    best_offset = 0

    for z_offset in range(z_start, z_end):
        total_ncc = 0
        n_valid = 0

        for bz in range(bold_data.shape[2]):
            t2w_zi = int(round(z_offset + bz * z_scale))
            if t2w_zi < 0 or t2w_zi >= t2w_data.shape[2]:
                continue

            # Compare in common region
            min_x = min(bold_resampled.shape[0], t2w_data.shape[0])
            min_y = min(bold_resampled.shape[1], t2w_data.shape[1])

            b_slice = bold_resampled[:min_x, :min_y, bz]
            t_slice = t2w_data[:min_x, :min_y, t2w_zi]

            mask = b_slice > 0
            if np.sum(mask) < 500:
                continue

            b_vals = b_slice[mask]
            t_vals = t_slice[mask]

            b_norm = (b_vals - b_vals.mean()) / (b_vals.std() + 1e-10)
            t_norm = (t_vals - t_vals.mean()) / (t_vals.std() + 1e-10)
            ncc = np.mean(b_norm * t_norm)
            total_ncc += ncc
            n_valid += 1

        if n_valid > 0:
            avg_ncc = total_ncc / n_valid
            if avg_ncc > best_ncc:
                best_ncc = avg_ncc
                best_offset = z_offset

    # Convert slice offset to mm
    z_offset_mm = best_offset * float(t2w_zooms[2])
    print(f"  Best Z offset: T2w slice {best_offset} ({z_offset_mm:.1f} mm), NCC={best_ncc:.4f}")
    print(f"  BOLD maps to T2w slices {best_offset}-{best_offset + n_bold_in_t2w}")

    # Write ITK initial transform with Z translation
    # ANTs transform maps fixed (T2w) → moving (BOLD) coordinates
    # BOLD Z=0 should map to T2w Z=z_offset_mm, so tz = -z_offset_mm
    work_dir.mkdir(parents=True, exist_ok=True)
    initial_transform_file = work_dir / 'initial_z_offset.txt'
    with open(initial_transform_file, 'w') as f:
        f.write("#Insight Transform File V1.0\n")
        f.write("#Transform 0\n")
        f.write("Transform: AffineTransform_double_3_3\n")
        f.write(f"Parameters: 1 0 0 0 1 0 0 0 1 0 0 {-z_offset_mm}\n")
        f.write("FixedParameters: 0 0 0\n")

    return initial_transform_file, {
        'z_offset_slice': best_offset,
        'z_offset_mm': z_offset_mm,
        'ncc': best_ncc,
        'bold_t2w_range': (best_offset, best_offset + n_bold_in_t2w)
    }


def register_bold_to_template(
    bold_ref_file: Path,
    template_file: Path,
    output_dir: Path,
    subject: str,
    session: str,
    work_dir: Path,
    n_cores: int = 4
) -> Dict[str, Any]:
    """
    Register mean BOLD directly to the cohort template.

    Registers the brain-extracted BOLD reference volume to the age-matched
    template volume using NCC-based Z initialization and rigid registration.
    This produces better SIGMA atlas overlap than the old BOLD→T2w→Template
    chain.

    Parameters
    ----------
    bold_ref_file : Path
        Brain-extracted BOLD reference volume (3D)
    template_file : Path
        Cohort template (e.g., tpl-BPARat_p60_T2w.nii.gz)
    output_dir : Path
        Study root directory (transforms saved to transforms/{subject}/{session}/)
    subject : str
        Subject ID
    session : str
        Session ID
    work_dir : Path
        Working directory for intermediate files
    n_cores : int
        Number of CPU cores for ANTs

    Returns
    -------
    dict
        Dictionary with transform paths and metadata
    """
    print("\n" + "="*60)
    print("BOLD to Template Registration")
    print("="*60)

    # Load images to get info
    bold_img = nib.load(bold_ref_file)
    template_img = nib.load(template_file)
    print(f"\n  BOLD ref: {bold_img.shape} voxels, {bold_img.header.get_zooms()[:3]} mm")
    print(f"  Template: {template_img.shape} voxels, {template_img.header.get_zooms()[:3]} mm")

    transforms_dir = output_dir / 'transforms' / subject / session
    transforms_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = transforms_dir / 'BOLD_to_template_'

    # Step 1: Find optimal Z offset via NCC scan
    print("\n  Finding optimal Z offset via NCC scan...")
    initial_transform, z_offset_info = _find_z_offset_ncc(
        bold_img, template_img, work_dir
    )

    # Step 2: Run antsRegistration directly with the original BOLD + initial transform
    print("\nRunning ANTs Rigid registration (BOLD → Template)...")

    warped_output = Path(str(output_prefix) + 'Warped.nii.gz')
    cmd = [
        'antsRegistration',
        '--dimensionality', '3',
        '--output', f'[{output_prefix},{warped_output}]',
        '--interpolation', 'Linear',
        '--use-histogram-matching', '1',
        '--winsorize-image-intensities', '[0.005,0.995]',
        '--initial-moving-transform', str(initial_transform),
        # Rigid only (6 DOF: translation + rotation)
        '--transform', 'Rigid[0.1]',
        '--metric', f'MI[{template_file},{bold_ref_file},1,32,Regular,0.25]',
        '--convergence', '[1000x500x250x100,1e-6,10]',
        '--shrink-factors', '4x2x1x1',
        '--smoothing-sigmas', '2x1x0x0vox',
    ]

    print(f"  Moving: {bold_ref_file.name}")
    print(f"  Fixed: {template_file.name}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if result.returncode != 0:
        print(f"  ERROR: Registration failed!")
        print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
        raise RuntimeError("BOLD to Template registration failed")

    # Check outputs
    affine_transform = Path(str(output_prefix) + '0GenericAffine.mat')
    warped_bold = Path(str(output_prefix) + 'Warped.nii.gz')

    if not affine_transform.exists():
        raise RuntimeError(f"Expected transform not found: {affine_transform}")

    print(f"  Rigid transform: {affine_transform.name}")
    if warped_bold.exists():
        print(f"  Warped BOLD ref: {warped_bold.name}")

        # Report which template slices have BOLD coverage
        warped_data = nib.load(warped_bold).get_fdata()
        slices_with_bold = [z for z in range(warped_data.shape[2])
                           if np.sum(warped_data[:, :, z] > 0) > 1000]
        if slices_with_bold:
            print(f"  BOLD covers template slices {slices_with_bold[0]}-{slices_with_bold[-1]} ({len(slices_with_bold)} slices)")

    return {
        'affine_transform': affine_transform,
        'warped_bold': warped_bold if warped_bold.exists() else None,
        'template_file': template_file,
        'bold_shape': bold_img.shape,
        'template_shape': template_img.shape,
    }


def register_bold_to_t2w(
    bold_ref_file: Path,
    t2w_file: Path,
    output_dir: Path,
    subject: str,
    session: str,
    work_dir: Path,
    n_cores: int = 4
) -> Dict[str, Any]:
    """
    Register mean BOLD to T2w within the same subject.

    .. deprecated::
        Use :func:`register_bold_to_template` instead for better atlas overlap.

    Registers the brain-extracted BOLD reference volume directly to the full
    T2w volume, letting ANTs find the optimal 3D alignment including the
    Z-offset for partial coverage fMRI.

    Parameters
    ----------
    bold_ref_file : Path
        Brain-extracted BOLD reference volume (3D)
    t2w_file : Path
        Preprocessed T2w from anatomical pipeline
    output_dir : Path
        Study root directory (transforms saved to transforms/{subject}/{session}/)
    subject : str
        Subject ID
    session : str
        Session ID
    work_dir : Path
        Working directory for intermediate files
    n_cores : int
        Number of CPU cores for ANTs

    Returns
    -------
    dict
        Dictionary with transform paths and metadata
    """
    import warnings
    warnings.warn(
        "register_bold_to_t2w() is deprecated. Use register_bold_to_template() "
        "for better SIGMA atlas overlap.",
        DeprecationWarning,
        stacklevel=2
    )

    print("\n" + "="*60)
    print("BOLD to T2w Registration")
    print("="*60)

    # Load images to get info
    bold_img = nib.load(bold_ref_file)
    t2w_img = nib.load(t2w_file)
    print(f"\n  BOLD ref: {bold_img.shape} voxels, {bold_img.header.get_zooms()[:3]} mm")
    print(f"  T2w: {t2w_img.shape} voxels, {t2w_img.header.get_zooms()[:3]} mm")

    transforms_dir = output_dir / 'transforms' / subject / session
    transforms_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = transforms_dir / 'BOLD_to_T2w_'

    # Step 1: Find optimal Z offset via NCC scan
    # (needed because partial-coverage BOLD + both origins at 0,0,0 gives
    #  ANTs a poor starting point for Z translation)
    print("\n  Finding optimal Z offset via NCC scan...")
    initial_transform, z_offset_info = _find_z_offset_ncc(
        bold_img, t2w_img, work_dir
    )

    # Step 2: Run antsRegistration directly with the original BOLD + initial transform
    # (antsRegistrationSyN.sh overrides initialization with COM alignment,
    #  so we call antsRegistration directly to control initialization)
    print("\nRunning ANTs Rigid registration (BOLD → full T2w)...")

    warped_output = Path(str(output_prefix) + 'Warped.nii.gz')
    cmd = [
        'antsRegistration',
        '--dimensionality', '3',
        '--output', f'[{output_prefix},{warped_output}]',
        '--interpolation', 'Linear',
        '--use-histogram-matching', '1',
        '--winsorize-image-intensities', '[0.005,0.995]',
        '--initial-moving-transform', str(initial_transform),
        # Rigid only (6 DOF: translation + rotation)
        # Affine is too unconstrained for 9-slice partial-coverage data
        '--transform', 'Rigid[0.1]',
        '--metric', f'MI[{t2w_file},{bold_ref_file},1,32,Regular,0.25]',
        '--convergence', '[1000x500x250x100,1e-6,10]',
        '--shrink-factors', '4x2x1x1',
        '--smoothing-sigmas', '2x1x0x0vox',
    ]

    print(f"  Moving: {bold_ref_file.name}")
    print(f"  Fixed: {t2w_file.name}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if result.returncode != 0:
        print(f"  ERROR: Registration failed!")
        print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
        raise RuntimeError("BOLD to T2w registration failed")

    # Check outputs
    affine_transform = Path(str(output_prefix) + '0GenericAffine.mat')
    warped_bold = Path(str(output_prefix) + 'Warped.nii.gz')

    if not affine_transform.exists():
        raise RuntimeError(f"Expected transform not found: {affine_transform}")

    print(f"  Rigid transform: {affine_transform.name}")
    if warped_bold.exists():
        print(f"  Warped BOLD ref: {warped_bold.name}")

        # Report which T2w slices have BOLD coverage
        warped_data = nib.load(warped_bold).get_fdata()
        slices_with_bold = [z for z in range(warped_data.shape[2])
                           if np.sum(warped_data[:, :, z] > 0) > 1000]
        if slices_with_bold:
            print(f"  BOLD covers T2w slices {slices_with_bold[0]}-{slices_with_bold[-1]} ({len(slices_with_bold)} slices)")

    return {
        'affine_transform': affine_transform,
        'warped_bold': warped_bold if warped_bold.exists() else None,
        't2w_file': t2w_file,
        'bold_shape': bold_img.shape,
        't2w_shape': t2w_img.shape,
    }


def discard_initial_volumes(
    input_file: Path,
    output_file: Path,
    n_discard: int = 0
) -> Tuple[Path, int]:
    """
    Discard initial volumes from fMRI timeseries for T1 equilibration.

    Parameters
    ----------
    input_file : Path
        Input 4D fMRI file
    output_file : Path
        Output file with discarded volumes
    n_discard : int
        Number of initial volumes to discard (default: 0)

    Returns
    -------
    Path
        Path to output file
    int
        Number of volumes discarded
    """
    if n_discard == 0:
        # No volumes to discard, just return input
        return input_file, 0

    print(f"Discarding first {n_discard} volumes for T1 equilibration...")

    # Load image
    img = nib.load(input_file)
    data = img.get_fdata()

    if len(data.shape) != 4:
        raise ValueError(f"Expected 4D image, got shape {data.shape}")

    # Discard volumes
    data_trimmed = data[..., n_discard:]

    # Save
    img_trimmed = nib.Nifti1Image(data_trimmed, img.affine, img.header)
    nib.save(img_trimmed, output_file)

    print(f"  Trimmed from {data.shape[3]} to {data_trimmed.shape[3]} volumes")

    return output_file, n_discard


def run_motion_correction(
    input_file: Path,
    output_dir: Path,
    reference: str = "middle",
    method: str = "mcflirt"
) -> Dict[str, Path]:
    """
    Perform motion correction on fMRI timeseries.

    Parameters
    ----------
    input_file : Path
        Input 4D fMRI file
    output_dir : Path
        Output directory for motion-corrected data
    reference : str
        Reference volume: 'first', 'middle', or 'mean' (default: 'middle')
    method : str
        Motion correction method: 'mcflirt' or 'ants' (default: 'mcflirt')

    Returns
    -------
    dict
        Paths to motion-corrected image and motion parameters
    """
    print(f"Running motion correction ({method})...")

    output_dir.mkdir(parents=True, exist_ok=True)

    if method == "mcflirt":
        # FSL MCFLIRT
        output_file = output_dir / "bold_mcf.nii.gz"
        motion_params = output_dir / "bold_mcf.nii.par"  # MCFLIRT creates .nii.par file

        # Determine reference volume
        if reference == "first":
            ref_vol = 0
        elif reference == "middle":
            img = nib.load(input_file)
            ref_vol = img.shape[3] // 2
        elif reference == "mean":
            ref_vol = "mean"
        else:
            raise ValueError(f"Unknown reference: {reference}")

        # Run MCFLIRT
        cmd = [
            "mcflirt",
            "-in", str(input_file),
            "-out", str(output_file.with_suffix('')),  # MCFLIRT adds .nii.gz
            "-plots"  # Generate motion plots
        ]

        if ref_vol != "mean":
            cmd.extend(["-refvol", str(ref_vol)])
        else:
            cmd.append("-meanvol")

        subprocess.run(cmd, check=True)

    elif method == "ants":
        # ANTs motion correction (slower but more accurate)
        raise NotImplementedError("ANTs motion correction not yet implemented")

    else:
        raise ValueError(f"Unknown motion correction method: {method}")

    print(f"  Motion correction complete")
    print(f"  Output: {output_file}")
    print(f"  Motion params: {motion_params}")

    return {
        'motion_corrected': output_file,
        'motion_params': motion_params
    }


def extract_brain_from_bold(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    frac: float = 0.3
) -> Dict[str, Path]:
    """
    Extract brain from BOLD image using BET.

    Parameters
    ----------
    input_file : Path
        Input BOLD image (preferably motion-corrected mean)
    output_file : Path
        Output brain-extracted image
    mask_file : Path
        Output brain mask
    frac : float
        BET fractional intensity threshold (default: 0.3 for rodents)

    Returns
    -------
    dict
        Paths to brain and mask
    """
    print(f"Extracting brain from BOLD image...")

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


def apply_mask_to_timeseries(
    input_file: Path,
    mask_file: Path,
    output_file: Path
) -> Path:
    """
    Apply brain mask to 4D fMRI timeseries.

    Parameters
    ----------
    input_file : Path
        Input 4D fMRI timeseries
    mask_file : Path
        Brain mask
    output_file : Path
        Output masked timeseries

    Returns
    -------
    Path
        Path to masked timeseries
    """
    print("Applying brain mask to timeseries...")

    # Use fslmaths to apply mask
    cmd = [
        "fslmaths",
        str(input_file),
        "-mas", str(mask_file),
        str(output_file)
    ]

    subprocess.run(cmd, check=True)

    print(f"  Masked timeseries: {output_file}")

    return output_file


def smooth_image(
    input_file: Path,
    output_file: Path,
    fwhm: float
) -> Path:
    """
    Apply spatial smoothing to fMRI image.

    Parameters
    ----------
    input_file : Path
        Input image
    output_file : Path
        Output smoothed image
    fwhm : float
        Full-width at half-maximum of Gaussian kernel (mm)

    Returns
    -------
    Path
        Path to smoothed image
    """
    if fwhm == 0:
        print("Skipping spatial smoothing (FWHM=0)")
        return input_file

    print(f"Applying spatial smoothing (FWHM={fwhm}mm)...")

    # Convert FWHM to sigma: sigma = FWHM / (2 * sqrt(2 * ln(2)))
    sigma = fwhm / 2.355

    # Use FSL's susan for edge-preserving smoothing
    cmd = [
        "fslmaths",
        str(input_file),
        "-s", str(sigma),
        str(output_file)
    ]

    subprocess.run(cmd, check=True)

    print(f"  Smoothed image: {output_file}")

    return output_file


def temporal_filter(
    input_file: Path,
    output_file: Path,
    tr: float,
    highpass: Optional[float] = None,
    lowpass: Optional[float] = None
) -> Path:
    """
    Apply temporal filtering to fMRI timeseries.

    Parameters
    ----------
    input_file : Path
        Input 4D timeseries
    output_file : Path
        Output filtered timeseries
    tr : float
        Repetition time (seconds)
    highpass : float, optional
        Highpass filter cutoff (Hz)
    lowpass : float, optional
        Lowpass filter cutoff (Hz)

    Returns
    -------
    Path
        Path to filtered timeseries
    """
    if highpass is None and lowpass is None:
        print("Skipping temporal filtering")
        return input_file

    print(f"Applying temporal filter (TR={tr}s)...")

    # FSL's fslmaths for temporal filtering
    # Highpass: -bptf <hp_sigma> <lp_sigma>
    # sigma = 1 / (2 * cutoff_freq * TR)

    if highpass is not None:
        hp_sigma = 1.0 / (2.0 * highpass * tr)
        print(f"  Highpass: {highpass} Hz (sigma={hp_sigma:.2f} volumes)")
    else:
        hp_sigma = -1  # No highpass

    if lowpass is not None:
        lp_sigma = 1.0 / (2.0 * lowpass * tr)
        print(f"  Lowpass: {lowpass} Hz (sigma={lp_sigma:.2f} volumes)")
    else:
        lp_sigma = -1  # No lowpass

    cmd = [
        "fslmaths",
        str(input_file),
        "-bptf", str(hp_sigma), str(lp_sigma),
        str(output_file)
    ]

    subprocess.run(cmd, check=True)

    print(f"  Filtered timeseries: {output_file}")

    return output_file


def calculate_mean_image(
    input_file: Path,
    output_file: Path
) -> Path:
    """
    Calculate mean image from 4D timeseries.

    Parameters
    ----------
    input_file : Path
        Input 4D timeseries
    output_file : Path
        Output mean image

    Returns
    -------
    Path
        Path to mean image
    """
    print("Calculating mean image...")

    cmd = [
        "fslmaths",
        str(input_file),
        "-Tmean",
        str(output_file)
    ]

    subprocess.run(cmd, check=True)

    print(f"  Mean image: {output_file}")

    return output_file


def extract_confounds(
    motion_params: Path,
    output_file: Path,
    derivatives: bool = True,
    squares: bool = True
) -> Path:
    """
    Extract confound regressors from motion parameters.

    Generates extended motion regressors including:
    - 6 motion parameters (3 translations, 3 rotations)
    - Temporal derivatives (optional)
    - Squared terms (optional)

    Total: 6 + 6 + 12 = 24 regressors (if derivatives and squares enabled)

    Parameters
    ----------
    motion_params : Path
        FSL motion parameters file (.par)
    output_file : Path
        Output confounds TSV file
    derivatives : bool
        Include temporal derivatives (default: True)
    squares : bool
        Include squared terms (default: True)

    Returns
    -------
    Path
        Path to confounds file
    """
    print("Extracting confound regressors from motion parameters...")

    # Load motion parameters (6 columns: 3 rotations, 3 translations)
    motion = np.loadtxt(motion_params)

    confounds = []
    headers = []

    # Original motion parameters
    confounds.append(motion)
    headers.extend(['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z'])

    # Temporal derivatives
    if derivatives:
        motion_deriv = np.vstack([np.zeros((1, 6)), np.diff(motion, axis=0)])
        confounds.append(motion_deriv)
        headers.extend([f'{h}_deriv' for h in headers[:6]])

    # Squared terms
    if squares:
        confounds.append(motion ** 2)
        headers.extend([f'{h}_sq' for h in headers[:6]])

        if derivatives:
            confounds.append(motion_deriv ** 2)
            headers.extend([f'{h}_deriv_sq' for h in headers[:6]])

    # Concatenate all confounds
    all_confounds = np.hstack(confounds)

    # Save as TSV
    np.savetxt(
        output_file,
        all_confounds,
        delimiter='\t',
        header='\t'.join(headers),
        comments=''
    )

    print(f"  Extracted {all_confounds.shape[1]} confound regressors")
    print(f"  Confounds file: {output_file}")

    return output_file


def run_tedana(
    echo_files: List[Path],
    echo_times_ms: List[float],
    output_dir: Path,
    mask_file: Path,
    tedpca = 0.95,
    tree: str = 'kundu',
) -> Dict[str, Path]:
    """Run TEDANA multi-echo ICA denoising.

    Parameters
    ----------
    echo_files : list of Path
        Motion-corrected echo NIfTI paths (one per echo)
    echo_times_ms : list of float
        Echo times in milliseconds
    output_dir : Path
        Directory for tedana outputs
    mask_file : Path
        Brain mask
    tedpca : str, int, or float
        PCA component selection: 'kundu', 'aic', 'kic', 'mdl',
        float 0-1 for variance % (default 0.95=95%), or int for exact count
    tree : str
        Decision tree method ('kundu' or 'kundu_tedort')

    Returns
    -------
    dict
        Paths to 'optcom', 'denoised', 'metrics', 'report'
    """
    from tedana import workflows as tedana_workflows

    output_dir.mkdir(parents=True, exist_ok=True)

    # tedana expects TEs in milliseconds
    print(f"  Running TEDANA multi-echo ICA denoising...")
    print(f"  Echo files: {[f.name for f in echo_files]}")
    print(f"  Echo times (ms): {echo_times_ms}")
    print(f"  Mask: {mask_file.name}")
    print(f"  PCA method: {tedpca}, Decision tree: {tree}")

    # tedana accepts str ('kundu','aic','kic','mdl'), float (variance), or int (n_components)
    tedana_workflows.tedana_workflow(
        data=[str(f) for f in echo_files],
        tes=[float(te) for te in echo_times_ms],
        out_dir=str(output_dir),
        mask=str(mask_file),
        tedpca=tedpca,
        tree=tree,
        verbose=True,
        prefix='tedana',
        overwrite=True,
    )

    # Collect output paths — tedana 23.x uses 'optcomDenoised' naming
    results = {}
    optcom = output_dir / 'tedana_desc-optcom_bold.nii.gz'
    # tedana 23.x: desc-optcomDenoised; older versions: desc-denoised
    denoised = output_dir / 'tedana_desc-optcomDenoised_bold.nii.gz'
    if not denoised.exists():
        denoised = output_dir / 'tedana_desc-denoised_bold.nii.gz'
    metrics = output_dir / 'tedana_desc-tedana_metrics.tsv'

    if optcom.exists():
        results['optcom'] = optcom
        print(f"  Optimally combined: {optcom.name}")
    if denoised.exists():
        results['denoised'] = denoised
        print(f"  Denoised output: {denoised.name}")
    if metrics.exists():
        results['metrics'] = metrics
        print(f"  Component metrics: {metrics.name}")

    # Count accepted/rejected components from metrics file
    if metrics.exists():
        import csv
        with open(metrics) as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)
        n_accepted = sum(1 for r in rows if r.get('classification') == 'accepted')
        n_rejected = sum(1 for r in rows if r.get('classification') == 'rejected')
        results['n_signal_components'] = n_accepted
        results['n_noise_components'] = n_rejected
        print(f"  Components: {n_accepted} accepted, {n_rejected} rejected (of {len(rows)} total)")

    # Check for report (may be in figures subdir)
    report = output_dir / 'tedana_report.html'
    if not report.exists():
        report = output_dir / 'figures' / 'tedana_report.html'
    if report.exists():
        results['report'] = report

    return results


def run_multiecho_motion_correction(
    echo_files: List[Path],
    work_dir: Path,
    brain_mask_out: Path,
    skull_strip_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Motion-correct multi-echo data using middle echo as reference.

    Strategy:
    1. MCFLIRT on middle echo -> per-volume .mat transforms
    2. Apply those transforms to all other echoes
    3. Skull strip on temporal mean of motion-corrected middle echo -> brain mask

    Parameters
    ----------
    echo_files : list of Path
        Per-echo 4D NIfTI files (trimmed, same number of volumes)
    work_dir : Path
        Working directory for intermediate files
    brain_mask_out : Path
        Where to save the output brain mask
    skull_strip_config : dict, optional
        Config for adaptive skull stripping (target_ratio, frac_range, etc.)

    Returns
    -------
    dict with:
        'corrected_echoes': list of Path (motion-corrected echo files)
        'brain_mask': Path
        'motion_params': Path (.par file from MCFLIRT)
        'mean_bold': Path (mean of motion-corrected middle echo)
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    n_echoes = len(echo_files)
    ref_echo_idx = n_echoes // 2  # middle echo

    print(f"  Multi-echo motion correction ({n_echoes} echoes)")
    print(f"  Reference echo: {ref_echo_idx} ({echo_files[ref_echo_idx].name})")

    # Step 1: MCFLIRT on reference echo
    ref_echo = echo_files[ref_echo_idx]
    mcf_dir = work_dir / 'mcflirt'
    mcf_dir.mkdir(exist_ok=True)

    ref_mcf_base = mcf_dir / f'echo{ref_echo_idx}_mcf'
    ref_mcf_out = Path(str(ref_mcf_base) + '.nii.gz')

    ref_img = nib.load(ref_echo)
    ref_vol = ref_img.shape[3] // 2

    cmd = [
        'mcflirt',
        '-in', str(ref_echo),
        '-out', str(ref_mcf_base),
        '-plots',
        '-mats',  # Save per-volume transform matrices
        '-refvol', str(ref_vol),
    ]
    print(f"  Running MCFLIRT on echo {ref_echo_idx} (refvol={ref_vol})...")
    subprocess.run(cmd, check=True)

    motion_params = Path(str(ref_mcf_base) + '.par')
    mat_dir = Path(str(ref_mcf_base) + '.mat')

    # Collect per-volume .mat files
    n_vols = ref_img.shape[3]
    mat_files = [mat_dir / f'MAT_{i:04d}' for i in range(n_vols)]
    has_mats = all(m.exists() for m in mat_files)

    if not has_mats:
        print(f"  WARNING: Per-volume .mat files not found in {mat_dir}")
        print(f"  Falling back to MCFLIRT on each echo independently")

    # Step 2: Apply transforms to all echoes
    corrected_echoes = []
    for echo_idx, echo_file in enumerate(echo_files):
        echo_mcf_out = mcf_dir / f'echo{echo_idx}_mcf.nii.gz'

        if echo_idx == ref_echo_idx:
            # Reference echo already motion-corrected
            corrected_echoes.append(ref_mcf_out)
            continue

        if has_mats:
            # Use applyxfm4D to apply reference echo transforms
            print(f"  Applying transforms to echo {echo_idx}...")
            try:
                cmd = [
                    'applyxfm4D',
                    str(echo_file),
                    str(ref_echo),  # reference image
                    str(echo_mcf_out),
                    str(mat_dir),
                    '-fourdigit',
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                corrected_echoes.append(echo_mcf_out)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback: split, transform each volume, merge
                print(f"  applyxfm4D failed, using split/flirt/merge fallback...")
                corrected_echoes.append(
                    _apply_transforms_fallback(
                        echo_file, ref_echo, mat_files, echo_mcf_out,
                        mcf_dir / f'echo{echo_idx}_split', ref_vol
                    )
                )
        else:
            # Fallback: run MCFLIRT independently on this echo
            print(f"  Running MCFLIRT independently on echo {echo_idx}...")
            echo_mcf_base = mcf_dir / f'echo{echo_idx}_mcf'
            cmd = [
                'mcflirt',
                '-in', str(echo_file),
                '-out', str(echo_mcf_base),
                '-refvol', str(ref_vol),
            ]
            subprocess.run(cmd, check=True)
            corrected_echoes.append(Path(str(echo_mcf_base) + '.nii.gz'))

    # Step 3: Skull strip on temporal mean of motion-corrected middle echo
    print(f"\n  Computing temporal mean of motion-corrected middle echo...")
    mcf_ref_img = nib.load(corrected_echoes[ref_echo_idx])
    mcf_ref_data = mcf_ref_img.get_fdata()
    mean_data = np.mean(mcf_ref_data, axis=3)
    mean_bold = work_dir / 'mean_bold_ref.nii.gz'
    nib.save(
        nib.Nifti1Image(mean_data.astype(np.float32), mcf_ref_img.affine, mcf_ref_img.header),
        mean_bold
    )

    # N4 + normalization + skull strip (same as SE pipeline)
    ss_work = work_dir / 'skull_strip'
    ss_work.mkdir(exist_ok=True)

    print("  Running N4 bias correction on mean BOLD...")
    mean_n4 = ss_work / 'mean_bold_n4.nii.gz'
    subprocess.run([
        'N4BiasFieldCorrection', '-i', str(mean_bold), '-o', str(mean_n4)
    ], check=True, capture_output=True)

    print("  Running intensity normalization...")
    img_n4 = nib.load(mean_n4)
    data_n4 = img_n4.get_fdata()
    p2, p98 = np.percentile(data_n4[data_n4 > 0], [2, 98])
    data_norm = np.clip(data_n4, p2, p98)
    data_norm = (data_norm - p2) / (p98 - p2) * 1000
    mean_norm = ss_work / 'mean_bold_norm.nii.gz'
    nib.save(nib.Nifti1Image(data_norm, img_n4.affine, img_n4.header), mean_norm)

    print("  Running skull stripping...")
    brain_ref = work_dir / 'mean_bold_brain.nii.gz'

    # Use the same adaptive skull strip as SE pipeline
    ss_params = skull_strip_config or {}
    brain_ref, brain_mask_out, ss_info = skull_strip(
        input_file=mean_norm,
        output_file=brain_ref,
        mask_file=brain_mask_out,
        work_dir=ss_work,
        method='auto',
        target_ratio=ss_params.get('target_ratio', 0.15),
        frac_range=tuple(ss_params.get('frac_range', [0.30, 0.90])),
        frac_step=ss_params.get('frac_step', 0.05),
        invert_intensity=ss_params.get('invert_intensity', False),
    )

    print(f"  Skull strip: {ss_info.get('method', 'unknown')}, "
          f"{ss_info.get('total_voxels', 0):,} voxels, "
          f"ratio={ss_info.get('extraction_ratio', 0):.3f}")

    return {
        'corrected_echoes': corrected_echoes,
        'brain_mask': brain_mask_out,
        'brain_ref': brain_ref,
        'motion_params': motion_params,
        'mean_bold': mean_bold,
        'skull_strip_info': ss_info,
    }


def _apply_transforms_fallback(
    echo_file: Path,
    ref_file: Path,
    mat_files: List[Path],
    output_file: Path,
    split_dir: Path,
    ref_vol: int,
) -> Path:
    """Apply per-volume transforms using fslsplit/flirt/fslmerge fallback."""
    split_dir.mkdir(exist_ok=True)

    # Split echo into individual volumes
    subprocess.run([
        'fslsplit', str(echo_file), str(split_dir / 'vol'), '-t'
    ], check=True)

    # Extract reference volume from ref_file for -ref
    ref_vol_file = split_dir / 'ref_vol.nii.gz'
    subprocess.run([
        'fslroi', str(ref_file), str(ref_vol_file), str(ref_vol), '1'
    ], check=True)

    # Apply transform to each volume
    transformed = []
    for i, mat_file in enumerate(mat_files):
        vol_in = split_dir / f'vol{i:04d}.nii.gz'
        vol_out = split_dir / f'vol{i:04d}_xfm.nii.gz'
        if not vol_in.exists():
            break
        subprocess.run([
            'flirt',
            '-in', str(vol_in),
            '-ref', str(ref_vol_file),
            '-applyxfm',
            '-init', str(mat_file),
            '-out', str(vol_out),
        ], check=True)
        transformed.append(vol_out)

    # Merge back
    subprocess.run(
        ['fslmerge', '-t', str(output_file)] + [str(f) for f in transformed],
        check=True
    )

    return output_file


def run_functional_preprocessing(
    config: Dict[str, Any],
    subject: str,
    session: str,
    bold_file: Union[Path, List[Path]],
    output_dir: Path,
    transform_registry: TransformRegistry,
    template_file: Optional[Path] = None,
    t2w_file: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    n_discard: int = 0,
    run_registration: bool = True,
    echo_times: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Run complete functional fMRI preprocessing workflow.

    Supports both single-echo (SE) and multi-echo (ME) fMRI data.
    Multi-echo is auto-detected when bold_file is a list of paths.

    **Single-echo path:**
    1. Image validation
    2. Discard initial volumes
    2.5. Slice timing correction (optional)
    3. Brain extraction from mean BOLD
    4. Motion correction (MCFLIRT)
    5. ICA denoising (optional)
    6-13. Smoothing, confounds, regression, filtering, QC, registration

    **Multi-echo path:**
    1. Image validation (all echoes)
    2. Discard initial volumes (all echoes)
    3-4. Multi-echo motion correction (MCFLIRT on middle echo, apply to others) + BET
    5. TEDANA multi-echo ICA denoising (replaces MELODIC)
    6-13. Smoothing, confounds, regression, filtering, QC, registration

    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config()
    subject : str
        Subject identifier (e.g., 'sub-Rat207')
    session : str
        Session identifier (e.g., 'ses-p60')
    bold_file : Path or list of Path
        Single-echo: single 4D NIfTI path.
        Multi-echo: list of 4D NIfTI paths, one per echo.
    output_dir : Path
        Study root directory (will create derivatives/{subject}/{session}/func/)
    transform_registry : TransformRegistry
        Transform registry for saving spatial transforms
    template_file : Path, optional
        Cohort template for direct BOLD→Template registration (preferred)
    t2w_file : Path, optional
        T2w anatomical reference (deprecated, use template_file instead)
    work_dir : Path, optional
        Working directory (defaults to output_dir/work/{subject}/{session}/func_preproc)
    n_discard : int
        Number of initial volumes to discard (default: 0)
    run_registration : bool
        Whether to register BOLD to template (default: True, requires template_file or t2w_file)
    echo_times : list of float, optional
        Echo times in milliseconds. Required for multi-echo data.

    Returns
    -------
    dict
        Dictionary with output file paths and processing info
    """
    # =====================================================================
    # Detect multi-echo mode
    # =====================================================================
    is_multiecho = isinstance(bold_file, list) and len(bold_file) > 1
    if is_multiecho:
        assert echo_times is not None and len(echo_times) == len(bold_file), (
            f"echo_times must match bold_file length: "
            f"got {len(echo_times) if echo_times else 0} TEs for {len(bold_file)} echoes"
        )
        echo_files = [Path(f) for f in bold_file]
        n_echoes = len(echo_files)
        # Use first echo file as representative for JSON sidecar / validation
        primary_bold = echo_files[0]
    else:
        if isinstance(bold_file, list):
            bold_file = bold_file[0]  # single-element list
        bold_file = Path(bold_file)
        primary_bold = bold_file
        echo_files = None
        n_echoes = 1

    print("="*80)
    print("FUNCTIONAL fMRI PREPROCESSING WORKFLOW")
    if is_multiecho:
        print(f"  MODE: MULTI-ECHO ({n_echoes} echoes, TEs={echo_times} ms)")
    else:
        print(f"  MODE: SINGLE-ECHO")
    print("="*80)
    print(f"Subject: {subject}")
    print(f"Session: {session}")
    if is_multiecho:
        for i, ef in enumerate(echo_files):
            print(f"  Echo {i+1}: {ef} (TE={echo_times[i]} ms)")
    else:
        print(f"Input BOLD: {bold_file}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Setup directories
    derivatives_dir = output_dir / 'derivatives' / subject / session / 'func'
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    qc_dir = get_subject_qc_dir(output_dir, subject, session, 'func')

    if work_dir is None:
        work_dir = output_dir / 'work' / subject / session / 'func_preproc'
    work_dir.mkdir(parents=True, exist_ok=True)

    # Get config parameters
    func_config = config.get('functional', {})
    tr = func_config.get('tr', 1.0)  # Will be overridden by JSON sidecar

    # Skull stripping parameters
    bet_config = func_config.get('bet', {})
    bet_method = bet_config.get('method', 'bet')  # 'bet' or 'bet4animal'
    bet_frac = bet_config.get('frac', 0.3)

    # bet4animal-specific parameters (for rodent brains)
    bet4animal_params = {}
    if bet_method == 'bet4animal':
        # Get age-specific brain center if available
        centers = bet_config.get('centers', {})
        if session in centers:
            center = centers[session]
        else:
            center = bet_config.get('center', [40, 25, 5])

        bet4animal_params = {
            'frac': bet_config.get('frac', 0.7),  # Higher for bet4animal
            'center': tuple(center),
            'radius': bet_config.get('radius', 125),
            'scale': tuple(bet_config.get('scale', [1, 1, 1.5])),
            'width': bet_config.get('width', 2.5)
        }

    motion_method = func_config.get('motion_correction', {}).get('method', 'mcflirt')
    motion_ref = func_config.get('motion_correction', {}).get('reference', 'middle')
    smoothing_fwhm = func_config.get('smoothing', {}).get('fwhm', 0.5)
    highpass_freq = func_config.get('filtering', {}).get('highpass', 0.01)
    lowpass_freq = func_config.get('filtering', {}).get('lowpass', None)

    # TEDANA config (for multi-echo)
    tedana_config = func_config.get('tedana', {})

    # Try to read TR from JSON sidecar
    # BIDS convention: RepetitionTime = effective volume TR
    # For multi-shot EPI: effective TR = n_shots * per-shot TR
    json_file = primary_bold.with_suffix('').with_suffix('.json')
    if json_file.exists():
        with open(json_file, 'r') as f:
            sidecar_meta = json.load(f)
            tr = sidecar_meta.get('RepetitionTime', tr)
            n_shots = sidecar_meta.get('NumberOfShots', 1)
            tr_per_shot = sidecar_meta.get('RepetitionTimePerShot', None)
            # Sanity check: if per-shot TR is available, verify effective TR
            if tr_per_shot is not None and n_shots > 1:
                expected_tr = n_shots * tr_per_shot
                if abs(tr - expected_tr) > 0.01:
                    print(f"WARNING: RepetitionTime ({tr}s) != NumberOfShots ({n_shots}) "
                          f"× RepetitionTimePerShot ({tr_per_shot}s) = {expected_tr}s")
                    print(f"  Using computed effective TR: {expected_tr}s")
                    tr = expected_tr
            print(f"Read TR from JSON sidecar: {tr}s"
                  f"{f' ({n_shots}-shot × {tr_per_shot}s)' if tr_per_shot and n_shots > 1 else ''}")

    results = {}

    # =========================================================================
    # STEP 1: Image Validation
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 1: Image Validation")
    print("="*60)

    if is_multiecho:
        for i, ef in enumerate(echo_files):
            print(f"\n  Validating echo {i+1} ({ef.name})...")
            vr = validate_image(ef)
            print_validation_results(vr)
            if not vr['valid']:
                raise ValueError(f"Echo {i+1} failed validation: {vr['errors']}")
    else:
        validation_results = validate_image(bold_file)
        print_validation_results(validation_results)
        if not validation_results['valid']:
            raise ValueError(f"Input BOLD image failed validation: {validation_results['errors']}")

    # =========================================================================
    # STEP 2: Discard Initial Volumes
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 2: Discard Initial Volumes")
    print("="*60)

    if is_multiecho:
        if n_discard > 0:
            trimmed_echoes = []
            for i, ef in enumerate(echo_files):
                trimmed = work_dir / f"{subject}_{session}_echo-{i+1}_bold_trimmed.nii.gz"
                trimmed_path, n_discarded = discard_initial_volumes(ef, trimmed, n_discard)
                trimmed_echoes.append(trimmed_path)
            echo_files_for_processing = trimmed_echoes
            results['n_volumes_discarded'] = n_discarded
            remaining = nib.load(echo_files_for_processing[0]).shape[3]
            if remaining < 2:
                raise ValueError(
                    f"Only {remaining} volume(s) remain after discarding {n_discarded}"
                )
        else:
            print("No volumes to discard")
            echo_files_for_processing = echo_files
            results['n_volumes_discarded'] = 0
    else:
        if n_discard > 0:
            trimmed_bold = work_dir / f"{subject}_{session}_bold_trimmed.nii.gz"
            bold_for_processing, n_discarded = discard_initial_volumes(
                bold_file, trimmed_bold, n_discard
            )
            results['n_volumes_discarded'] = n_discarded

            # Verify enough volumes remain after trimming
            remaining = nib.load(bold_for_processing).shape[3]
            if remaining < 2:
                raise ValueError(
                    f"Only {remaining} volume(s) remain after discarding {n_discarded} — "
                    f"need at least 2 for timeseries processing"
                )
        else:
            print("No volumes to discard")
            bold_for_processing = bold_file
            results['n_volumes_discarded'] = 0

    # =========================================================================
    # STEPS 2.5-5: Pathway fork — Multi-Echo vs Single-Echo
    # =========================================================================

    brain_mask = derivatives_dir / f"{subject}_{session}_desc-brain_mask.nii.gz"
    skull_strip_info = {}
    ref_norm = None  # Used for QC in SE path

    if is_multiecho:
        # =================================================================
        # MULTI-ECHO PATH: Steps 3-5
        # =================================================================
        print("\n" + "="*60)
        print("STEPS 3-4: Multi-Echo Motion Correction + Brain Extraction")
        print("="*60)

        skull_strip_params = config.get('functional', {}).get('skull_strip_adaptive', {})
        me_work_dir = work_dir / 'multiecho'

        me_results = run_multiecho_motion_correction(
            echo_files=echo_files_for_processing,
            work_dir=me_work_dir,
            brain_mask_out=brain_mask,
            skull_strip_config=skull_strip_params,
        )

        me_corrected_echoes = me_results['corrected_echoes']
        motion_params = me_results['motion_params']
        brain_ref = me_results['brain_ref']
        skull_strip_info = me_results['skull_strip_info']

        # Apply brain mask to all corrected echoes
        print("\nApplying brain mask to motion-corrected echoes...")
        me_masked_echoes = []
        for i, echo_mcf in enumerate(me_corrected_echoes):
            masked_out = me_work_dir / f'echo{i}_mcf_masked.nii.gz'
            apply_mask_to_timeseries(echo_mcf, brain_mask, masked_out)
            me_masked_echoes.append(masked_out)

        # Step 5: TEDANA multi-echo ICA denoising
        print("\n" + "="*60)
        print("STEP 5: TEDANA Multi-Echo ICA Denoising")
        print("="*60)

        tedana_dir = work_dir / 'tedana'
        tedana_results = run_tedana(
            echo_files=me_masked_echoes,
            echo_times_ms=echo_times,
            output_dir=tedana_dir,
            mask_file=brain_mask,
            tedpca=tedana_config.get('tedpca', 0.95),
            tree=tedana_config.get('tree', 'kundu'),
        )

        results['tedana'] = tedana_results
        results['multiecho'] = {
            'n_echoes': n_echoes,
            'echo_times_ms': echo_times,
            'corrected_echoes': me_corrected_echoes,
        }

        # Use TEDANA denoised output for downstream steps
        if 'denoised' in tedana_results:
            bold_for_smoothing = tedana_results['denoised']
            print(f"\n  Using TEDANA denoised output for smoothing")
        elif 'optcom' in tedana_results:
            bold_for_smoothing = tedana_results['optcom']
            print(f"\n  TEDANA denoised not found, using optimally combined")
        else:
            raise RuntimeError("TEDANA produced no output files")

        # For motion-corrected reference (used in registration later)
        bold_mcf = bold_for_smoothing

        # ICA/classification not used in ME path
        ica_enabled = False
        classification = None

    else:
        # =================================================================
        # SINGLE-ECHO PATH: Steps 2.5-5 (unchanged)
        # =================================================================

        # Step 2.5: Slice Timing Correction
        slice_timing_config = func_config.get('slice_timing', {})
        stc_enabled = slice_timing_config.get('enabled', False)

        if stc_enabled:
            print("\n" + "="*60)
            print("STEP 2.5: Slice Timing Correction")
            print("="*60)
            print("  IMPORTANT: Correcting for slice acquisition time differences")
            print("  This is done BEFORE motion correction to avoid interpolation artifacts")

            json_file_stc = primary_bold.with_suffix('.json')
            slice_order = slice_timing_config.get('order', 'interleaved')
            custom_order = slice_timing_config.get('custom_order', None)

            if json_file_stc.exists():
                print(f"  Checking for slice timing in: {json_file_stc.name}")
                detected_order, detected_custom = detect_slice_order(
                    json_file=json_file_stc,
                    method=slice_order
                )
                slice_order = detected_order
                if detected_custom is not None:
                    custom_order = detected_custom
                    print(f"  Detected custom slice order: {custom_order}")

            bold_stc = work_dir / f"{subject}_{session}_bold_stc.nii.gz"
            run_slice_timing_correction(
                input_file=bold_for_processing,
                output_file=bold_stc,
                tr=tr,
                slice_order=slice_order,
                custom_order=custom_order,
                reference_slice=slice_timing_config.get('reference_slice', 'middle')
            )

            bold_for_processing = bold_stc
            results['slice_timing_corrected'] = True
            results['slice_order'] = slice_order if custom_order is None else 'custom'

        else:
            print("\n" + "="*60)
            print("STEP 2.5: Slice Timing Correction (Skipped)")
            print("="*60)
            print("  Slice timing correction disabled in config")
            results['slice_timing_corrected'] = False

        # Step 3: Brain Extraction
        print("\n" + "="*60)
        print("STEP 3: Brain Extraction")
        print("="*60)

        print("\nExtracting reference volume for skull stripping...")
        img = nib.load(bold_for_processing)
        data = img.get_fdata()
        mid_vol = data.shape[3] // 2
        ref_data = data[..., mid_vol]
        ref_volume = work_dir / f"{subject}_{session}_bold_ref.nii.gz"
        nib.save(nib.Nifti1Image(ref_data, img.affine, img.header), ref_volume)
        print(f"  Extracted volume {mid_vol} of {data.shape[3]}")

        print("\nRunning optimized skull stripping...")
        print("  Strategy: N4 bias correction + intensity normalization + adaptive per-slice BET with -R flag")

        brain_ref = work_dir / f"{subject}_{session}_bold_brain_ref.nii.gz"

        skull_strip_params = config.get('functional', {}).get('skull_strip_adaptive', {})
        target_ratio = skull_strip_params.get('target_ratio', 0.15)
        frac_range = tuple(skull_strip_params.get('frac_range', [0.30, 0.90]))
        frac_step = skull_strip_params.get('frac_step', 0.05)
        use_R_flag = skull_strip_params.get('use_R_flag', True)
        invert_intensity = skull_strip_params.get('invert_intensity', False)

        skull_strip_work_dir = work_dir / 'skull_strip'
        skull_strip_work_dir.mkdir(exist_ok=True)

        print("  Running N4 bias correction...")
        ref_n4 = skull_strip_work_dir / f"{subject}_{session}_bold_ref_n4.nii.gz"
        subprocess.run([
            'N4BiasFieldCorrection',
            '-i', str(ref_volume),
            '-o', str(ref_n4)
        ], check=True, capture_output=True)

        print("  Running intensity normalization...")
        img_n4 = nib.load(ref_n4)
        data_n4 = img_n4.get_fdata()
        p2, p98 = np.percentile(data_n4[data_n4 > 0], [2, 98])
        data_norm = np.clip(data_n4, p2, p98)
        data_norm = (data_norm - p2) / (p98 - p2) * 1000
        ref_norm = skull_strip_work_dir / f"{subject}_{session}_bold_ref_norm.nii.gz"
        nib.save(nib.Nifti1Image(data_norm, img_n4.affine, img_n4.header), ref_norm)

        print("  Running skull stripping (auto-selects method based on slice count)...")
        brain_ref, brain_mask, skull_strip_info = skull_strip(
            input_file=ref_norm,
            output_file=brain_ref,
            mask_file=brain_mask,
            work_dir=skull_strip_work_dir,
            method='auto',
            target_ratio=target_ratio,
            frac_range=frac_range,
            frac_step=frac_step,
            invert_intensity=invert_intensity,
        )

        print(f"\n  Method: {skull_strip_info.get('method', 'unknown')}")
        print(f"  Mask created: {skull_strip_info.get('total_voxels', 0):,} voxels")
        print(f"  Extraction ratio: {skull_strip_info.get('extraction_ratio', 0):.3f}")
        if 'mean_frac' in skull_strip_info:
            print(f"  Mean frac: {skull_strip_info['mean_frac']:.3f} ± {skull_strip_info.get('std_frac', 0):.3f}")

        print("\nApplying mask to 4D timeseries...")
        bold_masked = work_dir / f"{subject}_{session}_bold_brain.nii.gz"
        apply_mask_to_timeseries(bold_for_processing, brain_mask, bold_masked)
        print(f"  Masked timeseries: {bold_masked}")

        # Step 4: Motion Correction
        print("\n" + "="*60)
        print("STEP 4: Motion Correction")
        print("="*60)

        motion_dir = work_dir / 'motion_correction'
        motion_results = run_motion_correction(
            bold_masked,
            motion_dir,
            reference=motion_ref,
            method=motion_method
        )

        bold_mcf = motion_results['motion_corrected']
        motion_params = motion_results['motion_params']

        # Step 5: ICA Denoising
        ica_config = func_config.get('denoising', {}).get('ica', {})
        ica_enabled = ica_config.get('enabled', False)

        if ica_enabled:
            print("\n" + "="*60)
            print("STEP 5: ICA Denoising")
            print("="*60)

            melodic_dir = work_dir / 'melodic'
            melodic_outputs = run_melodic_ica(
                input_file=bold_mcf,
                output_dir=melodic_dir,
                brain_mask=brain_mask,
                tr=tr,
                n_components=ica_config.get('n_components', 30)
            )

            csf_mask = None
            csf_mask_path = output_dir / 'derivatives' / subject / session / 'anat' / f'{subject}_{session}_label-CSF_probseg.nii.gz'
            if csf_mask_path.exists():
                from nibabel.processing import resample_from_to
                bold_ref_img = nib.load(brain_mask)
                csf_img = nib.load(csf_mask_path)
                if csf_img.shape[:3] != bold_ref_img.shape[:3]:
                    print(f"  Resampling CSF mask from {csf_img.shape[:3]} to BOLD space {bold_ref_img.shape[:3]}")
                    csf_resampled = resample_from_to(csf_img, bold_ref_img, order=1)
                    csf_mask_bold = work_dir / f'{subject}_{session}_space-bold_label-CSF_probseg.nii.gz'
                    nib.save(csf_resampled, csf_mask_bold)
                    csf_mask = csf_mask_bold
                else:
                    csf_mask = csf_mask_path

            classification = classify_ica_components(
                melodic_dir=melodic_dir,
                motion_params_file=motion_params,
                brain_mask_file=brain_mask,
                tr=tr,
                csf_mask_file=csf_mask,
                motion_threshold=ica_config.get('motion_threshold', 0.40),
                edge_threshold=ica_config.get('edge_threshold', 0.80),
                csf_threshold=ica_config.get('csf_threshold', 0.70),
                freq_threshold=ica_config.get('freq_threshold', 0.60),
                classification_mode=ica_config.get('classification_mode', 'score')
            )

            bold_denoised = work_dir / f"{subject}_{session}_desc-ica_denoised_bold.nii.gz"
            remove_noise_components(
                input_file=bold_mcf,
                output_file=bold_denoised,
                melodic_dir=melodic_dir,
                noise_components=classification['summary']['noise_components']
            )

            ica_qc_report = generate_ica_denoising_qc(
                subject=subject,
                session=session,
                classification_results=classification,
                melodic_dir=melodic_dir,
                output_dir=qc_dir
            )

            results['ica_denoising'] = {
                'denoised_bold': bold_denoised,
                'melodic_dir': melodic_dir,
                'classification': classification,
                'qc_report': ica_qc_report
            }

            bold_for_smoothing = bold_denoised

        else:
            print("\n" + "="*60)
            print("STEP 5: ICA Denoising (Skipped)")
            print("="*60)
            print("ICA denoising disabled in config")
            classification = None

            bold_for_smoothing = bold_mcf

    # =========================================================================
    # STEP 6: Spatial Smoothing (AFTER ICA - matches old pipeline)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 6: Spatial Smoothing")
    print("="*60)

    bold_smooth = work_dir / f"{subject}_{session}_bold_smooth.nii.gz"
    smooth_image(bold_for_smoothing, bold_smooth, smoothing_fwhm)

    # =========================================================================
    # STEP 7: Extract Motion Confounds
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 7: Extract Motion Confounds")
    print("="*60)

    confounds_file = derivatives_dir / f"{subject}_{session}_desc-confounds_timeseries.tsv"
    extract_confounds(
        motion_params,
        confounds_file,
        derivatives=True,
        squares=True
    )

    # =========================================================================
    # STEP 8: aCompCor Extraction (Optional)
    # =========================================================================
    acompcor_config = func_config.get('denoising', {}).get('acompcor', {})
    acompcor_enabled = acompcor_config.get('enabled', False)
    acompcor_file = None

    if acompcor_enabled:
        print("\n" + "="*60)
        print("STEP 8: aCompCor Extraction")
        print("="*60)

        # Find CSF and WM masks from anatomical preprocessing
        # These are in T2w space — resample to BOLD space if needed
        from nibabel.processing import resample_from_to as _resample_from_to
        anat_deriv_dir = output_dir / 'derivatives' / subject / session / 'anat'
        csf_mask = anat_deriv_dir / f'{subject}_{session}_label-CSF_probseg.nii.gz'
        wm_mask = anat_deriv_dir / f'{subject}_{session}_label-WM_probseg.nii.gz'

        if not csf_mask.exists() or not wm_mask.exists():
            print(f"  Warning: Tissue masks not found:")
            print(f"    CSF: {csf_mask.exists()}")
            print(f"    WM: {wm_mask.exists()}")
            print(f"  Skipping aCompCor extraction")
            print(f"  (Run anatomical preprocessing first to generate tissue masks)")
            acompcor_enabled = False
        else:
            bold_ref_img = nib.load(brain_mask)
            for label, mask_path in [('CSF', csf_mask), ('WM', wm_mask)]:
                mask_img = nib.load(mask_path)
                if mask_img.shape[:3] != bold_ref_img.shape[:3]:
                    print(f"  Resampling {label} mask from {mask_img.shape[:3]} to BOLD space {bold_ref_img.shape[:3]}")
                    resampled = _resample_from_to(mask_img, bold_ref_img, order=1)
                    out_path = work_dir / f'{subject}_{session}_space-bold_label-{label}_probseg.nii.gz'
                    nib.save(resampled, out_path)
                    if label == 'CSF':
                        csf_mask = out_path
                    else:
                        wm_mask = out_path

    if acompcor_enabled:
        acompcor_file = derivatives_dir / f"{subject}_{session}_desc-acompcor_timeseries.tsv"

        acompcor_results = extract_acompcor_components(
            bold_file=bold_smooth,
            csf_mask=csf_mask,
            wm_mask=wm_mask,
            n_components=acompcor_config.get('num_components', 5),
            variance_threshold=acompcor_config.get('variance_threshold', 0.5),
            output_file=acompcor_file
        )

        # Generate aCompCor QC
        acompcor_qc_report = generate_acompcor_qc(
            subject=subject,
            session=session,
            acompcor_results=acompcor_results,
            output_dir=qc_dir
        )

        results['acompcor'] = {
            'acompcor_file': acompcor_file,
            'qc_report': acompcor_qc_report,
            'n_components': acompcor_results['n_components_csf'] + acompcor_results['n_components_wm'],
            'n_voxels_csf': acompcor_results['n_voxels_csf'],
            'n_voxels_wm': acompcor_results['n_voxels_wm']
        }

        print(f"  ✓ aCompCor components extracted: {acompcor_file}")
    else:
        if not acompcor_config.get('enabled', False):
            print("\n" + "="*60)
            print("STEP 8: aCompCor (Skipped)")
            print("="*60)
            print("  aCompCor disabled in config")

    # =========================================================================
    # STEP 9: Nuisance Regression (motion + aCompCor)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 9: Nuisance Regression")
    print("="*60)

    # Build design matrix: 24 motion params + aCompCor (if available)
    confounds_matrix = np.loadtxt(confounds_file, skiprows=1)  # 24 motion regressors
    regressor_labels = ['24 motion']

    if acompcor_enabled and acompcor_file is not None:
        acompcor_matrix = np.loadtxt(acompcor_file, skiprows=1)
        confounds_matrix = np.column_stack([confounds_matrix, acompcor_matrix])
        n_acompcor = acompcor_matrix.shape[1]
        regressor_labels.append(f'{n_acompcor} aCompCor')

    print(f"  Regressors: {' + '.join(regressor_labels)} = {confounds_matrix.shape[1]} total")

    bold_img = nib.load(bold_smooth)
    bold_data = bold_img.get_fdata()
    mask_data = nib.load(brain_mask).get_fdata().astype(bool)

    n_timepoints = bold_data.shape[3]
    design = np.column_stack([confounds_matrix[:n_timepoints], np.ones(n_timepoints)])

    # Voxelwise OLS regression within mask, keep residuals
    clean_data = bold_data.copy()
    voxels = np.where(mask_data)
    timeseries = bold_data[voxels[0], voxels[1], voxels[2], :]  # (n_voxels, n_timepoints)

    pre_var = np.var(timeseries, axis=1).mean()
    betas = np.linalg.lstsq(design, timeseries.T, rcond=None)[0]  # (n_regressors, n_voxels)
    residuals = timeseries.T - design @ betas  # (n_timepoints, n_voxels)
    residuals = residuals + betas[-1, :]  # add intercept back
    post_var = np.var(residuals.T, axis=1).mean()
    clean_data[voxels[0], voxels[1], voxels[2], :] = residuals.T

    bold_regressed = work_dir / f"{subject}_{session}_bold_regressed.nii.gz"
    nib.save(nib.Nifti1Image(clean_data.astype(np.float32), bold_img.affine, bold_img.header), bold_regressed)
    print(f"  Regressed {confounds_matrix.shape[1]} nuisance regressors from {n_timepoints} volumes")
    print(f"  Variance reduction: {100*(1 - post_var/pre_var):.1f}%")

    # Save regressed (unfiltered) data to derivatives for fALFF analysis
    regressed_deriv = derivatives_dir / f"{subject}_{session}_desc-regressed_bold.nii.gz"
    shutil.copy(str(bold_regressed), str(regressed_deriv))
    print(f"  Saved unfiltered regressed data: {regressed_deriv.name}")

    # =========================================================================
    # STEP 10: Temporal Filtering (AFTER nuisance regression)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 10: Temporal Filtering")
    print("="*60)

    bold_filtered = work_dir / f"{subject}_{session}_bold_filtered.nii.gz"
    temporal_filter(
        bold_regressed,
        bold_filtered,
        tr,
        highpass=highpass_freq,
        lowpass=lowpass_freq
    )

    # Save filtered data as final output
    final_bold = derivatives_dir / f"{subject}_{session}_desc-preproc_bold.nii.gz"
    shutil.copy(str(bold_filtered), str(final_bold))

    # =========================================================================
    # STEP 11: Save Output Files and Metadata
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 11: Save Output Files")
    print("="*60)

    # Copy/save final outputs
    results.update({
        'bold_preproc': final_bold,
        'brain_mask': brain_mask,
        'mean_bold': brain_ref,  # Using brain-extracted reference volume
        'confounds': confounds_file,
        'motion_params': motion_params
    })

    # Save processing metadata
    if is_multiecho:
        input_files_str = [str(f) for f in (echo_files or bold_file)]
    else:
        input_files_str = str(bold_file)

    ica_config_local = func_config.get('denoising', {}).get('ica', {}) if not is_multiecho else {}

    metadata = {
        'subject': subject,
        'session': session,
        'input_file': input_files_str,
        'multi_echo': is_multiecho,
        'n_echoes': n_echoes,
        'echo_times_ms': echo_times if is_multiecho else None,
        'tr': tr,
        'n_volumes_discarded': results['n_volumes_discarded'],
        'motion_correction': {
            'method': 'mcflirt_multiecho' if is_multiecho else motion_method,
            'reference': motion_ref
        },
        'brain_extraction': {
            'method': skull_strip_info.get('method', 'auto'),
        },
        'smoothing': {
            'fwhm_mm': smoothing_fwhm
        },
        'temporal_filtering': {
            'highpass_hz': highpass_freq,
            'lowpass_hz': lowpass_freq
        },
        'denoising': {
            'method': 'tedana' if is_multiecho else ('ica' if ica_enabled else 'none'),
            'tedana': {
                'tedpca': str(tedana_config.get('tedpca', 'aic')),
                'tree': tedana_config.get('tree', 'kundu'),
            } if is_multiecho else None,
            'ica': {
                'enabled': ica_enabled,
                'n_components': ica_config_local.get('n_components', 30) if ica_enabled else None,
                'classification_mode': ica_config_local.get('classification_mode', 'score') if ica_enabled else None,
                'n_signal_components': classification['summary']['n_signal'] if (ica_enabled and classification) else None,
                'n_noise_components': classification['summary']['n_noise'] if (ica_enabled and classification) else None
            } if not is_multiecho else None,
        },
        'acompcor': {
            'enabled': acompcor_enabled,
            'n_components': results.get('acompcor', {}).get('n_components') if acompcor_enabled else None,
            'n_voxels_csf': results.get('acompcor', {}).get('n_voxels_csf') if acompcor_enabled else None,
            'n_voxels_wm': results.get('acompcor', {}).get('n_voxels_wm') if acompcor_enabled else None
        },
        'slice_timing': {
            'corrected': results.get('slice_timing_corrected', False),
            'slice_order': results.get('slice_order', None) if results.get('slice_timing_corrected', False) else None
        }
    }

    metadata_file = derivatives_dir / f"{subject}_{session}_desc-preproc_bold.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    results['metadata'] = metadata_file

    # =========================================================================
    # STEP 12: Generate QC Reports
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 12: Generate QC Reports")
    print("="*60)

    # Motion QC
    motion_qc_threshold = func_config.get('motion_qc', {}).get('fd_threshold', 0.05)
    # For ME path, ref_norm may be None — use brain_ref as fallback for original_file
    qc_original_file = ref_norm if ref_norm is not None else brain_ref
    motion_qc_report = generate_motion_qc_report(
        subject=subject,
        session=session,
        motion_params_file=motion_params,
        bold_file=final_bold,
        mask_file=brain_mask,
        output_dir=qc_dir,
        threshold_fd=motion_qc_threshold,
        original_file=qc_original_file,
        brain_file=brain_ref,
        skull_strip_mask_file=brain_mask,
        skull_strip_info=skull_strip_info,
    )

    # Confounds QC
    confounds_qc_report = generate_confounds_qc_report(
        subject=subject,
        session=session,
        confounds_file=confounds_file,
        output_dir=qc_dir
    )

    results['qc_reports'] = {
        'motion': motion_qc_report,
        'confounds': confounds_qc_report
    }

    # =========================================================================
    # STEP 13: BOLD to Template Registration
    # =========================================================================
    registration_results = None

    if run_registration:
        # Prefer template_file; fall back to t2w_file (deprecated)
        reg_target = template_file or t2w_file
        use_template = template_file is not None

        if reg_target is None:
            print("\n  Registration requested but no template/T2w file provided - skipping")
        elif not reg_target.exists():
            print(f"\n  Registration target not found: {reg_target} - skipping registration")
        else:
            # Compute temporal mean of motion-corrected BOLD + brain mask
            reg_work_dir = work_dir / 'bold_registration'
            reg_work_dir.mkdir(parents=True, exist_ok=True)
            mean_mcf_brain = reg_work_dir / f'{subject}_{session}_mean_mcf_brain.nii.gz'

            if not mean_mcf_brain.exists():
                print("  Computing temporal mean of motion-corrected BOLD...")
                mcf_img = nib.load(bold_mcf)
                mcf_data = mcf_img.get_fdata()
                mean_data = np.mean(mcf_data, axis=3)
                mask_data = nib.load(brain_mask).get_fdata() > 0
                mean_data = mean_data * mask_data
                nib.save(
                    nib.Nifti1Image(mean_data.astype(np.float32), mcf_img.affine, mcf_img.header),
                    mean_mcf_brain
                )

            if use_template:
                print("\n" + "="*60)
                print("STEP 12: BOLD to Template Registration")
                print("="*60)

                try:
                    registration_results = register_bold_to_template(
                        bold_ref_file=mean_mcf_brain,
                        template_file=template_file,
                        output_dir=output_dir,
                        subject=subject,
                        session=session,
                        work_dir=reg_work_dir,
                        n_cores=config.get('execution', {}).get('n_procs', 4)
                    )

                    reg_metadata_file = derivatives_dir / f'{subject}_{session}_BOLD_to_template_registration.json'
                    reg_metadata = {
                        'bold_ref_file': str(mean_mcf_brain),
                        'mcf_source': str(bold_mcf),
                        'brain_mask': str(brain_mask),
                        'template_file': str(registration_results['template_file']),
                        'affine_transform': str(registration_results['affine_transform']),
                        'warped_bold': str(registration_results['warped_bold']) if registration_results.get('warped_bold') else None,
                        'bold_shape': list(registration_results['bold_shape']),
                        'template_shape': list(registration_results['template_shape']),
                    }
                    with open(reg_metadata_file, 'w') as f:
                        json.dump(reg_metadata, f, indent=2)

                    results['registration'] = {
                        'affine_transform': registration_results['affine_transform'],
                        'warped_bold': registration_results.get('warped_bold'),
                        'metadata': reg_metadata_file,
                    }

                    print(f"\n  Registration complete:")
                    print(f"  - Transform: {registration_results['affine_transform']}")
                    print(f"  - Metadata: {reg_metadata_file}")

                except Exception as e:
                    print(f"\n  Registration failed: {e}")
                    print("  Continuing without registration...")
            else:
                # Legacy path: BOLD → T2w (deprecated)
                print("\n" + "="*60)
                print("STEP 12: BOLD to T2w Registration (deprecated)")
                print("="*60)

                try:
                    registration_results = register_bold_to_t2w(
                        bold_ref_file=mean_mcf_brain,
                        t2w_file=t2w_file,
                        output_dir=output_dir,
                        subject=subject,
                        session=session,
                        work_dir=reg_work_dir,
                        n_cores=config.get('execution', {}).get('n_procs', 4)
                    )

                    reg_metadata_file = derivatives_dir / f'{subject}_{session}_BOLD_to_T2w_registration.json'
                    reg_metadata = {
                        'bold_ref_file': str(mean_mcf_brain),
                        'mcf_source': str(bold_mcf),
                        'brain_mask': str(brain_mask),
                        't2w_file': str(registration_results['t2w_file']),
                        'affine_transform': str(registration_results['affine_transform']),
                        'warped_bold': str(registration_results['warped_bold']) if registration_results.get('warped_bold') else None,
                        'bold_shape': list(registration_results['bold_shape']),
                        't2w_shape': list(registration_results['t2w_shape']),
                    }
                    with open(reg_metadata_file, 'w') as f:
                        json.dump(reg_metadata, f, indent=2)

                    results['registration'] = {
                        'affine_transform': registration_results['affine_transform'],
                        'warped_bold': registration_results.get('warped_bold'),
                        'metadata': reg_metadata_file,
                    }

                    print(f"\n  Registration complete:")
                    print(f"  - Transform: {registration_results['affine_transform']}")
                    print(f"  - Metadata: {reg_metadata_file}")

                except Exception as e:
                    print(f"\n  Registration failed: {e}")
                    print("  Continuing without registration...")

    print("\n" + "="*80)
    print("FUNCTIONAL PREPROCESSING COMPLETE")
    if is_multiecho:
        print(f"  Mode: MULTI-ECHO ({n_echoes} echoes, TEDANA denoising)")
    else:
        print(f"  Mode: SINGLE-ECHO")
    print("="*80)
    print(f"Preprocessed BOLD: {final_bold}")
    print(f"Brain mask: {brain_mask}")
    print(f"Confounds: {confounds_file}")
    print(f"Metadata: {metadata_file}")

    if 'registration' in results:
        print(f"\nRegistration:")
        print(f"  Transform: {results['registration']['affine_transform']}")

    print(f"\nQC Reports:")
    print(f"  Motion QC: {motion_qc_report}")
    print(f"  Confounds QC: {confounds_qc_report}")

    if is_multiecho and 'tedana' in results:
        print(f"\nTEDANA Multi-Echo Denoising:")
        if 'denoised' in results['tedana']:
            print(f"  Denoised: {results['tedana']['denoised']}")
        if 'optcom' in results['tedana']:
            print(f"  Optimally combined: {results['tedana']['optcom']}")
        if 'report' in results['tedana']:
            print(f"  Report: {results['tedana']['report']}")
    elif ica_enabled and 'ica_denoising' in results:
        print(f"  ICA Denoising QC: {results['ica_denoising']['qc_report']}")
        print(f"\nICA Denoising:")
        print(f"  Signal components: {classification['summary']['n_signal']}")
        print(f"  Noise components: {classification['summary']['n_noise']}")

    if acompcor_enabled and 'acompcor' in results:
        print(f"  aCompCor QC: {results['acompcor']['qc_report']}")
        print(f"\naCompCor:")
        print(f"  Total components: {results['acompcor']['n_components']}")
        print(f"  CSF voxels: {results['acompcor']['n_voxels_csf']}")
        print(f"  WM voxels: {results['acompcor']['n_voxels_wm']}")
    print("="*80)

    return results
