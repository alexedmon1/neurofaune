"""
MSME (Multi-Slice Multi-Echo) T2 mapping and MWF calculation workflow.

This module provides preprocessing for multi-echo T2-weighted data,
including T2 fitting and Myelin Water Fraction (MWF) calculation using NNLS.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import subprocess
from scipy.optimize import curve_fit, nnls
import json

from neurofaune.preprocess.utils.validation import validate_image, print_validation_results
from neurofaune.utils.transforms import TransformRegistry
from neurofaune.config import get_config_value
from neurofaune.preprocess.qc import get_subject_qc_dir
from neurofaune.preprocess.workflows.func_preprocess import _find_z_offset_ncc
from neurofaune.preprocess.utils.skull_strip import skull_strip


def register_msme_to_template(
    msme_ref_file: Path,
    template_file: Path,
    output_dir: Path,
    subject: str,
    session: str,
    work_dir: Path,
    n_cores: int = 4,
    z_range: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Register MSME first echo directly to the cohort template.

    Uses NCC-based Z initialization followed by rigid-only registration.
    MSME has only 5 slices covering a small portion of the 41-slice template,
    requiring careful Z positioning. This produces better SIGMA atlas overlap
    than the old MSME→T2w→Template chain.

    Parameters
    ----------
    msme_ref_file : Path
        First echo volume (3D, skull-stripped or raw)
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
    z_range : tuple of (int, int), optional
        (min_slice, max_slice) to constrain the NCC Z search. Defaults to
        (14, 28) for MSME hippocampal slab positioning in template space.
        Set to None to search all positions (original behavior).

    Returns
    -------
    dict
        Dictionary with transform paths and metadata
    """
    print("\n" + "="*60)
    print("MSME to Template Registration")
    print("="*60)

    msme_img = nib.load(msme_ref_file)
    template_img = nib.load(template_file)
    print(f"\n  MSME ref: {msme_img.shape} voxels, {msme_img.header.get_zooms()[:3]} mm")
    print(f"  Template: {template_img.shape} voxels, {template_img.header.get_zooms()[:3]} mm")

    transforms_dir = output_dir / 'transforms' / subject / session
    transforms_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = transforms_dir / 'MSME_to_template_'

    # Step 1: Find optimal Z offset via NCC scan
    # Default z_range constrains to hippocampal region (slices 14-28)
    # to avoid spurious NCC peaks at extreme positions
    if z_range is None:
        z_range = (14, 28)
    print(f"\n  Finding optimal Z offset via NCC scan (z_range={z_range})...")
    initial_transform, z_offset_info = _find_z_offset_ncc(
        msme_img, template_img, work_dir, z_range=z_range
    )

    # Step 2: Rigid registration with conservative shrink factors
    # (5 slices is very partial — use 2x1x1 to avoid losing Z info)
    print("\nRunning ANTs Rigid registration (MSME → Template)...")

    warped_output = Path(str(output_prefix) + 'Warped.nii.gz')
    cmd = [
        'antsRegistration',
        '--dimensionality', '3',
        '--output', f'[{output_prefix},{warped_output}]',
        '--interpolation', 'Linear',
        '--use-histogram-matching', '1',
        '--winsorize-image-intensities', '[0.005,0.995]',
        '--initial-moving-transform', str(initial_transform),
        # Rigid only (6 DOF) — 5 slices is too few for affine
        '--transform', 'Rigid[0.1]',
        '--metric', f'MI[{template_file},{msme_ref_file},1,32,Regular,0.25]',
        '--convergence', '[500x250x100,1e-6,10]',
        '--shrink-factors', '2x1x1',
        '--smoothing-sigmas', '1x0x0vox',
    ]

    print(f"  Moving: {msme_ref_file.name}")
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
        raise RuntimeError("MSME to Template registration failed")

    # Check outputs
    rigid_transform = Path(str(output_prefix) + '0GenericAffine.mat')
    warped_msme = Path(str(output_prefix) + 'Warped.nii.gz')

    if not rigid_transform.exists():
        raise RuntimeError(f"Expected transform not found: {rigid_transform}")

    print(f"  Rigid transform: {rigid_transform.name}")
    if warped_msme.exists():
        print(f"  Warped MSME ref: {warped_msme.name}")
        # Report coverage
        warped_data = nib.load(warped_msme).get_fdata()
        threshold = warped_data.max() * 0.05
        slices_with_signal = [z for z in range(warped_data.shape[2])
                              if warped_data[:,:,z].max() > threshold]
        if slices_with_signal:
            print(f"  MSME covers template slices {slices_with_signal[0]}-{slices_with_signal[-1]} "
                  f"({len(slices_with_signal)} slices)")

    return {
        'affine_transform': rigid_transform,
        'warped_msme': warped_msme if warped_msme.exists() else None,
        'template_file': template_file,
        'msme_shape': msme_img.shape,
        'template_shape': template_img.shape,
    }


def register_msme_to_t2w(
    msme_ref_file: Path,
    t2w_file: Path,
    output_dir: Path,
    subject: str,
    session: str,
    work_dir: Path,
    n_cores: int = 4
) -> Dict[str, Any]:
    """
    Register MSME first echo to T2w within the same subject.

    .. deprecated::
        Use :func:`register_msme_to_template` instead for better atlas overlap.

    Uses NCC-based Z initialization (both images have origin at 0,0,0)
    followed by rigid-only registration. MSME has only 5 slices covering
    a small portion of the 41-slice T2w, requiring careful Z positioning.

    Parameters
    ----------
    msme_ref_file : Path
        First echo volume (3D, skull-stripped or raw)
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
        "register_msme_to_t2w() is deprecated. Use register_msme_to_template() "
        "for better SIGMA atlas overlap.",
        DeprecationWarning,
        stacklevel=2
    )

    print("\n" + "="*60)
    print("MSME to T2w Registration")
    print("="*60)

    msme_img = nib.load(msme_ref_file)
    t2w_img = nib.load(t2w_file)
    print(f"\n  MSME ref: {msme_img.shape} voxels, {msme_img.header.get_zooms()[:3]} mm")
    print(f"  T2w: {t2w_img.shape} voxels, {t2w_img.header.get_zooms()[:3]} mm")

    transforms_dir = output_dir / 'transforms' / subject / session
    transforms_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = transforms_dir / 'MSME_to_T2w_'

    # Step 1: Find optimal Z offset via NCC scan
    print("\n  Finding optimal Z offset via NCC scan...")
    initial_transform, z_offset_info = _find_z_offset_ncc(
        msme_img, t2w_img, work_dir
    )

    # Step 2: Rigid registration with conservative shrink factors
    # (5 slices is very partial — use 2x1x1 to avoid losing Z info)
    print("\nRunning ANTs Rigid registration (MSME → T2w)...")

    warped_output = Path(str(output_prefix) + 'Warped.nii.gz')
    cmd = [
        'antsRegistration',
        '--dimensionality', '3',
        '--output', f'[{output_prefix},{warped_output}]',
        '--interpolation', 'Linear',
        '--use-histogram-matching', '1',
        '--winsorize-image-intensities', '[0.005,0.995]',
        '--initial-moving-transform', str(initial_transform),
        # Rigid only (6 DOF) — 5 slices is too few for affine
        '--transform', 'Rigid[0.1]',
        '--metric', f'MI[{t2w_file},{msme_ref_file},1,32,Regular,0.25]',
        '--convergence', '[500x250x100,1e-6,10]',
        '--shrink-factors', '2x1x1',
        '--smoothing-sigmas', '1x0x0vox',
    ]

    print(f"  Moving: {msme_ref_file.name}")
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
        raise RuntimeError("MSME to T2w registration failed")

    # Check outputs
    rigid_transform = Path(str(output_prefix) + '0GenericAffine.mat')
    warped_msme = Path(str(output_prefix) + 'Warped.nii.gz')

    if not rigid_transform.exists():
        raise RuntimeError(f"Expected transform not found: {rigid_transform}")

    print(f"  Rigid transform: {rigid_transform.name}")
    if warped_msme.exists():
        print(f"  Warped MSME ref: {warped_msme.name}")
        # Report coverage
        warped_data = nib.load(warped_msme).get_fdata()
        threshold = warped_data.max() * 0.05
        slices_with_signal = [z for z in range(warped_data.shape[2])
                              if warped_data[:,:,z].max() > threshold]
        if slices_with_signal:
            print(f"  MSME covers T2w slices {slices_with_signal[0]}-{slices_with_signal[-1]} "
                  f"({len(slices_with_signal)} slices)")

    return {
        'affine_transform': rigid_transform,
        'warped_msme': warped_msme if warped_msme.exists() else None,
        't2w_file': t2w_file,
        'msme_shape': msme_img.shape,
        't2w_shape': t2w_img.shape,
    }


def run_msme_preprocessing(
    config: Dict[str, Any],
    subject: str,
    session: str,
    msme_file: Path,
    output_dir: Path,
    transform_registry: TransformRegistry,
    te_values: Optional[np.ndarray] = None,
    work_dir: Optional[Path] = None,
    template_file: Optional[Path] = None,
    t2w_file: Optional[Path] = None,
    run_registration: bool = True
) -> Dict[str, Any]:
    """
    Run MSME preprocessing workflow with T2 mapping and MWF calculation.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject identifier
    session : str
        Session identifier
    msme_file : Path
        Input MSME 4D NIfTI file (shape: X, Y, echoes, slices)
    output_dir : Path
        Study root directory
    transform_registry : TransformRegistry
        Transform registry
    te_values : np.ndarray, optional
        Echo times in ms (if None, assumes 10-320ms in 10ms steps)
    work_dir : Path, optional
        Working directory
    template_file : Path, optional
        Cohort template for direct MSME→Template registration (preferred)
    t2w_file : Path, optional
        T2w anatomical reference (deprecated, use template_file instead)
    run_registration : bool
        Whether to register MSME to template (default: True, requires template_file or t2w_file)

    Returns
    -------
    dict
        Dictionary with output paths and QC info
    """
    print("="*80)
    print(f"MSME T2 Mapping Workflow")
    print(f"Subject: {subject}, Session: {session}")
    print("="*80)

    # Setup directories
    derivatives_dir = output_dir / 'derivatives' / subject / session / 'msme'
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    qc_dir = get_subject_qc_dir(output_dir, subject, session, 'msme')

    if work_dir is None:
        work_dir = output_dir / 'work' / subject / session / 'msme_preproc'
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDirectories:")
    print(f"  Derivatives: {derivatives_dir}")
    print(f"  QC: {qc_dir}")
    print(f"  Work: {work_dir}")

    # Define output files
    brain_mask_file = derivatives_dir / f'{subject}_{session}_desc-brain_mask.nii.gz'
    msme_masked_file = work_dir / f'{subject}_{session}_msme_brain.nii.gz'
    mwf_file = derivatives_dir / f'{subject}_{session}_MWF.nii.gz'
    iwf_file = derivatives_dir / f'{subject}_{session}_IWF.nii.gz'
    csf_file = derivatives_dir / f'{subject}_{session}_CSFF.nii.gz'
    t2_file = derivatives_dir / f'{subject}_{session}_T2.nii.gz'

    # Default TE values (10-320ms in 10ms steps = 32 echoes)
    if te_values is None:
        te_values = np.arange(10, 330, 10)

    print(f"\nTE values: {len(te_values)} echoes from {te_values[0]:.1f} to {te_values[-1]:.1f} ms")

    # Read geometry from config (set during study init from Bruker method file)
    slice_thick_mm = get_config_value(config, 'msme.geometry.slice_thickness_mm', default=0.8)
    voxel_scale = get_config_value(config, 'msme.geometry.voxel_scale', default=10.0)
    slice_vox_z = float(slice_thick_mm * voxel_scale)
    print(f"Geometry: slice_thickness={slice_thick_mm}mm, voxel_scale={voxel_scale}x, Z voxel={slice_vox_z}mm")

    # ==========================================================================
    # Step 1: Skull stripping
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 1: Skull Stripping (Adaptive Slice-wise)")
    print("="*80)

    # Load MSME data
    img = nib.load(msme_file)
    data = img.get_fdata()

    print(f"MSME shape: {data.shape}")

    if len(data.shape) != 4:
        raise ValueError(f"Expected 4D MSME data, got shape: {data.shape}")

    # Use first echo for skull stripping
    # MSME shape: (X, Y, echoes, slices) — echoes in dim 2, slices in dim 3
    # Extract first echo (highest SNR) across all spatial slices
    first_echo = data[:, :, 0, :]  # First echo, all slices

    # Create 3D NIfTI with correct spatial header
    # Original 4D has echoes in dim2 and slices in dim3 — not a standard spatial layout.
    # Build a proper spatial affine: in-plane from header, Z = slice_thickness * scale
    in_plane = img.header.get_zooms()[:2]
    echo1_affine = np.diag([float(in_plane[0]), float(in_plane[1]), slice_vox_z, 1.0])

    first_echo_file = work_dir / f'{subject}_{session}_echo1.nii.gz'
    nib.save(nib.Nifti1Image(first_echo.astype(np.float32), echo1_affine), first_echo_file)
    print(f"First echo extracted: shape={first_echo.shape}, voxels=({in_plane[0]}, {in_plane[1]}, {slice_vox_z})mm")

    # Skull stripping with configurable method and parameters
    brain_extracted_file = work_dir / f'{subject}_{session}_echo1_brain.nii.gz'
    msme_ss_method = get_config_value(config, 'msme.skull_strip.method', default='adaptive')
    msme_ss_n_classes = get_config_value(config, 'msme.skull_strip.n_classes', default=3)
    msme_ss_target_ratio = get_config_value(config, 'msme.skull_strip.target_ratio', default=0.15)
    msme_ss_frac_min = get_config_value(config, 'msme.skull_strip.frac_min', default=0.3)
    msme_ss_frac_max = get_config_value(config, 'msme.skull_strip.frac_max', default=0.8)
    msme_ss_frac_step = get_config_value(config, 'msme.skull_strip.frac_step', default=0.05)
    msme_ss_cog_x = get_config_value(config, 'msme.skull_strip.cog_offset_x', default=0)
    msme_ss_cog_y = get_config_value(config, 'msme.skull_strip.cog_offset_y', default=-40)

    brain_file, mask_file_out, skull_strip_result = skull_strip(
        input_file=first_echo_file,
        output_file=brain_extracted_file,
        mask_file=brain_mask_file,
        work_dir=work_dir / 'skull_strip',
        method=msme_ss_method,
        n_classes=msme_ss_n_classes,
        target_ratio=msme_ss_target_ratio,
        frac_range=(msme_ss_frac_min, msme_ss_frac_max),
        frac_step=msme_ss_frac_step,
        cog_offset_x=msme_ss_cog_x,
        cog_offset_y=msme_ss_cog_y,
    )

    # Apply mask to all echoes
    # Mask shape: (X, Y, slices) = (160, 160, 5)
    # Data shape: (X, Y, echoes, slices) = (160, 160, 32, 5)
    mask_img = nib.load(brain_mask_file)
    mask_3d = mask_img.get_fdata() > 0  # Shape: (160, 160, 5)

    # Expand mask to match data dimensions: add echo dimension
    # mask_3d[:, :, np.newaxis, :] → (160, 160, 1, 5)
    # Broadcasting with data → (160, 160, 32, 5)
    mask_4d = mask_3d[:, :, np.newaxis, :]  # Add axis for echoes

    # Estimate noise sigma² from the air/noise region of the first echo.
    # The background outside the brain mask contains both noise (air) and
    # bright tissue (skull, muscle, fat). We isolate the noise-only voxels
    # by taking the lowest-intensity quartile of the background, which
    # should be pure Rayleigh-distributed noise from air.
    first_echo_all = data[:, :, 0, :]
    first_echo_bg = first_echo_all[~mask_3d]
    first_echo_bg = first_echo_bg[first_echo_bg > 0]  # exclude exact zeros

    # Keep only the lowest quartile — these are the air/noise voxels
    q25 = np.percentile(first_echo_bg, 25)
    noise_voxels = first_echo_bg[first_echo_bg <= q25]

    # For Rayleigh-distributed magnitude noise:
    #   mean = sigma * sqrt(pi/2)  →  sigma = mean / sqrt(pi/2)
    #   sigma² = mean² * 2/pi
    noise_mean = float(np.mean(noise_voxels))
    noise_sigma2 = noise_mean ** 2 * (2.0 / np.pi)
    print(f"Background noise estimation (first echo):")
    print(f"  Background voxels: {len(first_echo_bg)}, noise-floor voxels (≤Q25={q25:.0f}): {len(noise_voxels)}")
    print(f"  Noise mean={noise_mean:.1f}, Gaussian sigma²={noise_sigma2:.1f} "
          f"(sigma={np.sqrt(noise_sigma2):.1f})")

    data_masked = data * mask_4d  # Broadcasting applies mask to all echoes

    nib.save(nib.Nifti1Image(data_masked, img.affine, img.header), msme_masked_file)
    print(f"Masked MSME saved to: {msme_masked_file}")

    # ==========================================================================
    # Step 2: T2 Fitting and MWF Calculation
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 2: T2 Fitting and MWF Calculation (NNLS)")
    print("="*80)

    # Reorder data from (x, y, echoes, slices) to (x, y, slices, echoes)
    # calculate_mwf_nnls expects (x, y, z, echoes) format
    data_reordered = np.transpose(data_masked, (0, 1, 3, 2))
    print(f"  Data reordered: {data_masked.shape} → {data_reordered.shape}")

    t2_n_components = get_config_value(config, 'msme.t2_fitting.n_components', default=120)
    t2_range = get_config_value(config, 'msme.t2_fitting.t2_range', default=[10, 2000])
    t2_lambda_reg = get_config_value(config, 'msme.t2_fitting.lambda_reg', default=None)
    t2_mw_cutoff = get_config_value(config, 'msme.t2_fitting.myelin_water_cutoff', default=25.0)
    t2_ie_cutoff = get_config_value(config, 'msme.t2_fitting.intra_extra_cutoff', default=200.0)

    mwf_map, iwf_map, csf_map, t2_map, sample_data = calculate_mwf_nnls(
        data_reordered,
        mask_3d,
        te_values,
        lambda_reg=t2_lambda_reg,
        noise_sigma2=noise_sigma2,
        n_components=t2_n_components,
        t2_range=t2_range,
        myelin_water_cutoff=t2_mw_cutoff,
        intra_extra_cutoff=t2_ie_cutoff,
    )

    # Save output maps with correct spatial affine
    # Output maps have shape (x, y, slices) — use echo1_affine from config geometry
    nib.save(nib.Nifti1Image(mwf_map, echo1_affine), mwf_file)
    nib.save(nib.Nifti1Image(iwf_map, echo1_affine), iwf_file)
    nib.save(nib.Nifti1Image(csf_map, echo1_affine), csf_file)
    nib.save(nib.Nifti1Image(t2_map, echo1_affine), t2_file)

    print(f"\nOutput maps created:")
    print(f"  MWF: {mwf_file}")
    print(f"  IWF: {iwf_file}")
    print(f"  CSF Fraction: {csf_file}")
    print(f"  T2: {t2_file}")

    # ==========================================================================
    # Step 3: Quality Control
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 3: Quality Control")
    print("="*80)

    from neurofaune.preprocess.qc.msme import generate_msme_qc_report

    qc_results = generate_msme_qc_report(
        subject=subject,
        session=session,
        mwf_file=mwf_file,
        iwf_file=iwf_file,
        csf_file=csf_file,
        t2_file=t2_file,
        brain_mask=brain_mask_file,
        te_values=te_values,
        sample_data=sample_data,
        output_dir=qc_dir,
        original_file=first_echo_file,
        skull_strip_info=skull_strip_result,
    )

    print(f"\n✓ MSME QC report: {qc_results['html_report']}")

    # ==========================================================================
    # Step 4: MSME to Template Registration
    # ==========================================================================
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
            step_name = "MSME to Template" if use_template else "MSME to T2w (deprecated)"
            print("\n" + "="*80)
            print(f"Step 4: {step_name} Registration")
            print("="*80)

            try:
                # Extract first echo as registration reference
                reg_work_dir = work_dir / 'msme_registration'
                reg_work_dir.mkdir(parents=True, exist_ok=True)
                msme_ref = reg_work_dir / f'{subject}_{session}_msme_echo1.nii.gz'

                if not msme_ref.exists():
                    print("  Extracting first echo as registration reference...")
                    msme_data = img.get_fdata()
                    first_echo = msme_data[:, :, 0, :]  # First echo, all slices

                    in_plane = img.header.get_zooms()[:2]
                    ref_affine = np.diag([float(in_plane[0]), float(in_plane[1]), slice_vox_z, 1.0])

                    msme_ref_raw = reg_work_dir / f'{subject}_{session}_msme_echo1_raw.nii.gz'
                    nib.save(
                        nib.Nifti1Image(first_echo.astype(np.float32), ref_affine),
                        msme_ref_raw
                    )

                    print(f"  First echo ref: {first_echo.shape}, voxels=({in_plane[0]}, {in_plane[1]}, {slice_vox_z})mm")

                    # Skull stripping with configurable method
                    msme_mask = reg_work_dir / f'{subject}_{session}_msme_echo1_mask.nii.gz'
                    _, _, skull_strip_result = skull_strip(
                        input_file=msme_ref_raw,
                        output_file=msme_ref,
                        mask_file=msme_mask,
                        work_dir=reg_work_dir / 'skull_strip',
                        method=msme_ss_method,
                        n_classes=msme_ss_n_classes,
                    )

                if use_template:
                    registration_results = register_msme_to_template(
                        msme_ref_file=msme_ref,
                        template_file=template_file,
                        output_dir=output_dir,
                        subject=subject,
                        session=session,
                        work_dir=reg_work_dir,
                        n_cores=config.get('execution', {}).get('n_procs', 4)
                    )

                    reg_metadata_file = derivatives_dir / f'{subject}_{session}_MSME_to_template_registration.json'
                    reg_metadata = {
                        'msme_ref_file': str(msme_ref),
                        'msme_source': str(msme_file),
                        'template_file': str(registration_results['template_file']),
                        'affine_transform': str(registration_results['affine_transform']),
                        'warped_msme': str(registration_results['warped_msme']) if registration_results.get('warped_msme') else None,
                        'msme_shape': list(registration_results['msme_shape']),
                        'template_shape': list(registration_results['template_shape']),
                    }
                else:
                    registration_results = register_msme_to_t2w(
                        msme_ref_file=msme_ref,
                        t2w_file=t2w_file,
                        output_dir=output_dir,
                        subject=subject,
                        session=session,
                        work_dir=reg_work_dir,
                        n_cores=config.get('execution', {}).get('n_procs', 4)
                    )

                    reg_metadata_file = derivatives_dir / f'{subject}_{session}_MSME_to_T2w_registration.json'
                    reg_metadata = {
                        'msme_ref_file': str(msme_ref),
                        'msme_source': str(msme_file),
                        't2w_file': str(registration_results['t2w_file']),
                        'affine_transform': str(registration_results['affine_transform']),
                        'warped_msme': str(registration_results['warped_msme']) if registration_results.get('warped_msme') else None,
                        'msme_shape': list(registration_results['msme_shape']),
                        't2w_shape': list(registration_results['t2w_shape']),
                    }

                with open(reg_metadata_file, 'w') as f:
                    json.dump(reg_metadata, f, indent=2)

                print(f"\n  Registration complete:")
                print(f"  - Transform: {registration_results['affine_transform']}")
                print(f"  - Metadata: {reg_metadata_file}")

            except Exception as e:
                print(f"\n  Registration failed: {e}")
                print("  Continuing without registration...")

    # ==========================================================================
    # Workflow complete
    # ==========================================================================
    print("\n" + "="*80)
    print("MSME Preprocessing Complete!")
    print("="*80 + "\n")

    results = {
        'mwf': mwf_file,
        'iwf': iwf_file,
        'csf': csf_file,
        't2': t2_file,
        'brain_mask': brain_mask_file,
        'qc_results': qc_results
    }

    if registration_results:
        results['registration'] = {
            'affine_transform': registration_results['affine_transform'],
            'warped_msme': registration_results.get('warped_msme'),
        }

    return results


def _run_bet(input_file: Path, output_mask: Path, frac: float = 0.3):
    """Run FSL BET for skull stripping."""
    from nipype.interfaces import fsl

    bet = fsl.BET()
    bet.inputs.in_file = str(input_file)
    bet.inputs.out_file = str(output_mask.parent / output_mask.stem.replace('.nii', ''))
    bet.inputs.mask = True
    bet.inputs.frac = frac
    bet.inputs.robust = True

    bet.run()

    # BET creates files with _mask suffix
    mask_file = output_mask.parent / (output_mask.stem.replace('.nii', '') + '_mask.nii.gz')
    if mask_file.exists() and mask_file != output_mask:
        import shutil
        shutil.move(mask_file, output_mask)


def _skull_strip_msme_adaptive(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    work_dir: Path,
    target_ratio: float = 0.15,
    frac_range: Tuple[float, float] = (0.30, 0.80),
    frac_step: float = 0.05,
    invert_intensity: bool = False,
    use_R_flag: bool = False,
    cog_offset_x: Optional[int] = None,
    cog_offset_y: Optional[int] = None
) -> Dict[str, Any]:
    """
    Adaptive slice-wise skull stripping for MSME data.

    MSME has only 5 thick coronal slices (160x160x5 at 2.0x2.0x8.0mm).
    Standard 3D BET fails on this geometry. This function runs BET
    independently on each slice with per-slice frac optimization.

    Parameters
    ----------
    input_file : Path
        Input 3D MSME reference volume (first echo)
    output_file : Path
        Output brain-extracted volume
    mask_file : Path
        Output brain mask
    work_dir : Path
        Working directory for intermediate files
    target_ratio : float
        Target extraction ratio per slice (default 0.15 = 15% of slice)
    frac_range : Tuple[float, float]
        Range of BET frac values to test (default 0.30-0.80)
    frac_step : float
        Step size for frac search (default 0.05)
    invert_intensity : bool
        If True, invert intensity before BET (T2w → T1w-like)
    use_R_flag : bool
        If True, use BET's -R flag for robust center estimation
    cog_offset_x : int, optional
        X offset from image center for COG estimation
    cog_offset_y : int, optional
        Y offset from image center for COG estimation (negative = down/inferior)

    Returns
    -------
    dict
        Dictionary with skull stripping results and statistics
    """
    print("\n  Running adaptive slice-wise skull stripping for MSME...")
    print(f"    Target extraction ratio: {target_ratio*100:.0f}%")
    print(f"    Frac search range: {frac_range[0]:.2f} - {frac_range[1]:.2f}")
    if cog_offset_x is not None or cog_offset_y is not None:
        print(f"    COG offset: ({cog_offset_x or 0}, {cog_offset_y or 0})")
    else:
        print(f"    COG: intensity-weighted (use_R_flag={use_R_flag})")

    # Run adaptive skull stripping
    brain_file, mask_out, info = skull_strip_adaptive(
        input_file=input_file,
        output_file=output_file,
        mask_file=mask_file,
        work_dir=work_dir,
        target_ratio=target_ratio,
        frac_range=frac_range,
        frac_step=frac_step,
        invert_intensity=invert_intensity,
        use_R_flag=use_R_flag,
        cog_offset_x=cog_offset_x,
        cog_offset_y=cog_offset_y
    )

    # Report per-slice results
    print("\n  Per-slice skull stripping results:")
    for stat in info.get('slice_stats', []):
        print(f"    Slice {stat['slice']}: frac={stat['optimal_frac']:.2f}, "
              f"{stat['voxels']:,} voxels ({stat['ratio']*100:.1f}%)")

    print(f"\n  Overall: {info['total_voxels']:,} voxels, "
          f"extraction ratio={info['extraction_ratio']:.3f}")
    print(f"  Mean frac: {info['mean_frac']:.3f} ± {info['std_frac']:.3f}")

    return {
        'brain_file': brain_file,
        'mask_file': mask_out,
        'info': info
    }


def _skull_strip_msme_atropos(
    input_file: Path,
    output_file: Path,
    work_dir: Path,
    affine: np.ndarray,
    data: np.ndarray,
    n_classes: int = 5
):
    """
    Skull-strip MSME first echo using Atropos 5-component segmentation.

    Uses Atropos K-means with 5 classes (same as anatomical pipeline),
    then selects the top 3 brightest classes as brain tissue (WM, GM, CSF).
    The 2 darkest classes (background/air and skull/muscle) are excluded.

    This approach matches the anatomical pipeline's Atropos 5-component
    strategy adapted for T2w contrast.

    Parameters
    ----------
    input_file : Path
        Raw first echo volume
    output_file : Path
        Output skull-stripped volume
    work_dir : Path
        Working directory for Atropos intermediates
    affine : np.ndarray
        Affine matrix for output
    data : np.ndarray
        First echo data array
    n_classes : int
        Number of Atropos classes (default: 5)
    """
    # Create initial foreground mask (exclude obvious background)
    nonzero = data[data > 0]
    threshold = np.percentile(nonzero, 5)
    initial_mask = (data > threshold).astype(np.uint8)
    initial_mask_file = work_dir / 'atropos_initial_mask.nii.gz'
    nib.save(nib.Nifti1Image(initial_mask, affine), initial_mask_file)

    # Run Atropos K-means segmentation
    seg_file = work_dir / 'atropos_seg.nii.gz'
    output_prefix = work_dir / 'atropos_'
    cmd = [
        'Atropos', '-d', '3',
        '-a', str(input_file),
        '-x', str(initial_mask_file),
        '-i', f'KMeans[{n_classes}]',
        '-o', f'[{seg_file},{output_prefix}prob%02d.nii.gz]',
        '-v', '0'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Atropos segmentation failed: {result.stderr}")

    seg_data = nib.load(seg_file).get_fdata()

    # Find mean intensity per class
    class_means = {}
    for label in range(1, n_classes + 1):
        mask = seg_data == label
        if np.sum(mask) > 0:
            class_means[label] = data[mask].mean()
            print(f"    Class {label}: {int(np.sum(mask)):,} voxels, mean={class_means[label]:.1f}")

    # Sort classes by intensity: bottom 2 = non-brain, top 3 = brain
    sorted_classes = sorted(class_means, key=class_means.get)
    brain_classes = sorted_classes[2:]  # Top 3 brightest (WM, GM, CSF)
    excluded_classes = sorted_classes[:2]  # Bottom 2 (background, skull/muscle)
    print(f"    Brain classes: {brain_classes} (excluded: {excluded_classes})")

    # Brain mask = top 3 classes
    brain_mask = np.zeros_like(seg_data, dtype=np.uint8)
    for label in brain_classes:
        brain_mask[seg_data == label] = 1

    n_brain = int(np.sum(brain_mask))
    print(f"    Brain mask: {n_brain:,} voxels ({100*n_brain/brain_mask.size:.1f}%)")

    # Apply mask and save
    brain_data = data.astype(np.float32) * brain_mask
    nib.save(nib.Nifti1Image(brain_data, affine), output_file)


def _estimate_noise_sigma2(signals, n_tail=6):
    """Estimate noise variance from echo-to-echo differences in the decay tail.

    For each voxel, takes the last n_tail echoes, computes successive
    differences, and estimates sigma² from their variance (divided by 2,
    since var(diff of two measurements) = 2*sigma²).  Returns the median
    across voxels, which is robust to outliers from long-T2 voxels (CSF)
    where the tail still carries signal.

    Returns (sigma2, per_voxel_sigma2_list).
    """
    per_voxel_sigma2 = []
    for sig in signals:
        tail = sig[-n_tail:]
        diffs = np.diff(tail)
        per_voxel_sigma2.append(float(np.var(diffs) / 2.0))
    sigma2 = float(np.median(per_voxel_sigma2))
    return sigma2, per_voxel_sigma2


def calculate_mwf_nnls(
    data: np.ndarray,
    mask: np.ndarray,
    te_values: np.ndarray,
    lambda_reg: float = None,
    noise_sigma2: float = None,
    n_components: int = 120,
    t2_range: list = None,
    myelin_water_cutoff: float = 25.0,
    intra_extra_cutoff: float = 200.0,
    chi2_factor: float = 1.02,
) -> tuple:
    """
    Calculate MWF using regularized Non-Negative Least Squares (NNLS).

    Uses minimum curvature (2nd-order Tikhonov) regularization following
    Whittall & MacKay (1989) and Prasloski et al. (2012).

    When lambda_reg is None, uses per-voxel lambda estimation via the
    chi-square discrepancy principle: for each voxel, lambda is chosen
    so that the fit residual chi² ≈ n_echoes * sigma² * chi2_factor.
    This is the proper Whittall & MacKay approach — each voxel gets
    exactly the regularization it needs based on its SNR.

    Parameters
    ----------
    data : np.ndarray
        4D MSME data (x, y, z, echoes)
    mask : np.ndarray
        3D brain mask
    te_values : np.ndarray
        Echo times in ms
    lambda_reg : float or None
        Fixed regularization weight. If None (default), per-voxel lambda
        is estimated automatically using the chi-square criterion.
    noise_sigma2 : float or None
        Noise variance (Gaussian sigma²) for chi-square target. If None,
        estimated from echo-to-echo differences in the signal tail
        (less accurate for magnitude data due to Rician bias).
        Preferably estimated from the image background before masking.
    n_components : int
        Number of T2 components in log-spaced basis (default 120)
    t2_range : list
        [min, max] T2 values in ms (default [10, 2000])
    myelin_water_cutoff : float
        T2 boundary (ms) between myelin water and IEW (default 25ms)
    intra_extra_cutoff : float
        T2 boundary (ms) between IEW and CSF (default 200ms)
    chi2_factor : float
        Chi-square inflation factor for lambda estimation (default 1.02).
        Higher = stronger regularization = smoother spectra.

    Returns
    -------
    tuple
        (mwf_map, iwf_map, csf_map, t2_map, sample_data)
        sample_data contains representative T2 curves and NNLS spectra for QC
    """
    print("\nCalculating MWF using regularized NNLS (minimum curvature)...")

    shape_3d = data.shape[:3]
    n_echoes = data.shape[3]

    # Initialize output maps
    mwf_map = np.zeros(shape_3d)
    iwf_map = np.zeros(shape_3d)
    csf_map = np.zeros(shape_3d)
    t2_map = np.zeros(shape_3d)

    # Store sample data for QC visualization
    sample_data = {
        'voxels': [],  # Voxel coordinates
        'signals': [],  # T2 decay curves
        'spectra': [],  # NNLS T2 spectra
        'mwf_values': [],  # MWF values for these voxels
        't2_dist': None  # T2 distribution (same for all voxels)
    }

    # T2 distribution (log-spaced)
    if t2_range is None:
        t2_range = [10, 2000]
    t2_dist = np.geomspace(t2_range[0], t2_range[1], num=n_components)
    sample_data['t2_dist'] = t2_dist

    # Define water compartments using configurable cutoffs
    mw_cutoff = np.where(t2_dist < myelin_water_cutoff)[0][-1] + 1
    iw_cutoff = np.where(t2_dist < intra_extra_cutoff)[0][-1] + 1
    print(f"  Compartment cutoffs: MW < {myelin_water_cutoff}ms, "
          f"IEW {myelin_water_cutoff}-{intra_extra_cutoff}ms, "
          f"CSF > {intra_extra_cutoff}ms")

    # Create design matrix A for NNLS
    # A[i, j] = exp(-TE[i] / T2[j])
    A = np.zeros((n_echoes, len(t2_dist)))
    for i, te in enumerate(te_values):
        A[i, :] = np.exp(-te / t2_dist)

    # Minimum curvature regularization matrix (2nd-order Tikhonov)
    # Penalizes: Σ (x[j+2] - 2*x[j+1] + x[j])²
    # Whittall & MacKay (1989), Prasloski et al. (2012)
    n_t2 = len(t2_dist)
    H = np.zeros((n_t2 - 2, n_t2))
    for j in range(n_t2 - 2):
        H[j, j] = 1.0
        H[j, j + 1] = -2.0
        H[j, j + 2] = 1.0

    voxel_indices = np.argwhere(mask)
    n_voxels = len(voxel_indices)

    # --- Noise estimation and lambda mode ---
    use_per_voxel_lambda = (lambda_reg is None)

    if use_per_voxel_lambda:
        if noise_sigma2 is not None:
            # Use provided noise estimate (from image background)
            sigma2 = noise_sigma2
            print(f"  Per-voxel lambda via chi-square discrepancy principle")
            print(f"    Noise sigma²={sigma2:.2f} (provided from background)")
        else:
            # Fallback: estimate from echo-to-echo differences in signal tail.
            # NOTE: underestimates sigma² for magnitude data due to Rician bias.
            rng = np.random.default_rng(42)
            n_noise_subset = min(1000, n_voxels)
            noise_idx = rng.choice(n_voxels, size=n_noise_subset, replace=False)
            noise_signals = []
            for i in noise_idx:
                x, y, z = voxel_indices[i]
                sig = data[x, y, z, :]
                if sig[0] > 0:
                    noise_signals.append(sig)
            sigma2, pv_sigma2 = _estimate_noise_sigma2(noise_signals)
            # Apply Rician correction: measured var ≈ sigma² * (2 - pi/2)
            sigma2 = sigma2 / (2 - np.pi / 2)
            print(f"  Per-voxel lambda via chi-square discrepancy principle")
            print(f"    Noise sigma²={sigma2:.2f} (from signal tail, Rician-corrected)")

        chi2_target = n_echoes * sigma2 * chi2_factor
        print(f"    Chi² target={chi2_target:.2f} "
              f"(n_echoes={n_echoes} × sigma²={sigma2:.1f} × factor={chi2_factor})")
        print(f"    Bisection: 15 iterations per voxel, range [1e-6, 1e6]")

        # Pre-allocate augmented system to avoid repeated concatenation
        n_rows_reg = n_echoes + n_t2 - 2
        A_reg = np.empty((n_rows_reg, n_t2))
        A_reg[:n_echoes, :] = A
        b_reg = np.empty(n_rows_reg)
        b_reg[n_echoes:] = 0.0
    else:
        print(f"  Using fixed lambda: {lambda_reg}")
        A_reg_fixed = np.concatenate([A, np.sqrt(lambda_reg) * H])
        b_zeros = np.zeros(n_t2 - 2)

    print(f"  Processing {n_voxels} voxels...")

    # Progress and lambda tracking
    report_every = max(n_voxels // 10, 1)
    n_regularized = 0
    n_unreg = 0
    lambda_values = []  # track for diagnostics

    for idx, (x, y, z) in enumerate(voxel_indices):
        if idx % report_every == 0:
            progress = 100 * idx / n_voxels
            print(f"    Progress: {progress:.1f}%")

        # Get signal for this voxel
        signal = data[x, y, z, :]

        # Skip if no signal
        if np.sum(signal) == 0:
            continue

        # Solve NNLS with appropriate lambda
        try:
            if use_per_voxel_lambda:
                # --- Per-voxel lambda via chi-square bisection ---
                # Step 1: unregularized solve
                amp_unreg, _ = nnls(A, signal)
                chi2_unreg = np.sum((A @ amp_unreg - signal) ** 2)

                if chi2_unreg >= chi2_target:
                    # Already at/above noise floor — no regularization needed.
                    # Use minimal lambda for numerical stability.
                    amplitudes = amp_unreg
                    n_unreg += 1
                    lambda_values.append(0.0)
                else:
                    # Bisect: find lambda where chi² rises to target
                    lo, hi = 1e-6, 1e6
                    for _ in range(15):
                        mid = np.sqrt(lo * hi)
                        A_reg[n_echoes:, :] = np.sqrt(mid) * H
                        b_reg[:n_echoes] = signal
                        amp, _ = nnls(A_reg, b_reg)
                        chi2 = np.sum((A @ amp - signal) ** 2)
                        if chi2 < chi2_target:
                            lo = mid
                        else:
                            hi = mid
                    # Final solve at converged lambda
                    final_lam = np.sqrt(lo * hi)
                    A_reg[n_echoes:, :] = np.sqrt(final_lam) * H
                    b_reg[:n_echoes] = signal
                    amplitudes, _ = nnls(A_reg, b_reg)
                    n_regularized += 1
                    lambda_values.append(final_lam)
            else:
                # Fixed global lambda
                b_reg_fixed = np.concatenate([signal, b_zeros])
                amplitudes, _ = nnls(A_reg_fixed, b_reg_fixed)

            # Calculate fractions
            total_amp = np.sum(amplitudes)

            # Save sample voxels for QC (every 1000th voxel, up to 10 samples)
            if len(sample_data['voxels']) < 10 and idx % 1000 == 0:
                sample_data['voxels'].append((x, y, z))
                sample_data['signals'].append(signal.copy())
                sample_data['spectra'].append(amplitudes.copy())
            if total_amp > 0:
                mwf_val = np.sum(amplitudes[:mw_cutoff]) / total_amp
                mwf_map[x, y, z] = mwf_val
                iwf_map[x, y, z] = np.sum(amplitudes[mw_cutoff:iw_cutoff]) / total_amp
                csf_map[x, y, z] = np.sum(amplitudes[iw_cutoff:]) / total_amp

                # Store MWF value for sample voxels
                if len(sample_data['mwf_values']) < len(sample_data['voxels']):
                    sample_data['mwf_values'].append(mwf_val)

            # Calculate T2 using mono-exponential fit
            def t2_func(te, s0, t2):
                return s0 * np.exp(-te / t2)

            try:
                popt, _ = curve_fit(t2_func, te_values, signal,
                                    p0=[signal[0], 50],
                                    bounds=([0, 10], [np.inf, 500]),
                                    maxfev=1000)
                t2_map[x, y, z] = popt[1]
            except:
                t2_map[x, y, z] = 0

        except Exception as e:
            # Skip problematic voxels
            continue

    print("  100% complete!")

    if use_per_voxel_lambda and lambda_values:
        reg_lambdas = [l for l in lambda_values if l > 0]
        print(f"\n  Lambda statistics:")
        print(f"    Voxels regularized: {n_regularized}/{n_voxels} "
              f"({100*n_regularized/max(n_voxels,1):.1f}%)")
        print(f"    Voxels unregularized (chi²≥target): {n_unreg}/{n_voxels}")
        if reg_lambdas:
            print(f"    Regularized lambda: median={np.median(reg_lambdas):.4f}, "
                  f"range=[{min(reg_lambdas):.4f}, {max(reg_lambdas):.4f}]")

    # Calculate summary statistics
    mwf_masked = mwf_map[mask]
    print(f"\n  MWF statistics (within brain):")
    print(f"    Mean: {np.mean(mwf_masked):.3f}")
    print(f"    Std: {np.std(mwf_masked):.3f}")
    print(f"    Range: [{np.min(mwf_masked):.3f}, {np.max(mwf_masked):.3f}]")

    print(f"\n  Captured {len(sample_data['voxels'])} sample voxels for QC visualization")

    return mwf_map, iwf_map, csf_map, t2_map, sample_data
