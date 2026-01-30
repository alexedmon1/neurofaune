"""
DTI/DWI preprocessing workflow.

This module provides a complete preprocessing pipeline for diffusion MRI data,
including eddy correction and DTI fitting.

NOTE: Registration to study-specific FA template and SIGMA atlas will be added
separately in template building/registration modules.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import subprocess
import json

from neurofaune.preprocess.utils.dwi_utils import (
    convert_5d_to_4d,
    validate_gradient_table,
    extract_b0_volume,
    check_dwi_data_quality,
    pad_slices_for_eddy,
    pad_mask_for_eddy,
    crop_slices_after_eddy,
    normalize_dwi_intensity
)
from neurofaune.preprocess.utils.skull_strip import skull_strip
from neurofaune.preprocess.utils.validation import validate_image, print_validation_results
from neurofaune.preprocess.utils.orientation import (
    match_orientation_to_reference,
    save_orientation_metadata,
    print_orientation_info
)
from neurofaune.atlas.manager import AtlasManager
from neurofaune.utils.transforms import TransformRegistry
from neurofaune.preprocess.qc.dwi import generate_eddy_qc_report, generate_dti_qc_report
from neurofaune.preprocess.qc import get_subject_qc_dir


def register_fa_to_t2w(
    fa_file: Path,
    t2w_file: Path,
    output_dir: Path,
    subject: str,
    session: str,
    work_dir: Path,
    n_cores: int = 4
) -> Dict[str, Any]:
    """
    Register FA to T2w within the same subject.

    Registers FA directly to the full T2w volume, letting ANTs find the
    optimal 3D alignment including the Z-offset for partial coverage DWI.

    Parameters
    ----------
    fa_file : Path
        FA map from DTI fitting
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
    print("\n" + "="*60)
    print("FA to T2w Registration")
    print("="*60)

    # Load images to get info
    fa_img = nib.load(fa_file)
    t2w_img = nib.load(t2w_file)
    print(f"\n  FA: {fa_img.shape} voxels, {fa_img.header.get_zooms()[:3]} mm")
    print(f"  T2w: {t2w_img.shape} voxels, {t2w_img.header.get_zooms()[:3]} mm")

    # Register FA directly to full T2w - let ANTs find optimal alignment
    print("\nRunning ANTs Affine registration (FA → full T2w)...")
    transforms_dir = output_dir / 'transforms' / subject / session
    transforms_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = transforms_dir / 'FA_to_T2w_'

    # Use antsRegistrationSyN.sh with affine only
    cmd = [
        'antsRegistrationSyN.sh',
        '-d', '3',
        '-f', str(t2w_file),
        '-m', str(fa_file),
        '-o', str(output_prefix),
        '-n', str(n_cores),
        '-t', 'a'  # Affine only
    ]

    print(f"  Moving: {fa_file.name}")
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
        raise RuntimeError("FA to T2w registration failed")

    # Check outputs
    affine_transform = Path(str(output_prefix) + '0GenericAffine.mat')
    warped_fa = Path(str(output_prefix) + 'Warped.nii.gz')

    if not affine_transform.exists():
        raise RuntimeError(f"Expected transform not found: {affine_transform}")

    print(f"  ✓ Affine transform: {affine_transform.name}")
    if warped_fa.exists():
        print(f"  ✓ Warped FA: {warped_fa.name}")

        # Report which T2w slices have FA coverage
        warped_data = nib.load(warped_fa).get_fdata()
        slices_with_fa = [z for z in range(warped_data.shape[2])
                         if np.sum(warped_data[:, :, z] > 0.1) > 1000]
        if slices_with_fa:
            print(f"  FA covers T2w slices {slices_with_fa[0]}-{slices_with_fa[-1]} ({len(slices_with_fa)} slices)")

    return {
        'affine_transform': affine_transform,
        'warped_fa': warped_fa if warped_fa.exists() else None,
        't2w_file': t2w_file,
        'fa_shape': fa_img.shape,
        't2w_shape': t2w_img.shape,
    }


def run_dwi_preprocessing(
    config: Dict[str, Any],
    subject: str,
    session: str,
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    output_dir: Path,
    transform_registry: TransformRegistry,
    work_dir: Optional[Path] = None,
    use_gpu: bool = True,
    t2w_file: Optional[Path] = None,
    run_registration: bool = True
) -> Dict[str, Any]:
    """
    Run complete DTI/DWI preprocessing workflow.

    This workflow performs:
    1. Image validation and 5D→4D conversion (if needed)
    2. Gradient table validation
    3. GPU-accelerated eddy correction (motion + distortion)
    4. Brain masking from b0 volume
    5. DTI fitting (FA, MD, AD, RD)
    6. Save preprocessed outputs
    7. (Optional) Register FA to T2w for atlas propagation

    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config()
    subject : str
        Subject identifier (e.g., 'sub-Rat207')
    session : str
        Session identifier (e.g., 'ses-p60')
    dwi_file : Path
        Input DWI/DTI NIfTI file (may be 4D or 5D)
    bval_file : Path
        FSL-format bval file
    bvec_file : Path
        FSL-format bvec file (3xN)
    output_dir : Path
        Study root directory (will create derivatives/{subject}/{session}/dwi/)
    transform_registry : TransformRegistry
        Transform registry for saving spatial transforms
    work_dir : Path, optional
        Working directory (defaults to output_dir/work/{subject}/{session}/dwi_preproc)
    use_gpu : bool
        Use GPU-accelerated eddy_cuda (default: True)
    t2w_file : Path, optional
        Preprocessed T2w from anatomical pipeline (required if run_registration=True)
    run_registration : bool
        Whether to run FA→T2w registration (default: True)

    Returns
    -------
    dict
        Dictionary with output file paths and processing info:
        - 'dwi_preproc': Path to preprocessed DWI
        - 'dwi_mask': Path to brain mask
        - 'bval': Path to output bval file
        - 'bvec': Path to eddy-corrected bvec file
        - 'fa': Path to FA map
        - 'md': Path to MD map
        - 'ad': Path to AD map
        - 'rd': Path to RD map
        - 'qc_metrics': Dict with QC metrics
        - 'registration': Dict with registration outputs (if run_registration=True)

    Examples
    --------
    >>> from neurofaune.config import load_config
    >>> from neurofaune.utils.transforms import create_transform_registry
    >>> from pathlib import Path
    >>>
    >>> config = load_config(Path('config.yaml'))
    >>> registry = create_transform_registry(config, 'sub-Rat207', cohort='p60')
    >>>
    >>> results = run_dwi_preprocessing(
    ...     config=config,
    ...     subject='sub-Rat207',
    ...     session='ses-p60',
    ...     dwi_file=Path('dwi.nii.gz'),
    ...     bval_file=Path('dwi.bval'),
    ...     bvec_file=Path('dwi.bvec'),
    ...     output_dir=Path('/study'),
    ...     transform_registry=registry,
    ...     use_gpu=True
    ... )
    """
    print("="*80)
    print(f"DTI/DWI Preprocessing Workflow")
    print(f"Subject: {subject}, Session: {session}")
    print("="*80)

    # Setup directories
    derivatives_dir = output_dir / 'derivatives' / subject / session / 'dwi'
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    qc_dir = get_subject_qc_dir(output_dir, subject, session, 'dwi')

    if work_dir is None:
        work_dir = output_dir / 'work' / subject / session / 'dwi_preproc'
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDirectories:")
    print(f"  Derivatives: {derivatives_dir}")
    print(f"  QC: {qc_dir}")
    print(f"  Work: {work_dir}")

    # Define output files
    dwi_4d_file = work_dir / f'{subject}_{session}_dwi_4d.nii.gz'
    b0_file = work_dir / f'{subject}_{session}_b0.nii.gz'
    brain_mask_file = derivatives_dir / f'{subject}_{session}_desc-brain_mask.nii.gz'

    # Eddy-corrected outputs
    dwi_eddy_file = derivatives_dir / f'{subject}_{session}_desc-preproc_dwi.nii.gz'
    eddy_rotated_bvecs = derivatives_dir / f'{subject}_{session}_desc-preproc_dwi.bvec'

    # DTI outputs
    fa_file = derivatives_dir / f'{subject}_{session}_FA.nii.gz'
    md_file = derivatives_dir / f'{subject}_{session}_MD.nii.gz'
    ad_file = derivatives_dir / f'{subject}_{session}_AD.nii.gz'
    rd_file = derivatives_dir / f'{subject}_{session}_RD.nii.gz'

    # ==========================================================================
    # Step 1: Image validation and 5D→4D conversion
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 1: Image Validation and 5D→4D Conversion")
    print("="*80)

    # Validate input image
    validation = validate_image(dwi_file, modality='dwi', strict=False)
    print_validation_results(validation, name=f"{subject}_{session} DWI")

    if not validation['valid']:
        raise ValueError(f"DWI validation failed: {validation['errors']}")

    # Check if 5D and convert to 4D
    img = nib.load(dwi_file)
    if len(img.shape) == 5:
        print(f"\nDetected 5D data: {img.shape}")
        print("Converting 5D → 4D by averaging across 5th dimension...")
        convert_5d_to_4d(dwi_file, dwi_4d_file, method='mean')
        dwi_input = dwi_4d_file
    elif len(img.shape) == 4:
        print(f"\nData is already 4D: {img.shape}")
        dwi_input = dwi_file
    else:
        raise ValueError(f"Unexpected DWI shape: {img.shape}")

    # ==========================================================================
    # Step 2: Gradient table validation
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 2: Gradient Table Validation")
    print("="*80)

    img_4d = nib.load(dwi_input)
    n_volumes = img_4d.shape[3]

    bvals, bvecs = validate_gradient_table(bval_file, bvec_file, n_volumes)

    # Save validated gradient tables to work directory
    bval_validated = work_dir / 'dwi.bval'
    bvec_validated = work_dir / 'dwi.bvec'
    np.savetxt(bval_validated, bvals.reshape(1, -1), fmt='%.2f')
    np.savetxt(bvec_validated, bvecs, fmt='%.6f')

    # ==========================================================================
    # Step 2.5: Intensity Normalization (for robust brain extraction)
    # ==========================================================================
    # Check if intensity normalization is enabled (default: True)
    norm_config = config.get('diffusion', {}).get('intensity_normalization', {})
    normalize_enabled = norm_config.get('enabled', True)

    if normalize_enabled:
        print("\n" + "="*80)
        print("Step 2.5: Intensity Normalization")
        print("="*80)
        print("\nNormalizing DWI intensity for robust brain extraction...")
        print("  (Different Bruker ParaVision settings can cause vastly different")
        print("   intensity scales - normalization ensures consistent BET performance)")

        target_max = norm_config.get('target_max', 10000.0)
        dwi_normalized_file = work_dir / f'{subject}_{session}_dwi_4d_normalized.nii.gz'

        normalized_file, norm_params = normalize_dwi_intensity(
            dwi_input, dwi_normalized_file, target_max=target_max
        )

        print(f"\n  Original intensity range: [{norm_params['original_min']:.2f}, {norm_params['original_max']:.2f}]")
        print(f"  Percentile range used: [{norm_params['original_p_min']:.2f}, {norm_params['original_p_max']:.2f}]")
        print(f"  Scale factor applied: {norm_params['scale_factor']:.4f}")
        print(f"  Target max intensity: {norm_params['target_max']:.0f}")
        print(f"  Normalized DWI: {normalized_file}")

        # Use normalized data for subsequent steps
        dwi_input = normalized_file
    else:
        print("\n  [INFO] Intensity normalization disabled in config")

    # ==========================================================================
    # Step 3: Extract b0 volume and create brain mask
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 3: b0 Extraction and Brain Masking")
    print("="*80)

    extract_b0_volume(dwi_input, bval_validated, b0_file)

    # Create brain mask using unified skull strip dispatcher
    # DTI has 11 slices (>=10), so will auto-select atropos_bet two-pass method
    print(f"\nCreating brain mask (auto-selects method based on slice count)...")
    b0_brain_file = work_dir / f'{subject}_{session}_b0_brain.nii.gz'
    skull_strip_work_dir = work_dir / 'skull_strip'
    skull_strip_work_dir.mkdir(exist_ok=True)

    cohort = session.split('-')[1] if '-' in session else 'p60'
    _, _, skull_strip_info = skull_strip(
        input_file=b0_file,
        output_file=b0_brain_file,
        mask_file=brain_mask_file,
        work_dir=skull_strip_work_dir,
        method='auto',  # Will select 'atropos_bet' for 11-slice DTI
        cohort=cohort,
    )
    print(f"  Method: {skull_strip_info.get('method', 'unknown')}")
    print(f"  Extraction ratio: {skull_strip_info.get('extraction_ratio', 0):.3f}")

    # ==========================================================================
    # Step 4: GPU-accelerated eddy correction (with slice padding)
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 4: Eddy Correction (Motion + Distortion)")
    print("="*80)

    # Get slice padding config (default: 2 slices on each side)
    n_pad_slices = config.get('diffusion', {}).get('eddy', {}).get('slice_padding', 2)

    # Pad DWI and mask to prevent edge slice loss during eddy
    # This is critical for thin-slice acquisitions where motion correction
    # can cause edge slices to be interpolated from outside the volume
    print(f"\nPadding slices for eddy protection (n_pad={n_pad_slices})...")

    dwi_padded_file = work_dir / f'{subject}_{session}_dwi_4d_padded.nii.gz'
    mask_padded_file = work_dir / f'{subject}_{session}_mask_padded.nii.gz'

    dwi_padded_file, original_n_slices = pad_slices_for_eddy(
        dwi_input, dwi_padded_file, n_pad=n_pad_slices, method='reflect'
    )
    pad_mask_for_eddy(brain_mask_file, mask_padded_file, n_pad=n_pad_slices)

    # Check for eddy_cuda availability
    eddy_cmd = 'eddy_cuda' if use_gpu else 'eddy'

    try:
        subprocess.run([eddy_cmd, '--version'],
                      stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE,
                      check=True)
        print(f"Using {eddy_cmd} for eddy correction")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Warning: {eddy_cmd} not available, falling back to eddy")
        eddy_cmd = 'eddy'

    # Create index file (all volumes use same phase encoding)
    index_file = work_dir / 'index.txt'
    with open(index_file, 'w') as f:
        f.write(' '.join(['1'] * n_volumes))

    # Create acquisition parameters file (assuming PA acquisition)
    acqparams_file = work_dir / 'acqparams.txt'
    with open(acqparams_file, 'w') as f:
        # PA direction, 0.05s total readout time (adjust based on actual data)
        f.write('0 -1 0 0.05\n')

    # Run eddy on PADDED data
    eddy_basename = work_dir / 'eddy_corrected'
    eddy_cmd_full = [
        eddy_cmd,
        f'--imain={dwi_padded_file}',
        f'--mask={mask_padded_file}',
        f'--acqp={acqparams_file}',
        f'--index={index_file}',
        f'--bvecs={bvec_validated}',
        f'--bvals={bval_validated}',
        f'--out={eddy_basename}',
        '--repol',  # Replace outliers
        '--verbose'
    ]

    if use_gpu and eddy_cmd == 'eddy_cuda':
        eddy_cmd_full.append('--very_verbose')

    print(f"\nRunning eddy correction on padded data...")
    print(f"Command: {' '.join(eddy_cmd_full)}")

    result = subprocess.run(eddy_cmd_full,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           text=True)

    if result.returncode != 0:
        print(f"Eddy correction failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("Eddy correction failed")

    print("Eddy correction completed successfully")

    # Crop eddy output back to original slice count
    eddy_output_padded = work_dir / 'eddy_corrected.nii.gz'
    eddy_output_cropped = work_dir / 'eddy_corrected_cropped.nii.gz'

    crop_slices_after_eddy(
        eddy_output_padded, eddy_output_cropped,
        original_n_slices=original_n_slices, n_pad=n_pad_slices
    )

    # Copy cropped eddy output to derivatives
    eddy_bvecs_rotated = work_dir / 'eddy_corrected.eddy_rotated_bvecs'

    import shutil
    shutil.copy(eddy_output_cropped, dwi_eddy_file)

    # Fix eddy rotated bvecs - replace NaN with 0 (occurs for b0 volumes)
    bvecs_rotated = np.loadtxt(eddy_bvecs_rotated)
    if np.any(np.isnan(bvecs_rotated)):
        print("  Fixing NaN values in rotated bvecs (b0 volumes)...")
        bvecs_rotated = np.nan_to_num(bvecs_rotated, nan=0.0)
    np.savetxt(eddy_rotated_bvecs, bvecs_rotated, fmt='%.10g')

    # Copy bvals (unchanged by eddy)
    bval_output = derivatives_dir / f'{subject}_{session}_desc-preproc_dwi.bval'
    shutil.copy(bval_validated, bval_output)

    print(f"\nEddy-corrected DWI saved to: {dwi_eddy_file}")
    print(f"Rotated bvecs saved to: {eddy_rotated_bvecs}")

    # ==========================================================================
    # Step 5: DTI fitting
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 5: DTI Fitting (FA, MD, AD, RD)")
    print("="*80)

    fit_dti(
        dwi_file=dwi_eddy_file,
        mask_file=brain_mask_file,
        bval_file=bval_output,
        bvec_file=eddy_rotated_bvecs,
        output_prefix=derivatives_dir / f'{subject}_{session}'
    )

    print(f"\nDTI maps created:")
    print(f"  FA: {fa_file}")
    print(f"  MD: {md_file}")
    print(f"  AD: {ad_file}")
    print(f"  RD: {rd_file}")

    # NOTE: SIGMA registration removed - will be done separately in template registration module

    # ==========================================================================
    # Step 6: Quality control
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 6: Quality Control")
    print("="*80)

    qc_results = {}

    # Eddy QC (motion, signal quality)
    eddy_params_file = work_dir / 'eddy_corrected.eddy_parameters'
    if eddy_params_file.exists():
        eddy_qc = generate_eddy_qc_report(
            subject=subject,
            session=session,
            dwi_preproc=dwi_eddy_file,
            eddy_params=eddy_params_file,
            output_dir=qc_dir
        )
        qc_results['eddy_qc'] = eddy_qc
    else:
        print("Warning: Eddy parameters file not found, skipping motion QC")

    # DTI metrics QC
    dti_qc = generate_dti_qc_report(
        subject=subject,
        session=session,
        fa_file=fa_file,
        md_file=md_file,
        ad_file=ad_file,
        rd_file=rd_file,
        brain_mask=brain_mask_file,
        output_dir=qc_dir
    )
    qc_results['dti_qc'] = dti_qc

    # Basic data quality metrics
    qc_metrics = check_dwi_data_quality(dwi_eddy_file, brain_mask_file)
    qc_json = qc_dir / f'{subject}_{session}_dwi_basic_qc.json'
    with open(qc_json, 'w') as f:
        json.dump(qc_metrics, f, indent=2)

    print(f"\n✓ QC reports generated:")
    if 'eddy_qc' in qc_results:
        print(f"  - Eddy/Motion QC: {qc_results['eddy_qc']['html_report']}")
    print(f"  - DTI Metrics QC: {qc_results['dti_qc']['html_report']}")
    print(f"  - Basic metrics: {qc_json}")

    # ==========================================================================
    # Step 7: FA to T2w Registration (optional)
    # ==========================================================================
    registration_results = None

    if run_registration:
        if t2w_file is None:
            print("\n⚠ Registration requested but no T2w file provided - skipping")
        elif not t2w_file.exists():
            print(f"\n⚠ T2w file not found: {t2w_file} - skipping registration")
        else:
            print("\n" + "="*80)
            print("Step 7: FA to T2w Registration")
            print("="*80)

            try:
                registration_results = register_fa_to_t2w(
                    fa_file=fa_file,
                    t2w_file=t2w_file,
                    output_dir=output_dir,
                    subject=subject,
                    session=session,
                    work_dir=work_dir,
                    n_cores=4
                )

                # Save registration metadata to JSON
                reg_metadata_file = derivatives_dir / f'{subject}_{session}_FA_to_T2w_registration.json'
                reg_metadata = {
                    'fa_file': str(fa_file),
                    't2w_file': str(registration_results['t2w_file']),
                    'affine_transform': str(registration_results['affine_transform']),
                    'warped_fa': str(registration_results['warped_fa']) if registration_results.get('warped_fa') else None,
                    'fa_shape': list(registration_results['fa_shape']),
                    't2w_shape': list(registration_results['t2w_shape']),
                }
                with open(reg_metadata_file, 'w') as f:
                    json.dump(reg_metadata, f, indent=2)

                print(f"\n✓ Registration complete:")
                print(f"  - Transform: {registration_results['affine_transform']}")
                print(f"  - Metadata: {reg_metadata_file}")

            except Exception as e:
                print(f"\n✗ Registration failed: {e}")
                print("  Continuing without registration...")

    # ==========================================================================
    # Workflow complete
    # ==========================================================================
    print("\n" + "="*80)
    print("DTI/DWI Preprocessing Complete!")
    print("="*80)
    print(f"\nPreprocessed DWI: {dwi_eddy_file}")
    print(f"Brain mask: {brain_mask_file}")
    print(f"FA map: {fa_file}")
    print(f"MD map: {md_file}")
    print(f"AD map: {ad_file}")
    print(f"RD map: {rd_file}")
    if registration_results is not None:
        print(f"FA→T2w transform: {registration_results['affine_transform']}")
    else:
        print("\nNOTE: FA→T2w registration was skipped. Run with t2w_file to enable.")
    print("="*80 + "\n")

    results = {
        'dwi_preproc': dwi_eddy_file,
        'dwi_mask': brain_mask_file,
        'bval': bval_output,
        'bvec': eddy_rotated_bvecs,
        'fa': fa_file,
        'md': md_file,
        'ad': ad_file,
        'rd': rd_file,
        'qc_results': qc_results,
        'qc_metrics': qc_metrics
    }

    if registration_results is not None:
        results['registration'] = registration_results

    return results


def fit_dti(
    dwi_file: Path,
    mask_file: Path,
    bval_file: Path,
    bvec_file: Path,
    output_prefix: Path
) -> Tuple[Path, Path, Path, Path]:
    """
    Fit DTI model and compute FA, MD, AD, RD maps using FSL's dtifit.

    Parameters
    ----------
    dwi_file : Path
        Preprocessed DWI file
    mask_file : Path
        Brain mask
    bval_file : Path
        b-values
    bvec_file : Path
        b-vectors
    output_prefix : Path
        Output prefix (will create {prefix}_FA.nii.gz, etc.)

    Returns
    -------
    tuple
        Paths to (FA, MD, AD, RD) files
    """
    print("\nFitting DTI model with FSL dtifit...")

    # Use FSL's dtifit
    cmd = [
        'dtifit',
        f'--data={dwi_file}',
        f'--mask={mask_file}',
        f'--bvecs={bvec_file}',
        f'--bvals={bval_file}',
        f'--out={output_prefix}',
        '--sse',  # Save sum of squared errors
        '--save_tensor'  # Save tensor
    ]

    print(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(cmd,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           text=True)

    if result.returncode != 0:
        print(f"DTI fitting failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("FSL dtifit failed")

    print("  DTI fitting completed successfully")

    # Define output file paths (dtifit naming convention)
    fa_file = Path(str(output_prefix) + '_FA.nii.gz')
    md_file = Path(str(output_prefix) + '_MD.nii.gz')
    l1_file = Path(str(output_prefix) + '_L1.nii.gz')
    l2_file = Path(str(output_prefix) + '_L2.nii.gz')
    l3_file = Path(str(output_prefix) + '_L3.nii.gz')

    # Calculate AD and RD from eigenvalues
    # AD = L1 (axial diffusivity)
    # RD = (L2 + L3) / 2 (radial diffusivity)
    print("  Computing AD and RD from eigenvalues...")

    l1_img = nib.load(l1_file)
    l1_data = l1_img.get_fdata()

    l2_img = nib.load(l2_file)
    l2_data = l2_img.get_fdata()

    l3_img = nib.load(l3_file)
    l3_data = l3_img.get_fdata()

    # AD = L1
    ad_data = l1_data
    ad_file = Path(str(output_prefix) + '_AD.nii.gz')
    nib.save(nib.Nifti1Image(ad_data, l1_img.affine, l1_img.header), ad_file)

    # RD = (L2 + L3) / 2
    rd_data = (l2_data + l3_data) / 2.0
    rd_file = Path(str(output_prefix) + '_RD.nii.gz')
    nib.save(nib.Nifti1Image(rd_data, l1_img.affine, l1_img.header), rd_file)

    # Load FA and MD to check ranges
    fa_img = nib.load(fa_file)
    fa_data = fa_img.get_fdata()

    md_img = nib.load(md_file)
    md_data = md_img.get_fdata()

    print(f"  FA range: [{fa_data.min():.3f}, {fa_data.max():.3f}]")
    print(f"  MD range: [{md_data.min():.6f}, {md_data.max():.6f}]")
    print(f"  AD range: [{ad_data.min():.6f}, {ad_data.max():.6f}]")
    print(f"  RD range: [{rd_data.min():.6f}, {rd_data.max():.6f}]")

    return fa_file, md_file, ad_file, rd_file


def register_to_atlas_slices(
    moving_image: Path,
    fixed_image: Path,
    output_prefix: Path,
    output_warped: Path
) -> Path:
    """
    Register moving image to fixed atlas slices using ANTs SyN.

    Parameters
    ----------
    moving_image : Path
        Image to register (e.g., FA map)
    fixed_image : Path
        Fixed atlas image (slice-specific)
    output_prefix : Path
        Output prefix for transform files
    output_warped : Path
        Output path for warped image

    Returns
    -------
    Path
        Path to composite transform
    """
    print(f"\n  Moving: {moving_image.name}")
    print(f"  Fixed: {fixed_image.name}")

    # ANTs registration command (SyN)
    cmd = [
        'antsRegistration',
        '--dimensionality', '3',
        '--float', '1',
        '--output', f'[{output_prefix}_,{output_warped}]',
        '--interpolation', 'Linear',
        '--winsorize-image-intensities', '[0.005,0.995]',
        '--use-histogram-matching', '0',
        # Initial moving transform (center of mass)
        '--initial-moving-transform', f'[{fixed_image},{moving_image},1]',
        # Rigid registration
        '--transform', 'Rigid[0.1]',
        '--metric', f'MI[{fixed_image},{moving_image},1,32,Regular,0.25]',
        '--convergence', '[1000x500x250x100,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox',
        # Affine registration
        '--transform', 'Affine[0.1]',
        '--metric', f'MI[{fixed_image},{moving_image},1,32,Regular,0.25]',
        '--convergence', '[1000x500x250x100,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox',
        # SyN deformable registration
        '--transform', 'SyN[0.1,3,0]',
        '--metric', f'CC[{fixed_image},{moving_image},1,4]',
        '--convergence', '[100x70x50x20,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox'
    ]

    print("\n  Running ANTs registration...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f"Registration failed!")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("ANTs registration failed")

    print("  Registration completed successfully")

    composite_transform = Path(str(output_prefix) + '_Composite.h5')
    return composite_transform
