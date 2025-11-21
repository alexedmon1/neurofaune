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
    create_brain_mask_from_b0,
    check_dwi_data_quality
)
from neurofaune.preprocess.utils.validation import validate_image, print_validation_results
from neurofaune.preprocess.utils.orientation import (
    match_orientation_to_reference,
    save_orientation_metadata,
    print_orientation_info
)
from neurofaune.atlas.manager import AtlasManager
from neurofaune.utils.transforms import TransformRegistry


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
    use_gpu: bool = True
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

    NOTE: This workflow NO LONGER performs registration to SIGMA atlas.
    Registration will be done separately:
    - First to age-matched study FA template
    - Within-subject T2w ↔ FA registration (for label propagation)
    - T2w template → SIGMA (for parcellation access)

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

    qc_dir = output_dir / 'qc' / subject / session / 'dwi'
    qc_dir.mkdir(parents=True, exist_ok=True)

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
    # Step 3: Extract b0 volume and create brain mask
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 3: b0 Extraction and Brain Masking")
    print("="*80)

    extract_b0_volume(dwi_input, bval_validated, b0_file)

    # Get BET parameters from config
    bet_frac = config.get('diffusion', {}).get('bet', {}).get('frac', 0.3)
    print(f"\nCreating brain mask with BET (frac={bet_frac})...")
    create_brain_mask_from_b0(b0_file, brain_mask_file, frac=bet_frac)

    # ==========================================================================
    # Step 4: GPU-accelerated eddy correction
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 4: Eddy Correction (Motion + Distortion)")
    print("="*80)

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

    # Run eddy
    eddy_basename = work_dir / 'eddy_corrected'
    eddy_cmd_full = [
        eddy_cmd,
        f'--imain={dwi_input}',
        f'--mask={brain_mask_file}',
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

    print(f"\nRunning eddy correction...")
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

    # Copy eddy outputs to derivatives
    eddy_output = work_dir / 'eddy_corrected.nii.gz'
    eddy_bvecs_rotated = work_dir / 'eddy_corrected.eddy_rotated_bvecs'

    import shutil
    shutil.copy(eddy_output, dwi_eddy_file)
    shutil.copy(eddy_bvecs_rotated, eddy_rotated_bvecs)

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

    qc_metrics = check_dwi_data_quality(dwi_eddy_file, brain_mask_file)

    # Save QC metrics
    qc_json = qc_dir / f'{subject}_{session}_dwi_qc.json'
    with open(qc_json, 'w') as f:
        json.dump(qc_metrics, f, indent=2)

    print(f"\nQC metrics saved to: {qc_json}")
    print(f"\nQC Summary:")
    print(f"  Shape: {qc_metrics['shape']}")
    print(f"  Voxel size: {qc_metrics['voxel_size']}")
    print(f"  Number of volumes: {qc_metrics['n_volumes']}")
    if 'snr_estimate' in qc_metrics:
        print(f"  SNR estimate: {qc_metrics['snr_estimate']:.2f}")

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
    print("\nNOTE: Registration to study FA template and SIGMA will be done separately.")
    print("="*80 + "\n")

    return {
        'dwi_preproc': dwi_eddy_file,
        'dwi_mask': brain_mask_file,
        'bval': bval_output,
        'bvec': eddy_rotated_bvecs,
        'fa': fa_file,
        'md': md_file,
        'ad': ad_file,
        'rd': rd_file,
        'qc_metrics': qc_metrics
    }


def fit_dti(
    dwi_file: Path,
    mask_file: Path,
    bval_file: Path,
    bvec_file: Path,
    output_prefix: Path
) -> Tuple[Path, Path, Path, Path]:
    """
    Fit DTI model and compute FA, MD, AD, RD maps using dipy.

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
    print("\nFitting DTI model...")

    # Import dipy modules
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    # Load data
    img = nib.load(dwi_file)
    data = img.get_fdata()

    mask_img = nib.load(mask_file)
    mask = mask_img.get_fdata() > 0

    # Load gradient table
    bvals, bvecs = read_bvals_bvecs(str(bval_file), str(bvec_file))
    gtab = gradient_table(bvals, bvecs)

    print(f"  Data shape: {data.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Gradient table: {len(bvals)} volumes")

    # Fit DTI model
    print("  Fitting tensor model...")
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=mask)

    # Compute scalar maps
    print("  Computing scalar maps...")
    fa = tenfit.fa
    md = tenfit.md
    ad = tenfit.ad
    rd = tenfit.rd

    # Replace NaN with 0
    fa[np.isnan(fa)] = 0
    md[np.isnan(md)] = 0
    ad[np.isnan(ad)] = 0
    rd[np.isnan(rd)] = 0

    # Save maps
    fa_file = Path(str(output_prefix) + '_FA.nii.gz')
    md_file = Path(str(output_prefix) + '_MD.nii.gz')
    ad_file = Path(str(output_prefix) + '_AD.nii.gz')
    rd_file = Path(str(output_prefix) + '_RD.nii.gz')

    nib.save(nib.Nifti1Image(fa, img.affine, img.header), fa_file)
    nib.save(nib.Nifti1Image(md, img.affine, img.header), md_file)
    nib.save(nib.Nifti1Image(ad, img.affine, img.header), ad_file)
    nib.save(nib.Nifti1Image(rd, img.affine, img.header), rd_file)

    print(f"  FA range: [{fa.min():.3f}, {fa.max():.3f}]")
    print(f"  MD range: [{md.min():.6f}, {md.max():.6f}]")

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
