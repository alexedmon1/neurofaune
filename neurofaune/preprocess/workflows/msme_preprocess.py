"""
MSME (Multi-Slice Multi-Echo) T2 mapping and MWF calculation workflow.

This module provides preprocessing for multi-echo T2-weighted data,
including T2 fitting and Myelin Water Fraction (MWF) calculation using NNLS.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
from scipy.optimize import curve_fit, nnls
import json

from neurofaune.preprocess.utils.validation import validate_image, print_validation_results
from neurofaune.utils.transforms import TransformRegistry


def run_msme_preprocessing(
    config: Dict[str, Any],
    subject: str,
    session: str,
    msme_file: Path,
    output_dir: Path,
    transform_registry: TransformRegistry,
    te_values: Optional[np.ndarray] = None,
    work_dir: Optional[Path] = None
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
        Input MSME 4D NIfTI file (echoes in 4th dimension)
    output_dir : Path
        Study root directory
    transform_registry : TransformRegistry
        Transform registry
    te_values : np.ndarray, optional
        Echo times in ms (if None, assumes 10-320ms in 10ms steps)
    work_dir : Path, optional
        Working directory

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

    qc_dir = output_dir / 'qc' / subject / session / 'msme'
    qc_dir.mkdir(parents=True, exist_ok=True)

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

    # ==========================================================================
    # Step 1: Skull stripping
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 1: Skull Stripping")
    print("="*80)

    # Load MSME data
    img = nib.load(msme_file)
    data = img.get_fdata()

    print(f"MSME shape: {data.shape}")

    if len(data.shape) != 4:
        raise ValueError(f"Expected 4D MSME data, got shape: {data.shape}")

    # Use first echo for skull stripping
    first_echo = data[..., 0]
    first_echo_file = work_dir / f'{subject}_{session}_echo1.nii.gz'
    nib.save(nib.Nifti1Image(first_echo, img.affine, img.header), first_echo_file)

    # BET skull stripping
    bet_frac = config.get('msme', {}).get('bet', {}).get('frac', 0.3)
    print(f"\nRunning BET (frac={bet_frac})...")
    _run_bet(first_echo_file, brain_mask_file, frac=bet_frac)

    # Apply mask to all echoes
    mask_img = nib.load(brain_mask_file)
    mask = mask_img.get_fdata() > 0

    data_masked = data.copy()
    for echo_idx in range(data.shape[3]):
        data_masked[..., echo_idx] = data[..., echo_idx] * mask

    nib.save(nib.Nifti1Image(data_masked, img.affine, img.header), msme_masked_file)
    print(f"Masked MSME saved to: {msme_masked_file}")

    # ==========================================================================
    # Step 2: T2 Fitting and MWF Calculation
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 2: T2 Fitting and MWF Calculation (NNLS)")
    print("="*80)

    mwf_map, iwf_map, csf_map, t2_map, sample_data = calculate_mwf_nnls(
        data_masked,
        mask,
        te_values
    )

    # Save output maps
    nib.save(nib.Nifti1Image(mwf_map, img.affine, img.header), mwf_file)
    nib.save(nib.Nifti1Image(iwf_map, img.affine, img.header), iwf_file)
    nib.save(nib.Nifti1Image(csf_map, img.affine, img.header), csf_file)
    nib.save(nib.Nifti1Image(t2_map, img.affine, img.header), t2_file)

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
        output_dir=qc_dir
    )

    print(f"\n✓ MSME QC report: {qc_results['html_report']}")

    # ==========================================================================
    # Workflow complete
    # ==========================================================================
    print("\n" + "="*80)
    print("MSME Preprocessing Complete!")
    print("="*80 + "\n")

    return {
        'mwf': mwf_file,
        'iwf': iwf_file,
        'csf': csf_file,
        't2': t2_file,
        'brain_mask': brain_mask_file,
        'qc_results': qc_results
    }


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


def calculate_mwf_nnls(
    data: np.ndarray,
    mask: np.ndarray,
    te_values: np.ndarray,
    lambda_reg: float = 0.5
) -> tuple:
    """
    Calculate MWF using Non-Negative Least Squares (NNLS).

    Parameters
    ----------
    data : np.ndarray
        4D MSME data (x, y, z, echoes)
    mask : np.ndarray
        3D brain mask
    te_values : np.ndarray
        Echo times in ms
    lambda_reg : float
        Regularization parameter

    Returns
    -------
    tuple
        (mwf_map, iwf_map, csf_map, t2_map, sample_data)
        sample_data contains representative T2 curves and NNLS spectra for QC
    """
    print("\nCalculating MWF using NNLS...")
    print(f"  Lambda regularization: {lambda_reg}")

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

    # T2 distribution (log-spaced from 10 to 2000 ms)
    t2_dist = np.geomspace(10, 2000, num=120)
    sample_data['t2_dist'] = t2_dist

    # Define water compartments:
    # Myelin water: T2 < 25ms (indices 0:25)
    # Intra/extra-cellular: 25-40ms (indices 25:40)
    # CSF: 41-2000ms (indices 41:120)
    mw_cutoff = np.where(t2_dist < 25)[0][-1] + 1
    iw_cutoff = np.where(t2_dist < 40)[0][-1] + 1

    # Create design matrix A for NNLS
    # A[i, j] = exp(-TE[i] / T2[j])
    A = np.zeros((n_echoes, len(t2_dist)))
    for i, te in enumerate(te_values):
        A[i, :] = np.exp(-te / t2_dist)

    # Regularization matrix
    n_t2 = len(t2_dist)
    A_reg = np.concatenate([A, np.sqrt(lambda_reg) * np.eye(n_t2)])

    # Get voxel indices within mask
    voxel_indices = np.argwhere(mask)
    n_voxels = len(voxel_indices)

    print(f"  Processing {n_voxels} voxels...")

    # Progress reporting
    report_every = max(n_voxels // 10, 1)

    for idx, (x, y, z) in enumerate(voxel_indices):
        if idx % report_every == 0:
            progress = 100 * idx / n_voxels
            print(f"    Progress: {progress:.1f}%")

        # Get signal for this voxel
        signal = data[x, y, z, :]

        # Skip if no signal
        if np.sum(signal) == 0:
            continue

        # Regularized observation vector
        b_reg = np.concatenate([signal, np.zeros(n_t2)])

        # Solve NNLS
        try:
            amplitudes, residual = nnls(A_reg, b_reg)

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

    # Calculate summary statistics
    mwf_masked = mwf_map[mask]
    print(f"\n  MWF statistics (within brain):")
    print(f"    Mean: {np.mean(mwf_masked):.3f}")
    print(f"    Std: {np.std(mwf_masked):.3f}")
    print(f"    Range: [{np.min(mwf_masked):.3f}, {np.max(mwf_masked):.3f}]")

    print(f"\n  Captured {len(sample_data['voxels'])} sample voxels for QC visualization")

    return mwf_map, iwf_map, csf_map, t2_map, sample_data
