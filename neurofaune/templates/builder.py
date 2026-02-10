"""
Template building utilities for neurofaune.

This module provides functions for building age-specific templates from
preprocessed multi-modal MRI data.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import nibabel as nib
import numpy as np
import pandas as pd


def select_subjects_for_template(
    derivatives_dir: Path,
    cohort: str,
    modality: str,
    qc_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    top_percent: float = 1/3,
    min_subjects: int = 10
) -> List[str]:
    """
    Select best subjects for template building based on QC metrics.

    Parameters
    ----------
    derivatives_dir : Path
        Path to derivatives directory
    cohort : str
        Age cohort ('p30', 'p60', 'p90')
    modality : str
        Modality to select for ('anat', 'dwi', 'func')
    qc_metrics : dict, optional
        Dictionary of {subject_id: {metric: value}}
        If None, will attempt to load from QC directory
    top_percent : float
        Fraction of subjects to select (default: 0.25 = top 25%)
    min_subjects : int
        Minimum number of subjects required (default: 10)

    Returns
    -------
    list
        List of subject IDs selected for template building

    Examples
    --------
    >>> subjects = select_subjects_for_template(
    ...     derivatives_dir=Path('/study/derivatives'),
    ...     cohort='p60',
    ...     modality='anat',
    ...     top_percent=1/3
    ... )
    """
    session = f'ses-{cohort}'

    # Find all subjects with this session and modality
    subject_dirs = sorted(derivatives_dir.glob(f'sub-*/'))
    candidates = []

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        session_dir = subject_dir / session / modality

        if not session_dir.exists():
            continue

        # Check if preprocessed data exists
        if modality == 'anat':
            preproc_file = session_dir / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
        elif modality == 'dwi':
            preproc_file = session_dir / f'{subject}_{session}_FA.nii.gz'
        elif modality == 'func':
            preproc_file = session_dir / f'{subject}_{session}_desc-preproc_bold.nii.gz'
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if preproc_file.exists():
            candidates.append(subject)

    print(f"Found {len(candidates)} candidate subjects for {cohort} {modality}")

    if len(candidates) < min_subjects:
        raise ValueError(
            f"Not enough subjects found ({len(candidates)} < {min_subjects}). "
            f"Need at least {min_subjects} subjects for template building."
        )

    # If QC metrics provided, rank subjects
    if qc_metrics:
        # Calculate composite QC score for each subject
        scores = {}
        for subject in candidates:
            if subject in qc_metrics:
                metrics = qc_metrics[subject]
                # Simple scoring: average all metrics (assuming higher is better)
                # TODO: Make this more sophisticated with metric-specific weights
                scores[subject] = np.mean(list(metrics.values()))
            else:
                scores[subject] = 0.0

        # Sort by score descending
        ranked = sorted(candidates, key=lambda s: scores[s], reverse=True)

        # Select top percent
        n_select = max(min_subjects, int(len(ranked) * top_percent))
        selected = ranked[:n_select]

        print(f"Selected top {n_select} subjects ({top_percent*100:.0f}%) based on QC")
        for i, subject in enumerate(selected[:5]):
            print(f"  {i+1}. {subject} (score: {scores[subject]:.3f})")
        if len(selected) > 5:
            print(f"  ... and {len(selected)-5} more")
    else:
        # No QC metrics, select top N randomly
        n_select = max(min_subjects, int(len(candidates) * top_percent))
        selected = candidates[:n_select]
        print(f"No QC metrics provided. Using first {n_select} subjects.")
        print("⚠ WARNING: For best results, provide QC metrics for subject selection.")

    return selected


def extract_mean_bold(
    bold_file: Path,
    output_file: Path,
    method: str = 'median'
) -> Path:
    """
    Extract mean or median timepoint from 4D BOLD data.

    Parameters
    ----------
    bold_file : Path
        Input 4D BOLD file (time series)
    output_file : Path
        Output 3D file (single timepoint)
    method : str
        Aggregation method ('mean' or 'median')

    Returns
    -------
    Path
        Path to output file

    Examples
    --------
    >>> mean_bold = extract_mean_bold(
    ...     bold_file=Path('bold.nii.gz'),
    ...     output_file=Path('bold_mean.nii.gz'),
    ...     method='median'
    ... )
    """
    print(f"Extracting {method} BOLD timepoint from {bold_file.name}...")

    img = nib.load(bold_file)
    data = img.get_fdata()

    if len(data.shape) != 4:
        raise ValueError(f"Expected 4D BOLD data, got shape {data.shape}")

    # Extract mean or median across time
    if method == 'mean':
        mean_data = data.mean(axis=3)
    elif method == 'median':
        mean_data = np.median(data, axis=3)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mean' or 'median'.")

    # Save as 3D image
    mean_img = nib.Nifti1Image(mean_data, img.affine, img.header)
    nib.save(mean_img, output_file)

    print(f"  Saved to {output_file}")
    return output_file


def build_template(
    input_files: List[Path],
    output_prefix: Path,
    dimension: int = 3,
    n_iterations: int = 4,
    gradient_step: float = 0.2,
    n_cores: int = 8,
    use_float: bool = True
) -> Dict[str, Path]:
    """
    Build template using ANTs multivariate template construction.

    This is a wrapper around antsMultivariateTemplateConstruction.sh that
    handles the complexity of the ANTs script.

    Parameters
    ----------
    input_files : list of Path
        List of preprocessed images to use for template building
    output_prefix : Path
        Output prefix for template files (e.g., 'tpl-BPARatp60_T2w_')
    dimension : int
        Image dimension (default: 3)
    n_iterations : int
        Number of template building iterations (default: 4)
    gradient_step : float
        Gradient step size (default: 0.2)
    n_cores : int
        Number of CPU cores to use (default: 8)
    use_float : bool
        Use float precision (faster, default: True)

    Returns
    -------
    dict
        Dictionary with paths to template outputs:
        - 'template': Final template image
        - 'work_dir': Working directory with intermediate files
        - 'subjects': List of input files used

    Examples
    --------
    >>> results = build_template(
    ...     input_files=[Path('sub-001_T2w.nii.gz'), ...],
    ...     output_prefix=Path('/templates/p60/tpl-BPARatp60_T2w_'),
    ...     n_iterations=4,
    ...     n_cores=8
    ... )
    >>> template = results['template']
    """
    print("="*80)
    print("Building Template with ANTs")
    print("="*80)
    print(f"Input files: {len(input_files)}")
    print(f"Output prefix: {output_prefix}")
    print(f"Iterations: {n_iterations}")
    print(f"Cores: {n_cores}")
    print()

    # Validate inputs
    if len(input_files) < 3:
        raise ValueError(f"Need at least 3 subjects for template building, got {len(input_files)}")

    for f in input_files:
        if not f.exists():
            raise FileNotFoundError(f"Input file not found: {f}")

    # Create output directory
    output_dir = output_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        'antsMultivariateTemplateConstruction.sh',
        '-d', str(dimension),
        '-o', str(output_prefix),
        '-i', str(n_iterations),
        '-g', str(gradient_step),
        '-c', '2',  # Use parallel (2) for SyN
        '-j', str(n_cores),
        '-n', '0',  # Use all available cores for parallel
    ]

    if use_float:
        cmd.extend(['-r', '1'])  # Use float precision

    # Add input files
    cmd.extend([str(f) for f in input_files])

    print("Command:")
    print(' '.join(cmd))
    print()

    # Run template construction
    print("Running ANTs template construction (this may take a while)...")
    print("="*80)

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: Template construction failed!")
        print(result.stdout)
        raise RuntimeError("ANTs template construction failed")

    print(result.stdout)
    print("="*80)
    print("Template construction complete!")
    print()

    # Find output template file
    # ANTs creates: <prefix>template0.nii.gz
    template_file = Path(str(output_prefix) + 'template0.nii.gz')

    if not template_file.exists():
        raise FileNotFoundError(f"Expected template file not found: {template_file}")

    print(f"Template saved to: {template_file}")

    return {
        'template': template_file,
        'work_dir': output_dir,
        'subjects': input_files
    }


def register_template_to_sigma(
    template_file: Path,
    sigma_template: Path,
    output_prefix: Path,
    n_cores: int = 8
) -> Dict[str, Path]:
    """
    Register study template to SIGMA atlas (T2w only).

    This enables SIGMA parcellation access for the study template space.

    Parameters
    ----------
    template_file : Path
        Study template to register (e.g., tpl-BPARatp60_T2w.nii.gz)
    sigma_template : Path
        SIGMA atlas template (fixed image)
    output_prefix : Path
        Output prefix for transform files
    n_cores : int
        Number of CPU cores to use

    Returns
    -------
    dict
        Dictionary with paths to registration outputs:
        - 'affine_transform': Affine transform (.mat)
        - 'warp_transform': Deformation field (Template → SIGMA)
        - 'inverse_warp': Inverse deformation field (SIGMA → Template)
        - 'warped': Template in SIGMA space

    Examples
    --------
    >>> results = register_template_to_sigma(
    ...     template_file=Path('tpl-BPARatp60_T2w.nii.gz'),
    ...     sigma_template=Path('/atlas/SIGMA/SIGMA_InVivo_Brain.nii.gz'),
    ...     output_prefix=Path('tpl-to-SIGMA_'),
    ...     n_cores=8
    ... )
    """
    print("="*80)
    print("Registering Template to SIGMA Atlas")
    print("="*80)
    print(f"Template: {template_file.name}")
    print(f"SIGMA: {sigma_template.name}")
    print(f"Output: {output_prefix}")
    print()

    # Validate inputs
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")
    if not sigma_template.exists():
        raise FileNotFoundError(f"SIGMA template not found: {sigma_template}")

    # Create output directory
    output_dir = output_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use antsRegistrationSyN.sh for quick registration
    cmd = [
        'antsRegistrationSyN.sh',
        '-d', '3',
        '-f', str(sigma_template),  # Fixed (SIGMA)
        '-m', str(template_file),    # Moving (study template)
        '-o', str(output_prefix),
        '-n', str(n_cores),
        '-t', 's'  # SyN transform
    ]

    print("Command:")
    print(' '.join(cmd))
    print()

    print("Running ANTs registration...")
    print("="*80)

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: Registration failed!")
        print(result.stdout)
        raise RuntimeError("ANTs registration failed")

    print(result.stdout)
    print("="*80)
    print("Registration complete!")
    print()

    # Expected outputs from antsRegistrationSyN.sh
    # ANTs creates separate transform files by default:
    # - 0GenericAffine.mat: affine transform
    # - 1Warp.nii.gz: deformation field
    # - 1InverseWarp.nii.gz: inverse deformation field
    # - Warped.nii.gz: warped template
    affine_transform = Path(str(output_prefix) + '0GenericAffine.mat')
    warp_transform = Path(str(output_prefix) + '1Warp.nii.gz')
    inverse_warp = Path(str(output_prefix) + '1InverseWarp.nii.gz')
    warped = Path(str(output_prefix) + 'Warped.nii.gz')

    # Verify transforms exist
    if not affine_transform.exists():
        raise FileNotFoundError(f"Expected affine transform not found: {affine_transform}")
    if not warp_transform.exists():
        raise FileNotFoundError(f"Expected warp transform not found: {warp_transform}")

    print(f"Affine transform: {affine_transform}")
    print(f"Warp transform: {warp_transform}")
    print(f"Inverse warp: {inverse_warp}")
    print(f"Warped template: {warped}")
    print()

    return {
        'affine_transform': affine_transform,
        'warp_transform': warp_transform,
        'inverse_warp': inverse_warp,
        'warped': warped
    }


def save_template_metadata(
    template_dir: Path,
    cohort: str,
    modality: str,
    subjects_used: List[str],
    qc_metrics: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save metadata about template construction.

    Parameters
    ----------
    template_dir : Path
        Directory where template is saved
    cohort : str
        Age cohort ('p30', 'p60', 'p90')
    modality : str
        Modality ('anat', 'dwi', 'func')
    subjects_used : list
        List of subject IDs used for template
    qc_metrics : dict, optional
        QC metrics for subjects used

    Returns
    -------
    Path
        Path to metadata JSON file
    """
    metadata = {
        'cohort': cohort,
        'modality': modality,
        'n_subjects': len(subjects_used),
        'subjects': subjects_used,
        'qc_metrics': qc_metrics
    }

    metadata_file = template_dir / f'tpl-metadata_{cohort}_{modality}.json'

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Also save simple text file with subject list
    subjects_file = template_dir / f'subjects_used_{cohort}_{modality}.txt'
    with open(subjects_file, 'w') as f:
        f.write('\n'.join(subjects_used))

    print(f"Metadata saved to: {metadata_file}")
    print(f"Subject list saved to: {subjects_file}")

    return metadata_file
