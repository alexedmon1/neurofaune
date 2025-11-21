#!/usr/bin/env python3
"""
Bruker to BIDS conversion utilities.

Handles:
- Discovery of Bruker sessions across different cohort structures
- Bruker method classification (T2w, DTI, fMRI, spectroscopy, MSME, MTR)
- Conversion to NIfTI format
- Organization into BIDS-like structure
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from brukerapi.dataset import Dataset
import nibabel as nib
import numpy as np
from datetime import datetime

# Import voxel size correction utilities
from neurofaune.utils.fix_bruker_voxel_sizes import (
    parse_bruker_method,
    update_nifti_header
)

logger = logging.getLogger(__name__)


# Bruker method to BIDS modality mapping
# Note: Using separate folders for each scan type rather than grouping all in 'anat'
BRUKER_METHOD_MAP = {
    'Bruker:RARE': 'anat',        # T2-weighted anatomical
    'Bruker:DtiEpi': 'dwi',        # Diffusion weighted imaging
    'Bruker:EPI': 'func',          # Functional MRI
    'Bruker:PRESS': 'spec',        # MR Spectroscopy (skip - will use fsl-mrs in Phase 6)
    'Bruker:MSME': 'msme',         # Multi-slice multi-echo (T2 mapping) - separate folder
    'User:mt_Array_RARE': 'mtr',   # Magnetization transfer - separate folder
    'User:mt_Array_FLASH': 'mtr',  # Magnetization transfer - separate folder
    'Bruker:FLASH': 'flash',       # Fast Low Angle Shot - separate folder
    'Bruker:FieldMap': 'fmap',     # Field map for distortion correction
}

# BIDS suffix mapping
METHOD_SUFFIX_MAP = {
    'Bruker:RARE': 'T2w',
    'Bruker:DtiEpi': 'dwi',
    'Bruker:EPI': 'bold',
    'Bruker:PRESS': 'svs',  # Single voxel spectroscopy
    'Bruker:MSME': 'MSME',  # T2 mapping
    'User:mt_Array_RARE': 'MTR',
    'User:mt_Array_FLASH': 'MTR',
    'Bruker:FLASH': 'FLASH',
    'Bruker:FieldMap': 'fieldmap',
}


def parse_session_directory(session_dir: Path) -> Optional[Dict[str, str]]:
    """
    Parse Bruker session directory name to extract metadata.

    Handles three formats:
    1. Old format: YYYYMMDD_HHMMSS_IRC###_Rat###_1_1
    2. Old format variant: YYYYMMDD_HHMMSS_IRC###_Cohort#_Rat###_1_1
    3. New format: IRC####_Cohort#_Rat###_1__Rat###__p##_1_1_YYYYMMDD_HHMMSS

    Parameters
    ----------
    session_dir : Path
        Session directory path

    Returns
    -------
    dict or None
        Metadata dict with keys: subject, session, cohort, date
        Returns None if parsing fails
    """
    dirname = session_dir.name

    # Try new format first (IRC####_Cohort#_Rat###...)
    match = re.match(
        r'IRC\d+_Cohort(\d+)_Rat(\d+)_.*__(p\d+)_.*_(\d{8})_\d{6}',
        dirname
    )
    if match:
        cohort, rat_num, age, date = match.groups()
        return {
            'subject': f'Rat{rat_num}',
            'session': age,
            'cohort': f'Cohort{cohort}',
            'date': date
        }

    # Try old format (YYYYMMDD_HHMMSS_IRC###_Rat###_1_1)
    # Also handles: YYYYMMDD_HHMMSS_IRC###_Cohort#_Rat###_1_1
    match = re.match(
        r'(\d{8})_\d{6}_IRC\d+_(?:Cohort\d+_)?Rat(\d+)_',
        dirname
    )
    if match:
        date, rat_num = match.groups()
        # Need to infer session from parent directory
        parent = session_dir.parent
        if parent.name.startswith('p'):
            age = parent.name.split('_')[0]  # e.g., p30 from p30_202210
        else:
            age = 'unknown'

        # Infer cohort from path
        cohort_match = re.search(r'Cohort(\d+)', str(session_dir))
        cohort = f'Cohort{cohort_match.group(1)}' if cohort_match else 'unknown'

        return {
            'subject': f'Rat{rat_num}',
            'session': age,
            'cohort': cohort,
            'date': date
        }

    logger.warning(f"Could not parse session directory: {dirname}")
    return None


def get_bruker_method(scan_dir: Path) -> Optional[str]:
    """
    Read Bruker method from method file.

    Parameters
    ----------
    scan_dir : Path
        Scan directory containing method file

    Returns
    -------
    str or None
        Bruker method string (e.g., 'Bruker:RARE')
    """
    method_file = scan_dir / 'method'
    if not method_file.exists():
        return None

    try:
        with open(method_file, 'r') as f:
            for line in f:
                if line.startswith('##$Method='):
                    # Extract method from ##$Method=<Bruker:RARE>
                    match = re.search(r'<([^>]+)>', line)
                    if match:
                        return match.group(1)
        return None
    except Exception as e:
        logger.error(f"Error reading method file {method_file}: {e}")
        return None


def classify_scan(scan_dir: Path) -> Optional[Dict[str, str]]:
    """
    Classify Bruker scan by modality.

    Parameters
    ----------
    scan_dir : Path
        Scan directory

    Returns
    -------
    dict or None
        Classification with keys: method, modality, suffix
    """
    method = get_bruker_method(scan_dir)
    if not method:
        return None

    modality = BRUKER_METHOD_MAP.get(method)
    suffix = METHOD_SUFFIX_MAP.get(method)

    if not modality:
        logger.warning(f"Unknown Bruker method: {method}")
        return None

    return {
        'method': method,
        'modality': modality,
        'suffix': suffix,
        'scan_number': scan_dir.name
    }


def find_bruker_sessions(cohort_dirs: List[Path]) -> List[Dict[str, any]]:
    """
    Discover all Bruker sessions across cohorts.

    Parameters
    ----------
    cohort_dirs : list of Path
        List of cohort directories to scan

    Returns
    -------
    list of dict
        List of session metadata dicts
    """
    sessions = []

    for cohort_dir in cohort_dirs:
        if not cohort_dir.exists():
            logger.warning(f"Cohort directory not found: {cohort_dir}")
            continue

        # Find all potential session directories
        # Look for directories containing Bruker scans (numbered directories with 'method' files)
        for item in cohort_dir.rglob('*'):
            if not item.is_dir():
                continue

            # Check if this looks like a Bruker session
            # (contains numbered subdirectories with method files)
            numbered_dirs = [d for d in item.iterdir() if d.is_dir() and d.name.isdigit()]
            if numbered_dirs:
                # Check if at least one has a method file
                has_method = any((d / 'method').exists() for d in numbered_dirs[:3])
                if has_method:
                    metadata = parse_session_directory(item)
                    if metadata:
                        metadata['path'] = item
                        metadata['scans'] = sorted([int(d.name) for d in numbered_dirs])
                        sessions.append(metadata)

    return sessions


def extract_bids_metadata(scan_dir: Path, modality: str) -> Dict[str, any]:
    """
    Extract BIDS-relevant metadata from Bruker scan.

    Parameters
    ----------
    scan_dir : Path
        Bruker scan directory
    modality : str
        BIDS modality (anat, dwi, func, spec)

    Returns
    -------
    dict
        BIDS-compliant metadata dictionary
    """
    metadata = {}

    try:
        # Load dataset and parameter files
        pdata_path = scan_dir / 'pdata' / '1' / '2dseq'
        dataset = Dataset(str(pdata_path))

        # Load parameter files
        dataset.add_parameter_file('method')
        dataset.add_parameter_file('acqp')
        dataset.add_parameter_file('visu_pars')

        method = dataset.parameters['method'].to_dict()
        acqp = dataset.parameters['acqp'].to_dict()
        visu = dataset.parameters['visu_pars'].to_dict()

        # Common fields for all modalities
        metadata['Manufacturer'] = 'Bruker'
        metadata['ManufacturersModelName'] = acqp.get('ACQ_station', {}).get('value', 'Unknown')

        # Magnetic field strength
        if 'PVM_SPackArrGradOrient' in method:
            # Try to get field strength (typically 7T for this scanner)
            metadata['MagneticFieldStrength'] = 7.0  # Known from study

        # Acquisition date/time
        if 'ACQ_time' in acqp:
            acq_time = acqp['ACQ_time']['value']
            metadata['AcquisitionDateTime'] = acq_time

        # Scan name (more descriptive than method name)
        if 'ACQ_scan_name' in acqp:
            scan_name = acqp['ACQ_scan_name']['value']
            if scan_name:
                metadata['ScanName'] = scan_name

        # Protocol name
        if 'ACQ_protocol_name' in acqp:
            protocol_name = acqp['ACQ_protocol_name']['value']
            if protocol_name:
                metadata['ProtocolName'] = protocol_name

        # Common timing parameters
        if 'PVM_RepetitionTime' in method:
            metadata['RepetitionTime'] = float(method['PVM_RepetitionTime']['value']) / 1000.0  # Convert ms to s

        if 'PVM_EchoTime' in method:
            echo_time = method['PVM_EchoTime']['value']
            if isinstance(echo_time, list):
                metadata['EchoTime'] = float(echo_time[0]) / 1000.0  # First echo, ms to s
            else:
                metadata['EchoTime'] = float(echo_time) / 1000.0

        # Flip angle
        if 'PVM_ExcPulseAngle' in method:
            metadata['FlipAngle'] = float(method['PVM_ExcPulseAngle']['value'])

        # Imaging parameters
        if 'PVM_Matrix' in method:
            matrix = method['PVM_Matrix']['value']
            metadata['AcquisitionMatrixPE'] = int(matrix[1]) if len(matrix) > 1 else int(matrix[0])

        if 'PVM_SPackArrSliceDistance' in method:
            metadata['SliceThickness'] = float(method['PVM_SPackArrSliceDistance']['value'])

        # Modality-specific fields
        if modality == 'dwi':
            # Diffusion parameters
            if 'PVM_DwEffBval' in method:
                bvals = method['PVM_DwEffBval']['value']
                metadata['MaxBValue'] = float(np.max(bvals))
                metadata['NumberOfVolumes'] = len(bvals)

            if 'PVM_DwGradVec' in method:
                bvecs = np.array(method['PVM_DwGradVec']['value'])
                metadata['NumberOfDirections'] = bvecs.shape[0] if bvecs.ndim > 1 else 1

        elif modality == 'func':
            # Functional MRI parameters
            if 'PVM_NRepetitions' in method:
                metadata['NumberOfVolumesDiscardedByScanner'] = 0
                metadata['NumberOfVolumes'] = int(method['PVM_NRepetitions']['value'])

            # Task name (if this is task fMRI - for resting state it would be "rest")
            metadata['TaskName'] = 'rest'

        elif modality == 'anat':
            # Anatomical parameters
            if 'PVM_SPackArrNSlices' in method:
                metadata['NumberOfSlices'] = int(method['PVM_SPackArrNSlices']['value'])

            # Inversion time for T1w (if applicable)
            if 'PVM_InversionTime' in method:
                metadata['InversionTime'] = float(method['PVM_InversionTime']['value']) / 1000.0

        elif modality == 'spec':
            # Spectroscopy parameters
            if 'PVM_SpecSWH' in method:
                metadata['SpectralWidth'] = float(method['PVM_SpecSWH']['value'])

            if 'PVM_SpecMatrix' in method:
                metadata['NumberOfDataPoints'] = int(method['PVM_SpecMatrix']['value'])

        # Bruker-specific fields (optional, for reference)
        metadata['BrukerMethod'] = method.get('Method', {}).get('value', 'Unknown')

        # Protocol name from visu_pars
        if 'VisuAcqProtocolName' in visu:
            metadata['ProtocolName'] = visu['VisuAcqProtocolName']['value']

        if 'VisuSeriesExperimentComment' in visu:
            metadata['SeriesDescription'] = visu['VisuSeriesExperimentComment']['value']

    except Exception as e:
        logger.warning(f"Could not extract all metadata from {scan_dir}: {e}")
        # Return partial metadata

    return metadata


def extract_bvec_bval(scan_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract b-values and b-vectors from DTI scan.

    Adapted from rat-mri-preprocess/file_management/convert_bruker_files.py

    Parameters
    ----------
    scan_dir : Path
        Bruker scan directory

    Returns
    -------
    tuple of (bval, bvec) arrays or None
        b-values and gradient vectors, or None if not DTI
    """
    try:
        dataset = Dataset(str(scan_dir / 'pdata' / '1' / '2dseq'))
        dataset.add_parameter_file('method')
        method = dataset.parameters['method'].to_dict()

        # Extract b-values and gradient vectors
        bval = method['PVM_DwEffBval']['value']
        bvec = method['PVM_DwGradVec']['value']

        return np.array(bval), np.array(bvec)

    except (KeyError, Exception) as e:
        logger.debug(f"Could not extract b-values/vectors from {scan_dir}: {e}")
        return None


def convert_bruker_to_nifti(scan_dir: Path, output_file: Path) -> bool:
    """
    Convert Bruker scan to NIfTI using brukerapi.

    Adapted from rat-mri-preprocess/file_management/convert_bruker_files.py

    Parameters
    ----------
    scan_dir : Path
        Bruker scan directory (containing pdata/1/2dseq)
    output_file : Path
        Output NIfTI file path

    Returns
    -------
    bool
        True if conversion successful
    """
    try:
        # Load Bruker data from pdata/1/2dseq
        pdata_path = scan_dir / 'pdata' / '1' / '2dseq'
        if not pdata_path.exists():
            logger.error(f"2dseq file not found: {pdata_path}")
            return False

        dataset = Dataset(str(pdata_path))
        data = dataset.data

        # Handle complex data (take magnitude)
        if np.iscomplexobj(data):
            logger.debug(f"Converting complex data to magnitude for {scan_dir.name}")
            data = np.abs(data)

        # Squeeze extra dimensions
        data = np.squeeze(data)

        # Create NIfTI image with identity affine
        # (rat-mri-preprocess uses identity - works for their preprocessing)
        nifti_img = nib.Nifti1Image(data, np.eye(4))

        # Save
        output_file.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nifti_img, output_file)

        logger.info(f"Converted {scan_dir.name} → {output_file.name} (shape: {data.shape})")

        # Fix voxel sizes using Bruker method file
        method_file = scan_dir / 'method'
        if method_file.exists():
            try:
                params = parse_bruker_method(method_file)
                if 'voxel_size' in params:
                    logger.debug(f"Fixing voxel sizes: {params['voxel_size']} → scaled 10x")
                    update_nifti_header(output_file, params['voxel_size'], scale_factor=10.0)
                else:
                    logger.warning(f"Could not extract voxel size from {method_file}")
            except Exception as e:
                logger.warning(f"Failed to fix voxel sizes for {output_file.name}: {e}")
        else:
            logger.warning(f"Method file not found: {method_file}")

        return True

    except Exception as e:
        logger.error(f"Failed to convert {scan_dir}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def organize_to_bids(
    session_info: Dict[str, any],
    scan_classifications: List[Dict[str, str]],
    output_root: Path,
    convert: bool = True
) -> Dict[str, List[Path]]:
    """
    Organize Bruker session into BIDS-like structure.

    Parameters
    ----------
    session_info : dict
        Session metadata from parse_session_directory
    scan_classifications : list of dict
        List of scan classifications
    output_root : Path
        Output root directory (will create raw/bids/ structure)
    convert : bool
        Whether to convert files (or just organize existing NIfTI)

    Returns
    -------
    dict
        Mapping of modality → list of output files
    """
    subject = session_info['subject']
    session = session_info['session']
    session_path = session_info['path']

    # Create BIDS structure
    bids_root = output_root / 'raw' / 'bids'
    subject_dir = bids_root / f'sub-{subject}' / f'ses-{session}'

    output_files = {}

    for scan_class in scan_classifications:
        modality = scan_class['modality']
        suffix = scan_class['suffix']
        scan_num = scan_class['scan_number']

        # Skip spectroscopy for now - will handle in Phase 6 with fsl-mrs
        if modality == 'spec':
            logger.debug(f"Skipping spectroscopy scan {scan_num} (will process in Phase 6)")
            continue

        # Create modality directory
        modality_dir = subject_dir / modality
        modality_dir.mkdir(parents=True, exist_ok=True)

        # Output filename
        output_filename = f'sub-{subject}_ses-{session}_run-{scan_num}_{suffix}.nii.gz'
        output_file = modality_dir / output_filename

        # Convert if requested
        if convert:
            scan_dir = session_path / scan_num
            success = convert_bruker_to_nifti(scan_dir, output_file)
            if success:
                output_files.setdefault(modality, []).append(output_file)

                # Extract and save BIDS metadata as JSON sidecar
                json_file = output_file.with_suffix('').with_suffix('.json')
                metadata = extract_bids_metadata(scan_dir, modality)
                if metadata:
                    with open(json_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    logger.info(f"  Saved metadata: {json_file.name}")

        # For DTI, also extract bvecs/bvals
        if modality == 'dwi' and convert and success:
            bvec_file = output_file.with_suffix('').with_suffix('.bvec')
            bval_file = output_file.with_suffix('').with_suffix('.bval')

            # Extract gradient info
            gradient_info = extract_bvec_bval(scan_dir)
            if gradient_info is not None:
                bval, bvec = gradient_info

                # Transpose bvec for FSL format (3 x N)
                bvec_t = bvec.T if bvec.ndim > 1 else bvec.reshape(-1, 1).T

                # Save in FSL format
                np.savetxt(bvec_file, bvec_t, fmt='%.6f', delimiter=' ')
                np.savetxt(bval_file, bval.reshape(1, -1), fmt='%.1f', delimiter=' ')

                logger.info(f"  Saved gradients: {bvec_file.name}, {bval_file.name}")
            else:
                logger.warning(f"  Could not extract gradients for DTI scan {scan_num}")

    # Create dataset_description.json
    dataset_desc = {
        'Name': 'Rodent 7T MRI Study',
        'BIDSVersion': '1.6.0',
        'DatasetType': 'raw'
    }
    desc_file = bids_root / 'dataset_description.json'
    if not desc_file.exists():
        with open(desc_file, 'w') as f:
            json.dump(dataset_desc, f, indent=2)

    return output_files


def process_all_cohorts(
    cohort_root: Path,
    output_root: Path,
    cohorts: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Process all Bruker cohorts and convert to BIDS.

    Parameters
    ----------
    cohort_root : Path
        Root directory containing Cohort# directories
    output_root : Path
        Output root directory
    cohorts : list of str, optional
        Specific cohorts to process (e.g., ['Cohort1', 'Cohort2'])
        If None, processes all found cohorts

    Returns
    -------
    dict
        Summary statistics
    """
    # Find cohort directories
    if cohorts:
        cohort_dirs = [cohort_root / c for c in cohorts]
    else:
        cohort_dirs = sorted(cohort_root.glob('Cohort*'))

    logger.info(f"Found {len(cohort_dirs)} cohort directories")

    # Discover sessions
    sessions = find_bruker_sessions(cohort_dirs)
    logger.info(f"Found {len(sessions)} Bruker sessions")

    # Process each session
    stats = {
        'sessions_processed': 0,
        'scans_converted': 0,
        'failures': []
    }

    for session in sessions:
        logger.info(f"Processing {session['subject']} {session['session']} from {session['cohort']}")

        # Classify all scans
        classifications = []
        for scan_num in session['scans']:
            scan_dir = session['path'] / str(scan_num)
            scan_class = classify_scan(scan_dir)
            if scan_class:
                classifications.append(scan_class)

        # Organize to BIDS
        try:
            output_files = organize_to_bids(session, classifications, output_root)
            stats['sessions_processed'] += 1
            stats['scans_converted'] += sum(len(files) for files in output_files.values())
        except Exception as e:
            logger.error(f"Failed to process session {session['path']}: {e}")
            stats['failures'].append(str(session['path']))

    return stats
