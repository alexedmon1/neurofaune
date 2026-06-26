#!/usr/bin/env python3
"""
Per-scan Bruker readers used by the BIDS converter.

This module provides the low-level helpers the live converter
(:mod:`neurofaune.utils.bids` -- the ``neurofaune bids`` CLI) consumes:
- Bruker method classification (T2w, DTI, fMRI, spectroscopy, MSME, MTR)
- per-scan BIDS metadata + echo-time extraction
- conversion to NIfTI
- per-session inventory and best-scan selection

The old standalone study driver (``process_all_cohorts`` and the
``scripts/convert_bruker_to_bids.py`` entry point) has been removed; use
``neurofaune bids`` / :func:`neurofaune.utils.bids.convert_study` instead.
"""

import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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



def _resolve_echo_time(method: Dict[str, any],
                       echo_index: Optional[int] = None) -> Dict[str, any]:
    """Resolve BIDS ``EchoTime`` (seconds), and ``EchoNumber`` for multi-echo,
    from a parsed Bruker ``method`` dict (brukerapi ``{key: {'value': ...}}``).

    Multi-echo sequences (e.g. T2S_EPI) store the per-echo TEs in the
    ``EffectiveEchoTime`` array; there ``PVM_EchoTime`` is the echo SPACING
    (~0.29 ms) and must NOT be used as a TE. Prefer the per-echo array and index
    it by ``echo_index`` (0-based) when known; otherwise use the first echo.
    Falls back to scalar ``PVM_EchoTime`` for single-echo. Returns {} if no TE.
    """
    def _val(key):
        entry = method.get(key)
        return entry.get('value') if isinstance(entry, dict) else None

    n_echoes = 1
    ne = _val('PVM_NEchoImages')
    if ne is not None:
        try:
            n_echoes = int(ne)
        except (TypeError, ValueError):
            n_echoes = 1

    te_array = None
    for key in ('EffectiveEchoTime', 'GradientEchoTime'):
        val = _val(key)
        if isinstance(val, (list, tuple)) and len(val) >= 1:
            te_array = [float(x) for x in val]
            break

    out: Dict[str, any] = {}
    if te_array is not None:
        if n_echoes > 1 and echo_index is not None and 0 <= echo_index < len(te_array):
            out['EchoTime'] = te_array[echo_index] / 1000.0   # ms -> s
            out['EchoNumber'] = int(echo_index) + 1
        else:
            out['EchoTime'] = te_array[0] / 1000.0
    else:
        scalar = _val('PVM_EchoTime')
        if scalar is not None:
            te = scalar[0] if isinstance(scalar, (list, tuple)) else scalar
            out['EchoTime'] = float(te) / 1000.0
    return out


def extract_bids_metadata(scan_dir: Path, modality: str,
                          echo_index: Optional[int] = None) -> Dict[str, any]:
    """
    Extract BIDS-relevant metadata from Bruker scan.

    Parameters
    ----------
    scan_dir : Path
        Bruker scan directory
    modality : str
        BIDS modality (anat, dwi, func, spec)
    echo_index : int, optional
        0-based echo index of the image this sidecar describes. For multi-echo
        sequences the per-echo TE is taken from the ``EffectiveEchoTime`` array at
        this index; pass it when writing one sidecar per echo so each gets its own
        EchoTime (and EchoNumber). When None, the first echo's TE is used.

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

        # Echo time, incl. per-echo TEs for multi-echo (see _resolve_echo_time).
        metadata.update(_resolve_echo_time(method, echo_index))

        # Flip angle
        if 'PVM_ExcPulseAngle' in method:
            metadata['FlipAngle'] = float(method['PVM_ExcPulseAngle']['value'])

        # Imaging parameters
        if 'PVM_Matrix' in method:
            matrix = method['PVM_Matrix']['value']
            metadata['AcquisitionMatrixPE'] = int(matrix[1]) if len(matrix) > 1 else int(matrix[0])

        if 'PVM_SPackArrSliceDistance' in method:
            metadata['SliceThickness'] = float(method['PVM_SPackArrSliceDistance']['value'])

        # Extract in-plane voxel dimensions (CRITICAL for proper preprocessing)
        if 'PVM_SpatResol' in method:
            spat_resol = method['PVM_SpatResol']['value']
            if isinstance(spat_resol, (list, np.ndarray)) and len(spat_resol) >= 2:
                # In-plane resolution in mm [x, y]
                metadata['PixelSpacing'] = [float(spat_resol[0]), float(spat_resol[1])]

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



def inventory_session(
    session_dir: Path,
    output_csv: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Classify all scans in a Bruker session and extract key metadata.

    Walks numbered subdirectories that contain a ``method`` file, runs
    :func:`classify_scan` and :func:`parse_bruker_method` on each, and
    merges results into a single record per scan.

    Parameters
    ----------
    session_dir : Path
        Top-level Bruker session directory (contains numbered scan dirs).
    output_csv : Path, optional
        If provided, write the inventory to this CSV file.

    Returns
    -------
    list of dict
        One dict per scan with keys: scan_number, method, modality, suffix,
        matrix, n_slices, fov_mm, voxel_size_mm, slice_thickness_mm,
        n_volumes, max_bvalue, n_repetitions, n_directions.
    """
    records: List[Dict[str, Any]] = []

    # Find numbered subdirectories with a method file
    scan_dirs = sorted(
        (d for d in session_dir.iterdir()
         if d.is_dir() and d.name.isdigit() and (d / 'method').exists()),
        key=lambda d: int(d.name),
    )

    for scan_dir in scan_dirs:
        scan_num = int(scan_dir.name)

        # Classification (method/modality/suffix)
        classification = classify_scan(scan_dir)
        if classification is None:
            continue

        # Extended Bruker parameters (voxel size, DTI fields, etc.)
        params = parse_bruker_method(scan_dir / 'method')

        # Calculate total n_volumes from bval count or repetitions * directions
        n_volumes: Optional[int] = None
        if 'n_bvalues' in params:
            n_volumes = params['n_bvalues']
        elif 'n_repetitions' in params:
            n_volumes = params['n_repetitions']

        record: Dict[str, Any] = {
            'scan_number': scan_num,
            'method': classification['method'],
            'modality': classification['modality'],
            'suffix': classification['suffix'],
            'matrix': params.get('matrix'),
            'n_slices': params.get('n_slices'),
            'fov_mm': params.get('fov'),
            'voxel_size_mm': params.get('voxel_size'),
            'slice_thickness_mm': params.get('slice_thickness'),
            'n_volumes': n_volumes,
            'max_bvalue': params.get('max_bvalue'),
            'n_repetitions': params.get('n_repetitions'),
            'n_directions': params.get('n_directions'),
            'n_echoes': params.get('n_echoes'),
            'echo_times': params.get('echo_times'),
        }
        records.append(record)

    # Optionally write CSV
    if output_csv is not None and records:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(records[0].keys())
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)
        logger.info(f"Inventory CSV written to {output_csv}")

    return records


def select_best_t2w_from_inventory(
    inventory: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Select best T2w scan from an inventory list.

    Selection criteria (applied to raw Bruker metadata, no NIfTI needed):
    1. Filter to ``modality == 'anat'`` (Bruker:RARE).
    2. Penalize scans with method names containing 'localizer' or 'scout'.
    3. Prefer most slices.

    Parameters
    ----------
    inventory : list of dict
        Output of :func:`inventory_session`.

    Returns
    -------
    dict or None
        Best T2w scan record, or None if no anat scans found.
    """
    anat_scans = [r for r in inventory if r['modality'] == 'anat']
    if not anat_scans:
        return None

    def _score(rec: Dict[str, Any]) -> float:
        score = 0.0
        n_slices = rec.get('n_slices') or 0
        score += n_slices
        # Penalize localizer / scout scans
        method_lower = (rec.get('method') or '').lower()
        if 'localizer' in method_lower or 'scout' in method_lower:
            score -= 1000
        # Penalize very few slices
        if n_slices < 10:
            score -= 50
        return score

    return max(anat_scans, key=_score)


def select_best_dwi_from_inventory(
    inventory: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Select best DWI scan from an inventory list.

    Selection criteria:
    1. Filter to ``modality == 'dwi'`` (Bruker:DtiEpi).
    2. Prefer highest ``max_bvalue``.
    3. Tie-break by most ``n_volumes``.

    Parameters
    ----------
    inventory : list of dict
        Output of :func:`inventory_session`.

    Returns
    -------
    dict or None
        Best DWI scan record, or None if no DWI scans found.
    """
    dwi_scans = [r for r in inventory if r['modality'] == 'dwi']
    if not dwi_scans:
        return None

    def _score(rec: Dict[str, Any]) -> Tuple[float, int]:
        max_bval = rec.get('max_bvalue') or 0.0
        n_vols = rec.get('n_volumes') or 0
        return (max_bval, n_vols)

    return max(dwi_scans, key=_score)


def select_best_func_from_inventory(
    inventory: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Select best functional (BOLD) scan from an inventory list.

    Selection criteria:
    1. Filter to ``modality == 'func'`` (Bruker:EPI).
    2. Prefer most repetitions (longest timeseries).
    3. Tie-break by most slices.

    Parameters
    ----------
    inventory : list of dict
        Output of :func:`inventory_session`.

    Returns
    -------
    dict or None
        Best func scan record, or None if no func scans found.
    """
    func_scans = [r for r in inventory if r['modality'] == 'func']
    if not func_scans:
        return None

    def _score(rec: Dict[str, Any]) -> Tuple[int, int]:
        n_reps = rec.get('n_repetitions') or 0
        n_slices = rec.get('n_slices') or 0
        return (n_reps, n_slices)

    return max(func_scans, key=_score)


def select_best_msme_from_inventory(
    inventory: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Select best MSME (multi-echo T2 mapping) scan from an inventory list.

    Selection criteria:
    1. Filter to ``modality == 'msme'`` (Bruker:MSME).
    2. Prefer most echoes.
    3. Tie-break by most slices.

    Parameters
    ----------
    inventory : list of dict
        Output of :func:`inventory_session`.

    Returns
    -------
    dict or None
        Best MSME scan record, or None if no MSME scans found.
    """
    msme_scans = [r for r in inventory if r['modality'] == 'msme']
    if not msme_scans:
        return None

    def _score(rec: Dict[str, Any]) -> Tuple[int, int]:
        n_echoes = rec.get('n_echoes') or 0
        n_slices = rec.get('n_slices') or 0
        return (n_echoes, n_slices)

    return max(msme_scans, key=_score)

