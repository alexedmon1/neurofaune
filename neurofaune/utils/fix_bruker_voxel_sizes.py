#!/usr/bin/env python3
"""
Utility to extract voxel sizes from Bruker method files and update NIfTI headers.

The brukerapi conversion doesn't properly extract voxel sizes from Bruker data,
resulting in incorrect header information. This script fixes that by reading
the correct voxel sizes from the Bruker method files and updating the NIfTI headers.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import nibabel as nib
import json
import re


def parse_bruker_method(method_file: Path) -> Dict[str, any]:
    """
    Parse Bruker method file to extract acquisition parameters.

    Parameters
    ----------
    method_file : Path
        Path to Bruker method file

    Returns
    -------
    dict
        Dictionary with voxel_size (x, y, z in mm), matrix, fov, etc.
    """
    params = {}

    with open(method_file, 'r') as f:
        content = f.read()

    # Extract PVM_SpatResol (in-plane resolution in mm)
    match = re.search(r'##\$PVM_SpatResol=\( \d+ \)\n([\d\.\s]+)', content)
    if match:
        resol_values = [float(x) for x in match.group(1).strip().split()]
        params['in_plane_resolution'] = resol_values

    # Extract PVM_SliceThick (slice thickness in mm)
    match = re.search(r'##\$PVM_SliceThick=([\d\.]+)', content)
    if match:
        params['slice_thickness'] = float(match.group(1))

    # Extract PVM_Matrix (acquisition matrix)
    match = re.search(r'##\$PVM_Matrix=\( \d+ \)\n([\d\s]+)', content)
    if match:
        matrix_values = [int(x) for x in match.group(1).strip().split()]
        params['matrix'] = matrix_values

    # Extract PVM_Fov (field of view in mm)
    match = re.search(r'##\$PVM_Fov=\( \d+ \)\n([\d\.\s]+)', content)
    if match:
        fov_values = [float(x) for x in match.group(1).strip().split()]
        params['fov'] = fov_values

    # Extract method type
    match = re.search(r'##\$Method=<(.+?)>', content)
    if match:
        params['method'] = match.group(1)

    # Extract PVM_SPackArrNSlices (number of slices)
    match = re.search(r'##\$PVM_SPackArrNSlices=\( \d+ \)\n(\d+)', content)
    if match:
        params['n_slices'] = int(match.group(1))

    # Extract PVM_NRepetitions
    match = re.search(r'##\$PVM_NRepetitions=(\d+)', content)
    if match:
        params['n_repetitions'] = int(match.group(1))

    # Extract PVM_DwEffBval (effective b-values) for DTI scans
    match = re.search(r'##\$PVM_DwEffBval=\( (\d+) \)\n([\d\.\s\-e]+)', content)
    if match:
        n_bvals = int(match.group(1))
        bval_values = [float(x) for x in match.group(2).strip().split()]
        params['n_bvalues'] = len(bval_values)
        params['max_bvalue'] = max(bval_values) if bval_values else 0.0

    # Extract PVM_DwNDiffDir (number of diffusion directions)
    match = re.search(r'##\$PVM_DwNDiffDir=(\d+)', content)
    if match:
        params['n_directions'] = int(match.group(1))
    else:
        # Fallback: count rows in PVM_DwGradVec
        match = re.search(r'##\$PVM_DwGradVec=\( (\d+), 3 \)', content)
        if match:
            params['n_directions'] = int(match.group(1))

    # Extract PVM_EchoTime (single value or array for MSME)
    match = re.search(r'##\$PVM_EchoTime=([\d\.]+)', content)
    if match:
        params['echo_time'] = float(match.group(1))
    # For MSME: EffectiveTE is the array of echo times
    match = re.search(r'##\$EffectiveTE=\( (\d+) \)\n([\d\.\s]+)', content)
    if match:
        te_values = [float(x) for x in match.group(2).strip().split()]
        params['echo_times'] = te_values
        params['n_echoes'] = len(te_values)

    # Combine into voxel size (x, y, z)
    if 'in_plane_resolution' in params and 'slice_thickness' in params:
        if len(params['in_plane_resolution']) == 2:
            params['voxel_size'] = (
                params['in_plane_resolution'][0],
                params['in_plane_resolution'][1],
                params['slice_thickness']
            )
        elif len(params['in_plane_resolution']) == 3:
            # 3D acquisition
            params['voxel_size'] = tuple(params['in_plane_resolution'])

    return params


def update_nifti_header(
    nifti_file: Path,
    voxel_size: Tuple[float, float, float],
    output_file: Optional[Path] = None,
    scale_factor: float = 10.0
) -> Path:
    """
    Update NIfTI header with correct voxel sizes.

    Applies the correct voxel sizes from Bruker and scales them by 10x
    for FSL/ANTs compatibility (sub-mm → mm range).

    Parameters
    ----------
    nifti_file : Path
        Input NIfTI file
    voxel_size : Tuple[float, float, float]
        Voxel sizes in mm (x, y, z) from Bruker
    output_file : Path, optional
        Output file path (if None, overwrites input)
    scale_factor : float
        Scaling factor for FSL/ANTs compatibility (default: 10.0)

    Returns
    -------
    Path
        Path to updated file
    """
    if output_file is None:
        output_file = nifti_file

    # Load image
    img = nib.load(nifti_file)
    header = img.header.copy()

    # Get current voxel sizes
    current_zooms = header.get_zooms()

    # Scale voxel sizes for FSL/ANTs compatibility
    scaled_voxel_size = tuple(v * scale_factor for v in voxel_size)

    print(f"  Current voxel sizes:  {current_zooms[:3]}")
    print(f"  Bruker voxel sizes:   {voxel_size}")
    print(f"  Scaled voxel sizes:   {scaled_voxel_size}")

    # Update voxel sizes in header
    # Need to update the pixdim field
    header.set_zooms(scaled_voxel_size + current_zooms[3:])  # Preserve time dimension if present

    # Update affine matrix to match new voxel sizes
    affine = img.affine.copy()
    for i in range(3):
        # Scale each dimension by the ratio of new to old voxel size
        if current_zooms[i] != 0:
            ratio = scaled_voxel_size[i] / current_zooms[i]
            affine[:, i] = affine[:, i] * ratio

    # Create new image with updated header and affine
    new_img = nib.Nifti1Image(img.get_fdata(), affine, header)

    # Save
    nib.save(new_img, output_file)
    print(f"  ✓ Updated NIfTI: {output_file}")

    return output_file


def update_json_sidecar(
    json_file: Path,
    voxel_size: Tuple[float, float, float],
    scale_factor: float = 10.0
) -> Path:
    """
    Update JSON sidecar with PixelSpacing field.

    Parameters
    ----------
    json_file : Path
        JSON sidecar file
    voxel_size : Tuple[float, float, float]
        Voxel sizes in mm (x, y, z) from Bruker
    scale_factor : float
        Scaling factor for FSL/ANTs compatibility (default: 10.0)

    Returns
    -------
    Path
        Path to updated JSON file
    """
    # Load existing metadata
    with open(json_file, 'r') as f:
        metadata = json.load(f)

    # Scale voxel sizes
    scaled_voxel_size = tuple(v * scale_factor for v in voxel_size)

    # Add PixelSpacing (in-plane x, y) and update SliceThickness
    metadata['PixelSpacing'] = [scaled_voxel_size[0], scaled_voxel_size[1]]
    metadata['SliceThickness'] = scaled_voxel_size[2]

    # Save updated metadata
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Updated JSON: {json_file.name}")
    print(f"    PixelSpacing: {metadata['PixelSpacing']}")
    print(f"    SliceThickness: {metadata['SliceThickness']}")

    return json_file


def find_bruker_scan_dir(
    bruker_root: Path,
    subject: str,
    session: str,
    scan_name: str
) -> Optional[Path]:
    """
    Find the Bruker scan directory for a given BIDS scan.

    Parameters
    ----------
    bruker_root : Path
        Root directory of Bruker data
    subject : str
        Subject ID (e.g., 'Rat207')
    session : str
        Session/cohort (e.g., 'p60')
    scan_name : str
        Scan name from JSON sidecar (e.g., '5')

    Returns
    -------
    Path or None
        Path to Bruker scan directory containing method file
    """
    cohort_dirs = list(bruker_root.glob('Cohort*'))

    for cohort_dir in cohort_dirs:
        # Pattern 1: Cohorts 1-5 structure
        # bruker_root/Cohort*/session*/timestamp*subject*/scan_number/
        # Folder names like: 20221005_091305_IRC938_Rat150_1_1 or 20230413_..._Cohort2_Rat8_1_3
        session_dirs = list(cohort_dir.glob(f'{session}*'))

        for session_dir in session_dirs:
            # Look for subject directories - use exact match pattern to avoid substring matches
            # e.g., *_Rat8_* should not match *_Rat86_*
            subject_patterns_p1 = [
                f'*_{subject}_1_*',     # e.g., *_Rat8_1_1 or *_Rat8_1_2
                f'*_{subject}_*_*',     # e.g., *_Cohort2_Rat8_1_3
            ]

            for pattern in subject_patterns_p1:
                subject_dirs = list(session_dir.glob(pattern))
                for subject_dir in subject_dirs:
                    # Verify it's an exact subject match (not Rat8 matching Rat86)
                    dir_name = subject_dir.name
                    # Check that subject is followed by _ and a digit (not another letter)
                    if re.search(rf'_{subject}_\d', dir_name):
                        scan_dir = subject_dir / scan_name
                        if scan_dir.exists() and (scan_dir / 'method').exists():
                            return scan_dir

        # Pattern 2: Cohorts 7-8 structure (flat, all info in folder name)
        # bruker_root/Cohort*/IRC1200_Cohort7_Rat102_1__Rat_102__p60_1_1_20250407/scan_number/
        # Match patterns like *_Rat102_* or *_Rat_102_* and *_p60_*
        # Extract numeric part of subject ID for matching (e.g., "Rat102" -> "102")
        subject_num = subject.replace("Rat", "")

        subject_patterns = [
            f'*_{subject}_1_*_{session}_*',      # e.g., *_Rat102_1_*_p60_*
            f'*_{subject}_1__*_{session}_*',     # e.g., *_Rat102_1__*_p60_*
            f'*_Rat_{subject_num}_*_{session}_*',  # e.g., *_Rat_102_*_p60_*
        ]

        for pattern in subject_patterns:
            subject_dirs = list(cohort_dir.glob(pattern))
            for subject_dir in subject_dirs:
                if subject_dir.is_dir():
                    scan_dir = subject_dir / scan_name
                    if scan_dir.exists() and (scan_dir / 'method').exists():
                        return scan_dir

    return None


def fix_bids_nifti_headers(
    bids_root: Path,
    bruker_root: Path,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Fix voxel sizes for all NIfTI files in BIDS dataset.

    Parameters
    ----------
    bids_root : Path
        Root directory of BIDS dataset
    bruker_root : Path
        Root directory of Bruker data
    dry_run : bool
        If True, only report what would be done

    Returns
    -------
    dict
        Summary statistics
    """
    stats = {'processed': 0, 'updated': 0, 'failed': 0, 'skipped': 0}

    # Find all NIfTI files
    nifti_files = list(bids_root.glob('sub-*/ses-*/**/*.nii.gz'))

    print(f"Found {len(nifti_files)} NIfTI files")
    print("="*80)

    for nifti_file in nifti_files:
        stats['processed'] += 1

        # Get corresponding JSON sidecar
        json_file = nifti_file.with_suffix('').with_suffix('.json')

        if not json_file.exists():
            print(f"⚠ No JSON sidecar for {nifti_file.name}")
            stats['skipped'] += 1
            continue

        # Load JSON to get ScanName
        with open(json_file, 'r') as f:
            metadata = json.load(f)

        scan_name_full = metadata.get('ScanName')
        if not scan_name_full:
            print(f"⚠ No ScanName in JSON for {nifti_file.name}")
            stats['skipped'] += 1
            continue

        # Extract scan number from ScanName (format: "<Protocol (E#)>")
        match = re.search(r'\(E(\d+)\)', scan_name_full)
        if not match:
            print(f"⚠ Could not parse scan number from: {scan_name_full}")
            stats['skipped'] += 1
            continue

        scan_name = match.group(1)  # Just the number

        # Extract subject and session from path (exclude filename)
        parent_parts = nifti_file.parent.parts
        subject_id = None
        session_id = None
        for part in parent_parts:
            if part.startswith('sub-Rat'):
                subject_id = part.replace('sub-', '')
            if part.startswith('ses-'):
                session_id = part.replace('ses-', '')

        if not subject_id or not session_id:
            print(f"⚠ Could not extract subject/session from {nifti_file}")
            stats['skipped'] += 1
            continue

        print(f"\n{nifti_file.name}")
        print(f"  Subject: {subject_id}, Session: {session_id}, Scan: {scan_name}")

        # Find Bruker scan directory
        bruker_scan_dir = find_bruker_scan_dir(
            bruker_root, subject_id, session_id, scan_name
        )

        if not bruker_scan_dir:
            print(f"  ⚠ Could not find Bruker data for scan {scan_name}")
            stats['failed'] += 1
            continue

        method_file = bruker_scan_dir / 'method'
        print(f"  Bruker method: {method_file}")

        # Parse method file
        try:
            params = parse_bruker_method(method_file)

            if 'voxel_size' not in params:
                print(f"  ⚠ Could not extract voxel size from method file")
                stats['failed'] += 1
                continue

            voxel_size = params['voxel_size']
            scaled_voxel_size = tuple(v * 10.0 for v in voxel_size)

            # Update NIfTI header and JSON sidecar
            if not dry_run:
                update_nifti_header(nifti_file, voxel_size)
                update_json_sidecar(json_file, voxel_size)
                stats['updated'] += 1
            else:
                print(f"  [DRY RUN] Bruker voxel sizes: {voxel_size}")
                print(f"  [DRY RUN] Would update to (10x scaled): {scaled_voxel_size}")
                print(f"  [DRY RUN] Would add PixelSpacing: [{scaled_voxel_size[0]}, {scaled_voxel_size[1]}]")
                print(f"  [DRY RUN] Would update SliceThickness: {scaled_voxel_size[2]}")
                stats['updated'] += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            stats['failed'] += 1
            continue

    print("\n" + "="*80)
    print("SUMMARY")
    print(f"  Processed: {stats['processed']}")
    print(f"  Updated:   {stats['updated']}")
    print(f"  Failed:    {stats['failed']}")
    print(f"  Skipped:   {stats['skipped']}")

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Fix NIfTI voxel sizes using Bruker method files'
    )
    parser.add_argument(
        '--bids-root',
        type=Path,
        required=True,
        help='Root directory of BIDS dataset'
    )
    parser.add_argument(
        '--bruker-root',
        type=Path,
        required=True,
        help='Root directory of Bruker data'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only report what would be done, do not modify files'
    )

    args = parser.parse_args()

    fix_bids_nifti_headers(args.bids_root, args.bruker_root, args.dry_run)
