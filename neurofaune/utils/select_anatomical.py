#!/usr/bin/env python3
"""
Utility to automatically select the best anatomical (T2w) scan for preprocessing.

For rodent MRI, we typically have multiple T2w scans with different:
- Orientations (axial, coronal, sagittal)
- Resolutions
- Coverage (number of slices)
- Purposes (localizer, high-res, specific region)

This module implements a scoring system to select the most appropriate scan
for atlas registration.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import nibabel as nib
import re


def score_anatomical_scan(
    nifti_file: Path,
    json_file: Path,
    prefer_orientation: str = 'axial'
) -> Tuple[float, Dict[str, any]]:
    """
    Score an anatomical scan for suitability as primary registration target.

    Scoring criteria:
    - Number of slices (more is better)
    - Orientation match (axial preferred for reorienting to coronal)
    - Resolution (higher is better, but not too much)
    - Coverage (prefer whole-brain over targeted scans)

    Parameters
    ----------
    nifti_file : Path
        Path to NIfTI file
    json_file : Path
        Path to JSON sidecar
    prefer_orientation : str
        Preferred orientation ('axial', 'coronal', or 'sagittal')

    Returns
    -------
    Tuple[float, dict]
        (score, metadata_dict)
        Higher score = better candidate
    """
    # Load JSON metadata
    with open(json_file, 'r') as f:
        metadata = json.load(f)

    # Load NIfTI to get dimensions
    img = nib.load(nifti_file)
    shape = img.shape

    score = 0.0
    info = {
        'file': nifti_file.name,
        'scan_name': metadata.get('ScanName', ''),
        'slices': metadata.get('NumberOfSlices', shape[2]),
        'slice_thickness': metadata.get('SliceThickness', 1.0),
        'shape': shape,
    }

    # 1. Number of slices (weight: UP TO 5.0)
    # More slices = better coverage
    # STRONGLY penalize localizers and targeted scans
    n_slices = info['slices']
    if n_slices >= 40:
        score += 5.0  # Excellent full-brain coverage
    elif n_slices >= 30:
        score += 4.0  # Very good coverage
    elif n_slices >= 25:
        score += 3.0  # Good coverage
    elif n_slices >= 15:
        score += 2.0  # Moderate coverage
    elif n_slices >= 10:
        score += 1.0  # Minimal acceptable coverage
    else:
        score -= 5.0  # STRONG penalty for localizers/targeted scans (< 10 slices)

    # 2. Orientation (weight: 2.0)
    # Determine orientation from scan name
    scan_name = info['scan_name'].lower()
    if 'axial' in scan_name or 'ax' in scan_name:
        orientation = 'axial'
    elif 'cor' in scan_name or 'coronal' in scan_name:
        orientation = 'coronal'
    elif 'sag' in scan_name or 'sagittal' in scan_name:
        orientation = 'sagittal'
    else:
        orientation = 'unknown'

    info['orientation'] = orientation

    if orientation == prefer_orientation:
        score += 2.0
    elif orientation == 'coronal' and prefer_orientation == 'axial':
        score += 1.5  # Coronal is also good
    elif orientation != 'unknown':
        score += 0.5

    # 3. Resolution bonus (weight: 1.0)
    # Prefer slices between 0.5-1.0mm (not too thin, not too thick)
    thickness = info['slice_thickness']
    if 0.5 <= thickness <= 1.0:
        score += 1.0  # Ideal range
    elif 0.3 <= thickness < 0.5:
        score += 0.5  # High res but might be noisy
    elif 1.0 < thickness <= 1.5:
        score += 0.5  # Still acceptable

    # 4. Penalize targeted scans (weight: -2.0)
    # Look for keywords indicating non-whole-brain scans
    targeted_keywords = ['5slice', '3slice', 'hippo', 'cortex', 'localizer', 'scout']
    if any(kw in scan_name for kw in targeted_keywords):
        score -= 2.0

    # 5. STRONGLY penalize 3D T2w scans (weight: -10.0)
    # 3D TurboRARE scans from Cohorts 1-2 have skull stripping issues
    # Prefer 2D RARE scans for better preprocessing results
    if '3d' in scan_name or 'turborare_3d' in scan_name:
        score -= 10.0
        info['is_3d'] = True
    else:
        info['is_3d'] = False

    # 6. Bonus for "high res" or "125um" scans (weight: 1.0)
    if '125um' in scan_name or 'highres' in scan_name or 'hires' in scan_name:
        score += 1.0

    info['score'] = score
    return score, info


def select_best_anatomical(
    subject_dir: Path,
    session: str,
    modality: str = 'anat',
    prefer_orientation: str = 'axial'
) -> Optional[Dict[str, any]]:
    """
    Select the best anatomical scan from a subject/session.

    Parameters
    ----------
    subject_dir : Path
        Subject directory (e.g., /path/to/sub-Rat207)
    session : str
        Session identifier (e.g., 'ses-p60')
    modality : str
        Modality directory name (default: 'anat')
    prefer_orientation : str
        Preferred orientation

    Returns
    -------
    dict or None
        Dictionary with 'nifti', 'json', 'score', and metadata
        Returns None if no suitable scans found

    Examples
    --------
    >>> best = select_best_anatomical(
    ...     Path('/mnt/arborea/bpa-rat/raw/bids/sub-Rat207'),
    ...     'ses-p60'
    ... )
    >>> print(best['nifti'])
    >>> print(f"Score: {best['score']:.2f}")
    """
    anat_dir = subject_dir / session / modality

    if not anat_dir.exists():
        return None

    # Find all T2w NIfTI files
    nifti_files = list(anat_dir.glob('*_T2w.nii.gz'))

    if not nifti_files:
        return None

    candidates = []

    for nifti_file in nifti_files:
        json_file = nifti_file.with_suffix('').with_suffix('.json')

        if not json_file.exists():
            continue

        try:
            score, info = score_anatomical_scan(
                nifti_file, json_file, prefer_orientation
            )

            candidates.append({
                'nifti': nifti_file,
                'json': json_file,
                'score': score,
                'info': info
            })

        except Exception as e:
            print(f"Warning: Could not score {nifti_file.name}: {e}")
            continue

    if not candidates:
        return None

    # Sort by score (highest first)
    candidates.sort(key=lambda x: x['score'], reverse=True)

    return candidates[0]


def print_scan_ranking(
    subject_dir: Path,
    session: str,
    modality: str = 'anat'
) -> None:
    """
    Print ranked list of anatomical scans for a subject/session.

    Parameters
    ----------
    subject_dir : Path
        Subject directory
    session : str
        Session identifier
    modality : str
        Modality directory
    """
    anat_dir = subject_dir / session / modality
    nifti_files = list(anat_dir.glob('*_T2w.nii.gz'))

    print(f"\nAnatomical scans for {subject_dir.name} {session}:")
    print("=" * 80)

    candidates = []

    for nifti_file in nifti_files:
        json_file = nifti_file.with_suffix('').with_suffix('.json')
        if not json_file.exists():
            continue

        try:
            score, info = score_anatomical_scan(nifti_file, json_file)
            candidates.append((score, info))
        except:
            continue

    candidates.sort(key=lambda x: x[0], reverse=True)

    for i, (score, info) in enumerate(candidates, 1):
        marker = "⭐" if i == 1 else "  "
        print(f"{marker} [{score:4.1f}] {info['file']}")
        print(f"          {info['scan_name']}")
        print(f"          {info['orientation']:8s} | {info['slices']:2d} slices × {info['slice_thickness']:.2f}mm")
        print()

    if candidates:
        print(f"SELECTED: {candidates[0][1]['file']}")
    print("=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Select best anatomical scan for preprocessing'
    )
    parser.add_argument(
        '--subject-dir',
        type=Path,
        required=True,
        help='Subject directory (e.g., /path/to/sub-Rat207)'
    )
    parser.add_argument(
        '--session',
        type=str,
        required=True,
        help='Session identifier (e.g., ses-p60)'
    )

    args = parser.parse_args()

    print_scan_ranking(args.subject_dir, args.session)
