#!/usr/bin/env python3
"""
Convert 3D isotropic RARE acquisitions to BIDS T2w format.

Some subjects have 3D isotropic RARE instead of 2D multi-slice RARE for their
anatomical T2w scans. This script:
1. Identifies subjects missing T2w in BIDS but having 3D RARE in Bruker
2. Converts the 3D RARE to NIfTI
3. Places it in the BIDS anat folder with proper naming

The 3D acquisitions can then be processed through the standard pipeline.

Usage:
    python scripts/convert_3d_rare_to_bids.py \
        --bruker-root /mnt/arborea/bruker \
        --bids-root /mnt/arborea/bpa-rat/raw/bids \
        --dry-run
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.utils.bruker_convert import (
    convert_bruker_to_nifti,
    extract_bids_metadata,
    get_bruker_method,
)
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_3d_acquisition(method_file: Path) -> bool:
    """Check if a RARE scan is 3D isotropic (vs 2D multi-slice)."""
    with open(method_file, 'r') as f:
        content = f.read()

    # Check PVM_SpatResol - 3D has 3 values, 2D has 2
    match = re.search(r'##\$PVM_SpatResol=\( (\d+) \)', content)
    if match:
        n_dims = int(match.group(1))
        if n_dims == 3:
            return True

    # Also check matrix dimensions
    match = re.search(r'##\$PVM_Matrix=\( (\d+) \)', content)
    if match:
        n_dims = int(match.group(1))
        if n_dims == 3:
            return True

    return False


def get_rare_geometry(method_file: Path) -> Dict:
    """Extract geometry information from RARE scan."""
    with open(method_file, 'r') as f:
        content = f.read()

    info = {}

    # Matrix
    match = re.search(r'##\$PVM_Matrix=\( \d+ \)\n([\d\s]+)', content)
    if match:
        info['matrix'] = [int(x) for x in match.group(1).strip().split()]

    # Resolution
    match = re.search(r'##\$PVM_SpatResol=\( \d+ \)\n([\d\.\s]+)', content)
    if match:
        info['resolution'] = [float(x) for x in match.group(1).strip().split()]

    # Slice count (for 2D)
    match = re.search(r'##\$PVM_SPackArrNSlices=\( \d+ \)\n(\d+)', content)
    if match:
        info['n_slices'] = int(match.group(1))

    return info


def find_subjects_with_3d_rare(
    bids_root: Path,
    bruker_root: Path
) -> List[Dict]:
    """
    Find subjects that:
    - Have no T2w in BIDS (missing anat folder or no T2w files)
    - Have 3D RARE in Bruker data

    Returns list of dicts with subject/session info and Bruker scan path.
    """
    results = []

    # Find subjects with BIDS data but no T2w
    for subject_dir in sorted(bids_root.glob('sub-*')):
        subject = subject_dir.name.replace('sub-', '')

        for session_dir in sorted(subject_dir.glob('ses-*')):
            session = session_dir.name.replace('ses-', '')

            # Check if T2w exists
            anat_dir = session_dir / 'anat'
            has_t2w = anat_dir.exists() and list(anat_dir.glob('*T2w*.nii.gz'))

            if not has_t2w:
                # Look for 3D RARE in Bruker
                bruker_session = find_bruker_session(
                    bruker_root, subject, session
                )

                if bruker_session:
                    # Check each RARE scan for 3D
                    for scan_dir in sorted(bruker_session.glob('[0-9]*')):
                        method_file = scan_dir / 'method'
                        if not method_file.exists():
                            continue

                        method = get_bruker_method(scan_dir)
                        if method and 'RARE' in method:
                            if is_3d_acquisition(method_file):
                                geometry = get_rare_geometry(method_file)
                                results.append({
                                    'subject': subject,
                                    'session': session,
                                    'bruker_scan': scan_dir,
                                    'scan_number': scan_dir.name,
                                    'geometry': geometry,
                                    'bids_subject_dir': subject_dir,
                                    'bids_session_dir': session_dir,
                                })
                                break  # Take first 3D RARE found

    return results


def find_bruker_session(
    bruker_root: Path,
    subject: str,
    session: str
) -> Optional[Path]:
    """Find Bruker session directory for a subject/session."""
    # Extract rat number
    rat_num = subject.replace('Rat', '')

    # Search patterns for different cohort structures
    patterns = [
        # Cohort 1-5: Cohort*/session*/timestamp*_Rat###_*
        f'Cohort*/p*{session[-2:]}*/*_Rat{rat_num}_*',
        f'Cohort*/{session}*/*_Rat{rat_num}_*',
        # Cohort 7-8: Cohort*/*_Rat###_*_session_*
        f'Cohort*/*_Rat{rat_num}_*_{session}_*',
        f'Cohort*/*_Rat_{rat_num}_*_{session}_*',
    ]

    for pattern in patterns:
        matches = list(bruker_root.glob(pattern))
        for match in matches:
            if match.is_dir():
                # Verify it's a Bruker session (has numbered scan dirs)
                numbered_dirs = [d for d in match.iterdir()
                               if d.is_dir() and d.name.isdigit()]
                if numbered_dirs:
                    return match

    return None


def convert_3d_rare_to_bids(
    scan_info: Dict,
    bids_root: Path,
    dry_run: bool = False
) -> bool:
    """Convert a 3D RARE scan and place in BIDS structure."""
    subject = scan_info['subject']
    session = scan_info['session']
    scan_dir = scan_info['bruker_scan']
    scan_number = scan_info['scan_number']

    # Create output paths
    anat_dir = bids_root / f'sub-{subject}' / f'ses-{session}' / 'anat'
    output_file = anat_dir / f'sub-{subject}_ses-{session}_acq-3D_run-{scan_number}_T2w.nii.gz'
    json_file = output_file.with_suffix('').with_suffix('.json')

    logger.info(f"Converting sub-{subject}_ses-{session}")
    logger.info(f"  Source: {scan_dir}")
    logger.info(f"  Target: {output_file}")
    logger.info(f"  Geometry: {scan_info['geometry']}")

    if dry_run:
        logger.info("  [DRY RUN] Would convert")
        return True

    # Create directory
    anat_dir.mkdir(parents=True, exist_ok=True)

    # Convert
    success = convert_bruker_to_nifti(scan_dir, output_file)

    if success:
        # Extract and save metadata
        metadata = extract_bids_metadata(scan_dir, 'anat')
        metadata['Acquisition3D'] = True
        metadata['OriginalGeometry'] = scan_info['geometry']

        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  ✓ Converted successfully")
        return True
    else:
        logger.error(f"  ✗ Conversion failed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert 3D RARE acquisitions to BIDS T2w format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--bruker-root',
        type=Path,
        required=True,
        help='Root directory containing Cohort* directories with Bruker data'
    )

    parser.add_argument(
        '--bids-root',
        type=Path,
        required=True,
        help='BIDS root directory (raw/bids/)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without converting'
    )

    parser.add_argument(
        '--subjects',
        nargs='+',
        help='Only process specific subjects (e.g., Rat120 Rat166)'
    )

    args = parser.parse_args()

    # Validate paths
    if not args.bruker_root.exists():
        logger.error(f"Bruker root not found: {args.bruker_root}")
        sys.exit(1)

    if not args.bids_root.exists():
        logger.error(f"BIDS root not found: {args.bids_root}")
        sys.exit(1)

    # Find subjects with 3D RARE
    logger.info("Scanning for subjects with 3D RARE acquisitions...")
    subjects_3d = find_subjects_with_3d_rare(args.bids_root, args.bruker_root)

    # Filter if specific subjects requested
    if args.subjects:
        subjects_3d = [s for s in subjects_3d if s['subject'] in args.subjects]

    logger.info(f"Found {len(subjects_3d)} subjects with 3D RARE (missing 2D T2w)")

    if not subjects_3d:
        logger.info("No subjects to convert")
        return

    # Convert each
    success_count = 0
    for scan_info in subjects_3d:
        if convert_3d_rare_to_bids(scan_info, args.bids_root, args.dry_run):
            success_count += 1

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Converted: {success_count}/{len(subjects_3d)}")
    if args.dry_run:
        logger.info("(DRY RUN - no files were actually created)")


if __name__ == '__main__':
    main()
