#!/usr/bin/env python3
"""
001_explore_geometry.py

Explore the geometric differences between T2w and DWI acquisitions.
This helps us understand:
- How many slices each modality has
- Voxel sizes and orientations
- World coordinate overlap (which T2w slices correspond to DWI)

Usage:
    python 001_explore_geometry.py /path/to/bpa-rat

Output:
    Prints geometry comparison for sample subjects
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def get_world_extent(img):
    """Get world coordinate extent (min/max corners) of an image."""
    shape = img.shape[:3]
    affine = img.affine

    # Get all 8 corners in voxel space
    corners_vox = np.array([
        [0, 0, 0, 1],
        [shape[0]-1, 0, 0, 1],
        [0, shape[1]-1, 0, 1],
        [0, 0, shape[2]-1, 1],
        [shape[0]-1, shape[1]-1, 0, 1],
        [shape[0]-1, 0, shape[2]-1, 1],
        [0, shape[1]-1, shape[2]-1, 1],
        [shape[0]-1, shape[1]-1, shape[2]-1, 1],
    ])

    # Transform to world coordinates
    corners_world = (affine @ corners_vox.T).T[:, :3]

    return {
        'min': corners_world.min(axis=0),
        'max': corners_world.max(axis=0),
        'center': corners_world.mean(axis=0)
    }


def get_slice_world_coords(img, axis=2):
    """Get world z-coordinate for each slice along given axis."""
    shape = img.shape[:3]
    affine = img.affine

    # For each slice, get center point
    slice_coords = []
    for i in range(shape[axis]):
        # Create voxel coordinate at center of this slice
        vox = [shape[0]//2, shape[1]//2, shape[2]//2, 1]
        vox[axis] = i

        # Transform to world
        world = affine @ np.array(vox)
        slice_coords.append(world[axis])  # Get coordinate along slice axis

    return np.array(slice_coords)


def find_overlapping_slices(t2w_img, dwi_img, axis=2):
    """
    Find which T2w slices overlap with DWI coverage in world coordinates.

    Returns:
        dict with:
        - t2w_slice_indices: indices of T2w slices that overlap with DWI
        - overlap_fraction: what fraction of each T2w slice overlaps
    """
    t2w_z = get_slice_world_coords(t2w_img, axis)
    dwi_z = get_slice_world_coords(dwi_img, axis)

    # Get DWI z-range
    dwi_z_min = dwi_z.min()
    dwi_z_max = dwi_z.max()

    # Find T2w slices within DWI range
    # Account for slice thickness (approximate as spacing between slices)
    t2w_spacing = np.abs(np.diff(t2w_z)).mean() if len(t2w_z) > 1 else 1.0
    dwi_spacing = np.abs(np.diff(dwi_z)).mean() if len(dwi_z) > 1 else 1.0

    # T2w slices that fall within DWI extent (with half-slice tolerance)
    tolerance = max(t2w_spacing, dwi_spacing) / 2
    overlapping = (t2w_z >= dwi_z_min - tolerance) & (t2w_z <= dwi_z_max + tolerance)

    return {
        't2w_slice_indices': np.where(overlapping)[0],
        't2w_z_coords': t2w_z[overlapping],
        'dwi_z_range': (dwi_z_min, dwi_z_max),
        't2w_z_range': (t2w_z.min(), t2w_z.max()),
        't2w_spacing': t2w_spacing,
        'dwi_spacing': dwi_spacing
    }


def analyze_subject(bids_root, derivatives_root, subject, session):
    """Analyze geometry for a single subject/session."""

    # Find T2w (preprocessed)
    t2w_path = derivatives_root / subject / session / 'anat' / f'{subject}_{session}_desc-preproc_T2w.nii.gz'

    # Find DWI (raw - may need to check multiple runs)
    dwi_dir = bids_root / subject / session / 'dwi'
    dwi_files = list(dwi_dir.glob(f'{subject}_{session}_*_dwi.nii.gz')) if dwi_dir.exists() else []

    if not t2w_path.exists():
        return {'error': f'T2w not found: {t2w_path}'}

    if not dwi_files:
        return {'error': f'No DWI files found in {dwi_dir}'}

    # Load images
    t2w_img = nib.load(t2w_path)
    dwi_img = nib.load(dwi_files[0])  # Use first DWI file

    # Get basic info
    result = {
        'subject': subject,
        'session': session,
        't2w_file': t2w_path.name,
        'dwi_file': dwi_files[0].name,
        't2w_shape': t2w_img.shape[:3],
        'dwi_shape': dwi_img.shape[:3],
        't2w_voxel_size': tuple(round(v, 3) for v in t2w_img.header.get_zooms()[:3]),
        'dwi_voxel_size': tuple(round(v, 3) for v in dwi_img.header.get_zooms()[:3]),
    }

    # Get world extent
    t2w_extent = get_world_extent(t2w_img)
    dwi_extent = get_world_extent(dwi_img)

    result['t2w_world_extent'] = {
        'min': tuple(round(v, 2) for v in t2w_extent['min']),
        'max': tuple(round(v, 2) for v in t2w_extent['max'])
    }
    result['dwi_world_extent'] = {
        'min': tuple(round(v, 2) for v in dwi_extent['min']),
        'max': tuple(round(v, 2) for v in dwi_extent['max'])
    }

    # Find overlapping slices
    overlap = find_overlapping_slices(t2w_img, dwi_img)
    result['overlap'] = {
        't2w_slices_in_dwi_range': list(overlap['t2w_slice_indices']),
        'n_overlapping_slices': len(overlap['t2w_slice_indices']),
        'dwi_z_range_mm': tuple(round(v, 2) for v in overlap['dwi_z_range']),
        't2w_z_range_mm': tuple(round(v, 2) for v in overlap['t2w_z_range']),
        't2w_slice_spacing_mm': round(overlap['t2w_spacing'], 3),
        'dwi_slice_spacing_mm': round(overlap['dwi_spacing'], 3),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Explore T2w/DWI geometry differences')
    parser.add_argument('study_root', type=Path, help='Path to study root (e.g., /mnt/arborea/bpa-rat)')
    parser.add_argument('--subjects', nargs='+', help='Specific subjects to analyze')
    parser.add_argument('--max-subjects', type=int, default=5, help='Max subjects to analyze if not specified')
    args = parser.parse_args()

    bids_root = args.study_root / 'raw' / 'bids'
    derivatives_root = args.study_root / 'derivatives'

    if not bids_root.exists():
        print(f"ERROR: BIDS root not found: {bids_root}")
        sys.exit(1)

    # Find subjects with both T2w and DWI
    if args.subjects:
        subjects_to_check = [(f'sub-{s}' if not s.startswith('sub-') else s) for s in args.subjects]
    else:
        # Find subjects with DWI data
        dwi_subjects = set()
        for dwi_file in bids_root.glob('sub-*/ses-*/dwi/*.nii.gz'):
            parts = dwi_file.parts
            subject = [p for p in parts if p.startswith('sub-')][0]
            session = [p for p in parts if p.startswith('ses-')][0]
            dwi_subjects.add((subject, session))

        subjects_to_check = sorted(dwi_subjects)[:args.max_subjects]

    print("=" * 80)
    print("T2w / DWI Geometry Comparison")
    print("=" * 80)
    print(f"Study root: {args.study_root}")
    print(f"Analyzing {len(subjects_to_check)} subject/session pairs")
    print()

    results = []
    for item in subjects_to_check:
        if isinstance(item, tuple):
            subject, session = item
        else:
            # Find sessions for this subject
            subject = item
            sessions = list((derivatives_root / subject).glob('ses-*'))
            if not sessions:
                print(f"  {subject}: No sessions found")
                continue
            session = sessions[0].name

        print(f"Analyzing {subject} / {session}...")
        result = analyze_subject(bids_root, derivatives_root, subject, session)

        if 'error' in result:
            print(f"  ERROR: {result['error']}")
            continue

        results.append(result)

        # Print summary
        print(f"  T2w: {result['t2w_shape']} @ {result['t2w_voxel_size']} mm")
        print(f"  DWI: {result['dwi_shape']} @ {result['dwi_voxel_size']} mm")
        print(f"  DWI covers T2w slices: {result['overlap']['t2w_slices_in_dwi_range']}")
        print(f"  ({result['overlap']['n_overlapping_slices']} of {result['t2w_shape'][2]} T2w slices)")
        print()

    # Summary across subjects
    if results:
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Check consistency
        dwi_shapes = [r['dwi_shape'] for r in results]
        t2w_shapes = [r['t2w_shape'] for r in results]
        n_overlaps = [r['overlap']['n_overlapping_slices'] for r in results]

        print(f"DWI shapes: {set(dwi_shapes)}")
        print(f"T2w shapes: {set(t2w_shapes)}")
        print(f"Overlapping slices: min={min(n_overlaps)}, max={max(n_overlaps)}, mean={np.mean(n_overlaps):.1f}")

        # Show typical slice indices
        if results:
            typical_slices = results[0]['overlap']['t2w_slices_in_dwi_range']
            print(f"Typical T2w slice range for DWI: {min(typical_slices)} to {max(typical_slices)}")


if __name__ == '__main__':
    main()
