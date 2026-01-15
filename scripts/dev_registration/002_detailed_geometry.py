#!/usr/bin/env python3
"""
002_detailed_geometry.py

Detailed examination of T2w and DWI geometry for a single subject.
Looks at raw headers, affine matrices, and actual data to understand
the spatial relationship.

Usage:
    python 002_detailed_geometry.py /path/to/bpa-rat sub-Rat1 ses-p60
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def print_image_info(img_path, label):
    """Print detailed information about a NIfTI image."""
    print(f"\n{'='*60}")
    print(f"{label}: {img_path.name}")
    print(f"{'='*60}")

    img = nib.load(img_path)
    header = img.header

    print(f"\nShape: {img.shape}")
    print(f"Data type: {header.get_data_dtype()}")

    # Voxel sizes from header
    zooms = header.get_zooms()
    print(f"\nVoxel sizes (zooms): {zooms}")

    # Affine matrix
    print(f"\nAffine matrix:")
    print(img.affine)

    # Derive voxel sizes from affine (more reliable)
    affine_voxel_sizes = np.sqrt(np.sum(img.affine[:3, :3]**2, axis=0))
    print(f"\nVoxel sizes from affine: {affine_voxel_sizes}")

    # World coordinate extent
    shape = img.shape[:3]
    corners = np.array([
        [0, 0, 0, 1],
        [shape[0]-1, shape[1]-1, shape[2]-1, 1]
    ])
    world_corners = (img.affine @ corners.T).T[:, :3]
    print(f"\nWorld extent:")
    print(f"  Min corner: {world_corners[0]}")
    print(f"  Max corner: {world_corners[1]}")
    print(f"  Range: {world_corners[1] - world_corners[0]}")

    # Slice positions along Z
    print(f"\nZ-slice world positions (first 5, last 5):")
    for i in range(min(5, shape[2])):
        z_world = img.affine @ np.array([shape[0]//2, shape[1]//2, i, 1])
        print(f"  Slice {i}: z = {z_world[2]:.2f} mm")
    if shape[2] > 10:
        print("  ...")
        for i in range(max(5, shape[2]-5), shape[2]):
            z_world = img.affine @ np.array([shape[0]//2, shape[1]//2, i, 1])
            print(f"  Slice {i}: z = {z_world[2]:.2f} mm")

    # Orientation
    ornt = nib.orientations.io_orientation(img.affine)
    ornt_codes = nib.orientations.ornt2axcodes(ornt)
    print(f"\nOrientation: {ornt_codes}")

    # Check for scaling issues (rodent MRI often has sub-mm voxels)
    if any(v < 0.5 for v in zooms[:3]):
        print("\n⚠ WARNING: Very small voxel sizes detected - may need 10x scaling")
    if any(v > 5 for v in zooms[:3]):
        print("\n⚠ WARNING: Large voxel/slice thickness detected")

    return img


def compare_geometry(t2w_img, dwi_img):
    """Compare geometry between T2w and DWI."""
    print(f"\n{'='*60}")
    print("GEOMETRY COMPARISON")
    print(f"{'='*60}")

    # Get Z extents
    t2w_shape = t2w_img.shape[:3]
    dwi_shape = dwi_img.shape[:3]

    # Calculate Z range in world coordinates
    def get_z_range(img):
        shape = img.shape[:3]
        z_start = (img.affine @ np.array([0, 0, 0, 1]))[2]
        z_end = (img.affine @ np.array([0, 0, shape[2]-1, 1]))[2]
        return min(z_start, z_end), max(z_start, z_end)

    t2w_z = get_z_range(t2w_img)
    dwi_z = get_z_range(dwi_img)

    print(f"\nT2w Z range: {t2w_z[0]:.2f} to {t2w_z[1]:.2f} mm")
    print(f"DWI Z range: {dwi_z[0]:.2f} to {dwi_z[1]:.2f} mm")

    # Check overlap
    overlap_start = max(t2w_z[0], dwi_z[0])
    overlap_end = min(t2w_z[1], dwi_z[1])

    if overlap_start < overlap_end:
        print(f"\nOverlap: {overlap_start:.2f} to {overlap_end:.2f} mm ({overlap_end - overlap_start:.2f} mm)")

        # Which T2w slices fall in this range?
        t2w_slice_z = []
        for i in range(t2w_shape[2]):
            z = (t2w_img.affine @ np.array([0, 0, i, 1]))[2]
            t2w_slice_z.append(z)
        t2w_slice_z = np.array(t2w_slice_z)

        in_overlap = (t2w_slice_z >= overlap_start) & (t2w_slice_z <= overlap_end)
        print(f"T2w slices in overlap: {np.where(in_overlap)[0].tolist()}")
        print(f"  ({np.sum(in_overlap)} of {t2w_shape[2]} slices)")
    else:
        print("\n⚠ NO OVERLAP - images may have incorrect headers or different coordinate systems")

    # Check if images might need alignment
    print("\n--- Registration considerations ---")

    # In-plane resolution ratio
    t2w_inplane = np.sqrt(np.sum(t2w_img.affine[:3, 0]**2))
    dwi_inplane = np.sqrt(np.sum(dwi_img.affine[:3, 0]**2))
    print(f"In-plane resolution ratio (DWI/T2w): {dwi_inplane/t2w_inplane:.2f}")

    # Slice thickness ratio
    t2w_slice = np.sqrt(np.sum(t2w_img.affine[:3, 2]**2))
    dwi_slice = np.sqrt(np.sum(dwi_img.affine[:3, 2]**2))
    print(f"Slice thickness ratio (DWI/T2w): {dwi_slice/t2w_slice:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Detailed T2w/DWI geometry examination')
    parser.add_argument('study_root', type=Path, help='Path to study root')
    parser.add_argument('subject', type=str, help='Subject ID (e.g., sub-Rat1)')
    parser.add_argument('session', type=str, help='Session ID (e.g., ses-p60)')
    args = parser.parse_args()

    subject = args.subject if args.subject.startswith('sub-') else f'sub-{args.subject}'
    session = args.session if args.session.startswith('ses-') else f'ses-{args.session}'

    bids_root = args.study_root / 'raw' / 'bids'
    derivatives_root = args.study_root / 'derivatives'

    # Find T2w
    t2w_path = derivatives_root / subject / session / 'anat' / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
    if not t2w_path.exists():
        # Try without session
        t2w_candidates = list((derivatives_root / subject).glob(f'**/*desc-preproc_T2w.nii.gz'))
        if t2w_candidates:
            t2w_path = t2w_candidates[0]
            print(f"Using T2w: {t2w_path}")
        else:
            print(f"ERROR: T2w not found for {subject}")
            sys.exit(1)

    # Find DWI
    dwi_dir = bids_root / subject / session / 'dwi'
    if not dwi_dir.exists():
        # Try to find any DWI for this subject
        dwi_candidates = list((bids_root / subject).glob('**/dwi/*_dwi.nii.gz'))
        if dwi_candidates:
            dwi_path = dwi_candidates[0]
            print(f"Using DWI: {dwi_path}")
        else:
            print(f"ERROR: DWI not found for {subject}")
            sys.exit(1)
    else:
        dwi_files = list(dwi_dir.glob('*_dwi.nii.gz'))
        if not dwi_files:
            print(f"ERROR: No DWI files in {dwi_dir}")
            sys.exit(1)
        dwi_path = dwi_files[0]

    # Analyze both images
    t2w_img = print_image_info(t2w_path, "T2w (preprocessed)")
    dwi_img = print_image_info(dwi_path, "DWI (raw)")

    # Compare
    compare_geometry(t2w_img, dwi_img)


if __name__ == '__main__':
    main()
