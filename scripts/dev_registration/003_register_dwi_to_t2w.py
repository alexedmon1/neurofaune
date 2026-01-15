#!/usr/bin/env python3
"""
003_register_dwi_to_t2w.py

Register DWI (FA or b0) to corresponding T2w slices within the same subject.

This script:
1. Determines which T2w slices correspond to DWI coverage (based on world coordinates)
2. Extracts those T2w slices as registration target
3. Runs ANTs rigid/affine registration (no SyN - same subject, minor differences)
4. Saves transforms for later use in the registration chain

Prerequisites:
- DTI preprocessing must be complete (FA map available)
- T2w preprocessing must be complete
- Both images should have correct voxel sizes in headers

Usage:
    python 003_register_dwi_to_t2w.py /path/to/bpa-rat sub-Rat1 ses-p60

Output:
    - {output_dir}/transforms/{subject}/{session}/FA_to_T2w_0GenericAffine.mat
    - {output_dir}/transforms/{subject}/{session}/FA_to_T2w_Warped.nii.gz
    - {output_dir}/work/{subject}/{session}/T2w_slices_for_dwi.nii.gz
"""

import argparse
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def get_z_world_range(img):
    """Get min/max z-coordinate in world space."""
    shape = img.shape[:3]
    z_coords = []
    for i in range(shape[2]):
        world = img.affine @ np.array([shape[0]//2, shape[1]//2, i, 1])
        z_coords.append(world[2])
    return min(z_coords), max(z_coords)


def extract_t2w_slices_for_dwi(t2w_img, dwi_img, output_path):
    """
    Extract T2w slices that correspond to DWI coverage.

    Parameters
    ----------
    t2w_img : Nifti1Image
        Full T2w image
    dwi_img : Nifti1Image
        DWI image (FA or b0)
    output_path : Path
        Where to save extracted slices

    Returns
    -------
    Path
        Path to extracted T2w slices
    dict
        Slice extraction metadata
    """
    t2w_shape = t2w_img.shape[:3]

    # Get DWI z-range in world coordinates
    dwi_z_min, dwi_z_max = get_z_world_range(dwi_img)

    # Get T2w slice z-coordinates
    t2w_slice_z = []
    for i in range(t2w_shape[2]):
        world = t2w_img.affine @ np.array([t2w_shape[0]//2, t2w_shape[1]//2, i, 1])
        t2w_slice_z.append(world[2])
    t2w_slice_z = np.array(t2w_slice_z)

    # Calculate slice spacing for tolerance
    t2w_spacing = np.abs(np.diff(t2w_slice_z)).mean() if len(t2w_slice_z) > 1 else 1.0
    tolerance = t2w_spacing / 2

    # Find T2w slices within DWI range
    in_range = (t2w_slice_z >= dwi_z_min - tolerance) & (t2w_slice_z <= dwi_z_max + tolerance)
    slice_indices = np.where(in_range)[0]

    if len(slice_indices) == 0:
        raise ValueError(f"No T2w slices found within DWI range ({dwi_z_min:.1f} to {dwi_z_max:.1f} mm)")

    print(f"  DWI z-range: {dwi_z_min:.1f} to {dwi_z_max:.1f} mm")
    print(f"  T2w slices in range: {slice_indices[0]} to {slice_indices[-1]} ({len(slice_indices)} slices)")

    # Extract slices
    t2w_data = t2w_img.get_fdata()
    extracted_data = t2w_data[:, :, slice_indices]

    # Update affine to reflect new origin
    new_affine = t2w_img.affine.copy()
    # Shift origin by the number of removed slices
    new_affine[:3, 3] += new_affine[:3, 2] * slice_indices[0]

    # Create new image
    extracted_img = nib.Nifti1Image(extracted_data, new_affine, t2w_img.header)

    # Update header dimensions
    extracted_img.header.set_data_shape(extracted_data.shape)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(extracted_img, output_path)
    print(f"  Saved extracted T2w: {output_path}")

    metadata = {
        'original_t2w_shape': t2w_shape,
        'extracted_shape': extracted_data.shape,
        'slice_indices': slice_indices.tolist(),
        'dwi_z_range': (dwi_z_min, dwi_z_max),
    }

    return output_path, metadata


def run_ants_registration(
    moving_image: Path,
    fixed_image: Path,
    output_prefix: Path,
    transform_type: str = 'Affine',
    n_cores: int = 4
):
    """
    Run ANTs registration (rigid or affine - no SyN for within-subject).

    Parameters
    ----------
    moving_image : Path
        Image to register (DWI FA/b0)
    fixed_image : Path
        Target image (extracted T2w slices)
    output_prefix : Path
        Output prefix for transform files
    transform_type : str
        'Rigid' or 'Affine' (default: Affine)
    n_cores : int
        Number of CPU cores

    Returns
    -------
    dict
        Paths to output files
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Use antsRegistrationSyN.sh with 'a' for affine or 'r' for rigid
    transform_flag = 'a' if transform_type == 'Affine' else 'r'

    cmd = [
        'antsRegistrationSyN.sh',
        '-d', '3',
        '-f', str(fixed_image),
        '-m', str(moving_image),
        '-o', str(output_prefix),
        '-n', str(n_cores),
        '-t', transform_flag
    ]

    print(f"\n  Running ANTs {transform_type} registration...")
    print(f"  Moving: {moving_image.name}")
    print(f"  Fixed: {fixed_image.name}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if result.returncode != 0:
        print(f"  ERROR: Registration failed!")
        print(result.stdout)
        raise RuntimeError("ANTs registration failed")

    # Expected outputs
    outputs = {
        'affine': Path(str(output_prefix) + '0GenericAffine.mat'),
        'warped': Path(str(output_prefix) + 'Warped.nii.gz'),
    }

    # Check outputs exist
    for name, path in outputs.items():
        if not path.exists():
            print(f"  WARNING: Expected output not found: {path}")
        else:
            print(f"  ✓ {name}: {path.name}")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description='Register DWI to T2w within subject'
    )
    parser.add_argument('study_root', type=Path, help='Path to study root')
    parser.add_argument('subject', type=str, help='Subject ID')
    parser.add_argument('session', type=str, help='Session ID')
    parser.add_argument('--transform-type', choices=['Rigid', 'Affine'],
                        default='Affine', help='Registration type (default: Affine)')
    parser.add_argument('--n-cores', type=int, default=4, help='Number of CPU cores')
    parser.add_argument('--use-b0', action='store_true',
                        help='Use b0 instead of FA for registration')
    args = parser.parse_args()

    # Normalize IDs
    subject = args.subject if args.subject.startswith('sub-') else f'sub-{args.subject}'
    session = args.session if args.session.startswith('ses-') else f'ses-{args.session}'

    derivatives_root = args.study_root / 'derivatives'
    transforms_root = args.study_root / 'transforms'
    work_root = args.study_root / 'work'

    print("=" * 70)
    print(f"DWI to T2w Registration: {subject} / {session}")
    print("=" * 70)

    # Find T2w
    t2w_path = derivatives_root / subject / session / 'anat' / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
    if not t2w_path.exists():
        print(f"ERROR: T2w not found: {t2w_path}")
        sys.exit(1)

    # Find FA (or b0)
    dwi_dir = derivatives_root / subject / session / 'dwi'
    if args.use_b0:
        dwi_path = dwi_dir / f'{subject}_{session}_b0.nii.gz'
        dwi_label = 'b0'
    else:
        dwi_path = dwi_dir / f'{subject}_{session}_FA.nii.gz'
        dwi_label = 'FA'

    if not dwi_path.exists():
        print(f"ERROR: {dwi_label} not found: {dwi_path}")
        print(f"  Has DTI preprocessing been run for this subject?")
        sys.exit(1)

    print(f"\nInputs:")
    print(f"  T2w: {t2w_path}")
    print(f"  {dwi_label}: {dwi_path}")

    # Load images
    t2w_img = nib.load(t2w_path)
    dwi_img = nib.load(dwi_path)

    # Check for identity affine (indicates header issue)
    if np.allclose(dwi_img.affine, np.eye(4)):
        print(f"\n⚠ WARNING: DWI has identity affine - voxel sizes may be incorrect!")
        print(f"  Run fix_bruker_voxel_sizes.py first, or results will be wrong.")
        sys.exit(1)

    # Step 1: Extract matching T2w slices
    print(f"\nStep 1: Extracting T2w slices matching DWI coverage...")
    t2w_slices_path = work_root / subject / session / 'T2w_slices_for_dwi.nii.gz'
    t2w_slices_path, slice_metadata = extract_t2w_slices_for_dwi(
        t2w_img, dwi_img, t2w_slices_path
    )

    # Step 2: Run registration
    print(f"\nStep 2: Running {args.transform_type} registration...")
    output_prefix = transforms_root / subject / session / f'{dwi_label}_to_T2w_'

    reg_outputs = run_ants_registration(
        moving_image=dwi_path,
        fixed_image=t2w_slices_path,
        output_prefix=output_prefix,
        transform_type=args.transform_type,
        n_cores=args.n_cores
    )

    # Summary
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Transform: {reg_outputs['affine']}")
    print(f"Warped {dwi_label}: {reg_outputs['warped']}")
    print(f"T2w slice indices used: {slice_metadata['slice_indices']}")
    print()
    print("Next steps:")
    print("  1. Visually QC the registration (overlay warped FA on T2w)")
    print("  2. Use this transform in the chain: FA → T2w → Template → SIGMA")


if __name__ == '__main__':
    main()
