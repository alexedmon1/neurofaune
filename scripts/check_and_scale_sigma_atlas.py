#!/usr/bin/env python3
"""
Check and scale SIGMA atlas voxel sizes if needed.

The SIGMA rat brain atlas has a native resolution of ~0.1mm isotropic.
For FSL/ANTs compatibility, voxel sizes should be >= 1mm, so we scale by 10x.
"""

from pathlib import Path
import nibabel as nib
import shutil


def check_and_scale_atlas_file(
    atlas_file: Path,
    output_file: Path,
    expected_native_voxel: float = 0.1,
    scale_factor: float = 10.0,
    dry_run: bool = False
) -> None:
    """
    Check and scale an atlas file if needed.

    Parameters
    ----------
    atlas_file : Path
        Path to atlas NIfTI file
    output_file : Path
        Path to output scaled file
    expected_native_voxel : float
        Expected native voxel size in mm
    scale_factor : float
        Scaling factor to apply
    dry_run : bool
        If True, only report what would be done
    """
    print(f"\n{atlas_file.name} → {output_file.name}")

    # Load image
    img = nib.load(atlas_file)
    header = img.header
    voxel_sizes = header.get_zooms()[:3]

    print(f"  Current voxel sizes: {voxel_sizes}")
    print(f"  Dimensions: {img.shape[:3]}")

    # Check if scaling is needed
    # If voxels are < 1mm, they need scaling
    needs_scaling = any(v < 1.0 for v in voxel_sizes[:3])

    if needs_scaling:
        native_size = max(voxel_sizes[:3])  # Use the largest dimension as reference
        print(f"  → Voxels ~{native_size:.2f}mm (< 1mm), scaling by {scale_factor}x needed")

        if not dry_run:
            # Scale affine matrix
            affine = img.affine.copy()
            affine[:3, :3] = affine[:3, :3] * scale_factor

            # Update header
            new_header = header.copy()
            scaled_zooms = tuple(v * scale_factor for v in voxel_sizes[:3])
            new_header.set_zooms(scaled_zooms + voxel_sizes[3:])

            # Create new image
            new_img = nib.Nifti1Image(img.get_fdata(), affine, new_header)

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save to new location
            nib.save(new_img, output_file)
            print(f"  ✓ Scaled to: {scaled_zooms}")
        else:
            scaled_zooms = tuple(v * scale_factor for v in voxel_sizes[:3])
            print(f"  [DRY RUN] Would update to: {scaled_zooms}")

    else:
        min_size = min(voxel_sizes[:3])
        print(f"  ✓ Already scaled (voxels >= 1mm, min: {min_size:.2f}mm)")

        if not dry_run:
            # Copy file as-is to output directory
            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(atlas_file, output_file)
            print(f"  ✓ Copied to output directory")


def check_sigma_atlas(
    atlas_dir: Path,
    output_dir: Path,
    dry_run: bool = False
) -> None:
    """
    Check and scale all SIGMA atlas files to a new directory.

    Parameters
    ----------
    atlas_dir : Path
        Root directory of SIGMA atlas
    output_dir : Path
        Output directory for scaled atlas files
    dry_run : bool
        If True, only report what would be done
    """
    print("="*80)
    print("SIGMA Atlas Voxel Size Check and Scaling")
    print("="*80)
    print(f"Input:  {atlas_dir}")
    print(f"Output: {output_dir}")
    print()

    # Find all NIfTI files in atlas
    nifti_files = list(atlas_dir.rglob("*.nii")) + list(atlas_dir.rglob("*.nii.gz"))

    # Filter out Mac resource fork files
    nifti_files = [f for f in nifti_files if not f.name.startswith('._')]

    print(f"Found {len(nifti_files)} atlas files\n")

    for nifti_file in sorted(nifti_files):
        try:
            # Preserve directory structure in output
            relative_path = nifti_file.relative_to(atlas_dir)
            output_file = output_dir / relative_path

            check_and_scale_atlas_file(nifti_file, output_file, dry_run=dry_run)
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "="*80)
    if dry_run:
        print("DRY RUN COMPLETE - No files were created")
    else:
        print(f"COMPLETE - Scaled atlas saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Check and scale SIGMA atlas voxel sizes for FSL/ANTs compatibility'
    )
    parser.add_argument(
        '--atlas-dir',
        type=Path,
        default=Path('/mnt/arborea/atlases/SIGMA'),
        help='Root directory of SIGMA atlas (input)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/mnt/arborea/atlases/SIGMA_scaled'),
        help='Output directory for scaled atlas files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only report what would be done, do not create files'
    )

    args = parser.parse_args()

    check_sigma_atlas(args.atlas_dir, args.output_dir, args.dry_run)
