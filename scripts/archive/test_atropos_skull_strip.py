#!/usr/bin/env python3
"""
Test Atropos-based skull stripping on functional BOLD data.

This script tests the robust skull stripping approach (Atropos + BET + morphology)
adapted from anatomical preprocessing.
"""

import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add neurofaune to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.preprocess.utils.func.skull_strip_atropos import skull_strip_bold_atropos


def extract_middle_volume(bold_file: Path, output_file: Path) -> Path:
    """Extract middle volume from 4D BOLD timeseries."""
    print(f"Extracting middle volume from {bold_file.name}...")

    img = nib.load(bold_file)
    data = img.get_fdata()

    if len(data.shape) == 4:
        middle_vol = data.shape[3] // 2
        print(f"  Extracting volume {middle_vol} of {data.shape[3]}")
        ref_data = data[..., middle_vol]
    else:
        print(f"  Input is already 3D")
        ref_data = data

    ref_img = nib.Nifti1Image(ref_data, img.affine, img.header)
    nib.save(ref_img, output_file)

    print(f"  Saved: {output_file}")
    return output_file


def create_qc_visualization(
    original_file: Path,
    brain_file: Path,
    mask_file: Path,
    output_file: Path,
    subject: str
):
    """Create QC visualization comparing original, mask, and brain."""
    print(f"\nCreating QC visualization...")

    # Load images
    orig_img = nib.load(original_file)
    orig_data = orig_img.get_fdata()

    brain_img = nib.load(brain_file)
    brain_data = brain_img.get_fdata()

    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()

    # Get middle slices
    mid_axial = orig_data.shape[2] // 2
    mid_coronal = orig_data.shape[1] // 2
    mid_sagittal = orig_data.shape[0] // 2

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig)

    fig.suptitle(f'{subject} - Atropos-Based BOLD Skull Stripping',
                 fontsize=16, fontweight='bold')

    # Row 1: Axial views
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_data[:, :, mid_axial].T, cmap='gray', origin='lower')
    ax1.set_title('Original (Axial)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(orig_data[:, :, mid_axial].T, cmap='gray', origin='lower')
    ax2.imshow(mask_data[:, :, mid_axial].T, cmap='Reds', alpha=0.3, origin='lower')
    ax2.set_title('Mask Overlay (Axial)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(brain_data[:, :, mid_axial].T, cmap='gray', origin='lower')
    ax3.set_title('Brain Extracted (Axial)', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Row 2: Coronal views
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(orig_data[:, mid_coronal, :].T, cmap='gray', origin='lower')
    ax4.set_title('Original (Coronal)', fontsize=12, fontweight='bold')
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(orig_data[:, mid_coronal, :].T, cmap='gray', origin='lower')
    ax5.imshow(mask_data[:, mid_coronal, :].T, cmap='Reds', alpha=0.3, origin='lower')
    ax5.set_title('Mask Overlay (Coronal)', fontsize=12, fontweight='bold')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(brain_data[:, mid_coronal, :].T, cmap='gray', origin='lower')
    ax6.set_title('Brain Extracted (Coronal)', fontsize=12, fontweight='bold')
    ax6.axis('off')

    # Row 3: Sagittal views
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.imshow(orig_data[mid_sagittal, :, :].T, cmap='gray', origin='lower')
    ax7.set_title('Original (Sagittal)', fontsize=12, fontweight='bold')
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.imshow(orig_data[mid_sagittal, :, :].T, cmap='gray', origin='lower')
    ax8.imshow(mask_data[mid_sagittal, :, :].T, cmap='Reds', alpha=0.3, origin='lower')
    ax8.set_title('Mask Overlay (Sagittal)', fontsize=12, fontweight='bold')
    ax8.axis('off')

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.imshow(brain_data[mid_sagittal, :, :].T, cmap='gray', origin='lower')
    ax9.set_title('Brain Extracted (Sagittal)', fontsize=12, fontweight='bold')
    ax9.axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved QC figure: {output_file}")


def main():
    """Run test on a single subject."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Test Atropos-based skull stripping on BOLD data'
    )
    parser.add_argument('--subject', default='sub-Rat108', help='Subject ID')
    parser.add_argument('--session', default='ses-p30', help='Session ID')
    parser.add_argument('--bold-file', type=Path,
                       help='Path to BOLD file (default: auto-detect)')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('/mnt/arborea/bpa-rat/test/atropos_skull_strip'),
                       help='Output directory')

    args = parser.parse_args()

    # Auto-detect BOLD file if not specified
    if args.bold_file is None:
        bold_file = Path(f'/mnt/arborea/bpa-rat/raw/bids/{args.subject}/{args.session}/func/'
                        f'{args.subject}_{args.session}_run-13_bold.nii.gz')
    else:
        bold_file = args.bold_file

    if not bold_file.exists():
        print(f"ERROR: BOLD file not found: {bold_file}")
        return 1

    # Create output directories
    work_dir = args.output_dir / 'work' / args.subject / args.session
    work_dir.mkdir(parents=True, exist_ok=True)

    qc_dir = args.output_dir / 'qc'
    qc_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TESTING ATROPOS-BASED SKULL STRIPPING ON BOLD DATA")
    print("="*80)
    print(f"Subject: {args.subject}")
    print(f"Session: {args.session}")
    print(f"BOLD file: {bold_file}")
    print(f"Output: {args.output_dir}")
    print()

    # Extract middle volume
    ref_volume = work_dir / f"{args.subject}_{args.session}_bold_ref.nii.gz"
    extract_middle_volume(bold_file, ref_volume)

    # Run Atropos-based skull stripping
    brain_file = work_dir / f"{args.subject}_{args.session}_bold_brain.nii.gz"
    mask_file = work_dir / f"{args.subject}_{args.session}_bold_brain_mask.nii.gz"

    try:
        brain_file, mask_file = skull_strip_bold_atropos(
            input_file=ref_volume,
            output_file=brain_file,
            mask_file=mask_file
        )

        # Calculate statistics
        print("\nCalculating statistics...")
        orig_data = nib.load(ref_volume).get_fdata()
        mask_data = nib.load(mask_file).get_fdata()
        brain_data = nib.load(brain_file).get_fdata()

        orig_nonzero = (orig_data > 0).sum()
        mask_voxels = (mask_data > 0).sum()
        extraction_ratio = mask_voxels / orig_nonzero if orig_nonzero > 0 else 0

        print(f"  Original non-zero voxels: {orig_nonzero:,}")
        print(f"  Mask voxels: {mask_voxels:,}")
        print(f"  Extraction ratio: {extraction_ratio:.3f}")

        # Create QC visualization
        qc_file = qc_dir / f"{args.subject}_{args.session}_atropos_skull_strip_qc.png"
        create_qc_visualization(
            ref_volume,
            brain_file,
            mask_file,
            qc_file,
            f"{args.subject} {args.session}"
        )

        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"Brain: {brain_file}")
        print(f"Mask: {mask_file}")
        print(f"QC: {qc_file}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
