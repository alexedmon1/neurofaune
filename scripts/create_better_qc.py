#!/usr/bin/env python3
"""
Create improved QC visualization for skull stripping with better mask visibility.
"""

import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_improved_qc(
    original_file: Path,
    mask_file: Path,
    output_file: Path,
    title: str,
    n_slices: int = 9
):
    """Create improved mosaic QC with better mask visualization."""
    orig_img = nib.load(original_file)
    orig_data = orig_img.get_fdata()

    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata().astype(bool)

    # Find slices where mask exists
    coronal_mask_counts = mask_data.sum(axis=(0, 2))  # Sum over x and z
    slices_with_mask = np.where(coronal_mask_counts > 0)[0]

    if len(slices_with_mask) == 0:
        print(f"ERROR: No mask found in any slice!")
        return

    print(f"Mask present in {len(slices_with_mask)} coronal slices "
          f"(indices {slices_with_mask.min()}-{slices_with_mask.max()})")

    # Select evenly spaced slices from those that have mask
    if len(slices_with_mask) >= n_slices:
        slice_indices = np.linspace(0, len(slices_with_mask)-1, n_slices, dtype=int)
        slice_indices = slices_with_mask[slice_indices]
    else:
        slice_indices = slices_with_mask

    # Calculate grid dimensions
    n_cols = 3
    n_rows = (len(slice_indices) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes = axes.flatten() if len(slice_indices) > 1 else [axes]

    for idx, slice_idx in enumerate(slice_indices):
        if idx < len(axes):
            ax = axes[idx]

            # Get slice data
            orig_slice = orig_data[:, slice_idx, :].T
            mask_slice = mask_data[:, slice_idx, :].T

            # Show original
            ax.imshow(orig_slice, cmap='gray', origin='lower')

            # Overlay mask with higher alpha
            if mask_slice.sum() > 0:
                ax.imshow(mask_slice, cmap='Reds', alpha=0.4, origin='lower')

                # Add edge contour for better visibility
                edges = ndimage.binary_dilation(mask_slice) & ~mask_slice
                if edges.sum() > 0:
                    ax.contour(mask_slice, levels=[0.5], colors='red', linewidths=2)

            # Add statistics
            mask_count = mask_slice.sum()
            ax.set_title(f'Coronal {slice_idx} ({mask_count:,} mask voxels)',
                        fontsize=10)
            ax.axis('off')

    # Hide unused subplots
    for idx in range(len(slice_indices), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")


def main():
    """Generate improved QC for all parameter combinations."""

    base_dir = Path('/mnt/arborea/bpa-rat/test/skull_strip_params_comparison')
    qc_dir = base_dir / 'qc_improved'
    qc_dir.mkdir(exist_ok=True)

    configs = ['baseline', 'lower_frac', 'with_B_flag', 'aggressive', 'very_aggressive']

    subject = 'sub-Rat108'
    session = 'ses-p30'

    ref_volume = base_dir / 'work' / subject / session / f'{subject}_{session}_bold_ref.nii.gz'

    for config_name in configs:
        print(f"\n{'='*80}")
        print(f"Creating improved QC for: {config_name}")
        print('='*80)

        mask_file = base_dir / 'work' / subject / session / config_name / f'{subject}_{session}_mask.nii.gz'

        if not mask_file.exists():
            print(f"  Mask not found: {mask_file}")
            continue

        # Load mask to get stats
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata().astype(bool)
        mask_voxels = mask_data.sum()

        output_file = qc_dir / f'{config_name}_improved_qc.png'
        title = f'{subject} {session} - {config_name}\nMask: {mask_voxels:,} voxels'

        create_improved_qc(ref_volume, mask_file, output_file, title)

    print(f"\n{'='*80}")
    print("IMPROVED QC COMPLETE")
    print('='*80)
    print(f"Output directory: {qc_dir}")
    print(f"\nView images:")
    print(f"  ls {qc_dir}/*.png")
    print('='*80)


if __name__ == '__main__':
    main()
