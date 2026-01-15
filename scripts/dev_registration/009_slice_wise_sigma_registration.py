#!/usr/bin/env python
"""
Test slice-wise registration of p60 template to SIGMA atlas.

This script tests the 2D slice-wise registration approach for handling
thick-slice coronal templates.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from neurofaune.templates.slice_registration import (
    compute_slice_correspondence,
    extract_coronal_slab_atlas,
    extract_coronal_slice_template,
    get_slice_geometry,
    slice_wise_registration,
)


def visualize_slice_correspondence(
    template_path: Path,
    atlas_path: Path,
    output_path: Path,
    template_coronal_axis: int = 2,
    atlas_coronal_axis: int = 1
):
    """Visualize the slice correspondence between template and atlas."""
    template_img = nib.load(template_path)
    atlas_img = nib.load(atlas_path)

    template_data = np.squeeze(template_img.get_fdata())
    atlas_data = np.squeeze(atlas_img.get_fdata())

    template_geom = get_slice_geometry(template_img)
    atlas_geom = get_slice_geometry(atlas_img)

    correspondences = compute_slice_correspondence(
        template_geom, atlas_geom,
        template_data, atlas_data,
        template_coronal_axis, atlas_coronal_axis
    )

    # Select a few slices to visualize
    slice_indices = [5, 15, 20, 30, 35]
    slice_indices = [i for i in slice_indices if i < template_geom['shape'][template_coronal_axis]]

    fig, axes = plt.subplots(len(slice_indices), 3, figsize=(12, 4 * len(slice_indices)))

    atlas_voxel = atlas_geom['voxel_size'][atlas_coronal_axis]

    for row, slice_idx in enumerate(slice_indices):
        corr = correspondences[slice_idx]

        # Extract template slice
        template_slice = extract_coronal_slice_template(
            template_data, slice_idx, template_coronal_axis
        )

        # Extract atlas slab
        atlas_slice = extract_coronal_slab_atlas(
            atlas_data,
            corr['atlas_center_mm'],
            corr['atlas_thickness_mm'],
            atlas_voxel,
            atlas_coronal_axis
        )

        # Show template slice
        axes[row, 0].imshow(template_slice.T, cmap='gray', origin='lower')
        axes[row, 0].set_title(f'Template slice {slice_idx}\n({corr["template_mm"]:.1f}mm)')
        axes[row, 0].set_ylabel(f'Slice {slice_idx}')

        # Show atlas slice
        axes[row, 1].imshow(atlas_slice.T, cmap='gray', origin='lower')
        axes[row, 1].set_title(f'Atlas slab\n(center={corr["atlas_center_mm"]:.1f}mm)')

        # Show overlay (resample template to atlas size)
        from scipy.ndimage import zoom
        if template_slice.shape != atlas_slice.shape:
            zoom_factors = [atlas_slice.shape[i] / template_slice.shape[i] for i in range(2)]
            template_resampled = zoom(template_slice, zoom_factors, order=1)
        else:
            template_resampled = template_slice

        # Normalize for overlay
        t_norm = template_resampled / (template_resampled.max() + 1e-10)
        a_norm = atlas_slice / (atlas_slice.max() + 1e-10)

        rgb = np.zeros((*t_norm.T.shape, 3))
        rgb[..., 0] = a_norm.T  # Red = atlas
        rgb[..., 1] = t_norm.T  # Green = template
        axes[row, 2].imshow(np.clip(rgb, 0, 1), origin='lower')
        axes[row, 2].set_title('Overlay\nR=Atlas, G=Template')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Slice Correspondence: Template vs SIGMA Atlas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {output_path}')


def main():
    # Paths
    template_path = Path('/mnt/arborea/bpa-rat/templates/anat/p60/tpl-BPARat_p60_T2w_template0.nii.gz')
    atlas_path = Path('/mnt/arborea/bpa-rat/templates/anat/p60/SIGMA_InVivo_Brain.nii.gz')
    output_dir = Path('/mnt/arborea/bpa-rat/templates/anat/p60/transforms/slice_wise')

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Slice-wise Registration Test")
    print("=" * 60)

    # First, visualize the slice correspondence
    print("\n1. Visualizing slice correspondence...")
    visualize_slice_correspondence(
        template_path, atlas_path,
        output_dir / 'slice_correspondence.png'
    )

    # Run slice-wise registration
    print("\n2. Running slice-wise registration...")
    results = slice_wise_registration(
        template_path, atlas_path,
        output_dir,
        template_coronal_axis=2,  # p60 Z axis is coronal slices
        atlas_coronal_axis=1,     # SIGMA Y axis is A-P
        use_ants=True,
        verbose=True
    )

    print(f"\nResults saved to: {output_dir}")
    print(f"Successful slices: {results['n_successful']}/{results['n_slices']}")
    print(f"Mean correlation: {results['mean_correlation']:.3f}")


if __name__ == '__main__':
    main()
