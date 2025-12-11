#!/usr/bin/env python3
"""Visualize individual Atropos posteriors to debug skull stripping."""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

work_dir = Path('/mnt/arborea/bpa-rat/test/atropos_skull_strip/work/sub-Rat108/ses-p30')

# Load original image
orig_img = nib.load(work_dir / 'sub-Rat108_ses-p30_bold_ref.nii.gz')
orig_data = orig_img.get_fdata()

# Load posteriors
posteriors = []
for i in range(1, 6):
    post_file = work_dir / f'POSTERIOR_0{i}.nii.gz'
    if post_file.exists():
        posteriors.append(nib.load(post_file).get_fdata())

# Middle slice
mid_slice = orig_data.shape[2] // 2

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Atropos Posteriors - Axial Slice', fontsize=16, fontweight='bold')

# Original
axes[0, 0].imshow(orig_data[:, :, mid_slice].T, cmap='gray', origin='lower')
axes[0, 0].set_title('Original BOLD', fontsize=12)
axes[0, 0].axis('off')

# Posteriors 1-5
for i in range(5):
    row = (i + 1) // 3
    col = (i + 1) % 3

    post_mask = posteriors[i][:, :, mid_slice] > 0.5

    axes[row, col].imshow(orig_data[:, :, mid_slice].T, cmap='gray', origin='lower', alpha=0.5)
    axes[row, col].imshow(post_mask.T, cmap='Reds', alpha=0.5, origin='lower')
    axes[row, col].set_title(f'Posterior {i+1}', fontsize=12, fontweight='bold')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(work_dir.parent.parent.parent / 'qc' / 'posteriors_debug.png', dpi=150, bbox_inches='tight')
print(f"Saved: {work_dir.parent.parent.parent / 'qc' / 'posteriors_debug.png'}")
plt.close()

# Also show which ones we're keeping
kept_mask = (posteriors[1] > 0.5) | (posteriors[2] > 0.5) | (posteriors[3] > 0.5)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Posterior Selection: Keeping 2+3+4, Excluding 1+5', fontsize=16, fontweight='bold')

axes[0].imshow(orig_data[:, :, mid_slice].T, cmap='gray', origin='lower')
axes[0].set_title('Original', fontsize=12)
axes[0].axis('off')

axes[1].imshow(orig_data[:, :, mid_slice].T, cmap='gray', origin='lower')
axes[1].imshow(kept_mask[:, :, mid_slice].T, cmap='Greens', alpha=0.4, origin='lower')
axes[1].set_title('Kept (Posteriors 2+3+4)', fontsize=12)
axes[1].axis('off')

# Show what we excluded
excluded_mask = (posteriors[0] > 0.5) | (posteriors[4] > 0.5)
axes[2].imshow(orig_data[:, :, mid_slice].T, cmap='gray', origin='lower')
axes[2].imshow(excluded_mask[:, :, mid_slice].T, cmap='Reds', alpha=0.4, origin='lower')
axes[2].set_title('Excluded (Posteriors 1+5)', fontsize=12)
axes[2].axis('off')

plt.tight_layout()
plt.savefig(work_dir.parent.parent.parent / 'qc' / 'posterior_selection_debug.png', dpi=150, bbox_inches='tight')
print(f"Saved: {work_dir.parent.parent.parent / 'qc' / 'posterior_selection_debug.png'}")
