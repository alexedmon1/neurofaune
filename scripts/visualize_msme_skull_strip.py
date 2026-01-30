#!/usr/bin/env python3
"""
Visualize MSME skull stripping and registration quality.

Shows:
1. Brain mask outline overlaid on first echo (per slice)
2. Registration result: warped MSME on T2w with edge overlay
"""

import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_mask_outline(mask_slice, thickness=1):
    """Get outline of a binary mask using morphological operations."""
    dilated = ndimage.binary_dilation(mask_slice, iterations=thickness)
    eroded = ndimage.binary_erosion(mask_slice, iterations=thickness)
    outline = dilated.astype(float) - eroded.astype(float)
    return outline > 0


def plot_mask_overlay(echo1_file: Path, mask_file: Path, output_file: Path, subject: str, session: str):
    """Plot brain mask outline overlaid on first echo for each slice."""
    echo1_img = nib.load(echo1_file)
    mask_img = nib.load(mask_file)

    echo1_data = echo1_img.get_fdata()
    mask_data = mask_img.get_fdata() > 0

    n_slices = echo1_data.shape[2]

    # Calculate grid size
    n_cols = min(5, n_slices)
    n_rows = int(np.ceil(n_slices / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(n_slices):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        slice_data = np.rot90(echo1_data[:, :, idx])
        slice_mask = np.rot90(mask_data[:, :, idx])

        # Normalize intensity for display
        vmin, vmax = np.percentile(slice_data[slice_data > 0], [2, 98]) if slice_data.max() > 0 else (0, 1)

        ax.imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax)

        # Get and overlay mask outline
        if slice_mask.sum() > 0:
            outline = get_mask_outline(slice_mask, thickness=1)
            outline_rgba = np.zeros((*outline.shape, 4))
            outline_rgba[outline, :] = [1, 0, 0, 1]  # Red outline
            ax.imshow(outline_rgba)

            # Calculate extraction percentage for this slice
            pct = 100 * slice_mask.sum() / slice_mask.size
            ax.set_title(f'Slice {idx} ({pct:.1f}%)', fontsize=10)
        else:
            ax.set_title(f'Slice {idx} (no mask)', fontsize=10)

        ax.axis('off')

    # Hide empty subplots
    for idx in range(n_slices, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    # Overall extraction percentage
    total_pct = 100 * mask_data.sum() / mask_data.size
    fig.suptitle(f'MSME Skull Strip: {subject} {session}\nTotal brain extraction: {total_pct:.1f}%',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved mask overlay: {output_file}")
    return total_pct


def plot_registration_result(warped_file: Path, t2w_file: Path, output_file: Path,
                             subject: str, session: str):
    """Plot registration result with edge overlay on T2w."""
    warped_img = nib.load(warped_file)
    t2w_img = nib.load(t2w_file)

    warped_data = warped_img.get_fdata()
    t2w_data = t2w_img.get_fdata()

    # MSME is warped to T2w space, so they should have same dimensions
    # Find which T2w slices have MSME data (non-zero)
    msme_slices = []
    for z in range(warped_data.shape[2]):
        if warped_data[:, :, z].max() > 0:
            msme_slices.append(z)

    if not msme_slices:
        print(f"  Warning: No non-zero slices in warped MSME")
        return

    n_slices = len(msme_slices)
    n_cols = min(5, n_slices)
    n_rows = int(np.ceil(n_slices / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, z in enumerate(msme_slices):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        t2w_slice = np.rot90(t2w_data[:, :, z])
        warped_slice = np.rot90(warped_data[:, :, z])

        # Normalize T2w
        t2w_vmin, t2w_vmax = np.percentile(t2w_slice[t2w_slice > 0], [2, 98]) if t2w_slice.max() > 0 else (0, 1)
        t2w_norm = np.clip((t2w_slice - t2w_vmin) / (t2w_vmax - t2w_vmin + 1e-8), 0, 1)

        # Show T2w as background
        ax.imshow(t2w_norm, cmap='gray')

        # Get edges of warped MSME using Sobel
        if warped_slice.max() > 0:
            # Normalize warped MSME
            w_vmin, w_vmax = np.percentile(warped_slice[warped_slice > 0], [2, 98])
            warped_norm = np.clip((warped_slice - w_vmin) / (w_vmax - w_vmin + 1e-8), 0, 1)

            # Compute edges
            sobel_x = ndimage.sobel(warped_norm, axis=0)
            sobel_y = ndimage.sobel(warped_norm, axis=1)
            edges = np.hypot(sobel_x, sobel_y)

            # Threshold edges
            edge_thresh = np.percentile(edges[edges > 0], 70) if edges.max() > 0 else 0.1
            edge_mask = edges > edge_thresh

            # Create colored edge overlay (cyan)
            edge_rgba = np.zeros((*edges.shape, 4))
            edge_rgba[edge_mask, :] = [0, 1, 1, 0.8]  # Cyan edges
            ax.imshow(edge_rgba)

            # Calculate NCC for this slice
            t2w_flat = t2w_slice.flatten()
            warped_flat = warped_slice.flatten()
            valid = (t2w_flat > 0) & (warped_flat > 0)
            if valid.sum() > 100:
                t2w_valid = t2w_flat[valid]
                warped_valid = warped_flat[valid]
                ncc = np.corrcoef(t2w_valid, warped_valid)[0, 1]
                ax.set_title(f'T2w slice {z} (NCC={ncc:.2f})', fontsize=10)
            else:
                ax.set_title(f'T2w slice {z}', fontsize=10)
        else:
            ax.set_title(f'T2w slice {z} (no MSME)', fontsize=10)

        ax.axis('off')

    # Hide empty subplots
    for idx in range(n_slices, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    fig.suptitle(f'MSME→T2w Registration: {subject} {session}\nCyan edges = warped MSME on T2w background',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved registration overlay: {output_file}")


def visualize_subject(subject: str, session: str, study_root: Path, output_dir: Path):
    """Generate visualizations for a single subject."""
    print(f"\n{'='*60}")
    print(f"Visualizing: {subject} {session}")
    print(f"{'='*60}")

    # Find files
    work_dir = study_root / 'work' / subject / session
    transforms_dir = study_root / 'transforms' / subject / session
    derivatives_dir = study_root / 'derivatives' / subject / session

    # Look for skull-stripped echo1 and mask
    echo1_brain = None
    echo1_mask = None

    # Check different possible work directories
    for subdir in ['msme_adaptive_test', 'msme_multi_test', 'msme_preproc']:
        candidate_dir = work_dir / subdir
        if candidate_dir.exists():
            # Look for echo1 brain file
            for pattern in ['*echo1_brain.nii.gz', '*echo1.nii.gz']:
                matches = list(candidate_dir.glob(pattern))
                if matches and 'brain' in str(matches[0]):
                    echo1_brain = matches[0]
                    break

            # Look for mask in registration subdirectory
            reg_dir = candidate_dir / 'msme_registration'
            if reg_dir.exists():
                mask_matches = list(reg_dir.glob('*mask.nii.gz'))
                if mask_matches:
                    echo1_mask = mask_matches[0]
                # Also check for echo1 there
                echo1_matches = list(reg_dir.glob('*echo1.nii.gz'))
                if echo1_matches and not any('raw' in str(m) for m in echo1_matches):
                    for m in echo1_matches:
                        if 'raw' not in str(m):
                            echo1_brain = m
                            break

            if echo1_brain and echo1_mask:
                break

    # Find warped MSME and T2w
    warped_msme = transforms_dir / 'MSME_to_T2w_Warped.nii.gz' if transforms_dir.exists() else None
    if warped_msme and not warped_msme.exists():
        warped_msme = None

    t2w_file = None
    if derivatives_dir.exists():
        anat_dir = derivatives_dir / 'anat'
        if anat_dir.exists():
            t2w_matches = list(anat_dir.glob('*T2w_brain.nii.gz'))
            if not t2w_matches:
                t2w_matches = list(anat_dir.glob('*skullstrip*T2w.nii.gz'))
            if not t2w_matches:
                t2w_matches = list(anat_dir.glob('*T2w*.nii.gz'))
            if t2w_matches:
                t2w_file = t2w_matches[0]

    # Create output directory
    subject_output = output_dir / subject / session
    subject_output.mkdir(parents=True, exist_ok=True)

    results = {'subject': subject, 'session': session}

    # Plot mask overlay
    if echo1_brain and echo1_mask:
        print(f"  Echo1: {echo1_brain.name}")
        print(f"  Mask: {echo1_mask.name}")
        mask_output = subject_output / f'{subject}_{session}_msme_mask_overlay.png'
        pct = plot_mask_overlay(echo1_brain, echo1_mask, mask_output, subject, session)
        results['mask_overlay'] = str(mask_output)
        results['extraction_pct'] = pct
    else:
        print(f"  Warning: Could not find echo1 brain ({echo1_brain}) or mask ({echo1_mask})")

    # Plot registration result
    if warped_msme and t2w_file:
        print(f"  Warped MSME: {warped_msme.name}")
        print(f"  T2w: {t2w_file.name}")
        reg_output = subject_output / f'{subject}_{session}_msme_registration_overlay.png'
        plot_registration_result(warped_msme, t2w_file, reg_output, subject, session)
        results['registration_overlay'] = str(reg_output)
    else:
        print(f"  Warning: Could not find warped MSME ({warped_msme}) or T2w ({t2w_file})")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize MSME skull stripping and registration')
    parser.add_argument('--study-root', type=Path, default=Path('/mnt/arborea/bpa-rat'),
                        help='Study root directory')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory for figures (default: study_root/qc/msme_visualization)')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Specific subjects to visualize (default: find all with MSME transforms)')
    args = parser.parse_args()

    study_root = args.study_root
    output_dir = args.output_dir or study_root / 'qc' / 'msme_visualization'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find subjects with MSME transforms
    if args.subjects:
        subjects_sessions = []
        for subj in args.subjects:
            # Find sessions for this subject
            subj_transform_dir = study_root / 'transforms' / subj
            if subj_transform_dir.exists():
                for sess_dir in subj_transform_dir.iterdir():
                    if sess_dir.is_dir() and (sess_dir / 'MSME_to_T2w_0GenericAffine.mat').exists():
                        subjects_sessions.append((subj, sess_dir.name))
    else:
        # Find all subjects with MSME transforms
        transforms_root = study_root / 'transforms'
        subjects_sessions = []
        if transforms_root.exists():
            for subj_dir in sorted(transforms_root.iterdir()):
                if subj_dir.is_dir() and subj_dir.name.startswith('sub-'):
                    for sess_dir in subj_dir.iterdir():
                        if sess_dir.is_dir() and (sess_dir / 'MSME_to_T2w_0GenericAffine.mat').exists():
                            subjects_sessions.append((subj_dir.name, sess_dir.name))

    if not subjects_sessions:
        print("No subjects found with MSME transforms.")
        print("Looking for work directories with MSME data...")

        # Fall back to checking work directories
        work_root = study_root / 'work'
        if work_root.exists():
            for subj_dir in sorted(work_root.iterdir()):
                if subj_dir.is_dir() and subj_dir.name.startswith('sub-'):
                    for sess_dir in subj_dir.iterdir():
                        if sess_dir.is_dir():
                            for msme_dir in ['msme_adaptive_test', 'msme_multi_test']:
                                if (sess_dir / msme_dir).exists():
                                    subjects_sessions.append((subj_dir.name, sess_dir.name))
                                    break

    print(f"Found {len(subjects_sessions)} subject/session combinations with MSME data")

    all_results = []
    for subject, session in subjects_sessions:
        try:
            results = visualize_subject(subject, session, study_root, output_dir)
            all_results.append(results)
        except Exception as e:
            print(f"  Error: {e}")
            all_results.append({'subject': subject, 'session': session, 'error': str(e)})

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for r in all_results:
        subject, session = r['subject'], r['session']
        if 'error' in r:
            print(f"  ✗ {subject} {session}: {r['error']}")
        else:
            pct = r.get('extraction_pct', 'N/A')
            print(f"  ✓ {subject} {session}: {pct:.1f}% extraction" if isinstance(pct, float) else f"  ✓ {subject} {session}")

    print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()
