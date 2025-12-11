#!/usr/bin/env python3
"""
Test multiple skull stripping parameter combinations and create mosaic QC.
"""

import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.preprocess.utils.func.skull_strip_preprocessed import skull_strip_bold_preprocessed


def extract_middle_volume(bold_file: Path, output_file: Path) -> Path:
    """Extract middle volume."""
    img = nib.load(bold_file)
    data = img.get_fdata()

    if len(data.shape) == 4:
        middle_vol = data.shape[3] // 2
        ref_data = data[..., middle_vol]
    else:
        ref_data = data

    nib.save(nib.Nifti1Image(ref_data, img.affine, img.header), output_file)
    return output_file


def create_mosaic_qc(
    original_file: Path,
    mask_file: Path,
    output_file: Path,
    title: str,
    n_slices: int = 9
):
    """Create mosaic QC with multiple coronal slices."""
    orig_img = nib.load(original_file)
    orig_data = orig_img.get_fdata()

    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()

    # Select evenly spaced coronal slices
    n_coronal = orig_data.shape[1]
    slice_indices = np.linspace(
        int(n_coronal * 0.2),  # Start at 20%
        int(n_coronal * 0.8),  # End at 80%
        n_slices,
        dtype=int
    )

    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_slices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes = axes.flatten() if n_slices > 1 else [axes]

    for idx, slice_idx in enumerate(slice_indices):
        if idx < len(axes):
            ax = axes[idx]

            # Show original with mask overlay
            ax.imshow(orig_data[:, slice_idx, :].T, cmap='gray', origin='lower')
            ax.imshow(mask_data[:, slice_idx, :].T, cmap='Reds', alpha=0.3, origin='lower')
            ax.set_title(f'Coronal Slice {slice_idx}', fontsize=10)
            ax.axis('off')

    # Hide unused subplots
    for idx in range(n_slices, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def test_parameters(
    ref_volume: Path,
    work_base: Path,
    qc_dir: Path,
    subject: str,
    session: str
):
    """Test multiple parameter combinations."""

    # Parameter combinations to test
    test_configs = [
        {
            'name': 'baseline',
            'frac': 0.30,
            'use_F': True,
            'use_B': False,
            'erode': 2,
            'dilate': 2
        },
        {
            'name': 'lower_frac',
            'frac': 0.25,
            'use_F': True,
            'use_B': False,
            'erode': 2,
            'dilate': 2
        },
        {
            'name': 'with_B_flag',
            'frac': 0.30,
            'use_F': True,
            'use_B': True,
            'erode': 2,
            'dilate': 2
        },
        {
            'name': 'aggressive',
            'frac': 0.25,
            'use_F': True,
            'use_B': True,
            'erode': 3,
            'dilate': 3
        },
        {
            'name': 'very_aggressive',
            'frac': 0.20,
            'use_F': True,
            'use_B': True,
            'erode': 3,
            'dilate': 3
        }
    ]

    results = []

    for config in test_configs:
        print("\n" + "="*80)
        print(f"Testing: {config['name']}")
        print("="*80)
        print(f"  frac={config['frac']}, -F={config['use_F']}, -B={config['use_B']}")
        print(f"  erode={config['erode']}x, dilate={config['dilate']}x")

        work_dir = work_base / config['name']
        work_dir.mkdir(parents=True, exist_ok=True)

        brain_file = work_dir / f"{subject}_{session}_brain.nii.gz"
        mask_file = work_dir / f"{subject}_{session}_mask.nii.gz"

        try:
            brain_file, mask_file, info = skull_strip_bold_preprocessed(
                input_file=ref_volume,
                output_file=brain_file,
                mask_file=mask_file,
                work_dir=work_dir,
                bet_method='bet',
                bet_frac=config['frac'],
                use_functional_flag=config['use_F'],
                use_bias_cleanup=config['use_B'],
                norm_method='percentile',
                clean_mask=True,
                erode_iter=config['erode'],
                dilate_iter=config['dilate']
            )

            # Create mosaic QC
            qc_file = qc_dir / f"{config['name']}_mosaic_qc.png"
            title = (f"{subject} {session} - {config['name']}\n"
                    f"frac={config['frac']}, -F={config['use_F']}, -B={config['use_B']}, "
                    f"erode={config['erode']}x, dilate={config['dilate']}x\n"
                    f"Extraction ratio: {info['extraction_ratio']:.3f}")

            create_mosaic_qc(ref_volume, mask_file, qc_file, title)

            results.append({
                'config': config,
                'info': info,
                'qc_file': qc_file
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    return results


def main():
    subject = 'sub-Rat108'
    session = 'ses-p30'
    bold_file = Path(f'/mnt/arborea/bpa-rat/raw/bids/{subject}/{session}/func/'
                    f'{subject}_{session}_run-13_bold.nii.gz')

    output_dir = Path('/mnt/arborea/bpa-rat/test/skull_strip_params_comparison')

    if not bold_file.exists():
        print(f"ERROR: {bold_file} not found")
        return 1

    # Setup directories
    work_base = output_dir / 'work' / subject / session
    work_base.mkdir(parents=True, exist_ok=True)

    qc_dir = output_dir / 'qc'
    qc_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TESTING MULTIPLE SKULL STRIPPING PARAMETER COMBINATIONS")
    print("="*80)
    print(f"Subject: {subject}")
    print(f"Session: {session}")
    print(f"Output: {output_dir}")
    print()

    # Extract reference volume
    ref_volume = work_base / f"{subject}_{session}_bold_ref.nii.gz"
    print(f"Extracting reference volume...")
    extract_middle_volume(bold_file, ref_volume)

    # Test all parameter combinations
    results = test_parameters(ref_volume, work_base, qc_dir, subject, session)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for result in results:
        config = result['config']
        info = result['info']
        print(f"\n{config['name']}:")
        print(f"  frac={config['frac']}, -F={config['use_F']}, -B={config['use_B']}, "
              f"erode={config['erode']}x, dilate={config['dilate']}x")
        print(f"  Extraction ratio: {info['extraction_ratio']:.3f}")
        print(f"  Mask voxels: {info['mask_voxels']:,}")
        print(f"  QC: {result['qc_file']}")

    print("\n" + "="*80)
    print("View QC mosaics:")
    print(f"  ls {qc_dir}/*.png")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
