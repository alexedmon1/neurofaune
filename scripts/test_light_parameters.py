#!/usr/bin/env python3
"""
Test light parameters - NO -B flag, minimal/no morphology.

Strategy: Preserve brain, use light touch to clean up skull.
"""

import sys
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.preprocess.utils.func.skull_strip_preprocessed import skull_strip_bold_preprocessed


def extract_middle_volume(bold_file: Path, output_file: Path) -> Path:
    """Extract middle volume."""
    img = nib.load(bold_file)
    data = img.get_fdata()
    if len(data.shape) == 4:
        ref_data = data[..., data.shape[3] // 2]
    else:
        ref_data = data
    nib.save(nib.Nifti1Image(ref_data, img.affine, img.header), output_file)
    return output_file


def create_9_slice_qc(ref_file: Path, mask_file: Path, output_file: Path, title: str):
    """Create QC showing all 9 axial slices."""
    ref = nib.load(ref_file).get_fdata()
    mask = nib.load(mask_file).get_fdata().astype(bool)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for z in range(9):
        ax = axes[z]
        img_slice = ref[:, :, z].T
        mask_slice = mask[:, :, z].T

        ax.imshow(img_slice, cmap='gray', origin='lower')
        if mask_slice.sum() > 0:
            ax.imshow(mask_slice, cmap='Reds', alpha=0.4, origin='lower')
            ax.contour(mask_slice, levels=[0.5], colors='red', linewidths=2)

        mask_count = mask_slice.sum()
        ax.set_title(f'Slice {z}: {mask_count:,} voxels', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def main():
    subject = 'sub-Rat108'
    session = 'ses-p30'
    bold_file = Path(f'/mnt/arborea/bpa-rat/raw/bids/{subject}/{session}/func/'
                    f'{subject}_{session}_run-13_bold.nii.gz')

    output_dir = Path('/mnt/arborea/bpa-rat/test/skull_strip_light')
    work_base = output_dir / 'work' / subject / session
    work_base.mkdir(parents=True, exist_ok=True)
    qc_dir = output_dir / 'qc'
    qc_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TESTING LIGHT PARAMETERS - NO -B FLAG")
    print("="*80)
    print("Strategy: Preserve brain, minimal skull cleanup")
    print()

    # Extract reference
    ref_volume = work_base / f"{subject}_{session}_bold_ref.nii.gz"
    extract_middle_volume(bold_file, ref_volume)

    # Test configurations - NO -B flag
    configs = [
        {
            'name': 'no_morphology_frac30',
            'frac': 0.30,
            'use_B': False,
            'erode': 0,
            'dilate': 0,
            'desc': 'frac=0.30, no morphology'
        },
        {
            'name': 'no_morphology_frac25',
            'frac': 0.25,
            'use_B': False,
            'erode': 0,
            'dilate': 0,
            'desc': 'frac=0.25, no morphology'
        },
        {
            'name': 'light_morph_frac30',
            'frac': 0.30,
            'use_B': False,
            'erode': 1,
            'dilate': 1,
            'desc': 'frac=0.30, erode/dilate 1x'
        },
        {
            'name': 'light_morph_frac25',
            'frac': 0.25,
            'use_B': False,
            'erode': 1,
            'dilate': 1,
            'desc': 'frac=0.25, erode/dilate 1x'
        },
        {
            'name': 'light_morph_frac35',
            'frac': 0.35,
            'use_B': False,
            'erode': 1,
            'dilate': 1,
            'desc': 'frac=0.35, erode/dilate 1x'
        },
        {
            'name': 'asymmetric_light',
            'frac': 0.30,
            'use_B': False,
            'erode': 2,
            'dilate': 1,
            'desc': 'frac=0.30, erode 2x/dilate 1x'
        }
    ]

    results = []

    for config in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {config['name']}")
        print(f"  {config['desc']}")
        print('='*80)

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
                use_functional_flag=True,
                use_bias_cleanup=config['use_B'],
                norm_method='percentile',
                clean_mask=config['erode'] > 0,
                erode_iter=config['erode'],
                dilate_iter=config['dilate']
            )

            # Check coverage across all 9 axial slices
            mask_img = nib.load(mask_file)
            mask_data = mask_img.get_fdata().astype(bool)

            coverage_per_slice = []
            for z in range(9):
                slice_mask = mask_data[:, :, z]
                coverage_per_slice.append(slice_mask.sum())

            slices_with_mask = sum(1 for c in coverage_per_slice if c > 0)

            print(f"\nCoverage across 9 axial slices:")
            print(f"  Slices with mask: {slices_with_mask}/9")
            print(f"  Voxels per slice: {coverage_per_slice}")
            print(f"  Total voxels: {info['mask_voxels']:,}")
            print(f"  Extraction ratio: {info['extraction_ratio']:.3f}")

            # Create QC
            qc_file = qc_dir / f"{config['name']}_9slices.png"
            title = (f"{config['name']}\n{config['desc']}\n"
                    f"Slices: {slices_with_mask}/9, Ratio: {info['extraction_ratio']:.3f}")
            create_9_slice_qc(ref_volume, mask_file, qc_file, title)
            print(f"  QC: {qc_file}")

            results.append({
                'config': config,
                'info': info,
                'coverage': {
                    'slices_with_mask': slices_with_mask,
                    'per_slice': coverage_per_slice
                },
                'qc_file': qc_file
            })

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY - Light Parameters (NO -B flag)")
    print('='*80)

    for result in results:
        config = result['config']
        info = result['info']
        cov = result['coverage']
        print(f"\n{config['name']}:")
        print(f"  {config['desc']}")
        print(f"  Parameters: frac={config['frac']}, -B=False, erode={config['erode']}x, dilate={config['dilate']}x")
        print(f"  Coverage: {cov['slices_with_mask']}/9 axial slices")
        print(f"  Extraction ratio: {info['extraction_ratio']:.3f} ({info['mask_voxels']:,} voxels)")

    print(f"\n{'='*80}")
    print(f"View all: {qc_dir}/*.png")
    print('='*80)


if __name__ == '__main__':
    main()
