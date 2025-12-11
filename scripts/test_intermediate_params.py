#!/usr/bin/env python3
"""
Test intermediate skull stripping parameters.

Goal: Remove more skull than baseline without losing brain coverage.
Strategy: NO -B flag, but more aggressive morphology.
"""

import sys
from pathlib import Path
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.preprocess.utils.func.skull_strip_preprocessed import skull_strip_bold_preprocessed
from create_better_qc import create_improved_qc


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


def main():
    subject = 'sub-Rat108'
    session = 'ses-p30'
    bold_file = Path(f'/mnt/arborea/bpa-rat/raw/bids/{subject}/{session}/func/'
                    f'{subject}_{session}_run-13_bold.nii.gz')

    output_dir = Path('/mnt/arborea/bpa-rat/test/skull_strip_intermediate')
    work_base = output_dir / 'work' / subject / session
    work_base.mkdir(parents=True, exist_ok=True)
    qc_dir = output_dir / 'qc'
    qc_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TESTING INTERMEDIATE SKULL STRIPPING PARAMETERS")
    print("="*80)
    print("Strategy: NO -B flag (preserve brain coverage)")
    print("          More aggressive morphology (remove skull)")
    print()

    # Extract reference
    ref_volume = work_base / f"{subject}_{session}_bold_ref.nii.gz"
    extract_middle_volume(bold_file, ref_volume)

    # Test configurations - NO -B flag, varying morphology
    configs = [
        {
            'name': 'baseline',
            'frac': 0.30,
            'use_B': False,
            'erode': 2,
            'dilate': 2,
            'desc': 'Original (12%)'
        },
        {
            'name': 'more_erosion',
            'frac': 0.30,
            'use_B': False,
            'erode': 3,
            'dilate': 3,
            'desc': 'More erosion to remove skull'
        },
        {
            'name': 'heavy_erosion',
            'frac': 0.30,
            'use_B': False,
            'erode': 4,
            'dilate': 4,
            'desc': 'Heavy erosion'
        },
        {
            'name': 'lower_frac_more_erosion',
            'frac': 0.25,
            'use_B': False,
            'erode': 3,
            'dilate': 3,
            'desc': 'Lower frac + more erosion'
        },
        {
            'name': 'asymmetric_morphology',
            'frac': 0.30,
            'use_B': False,
            'erode': 3,
            'dilate': 2,
            'desc': 'Erode 3x, dilate 2x (net removal)'
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
                clean_mask=True,
                erode_iter=config['erode'],
                dilate_iter=config['dilate']
            )

            # Check mask coverage
            mask_img = nib.load(mask_file)
            mask_data = mask_img.get_fdata().astype(bool)
            coronal_mask = mask_data.sum(axis=(0, 2)) > 0
            n_coronal = coronal_mask.sum()
            coronal_range = (np.where(coronal_mask)[0].min(),
                           np.where(coronal_mask)[0].max())

            print(f"\nMask coverage:")
            print(f"  Coronal slices: {n_coronal} (indices {coronal_range[0]}-{coronal_range[1]})")
            print(f"  Extraction ratio: {info['extraction_ratio']:.3f}")
            print(f"  Total voxels: {info['mask_voxels']:,}")

            # Create QC
            qc_file = qc_dir / f"{config['name']}_qc.png"
            title = (f"{subject} {session} - {config['name']}\n"
                    f"{config['desc']}\n"
                    f"Slices: {n_coronal}, Ratio: {info['extraction_ratio']:.3f}")
            create_improved_qc(ref_volume, mask_file, qc_file, title)

            results.append({
                'config': config,
                'info': info,
                'coverage': {
                    'n_slices': n_coronal,
                    'range': coronal_range
                },
                'qc_file': qc_file
            })

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY - Intermediate Parameters (NO -B flag)")
    print('='*80)

    for result in results:
        config = result['config']
        info = result['info']
        cov = result['coverage']
        print(f"\n{config['name']}:")
        print(f"  {config['desc']}")
        print(f"  Parameters: frac={config['frac']}, erode={config['erode']}x, dilate={config['dilate']}x")
        print(f"  Coronal coverage: {cov['n_slices']} slices ({cov['range'][0]}-{cov['range'][1]})")
        print(f"  Extraction ratio: {info['extraction_ratio']:.3f} ({info['mask_voxels']:,} voxels)")
        print(f"  QC: {result['qc_file']}")

    print(f"\n{'='*80}")
    print(f"All QC images: {qc_dir}/*.png")
    print('='*80)


if __name__ == '__main__':
    main()
