#!/usr/bin/env python
"""
Test DTI preprocessing workflow on BPA-Rat sub-Rat207, ses-p60 data.
"""

from pathlib import Path
import sys

# Add neurofaune to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.preprocess.workflows.dwi_preprocess import run_dwi_preprocessing


def main():
    """Test DTI preprocessing workflow."""

    # Configuration
    config = {
        'study': {
            'name': 'BPA-Rat Test',
            'code': 'BPARAT',
            'species': 'rat'
        },
        'paths': {
            'study_root': '/mnt/arborea/bpa-rat/test',
            'derivatives': '/mnt/arborea/bpa-rat/test/derivatives',
            'transforms': '/mnt/arborea/bpa-rat/test/transforms',
            'qc': '/mnt/arborea/bpa-rat/test/qc',
            'work': '/mnt/arborea/bpa-rat/test/work'
        },
        'atlas': {
            'name': 'SIGMA',
            'base_path': '/mnt/arborea/atlases/SIGMA',
            'slice_definitions': {
                'dwi': {
                    'start': 15,
                    'end': 25  # 11 hippocampal slices
                }
            }
        },
        'execution': {
            'n_procs': 6
        },
        'diffusion': {
            'bet': {
                'frac': 0.3  # Rodent-optimized
            },
            'eddy': {
                'use_cuda': True
            }
        }
    }

    # Input data
    subject = 'sub-Rat207'
    session = 'ses-p60'

    dwi_file = Path('/mnt/arborea/bpa-rat/raw/bids') / subject / session / 'dwi' / f'{subject}_{session}_run-12_dwi.nii.gz'
    bval_file = Path('/mnt/arborea/bpa-rat/raw/bids') / subject / session / 'dwi' / f'{subject}_{session}_run-12_dwi.bval'
    bvec_file = Path('/mnt/arborea/bpa-rat/raw/bids') / subject / session / 'dwi' / f'{subject}_{session}_run-12_dwi.bvec'

    # Check files exist
    for f in [dwi_file, bval_file, bvec_file]:
        if not f.exists():
            print(f"ERROR: File not found: {f}")
            return 1

    print("="*80)
    print("Testing DTI Preprocessing Workflow")
    print("="*80)
    print(f"Subject: {subject}")
    print(f"Session: {session}")
    print(f"DWI file: {dwi_file}")
    print(f"Bval file: {bval_file}")
    print(f"Bvec file: {bvec_file}")
    print()

    # Create transform registry
    study_root = Path(config['paths']['study_root'])
    registry = create_transform_registry(
        config,
        subject=subject,
        cohort='p60'
    )

    # Run workflow
    try:
        results = run_dwi_preprocessing(
            config=config,
            subject=subject,
            session=session,
            dwi_file=dwi_file,
            bval_file=bval_file,
            bvec_file=bvec_file,
            output_dir=study_root,
            transform_registry=registry,
            use_gpu=True
        )

        print("\n" + "="*80)
        print("Workflow completed successfully!")
        print("="*80)
        print("\nOutput files:")
        for key, value in results.items():
            if isinstance(value, Path):
                status = "✓" if value.exists() else "✗"
                print(f"  {status} {key}: {value}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    status = "✓" if v.exists() else "✗"
                    print(f"    {status} {k}: {v}")

        return 0

    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR: Workflow failed!")
        print("="*80)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
