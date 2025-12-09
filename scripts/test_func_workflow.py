#!/usr/bin/env python3
"""
Test functional preprocessing workflow on BPA-Rat data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.preprocess.workflows.func_preprocess import run_functional_preprocessing
from neurofaune.utils.transforms import create_transform_registry


def main():
    # Test subject with functional data
    subject = 'sub-Rat49'
    session = 'ses-p90'
    
    # Paths
    bids_dir = Path('/mnt/arborea/bpa-rat/raw/bids')
    output_dir = Path('/mnt/arborea/bpa-rat/test/func_test')
    
    # Find BOLD file
    func_dir = bids_dir / subject / session / 'func'
    bold_files = list(func_dir.glob(f'{subject}_{session}_*_bold.nii.gz'))
    
    if not bold_files:
        print(f"ERROR: No BOLD files found in {func_dir}")
        return 1
    
    # Use first run
    bold_file = bold_files[0]
    print(f"Testing with: {bold_file}")
    
    # Load config
    config_file = Path(__file__).parent.parent / 'configs' / 'default.yaml'
    config = load_config(config_file)
    
    # Update paths for test
    config['paths']['study_root'] = str(output_dir)
    config['paths']['derivatives'] = str(output_dir / 'derivatives')
    config['paths']['transforms'] = str(output_dir / 'transforms')
    config['paths']['qc'] = str(output_dir / 'qc')
    config['paths']['work'] = str(output_dir / 'work')
    
    # Create transform registry
    registry = create_transform_registry(config, subject, cohort='p90')
    
    # Run functional preprocessing
    try:
        results = run_functional_preprocessing(
            config=config,
            subject=subject,
            session=session,
            bold_file=bold_file,
            output_dir=output_dir,
            transform_registry=registry,
            n_discard=5  # Discard first 5 volumes
        )
        
        print("\n" + "="*80)
        print("TEST SUCCESSFUL!")
        print("="*80)
        print("\nOutput files:")
        for key, value in results.items():
            if isinstance(value, Path):
                print(f"  {key}: {value}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
