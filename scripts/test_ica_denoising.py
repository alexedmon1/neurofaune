#!/usr/bin/env python3
"""
Test rodent-specific ICA denoising on BPA-Rat data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.preprocess.utils.func.ica_denoising import (
    run_melodic_ica,
    classify_ica_components,
    remove_noise_components,
    generate_ica_denoising_qc
)


def main():
    # Use preprocessed data from previous functional test
    test_dir = Path('/mnt/arborea/bpa-rat/test/func_test')
    subject = 'sub-Rat49'
    session = 'ses-p90'
    
    # Input files from functional preprocessing
    bold_file = test_dir / 'derivatives' / subject / session / 'func' / f'{subject}_{session}_desc-preproc_bold.nii.gz'
    brain_mask = test_dir / 'derivatives' / subject / session / 'func' / f'{subject}_{session}_desc-brain_mask.nii.gz'
    motion_params = test_dir / 'work' / subject / session / 'func_preproc' / 'motion_correction' / 'bold_mcf.nii.par'
    
    # Check files exist
    if not bold_file.exists():
        print(f"ERROR: Preprocessed BOLD file not found: {bold_file}")
        print("Run test_func_workflow.py first!")
        return 1
    
    # Output directory
    ica_test_dir = test_dir / 'ica_test'
    melodic_dir = ica_test_dir / 'melodic'
    qc_dir = ica_test_dir / 'qc'
    
    print("="*80)
    print("TESTING RODENT-SPECIFIC ICA DENOISING")
    print("="*80)
    print(f"Subject: {subject}")
    print(f"Session: {session}")
    print(f"Input BOLD: {bold_file}")
    print("="*80)
    
    # =========================================================================
    # STEP 1: Run MELODIC ICA
    # =========================================================================
    print("\nSTEP 1: Running MELODIC ICA...")
    
    melodic_outputs = run_melodic_ica(
        input_file=bold_file,
        output_dir=melodic_dir,
        brain_mask=brain_mask,
        tr=0.5,  # From BPA-Rat data
        n_components=30  # Use fewer components for faster testing
    )
    
    print(f"MELODIC outputs:")
    for key, path in melodic_outputs.items():
        print(f"  {key}: {path}")
    
    # =========================================================================
    # STEP 2: Classify Components
    # =========================================================================
    print("\nSTEP 2: Classifying ICA components...")
    
    # Check if CSF mask exists (from anatomical preprocessing)
    csf_mask = test_dir / 'derivatives' / subject / session / 'anat' / f'{subject}_{session}_label-CSF_probseg.nii.gz'
    if not csf_mask.exists():
        print(f"  WARNING: CSF mask not found, skipping CSF overlap feature")
        csf_mask = None
    
    classification = classify_ica_components(
        melodic_dir=melodic_dir,
        motion_params_file=motion_params,
        brain_mask_file=brain_mask,
        tr=0.5,
        csf_mask_file=csf_mask
    )
    
    print(f"\nClassification summary:")
    print(f"  Signal components: {classification['summary']['n_signal']}")
    print(f"  Noise components: {classification['summary']['n_noise']}")
    print(f"  Signal indices: {classification['summary']['signal_components']}")
    print(f"  Noise indices: {classification['summary']['noise_components']}")
    
    # =========================================================================
    # STEP 3: Remove Noise Components
    # =========================================================================
    print("\nSTEP 3: Removing noise components...")
    
    denoised_file = ica_test_dir / f'{subject}_{session}_desc-denoised_bold.nii.gz'
    
    remove_noise_components(
        input_file=bold_file,
        output_file=denoised_file,
        melodic_dir=melodic_dir,
        noise_components=classification['summary']['noise_components']
    )
    
    # =========================================================================
    # STEP 4: Generate QC Report
    # =========================================================================
    print("\nSTEP 4: Generating QC report...")
    
    qc_report = generate_ica_denoising_qc(
        subject=subject,
        session=session,
        classification_results=classification,
        melodic_dir=melodic_dir,
        output_dir=qc_dir
    )
    
    print("\n" + "="*80)
    print("ICA DENOISING TEST COMPLETE!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  Denoised BOLD: {denoised_file}")
    print(f"  QC report: {qc_report}")
    print(f"  MELODIC report: {melodic_outputs['report']}")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
