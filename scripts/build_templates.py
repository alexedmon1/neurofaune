#!/usr/bin/env python3
"""
Build age-specific templates from preprocessed BPA-Rat data.

This script builds templates for each cohort (p30, p60, p90) and modality:
- T2w: Preprocessed, brain-extracted anatomical images
- FA: DTI scalar maps from eddy-corrected data
- BOLD: Mean timepoint from motion-corrected fMRI data

Usage:
    python scripts/build_templates.py --config config.yaml --cohort p60 --modality anat
    python scripts/build_templates.py --config config.yaml --cohort p60 --modality dwi
    python scripts/build_templates.py --config config.yaml --cohort p60 --modality func
    python scripts/build_templates.py --config config.yaml --cohort all --modality all
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.templates.builder import (
    select_subjects_for_template,
    extract_mean_bold,
    build_template,
    register_template_to_sigma,
    save_template_metadata
)
from neurofaune.atlas.manager import AtlasManager


def build_anatomical_template(
    config: dict,
    cohort: str,
    derivatives_dir: Path,
    template_dir: Path,
    top_percent: float = 1/3,
    n_cores: int = 8
):
    """Build T2w anatomical template for a cohort."""
    print("\n" + "="*80)
    print(f"Building T2w Template for Cohort {cohort}")
    print("="*80)

    # Select subjects
    subjects = select_subjects_for_template(
        derivatives_dir=derivatives_dir,
        cohort=cohort,
        modality='anat',
        top_percent=top_percent
    )

    # Collect preprocessed T2w files
    session = f'ses-{cohort}'
    input_files = []

    for subject in subjects:
        t2w_file = derivatives_dir / subject / session / 'anat' / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
        if t2w_file.exists():
            input_files.append(t2w_file)
        else:
            print(f"⚠ Warning: File not found for {subject}: {t2w_file}")

    if len(input_files) < 10:
        raise ValueError(f"Not enough input files ({len(input_files)} < 10)")

    print(f"\nUsing {len(input_files)} subjects for template building")

    # Build template - organize by modality/cohort
    cohort_dir = template_dir / 'anat' / cohort
    cohort_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = cohort_dir / f'tpl-BPARat_{cohort}_T2w_'

    results = build_template(
        input_files=input_files,
        output_prefix=output_prefix,
        n_iterations=4,
        n_cores=n_cores
    )

    # Rename template to standard name
    template_file = cohort_dir / f'tpl-BPARat_{cohort}_T2w.nii.gz'
    results['template'].rename(template_file)
    print(f"\nFinal T2w template: {template_file}")

    # Register to SIGMA
    print("\n" + "="*80)
    print("Registering T2w Template to SIGMA Atlas")
    print("="*80)

    atlas_mgr = AtlasManager(config)
    sigma_template = atlas_mgr.get_template(modality=None, masked=True, coronal=True)

    # Save SIGMA template to work directory
    sigma_file = cohort_dir / 'SIGMA_InVivo_Brain.nii.gz'
    import nibabel as nib
    nib.save(sigma_template, sigma_file)

    transforms_dir = cohort_dir / 'transforms'
    transforms_dir.mkdir(exist_ok=True)

    reg_results = register_template_to_sigma(
        template_file=template_file,
        sigma_template=sigma_file,
        output_prefix=transforms_dir / 'tpl-to-SIGMA_',
        n_cores=n_cores
    )

    # Rename warped template
    warped_template = cohort_dir / f'tpl-BPARat_{cohort}_space-SIGMA_T2w.nii.gz'
    reg_results['warped'].rename(warped_template)
    print(f"\nT2w template in SIGMA space: {warped_template}")

    # Rename transforms to standard names
    composite = transforms_dir / 'tpl-to-SIGMA_Composite.h5'
    inverse = transforms_dir / 'SIGMA-to-tpl_Composite.h5'
    reg_results['composite_transform'].rename(composite)
    reg_results['inverse_composite_transform'].rename(inverse)

    # Build tissue-specific templates (GM, WM, CSF)
    print("\n" + "="*80)
    print("Building Tissue-Specific Templates (GM, WM, CSF)")
    print("="*80)

    for tissue in ['GM', 'WM', 'CSF']:
        print(f"\nBuilding {tissue} template...")

        # Collect tissue probability maps
        tissue_files = []
        for subject in subjects:
            tissue_file = derivatives_dir / subject / session / 'anat' / f'{subject}_{session}_label-{tissue}_probseg.nii.gz'
            if tissue_file.exists():
                tissue_files.append(tissue_file)
            else:
                print(f"⚠ Warning: {tissue} file not found for {subject}: {tissue_file}")

        if len(tissue_files) < 10:
            print(f"⚠ Warning: Not enough {tissue} files ({len(tissue_files)} < 10), skipping {tissue} template")
            continue

        print(f"  Using {len(tissue_files)} subjects for {tissue} template")

        # Build tissue template
        tissue_prefix = cohort_dir / f'tpl-BPARat_{cohort}_{tissue}_'

        tissue_results = build_template(
            input_files=tissue_files,
            output_prefix=tissue_prefix,
            n_iterations=4,
            n_cores=n_cores
        )

        # Rename to standard name
        tissue_template = cohort_dir / f'tpl-BPARat_{cohort}_label-{tissue}_probseg.nii.gz'
        tissue_results['template'].rename(tissue_template)
        print(f"  ✓ {tissue} template: {tissue_template.name}")

    # Save metadata
    save_template_metadata(
        template_dir=cohort_dir,
        cohort=cohort,
        modality='anat',
        subjects_used=subjects
    )

    print("\n" + "="*80)
    print("Anatomical Template Building Complete!")
    print("="*80)
    print(f"T2w template: {template_file}")
    print(f"GM template: {cohort_dir / f'tpl-BPARat_{cohort}_label-GM_probseg.nii.gz'}")
    print(f"WM template: {cohort_dir / f'tpl-BPARat_{cohort}_label-WM_probseg.nii.gz'}")
    print(f"CSF template: {cohort_dir / f'tpl-BPARat_{cohort}_label-CSF_probseg.nii.gz'}")
    print(f"SIGMA space: {warped_template}")
    print(f"Transform: {composite}")
    print(f"Inverse: {inverse}")
    print("="*80 + "\n")


def build_dti_template(
    config: dict,
    cohort: str,
    derivatives_dir: Path,
    template_dir: Path,
    top_percent: float = 1/3,
    n_cores: int = 8
):
    """Build FA template for DTI."""
    print("\n" + "="*80)
    print(f"Building FA Template for Cohort {cohort}")
    print("="*80)

    # Select subjects
    subjects = select_subjects_for_template(
        derivatives_dir=derivatives_dir,
        cohort=cohort,
        modality='dwi',
        top_percent=top_percent
    )

    # Collect FA files
    session = f'ses-{cohort}'
    input_files = []

    for subject in subjects:
        fa_file = derivatives_dir / subject / session / 'dwi' / f'{subject}_{session}_FA.nii.gz'
        if fa_file.exists():
            input_files.append(fa_file)
        else:
            print(f"⚠ Warning: File not found for {subject}: {fa_file}")

    if len(input_files) < 10:
        raise ValueError(f"Not enough input files ({len(input_files)} < 10)")

    print(f"\nUsing {len(input_files)} subjects for template building")

    # Build template - organize by modality/cohort
    cohort_dir = template_dir / 'dwi' / cohort
    cohort_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = cohort_dir / f'tpl-BPARat_{cohort}_FA_'

    results = build_template(
        input_files=input_files,
        output_prefix=output_prefix,
        n_iterations=4,
        n_cores=n_cores
    )

    # Rename template to standard name
    template_file = cohort_dir / f'tpl-BPARat_{cohort}_FA.nii.gz'
    results['template'].rename(template_file)
    print(f"\nFinal FA template: {template_file}")

    # Note: FA template stays in FA space (no SIGMA registration)
    print("\nNote: FA template remains in FA space (not registered to SIGMA)")

    # Save metadata
    save_template_metadata(
        template_dir=cohort_dir,
        cohort=cohort,
        modality='dwi',
        subjects_used=subjects
    )

    print("\n" + "="*80)
    print("FA Template Building Complete!")
    print("="*80)
    print(f"Template: {template_file}")
    print("="*80 + "\n")


def build_func_template(
    config: dict,
    cohort: str,
    derivatives_dir: Path,
    template_dir: Path,
    top_percent: float = 1/3,
    n_cores: int = 8
):
    """Build BOLD template for fMRI."""
    print("\n" + "="*80)
    print(f"Building BOLD Template for Cohort {cohort}")
    print("="*80)

    # Select subjects
    subjects = select_subjects_for_template(
        derivatives_dir=derivatives_dir,
        cohort=cohort,
        modality='func',
        top_percent=top_percent
    )

    # Extract mean BOLD for each subject - organize by modality/cohort
    session = f'ses-{cohort}'
    cohort_dir = template_dir / 'func' / cohort
    work_dir = cohort_dir / 'work'
    work_dir.mkdir(parents=True, exist_ok=True)

    input_files = []

    for subject in subjects:
        bold_file = derivatives_dir / subject / session / 'func' / f'{subject}_{session}_desc-preproc_bold.nii.gz'

        if not bold_file.exists():
            print(f"⚠ Warning: File not found for {subject}: {bold_file}")
            continue

        # Extract mean timepoint
        mean_bold = work_dir / f'{subject}_{session}_bold_mean.nii.gz'

        if not mean_bold.exists():
            extract_mean_bold(bold_file, mean_bold, method='median')

        input_files.append(mean_bold)

    if len(input_files) < 10:
        raise ValueError(f"Not enough input files ({len(input_files)} < 10)")

    print(f"\nUsing {len(input_files)} subjects for template building")

    # Build template
    output_prefix = cohort_dir / f'tpl-BPARat_{cohort}_bold_'

    results = build_template(
        input_files=input_files,
        output_prefix=output_prefix,
        n_iterations=4,
        n_cores=n_cores
    )

    # Rename template to standard name
    template_file = cohort_dir / f'tpl-BPARat_{cohort}_bold.nii.gz'
    results['template'].rename(template_file)
    print(f"\nFinal BOLD template: {template_file}")

    # Note: BOLD template stays in BOLD space (no SIGMA registration)
    print("\nNote: BOLD template remains in BOLD space (not registered to SIGMA)")

    # Save metadata
    save_template_metadata(
        template_dir=cohort_dir,
        cohort=cohort,
        modality='func',
        subjects_used=subjects
    )

    print("\n" + "="*80)
    print("BOLD Template Building Complete!")
    print("="*80)
    print(f"Template: {template_file}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Build age-specific templates for BPA-Rat study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--config', type=Path, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--cohort', type=str, required=True,
                        choices=['p30', 'p60', 'p90', 'all'],
                        help='Age cohort to build template for')
    parser.add_argument('--modality', type=str, required=True,
                        choices=['anat', 'dwi', 'func', 'all'],
                        help='Modality to build template for')
    parser.add_argument('--top-percent', type=float, default=1/3,
                        help='Fraction of subjects to use (default: 1/3)')
    parser.add_argument('--n-cores', type=int, default=8,
                        help='Number of CPU cores (default: 8)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get paths
    study_root = Path(config['paths']['study_root'])
    derivatives_dir = Path(config['paths']['derivatives'])
    template_dir = study_root / 'templates'

    # Determine cohorts to process
    if args.cohort == 'all':
        cohorts = ['p30', 'p60', 'p90']
    else:
        cohorts = [args.cohort]

    # Determine modalities to process
    if args.modality == 'all':
        modalities = ['anat', 'dwi', 'func']
    else:
        modalities = [args.modality]

    # Build templates
    for cohort in cohorts:
        for modality in modalities:
            try:
                if modality == 'anat':
                    build_anatomical_template(
                        config, cohort, derivatives_dir, template_dir,
                        args.top_percent, args.n_cores
                    )
                elif modality == 'dwi':
                    build_dti_template(
                        config, cohort, derivatives_dir, template_dir,
                        args.top_percent, args.n_cores
                    )
                elif modality == 'func':
                    build_func_template(
                        config, cohort, derivatives_dir, template_dir,
                        args.top_percent, args.n_cores
                    )
            except Exception as e:
                print(f"\n❌ ERROR building {modality} template for {cohort}: {e}\n")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "="*80)
    print("All Template Building Complete!")
    print("="*80)
    print(f"Templates saved to: {template_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
