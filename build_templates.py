#!/usr/bin/env python3
"""
Build age-specific T2w templates from preprocessed anatomical data.

This script creates study-specific templates for each age cohort (p30, p60, p90)
using ANTs buildtemplateparallel.

Usage:
    python build_templates.py <study_root>

Example:
    python build_templates.py /mnt/arborea/bpa-rat
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Dict
import json


def find_preprocessed_subjects(study_root: Path, cohort: str) -> List[Path]:
    """
    Find all successfully preprocessed subjects for a given cohort.

    Parameters
    ----------
    study_root : Path
        Study root directory
    cohort : str
        Cohort identifier ('p30', 'p60', 'p90')

    Returns
    -------
    List[Path]
        List of preprocessed T2w brain files
    """
    derivatives_dir = study_root / 'derivatives'
    brain_files = []

    # Find all subjects
    for subject_dir in sorted(derivatives_dir.glob('sub-*')):
        # Check for session directories
        session_dir = subject_dir / f'ses-{cohort}'

        if not session_dir.exists():
            continue

        # Check for exclusion marker
        exclusion_markers = list(session_dir.glob('*_EXCLUDE.txt'))
        if exclusion_markers:
            print(f"  Skipping {subject_dir.name}/ses-{cohort}: Exclusion marker found")
            continue

        # Find preprocessed brain file
        anat_dir = session_dir / 'anat'
        if not anat_dir.exists():
            continue

        brain_file = anat_dir / f'{subject_dir.name}_ses-{cohort}_desc-preproc_T2w.nii.gz'

        if brain_file.exists():
            brain_files.append(brain_file)
        else:
            print(f"  Warning: Brain file not found for {subject_dir.name}/ses-{cohort}")

    return brain_files


def build_template_ants(
    input_files: List[Path],
    output_prefix: Path,
    n_iterations: int = 4,
    n_cores: int = 6
) -> Path:
    """
    Build template using ANTs buildtemplateparallel.sh.

    Parameters
    ----------
    input_files : List[Path]
        List of input brain images
    output_prefix : Path
        Output prefix for template
    n_iterations : int
        Number of iterations (default: 4)
    n_cores : int
        Number of CPU cores to use

    Returns
    -------
    Path
        Path to final template
    """
    print(f"\nBuilding template with ANTs...")
    print(f"  Input files: {len(input_files)}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Cores: {n_cores}")

    # Create input file list
    input_list_file = output_prefix.parent / f'{output_prefix.name}_input_list.txt'
    with open(input_list_file, 'w') as f:
        for img_file in input_files:
            f.write(f"{img_file}\n")

    print(f"  Input list: {input_list_file}")

    # ANTs buildtemplateparallel command
    cmd = [
        'buildtemplateparallel.sh',
        '-d', '3',                    # 3D
        '-i', str(n_iterations),      # Number of iterations
        '-j', str(n_cores),           # CPU cores
        '-c', '2',                    # Control for intermediate templates (2=use existing)
        '-o', f'{output_prefix}_',    # Output prefix
        *[str(f) for f in input_files]  # Input files
    ]

    print(f"\nRunning: {' '.join(cmd[:10])} ... ({len(input_files)} files)")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        print("✓ Template building complete!")

        # Final template should be {prefix}_template.nii.gz
        final_template = Path(f'{output_prefix}_template.nii.gz')

        if not final_template.exists():
            raise FileNotFoundError(f"Template not created: {final_template}")

        return final_template

    except subprocess.CalledProcessError as e:
        print(f"✗ Template building failed!")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise
    except FileNotFoundError:
        print("\n✗ buildtemplateparallel.sh not found!")
        print("Please ensure ANTs is installed and in your PATH")
        print("See: https://github.com/ANTsX/ANTs")
        raise


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_templates.py <study_root>")
        print("\nExample:")
        print("  python build_templates.py /mnt/arborea/bpa-rat")
        print("\nOptions:")
        print("  --cohorts p30,p60,p90  : Specify cohorts (default: all)")
        print("  --iterations 4         : Number of iterations (default: 4)")
        print("  --cores 6              : Number of CPU cores (default: 6)")
        sys.exit(1)

    study_root = Path(sys.argv[1])

    if not study_root.exists():
        print(f"Error: Study root not found: {study_root}")
        sys.exit(1)

    # Parse options
    cohorts = ['p30', 'p60', 'p90']
    if '--cohorts' in sys.argv:
        idx = sys.argv.index('--cohorts')
        cohorts = sys.argv[idx + 1].split(',')

    n_iterations = 4
    if '--iterations' in sys.argv:
        idx = sys.argv.index('--iterations')
        n_iterations = int(sys.argv[idx + 1])

    n_cores = 6
    if '--cores' in sys.argv:
        idx = sys.argv.index('--cores')
        n_cores = int(sys.argv[idx + 1])

    print("="*80)
    print("Template Building")
    print("="*80)
    print(f"Study root: {study_root}")
    print(f"Cohorts: {', '.join(cohorts)}")
    print(f"Iterations: {n_iterations}")
    print(f"Cores: {n_cores}")
    print("="*80)

    # Create templates directory
    templates_dir = study_root / 'templates'
    templates_dir.mkdir(parents=True, exist_ok=True)

    # Build template for each cohort
    results = {}

    for cohort in cohorts:
        print(f"\n{'='*80}")
        print(f"Processing cohort: {cohort}")
        print(f"{'='*80}")

        # Find preprocessed subjects
        print(f"\nFinding preprocessed subjects for {cohort}...")
        brain_files = find_preprocessed_subjects(study_root, cohort)

        print(f"Found {len(brain_files)} subjects")

        if len(brain_files) < 3:
            print(f"✗ Not enough subjects for {cohort} (need at least 3, found {len(brain_files)})")
            results[cohort] = {
                'status': 'skipped',
                'reason': 'Insufficient subjects',
                'n_subjects': len(brain_files)
            }
            continue

        # Build template
        output_prefix = templates_dir / f'template_{cohort}'

        try:
            template_file = build_template_ants(
                input_files=brain_files,
                output_prefix=output_prefix,
                n_iterations=n_iterations,
                n_cores=n_cores
            )

            print(f"\n✓ Template created: {template_file}")

            results[cohort] = {
                'status': 'success',
                'template': str(template_file),
                'n_subjects': len(brain_files)
            }

        except Exception as e:
            print(f"\n✗ Failed to build template for {cohort}: {e}")
            results[cohort] = {
                'status': 'failed',
                'error': str(e),
                'n_subjects': len(brain_files)
            }

    # Print summary
    print("\n" + "="*80)
    print("TEMPLATE BUILDING COMPLETE")
    print("="*80)

    for cohort, result in results.items():
        status = result['status']
        n_subjects = result['n_subjects']

        if status == 'success':
            print(f"✓ {cohort}: Success ({n_subjects} subjects)")
            print(f"  Template: {result['template']}")
        elif status == 'skipped':
            print(f"⊘ {cohort}: Skipped - {result['reason']} ({n_subjects} subjects)")
        elif status == 'failed':
            print(f"✗ {cohort}: Failed - {result['error']} ({n_subjects} subjects)")

    # Save results
    results_file = templates_dir / 'template_building_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("="*80)

    # Exit with error if any failures
    if any(r['status'] == 'failed' for r in results.values()):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
