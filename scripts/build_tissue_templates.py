#!/usr/bin/env python3
"""
build_tissue_templates.py

Build tissue probability templates (GM, WM, CSF) for each cohort by:
1. Warping each subject's tissue maps to template space using existing transforms
2. Averaging across subjects

This is faster than full ANTs template building since we reuse the subject-to-template
transforms computed during T2w template construction.

Usage:
    python build_tissue_templates.py /path/to/bpa-rat [--cohorts p30 p60 p90] [--n-cores 4]

Prerequisites:
    - T2w templates already built with subject transforms
    - Subject tissue probability maps in derivatives/
"""

import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np


def find_subject_transforms(
    template_dir: Path,
    subject: str,
    session: str
) -> Optional[dict]:
    """
    Find the transforms from subject space to template space.

    ANTs template building creates transforms with numbered suffixes.
    """
    # Look for transform files with various naming patterns
    pattern_base = f"tpl-*_{subject}_{session}_desc-preproc_T2w"

    # Find affine and warp files
    affine_files = list(template_dir.glob(f"{pattern_base}*Affine.txt"))
    warp_files = list(template_dir.glob(f"{pattern_base}*Warp.nii.gz"))

    # Filter out InverseWarp
    warp_files = [f for f in warp_files if 'Inverse' not in f.name]

    if not affine_files or not warp_files:
        return None

    # Use first match (should only be one per subject)
    return {
        'affine': affine_files[0],
        'warp': warp_files[0]
    }


def warp_tissue_map(
    tissue_map: Path,
    reference: Path,
    warp: Path,
    affine: Path,
    output: Path
) -> bool:
    """
    Apply transforms to warp tissue probability map to template space.
    """
    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(tissue_map),
        '-r', str(reference),
        '-o', str(output),
        '-t', str(warp),
        '-t', str(affine),
        '-n', 'Linear',  # Linear interpolation for probability maps
        '--float'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def build_tissue_template(
    study_root: Path,
    cohort: str,
    tissue: str,
    n_cores: int = 4
) -> Optional[Path]:
    """
    Build a tissue probability template for a single cohort and tissue type.

    Parameters
    ----------
    study_root : Path
        Study root directory
    cohort : str
        Cohort ID (e.g., 'p60')
    tissue : str
        Tissue type ('GM', 'WM', or 'CSF')
    n_cores : int
        Number of cores (not used currently, but for future parallelization)

    Returns
    -------
    Path or None
        Path to output template, or None if failed
    """
    templates_dir = study_root / 'templates' / 'anat' / cohort
    derivatives_dir = study_root / 'derivatives'

    # Load subjects list
    subjects_file = templates_dir / f'subjects_used_{cohort}_anat.txt'
    if not subjects_file.exists():
        print(f"  ERROR: Subjects file not found: {subjects_file}")
        return None

    subjects = subjects_file.read_text().strip().split('\n')
    session = f'ses-{cohort}'

    # Reference template
    template_file = templates_dir / f'tpl-BPARat_{cohort}_T2w.nii.gz'
    if not template_file.exists():
        print(f"  ERROR: Template not found: {template_file}")
        return None

    print(f"  Building {tissue} template from {len(subjects)} subjects...")

    # Warp each subject's tissue map to template space
    warped_maps = []
    work_dir = templates_dir / 'work_tissue'
    work_dir.mkdir(exist_ok=True)

    for subject in subjects:
        # Find tissue map
        tissue_map = derivatives_dir / subject / session / 'anat' / f'{subject}_{session}_label-{tissue}_probseg.nii.gz'
        if not tissue_map.exists():
            print(f"    ⚠ Tissue map not found: {tissue_map.name}")
            continue

        # Find transforms
        transforms = find_subject_transforms(templates_dir, subject, session)
        if transforms is None:
            print(f"    ⚠ Transforms not found for {subject}")
            continue

        # Warp tissue map
        warped_output = work_dir / f'{subject}_{tissue}_warped.nii.gz'

        if warp_tissue_map(
            tissue_map=tissue_map,
            reference=template_file,
            warp=transforms['warp'],
            affine=transforms['affine'],
            output=warped_output
        ):
            warped_maps.append(warped_output)
        else:
            print(f"    ⚠ Failed to warp {subject}")

    if len(warped_maps) < 5:
        print(f"  ERROR: Only {len(warped_maps)} subjects warped successfully, need at least 5")
        return None

    print(f"    Successfully warped {len(warped_maps)}/{len(subjects)} subjects")

    # Average the warped maps
    print(f"    Averaging tissue maps...")

    # Load all warped maps and compute mean
    first_img = nib.load(warped_maps[0])
    shape = first_img.shape
    affine = first_img.affine

    sum_data = np.zeros(shape, dtype=np.float64)
    count = 0

    for warped_path in warped_maps:
        img = nib.load(warped_path)
        data = img.get_fdata()

        # Sanity check - values should be probability (0-1)
        if data.max() > 1.5:
            data = data / data.max()  # Normalize if needed

        sum_data += data
        count += 1

    # Compute mean
    mean_data = sum_data / count

    # Ensure valid probability range
    mean_data = np.clip(mean_data, 0, 1)

    # Save
    output_file = templates_dir / f'tpl-BPARat_{cohort}_label-{tissue}_probseg.nii.gz'
    out_img = nib.Nifti1Image(mean_data.astype(np.float32), affine)
    nib.save(out_img, output_file)

    print(f"    ✓ Saved: {output_file.name}")

    # Cleanup work directory
    for f in warped_maps:
        f.unlink()
    work_dir.rmdir()

    return output_file


def build_all_tissue_templates(
    study_root: Path,
    cohorts: List[str],
    n_cores: int = 4
) -> dict:
    """
    Build tissue probability templates for all cohorts.
    """
    results = {}

    for cohort in cohorts:
        print(f"\n{'='*70}")
        print(f"Building tissue templates for {cohort}")
        print('='*70)

        results[cohort] = {}

        for tissue in ['GM', 'WM', 'CSF']:
            output = build_tissue_template(
                study_root=study_root,
                cohort=cohort,
                tissue=tissue,
                n_cores=n_cores
            )
            results[cohort][tissue] = output

    return results


def create_tissue_qc(study_root: Path, cohort: str) -> Path:
    """
    Create QC visualization of tissue probability templates.
    """
    import matplotlib.pyplot as plt

    templates_dir = study_root / 'templates' / 'anat' / cohort

    # Load template and tissue maps
    template = nib.load(templates_dir / f'tpl-BPARat_{cohort}_T2w.nii.gz')
    template_data = template.get_fdata()

    tissue_data = {}
    for tissue in ['GM', 'WM', 'CSF']:
        tissue_file = templates_dir / f'tpl-BPARat_{cohort}_label-{tissue}_probseg.nii.gz'
        if tissue_file.exists():
            tissue_data[tissue] = nib.load(tissue_file).get_fdata()

    if not tissue_data:
        print(f"  No tissue templates found for QC")
        return None

    # Create visualization
    n_slices = 6
    slice_indices = np.linspace(5, template_data.shape[2] - 5, n_slices).astype(int)

    fig, axes = plt.subplots(4, n_slices, figsize=(18, 12))

    for col, slice_idx in enumerate(slice_indices):
        # T2w template
        axes[0, col].imshow(template_data[:, :, slice_idx].T, cmap='gray', origin='lower')
        axes[0, col].set_title(f'z={slice_idx}')
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_ylabel('T2w', fontsize=12, fontweight='bold')

        # GM
        if 'GM' in tissue_data:
            axes[1, col].imshow(tissue_data['GM'][:, :, slice_idx].T, cmap='Reds', origin='lower', vmin=0, vmax=1)
            axes[1, col].axis('off')
            if col == 0:
                axes[1, col].set_ylabel('GM', fontsize=12, fontweight='bold')

        # WM
        if 'WM' in tissue_data:
            axes[2, col].imshow(tissue_data['WM'][:, :, slice_idx].T, cmap='Blues', origin='lower', vmin=0, vmax=1)
            axes[2, col].axis('off')
            if col == 0:
                axes[2, col].set_ylabel('WM', fontsize=12, fontweight='bold')

        # CSF
        if 'CSF' in tissue_data:
            axes[3, col].imshow(tissue_data['CSF'][:, :, slice_idx].T, cmap='Greens', origin='lower', vmin=0, vmax=1)
            axes[3, col].axis('off')
            if col == 0:
                axes[3, col].set_ylabel('CSF', fontsize=12, fontweight='bold')

    fig.suptitle(f'{cohort.upper()} Tissue Probability Templates', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = templates_dir / f'tissue_templates_qc_{cohort}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Build tissue probability templates for cohorts'
    )
    parser.add_argument('study_root', type=Path, help='Path to study root')
    parser.add_argument('--cohorts', nargs='+', default=['p30', 'p60', 'p90'],
                        help='Cohorts to process (default: p30 p60 p90)')
    parser.add_argument('--n-cores', type=int, default=4, help='Number of CPU cores')
    args = parser.parse_args()

    print("="*70)
    print("Building Tissue Probability Templates")
    print("="*70)
    print(f"Study root: {args.study_root}")
    print(f"Cohorts: {args.cohorts}")

    # Build templates
    results = build_all_tissue_templates(
        study_root=args.study_root,
        cohorts=args.cohorts,
        n_cores=args.n_cores
    )

    # Generate QC
    print(f"\n{'='*70}")
    print("Generating QC Visualizations")
    print('='*70)

    for cohort in args.cohorts:
        qc_file = create_tissue_qc(args.study_root, cohort)
        if qc_file:
            print(f"  {cohort}: {qc_file.name}")

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print('='*70)

    for cohort, tissues in results.items():
        print(f"\n{cohort}:")
        for tissue, path in tissues.items():
            status = "✓" if path else "✗"
            print(f"  {status} {tissue}: {path.name if path else 'FAILED'}")

    print("\nDone!")


if __name__ == '__main__':
    main()
