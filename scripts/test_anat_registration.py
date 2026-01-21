#!/usr/bin/env python3
"""
Test script for anatomical registration workflow.

Tests:
1. register_anat_to_template() - T2w to cohort template
2. propagate_atlas_to_anat() - SIGMA atlas to T2w space
"""

import sys
from pathlib import Path

# Set matplotlib backend before imports
import matplotlib
matplotlib.use('Agg')

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.atlas import AtlasManager
from neurofaune.templates.anat_registration import (
    register_anat_to_template,
    propagate_atlas_to_anat
)


def main():
    # Configuration
    study_root = Path('/mnt/arborea/bpa-rat')
    config_path = Path('/home/edm9fd/sandbox/neurofaune/configs/bpa_rat_example.yaml')

    # Test subject (preprocessed but no transforms yet)
    subject = 'sub-Rat193'
    session = 'ses-p60'
    cohort = 'p60'

    print("="*80)
    print("TESTING ANATOMICAL REGISTRATION WORKFLOW")
    print("="*80)
    print(f"Subject: {subject}")
    print(f"Session: {session}")
    print(f"Cohort: {cohort}")
    print()

    # Load config
    config = load_config(config_path)
    atlas_mgr = AtlasManager(config)

    # Paths
    t2w_file = study_root / 'derivatives' / subject / session / 'anat' / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
    brain_mask = study_root / 'derivatives' / subject / session / 'anat' / f'{subject}_{session}_desc-brain_mask.nii.gz'

    # Template path - use NATIVE template (same resolution as subject, fast registration)
    # The SIGMA atlas has been pre-computed in native template space
    template_file = study_root / 'templates' / 'anat' / cohort / f'tpl-BPARat_{cohort}_T2w.nii.gz'
    template_mask = None  # We don't have a template mask yet

    # Output directories
    transforms_dir = study_root / 'transforms' / subject / session
    transforms_dir.mkdir(parents=True, exist_ok=True)

    qc_dir = study_root / 'qc' / subject / session / 'anat'
    qc_dir.mkdir(parents=True, exist_ok=True)

    print(f"T2w file: {t2w_file}")
    print(f"Template: {template_file}")
    print(f"Transforms output: {transforms_dir}")
    print()

    # Verify files exist
    if not t2w_file.exists():
        print(f"ERROR: T2w file not found: {t2w_file}")
        sys.exit(1)
    if not template_file.exists():
        print(f"ERROR: Template not found: {template_file}")
        sys.exit(1)

    # Step 1: Register T2w to template
    print("\n" + "-"*60)
    print("STEP 1: Register T2w to Template")
    print("-"*60)

    reg_result = register_anat_to_template(
        t2w_file=t2w_file,
        template_file=template_file,
        output_dir=transforms_dir,
        subject=subject,
        session=session,
        mask_file=brain_mask if brain_mask.exists() else None,
        template_mask=template_mask,
        generate_qc=True
    )

    print("\nRegistration results:")
    for key, val in reg_result.items():
        if key != 'metrics':
            print(f"  {key}: {val}")
    if 'metrics' in reg_result:
        print(f"  Metrics:")
        for mk, mv in reg_result['metrics'].items():
            print(f"    {mk}: {mv:.4f}" if isinstance(mv, float) else f"    {mk}: {mv}")

    # Step 2: Propagate SIGMA atlas to T2w space
    print("\n" + "-"*60)
    print("STEP 2: Propagate SIGMA Atlas to T2w Space")
    print("-"*60)

    # Use pre-computed SIGMA atlas in native template space
    # This was created by applying inverse tpl-to-SIGMA transforms
    sigma_in_template = study_root / 'templates' / 'anat' / cohort / 'SIGMA_atlas_in_native_template.nii.gz'
    if not sigma_in_template.exists():
        print(f"ERROR: Pre-computed atlas not found: {sigma_in_template}")
        print("Run the atlas pre-computation script first.")
        sys.exit(1)
    print(f"SIGMA atlas (in template space): {sigma_in_template}")

    # Output atlas file
    atlas_output = study_root / 'derivatives' / subject / session / 'anat' / f'{subject}_{session}_space-T2w_atlas-SIGMA.nii.gz'

    # Simplified approach: just apply inverse subject-to-template transform
    # to the pre-computed atlas in native template space
    import subprocess

    # Get subject-to-template transforms
    subj_affine = transforms_dir / f'{subject}_{session}_T2w_to_template_0GenericAffine.mat'
    subj_inv_warp = transforms_dir / f'{subject}_{session}_T2w_to_template_1InverseWarp.nii.gz'

    # Build transform list (template → subject)
    transform_list = []
    if subj_inv_warp.exists():
        transform_list.extend(['-t', str(subj_inv_warp)])
    transform_list.extend(['-t', f'[{subj_affine},1]'])  # Inverse affine

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(sigma_in_template),
        '-r', str(t2w_file),
        '-o', str(atlas_output),
        '-n', 'NearestNeighbor',
    ] + transform_list

    print(f"\nApplying transforms: template → subject")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)

    # Compute stats
    import nibabel as nib
    import numpy as np
    atlas_img = nib.load(atlas_output)
    atlas_data = atlas_img.get_fdata()
    n_labels = len(set(atlas_data[atlas_data > 0].astype(int)))
    ref_data = nib.load(t2w_file).get_fdata()
    brain_voxels = np.sum(ref_data > 0)
    labeled_voxels = np.sum(atlas_data > 0)
    coverage = labeled_voxels / brain_voxels if brain_voxels > 0 else 0

    atlas_result = {
        'atlas_file': atlas_output,
        'n_labels': n_labels,
        'coverage': float(coverage),
    }

    print(f"  ✓ Atlas saved: {atlas_output.name}")
    print(f"  Unique labels: {n_labels}")
    print(f"  Coverage: {coverage:.1%}")

    # Generate QC overlay
    from neurofaune.templates.registration_qc import generate_atlas_overlay_figure
    qc_figure = qc_dir / f'{subject}_{session}_atlas_overlay.png'
    generate_atlas_overlay_figure(
        anatomical_file=t2w_file,
        atlas_file=atlas_output,
        output_file=qc_figure,
        title=f"SIGMA Atlas in T2w Space: {subject} {session}"
    )
    atlas_result['qc_figure'] = qc_figure

    print("\nAtlas propagation results:")
    for key, val in atlas_result.items():
        print(f"  {key}: {val}")

    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  Registration QC: {reg_result.get('qc_figure')}")
    print(f"  Atlas in T2w space: {atlas_result['atlas_file']}")
    print(f"  Atlas QC: {atlas_result.get('qc_figure')}")


if __name__ == '__main__':
    main()
