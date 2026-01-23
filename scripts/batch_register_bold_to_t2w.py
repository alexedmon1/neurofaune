#!/usr/bin/env python3
"""
Batch BOLD-to-T2w registration for all subjects with motion-corrected BOLD and preprocessed T2w.

Computes the temporal mean of the motion-corrected BOLD, applies the brain mask,
then runs ANTs affine registration to T2w for each subject/session.

Usage:
    python batch_register_bold_to_t2w.py
    python batch_register_bold_to_t2w.py --dry-run
    python batch_register_bold_to_t2w.py --force  # Re-run even if transform exists
    python batch_register_bold_to_t2w.py --n-cores 8
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import traceback

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.preprocess.workflows.func_preprocess import register_bold_to_t2w


def find_registration_pairs(study_root: Path):
    """Find all subject/sessions with motion-corrected BOLD, brain mask, and preprocessed T2w."""
    work_root = study_root / 'work'
    derivatives = study_root / 'derivatives'
    pairs = []
    skipped_fov = []

    # Look for motion-corrected BOLD in work directory
    for mcf_file in sorted(work_root.glob('sub-*/ses-*/func_preproc/motion_correction/bold_mcf.nii.gz')):
        # Parse subject/session from path
        parts = mcf_file.parts
        mc_idx = parts.index('motion_correction')
        func_idx = mc_idx - 1  # func_preproc
        subject = parts[func_idx - 2]
        session = parts[func_idx - 1]

        # Check for brain mask in derivatives
        brain_mask = derivatives / subject / session / 'func' / f'{subject}_{session}_desc-brain_mask.nii.gz'
        if not brain_mask.exists():
            continue

        # Check for preprocessed T2w
        t2w_file = derivatives / subject / session / 'anat' / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
        if not t2w_file.exists():
            continue

        # Check FOV compatibility (BOLD in-plane FOV should be within 50% of T2w)
        bold_img = nib.load(mcf_file)
        t2w_img = nib.load(t2w_file)
        bold_fov_xy = bold_img.shape[0] * bold_img.header.get_zooms()[0]
        t2w_fov_xy = t2w_img.shape[0] * t2w_img.header.get_zooms()[0]
        if bold_fov_xy < t2w_fov_xy * 0.5:
            skipped_fov.append(f"{subject}/{session} (BOLD FOV={bold_fov_xy:.0f} vs T2w={t2w_fov_xy:.0f} mm)")
            continue

        # Check for existing transform
        transform_file = study_root / 'transforms' / subject / session / 'BOLD_to_T2w_0GenericAffine.mat'

        pairs.append({
            'subject': subject,
            'session': session,
            'mcf_file': mcf_file,
            'brain_mask': brain_mask,
            't2w_file': t2w_file,
            'transform_exists': transform_file.exists(),
        })

    if skipped_fov:
        print(f"  Skipped {len(skipped_fov)} subjects with mismatched FOV (needs header fix):")
        for s in skipped_fov:
            print(f"    {s}")

    return pairs


def compute_mean_bold(mcf_file: Path, brain_mask: Path, output_file: Path) -> Path:
    """Compute temporal mean of motion-corrected BOLD and apply brain mask."""
    mcf_img = nib.load(mcf_file)
    mcf_data = mcf_img.get_fdata()

    # Temporal mean
    mean_data = np.mean(mcf_data, axis=3)

    # Apply brain mask
    mask_data = nib.load(brain_mask).get_fdata() > 0
    mean_data = mean_data * mask_data

    output_file.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mean_data.astype(np.float32), mcf_img.affine, mcf_img.header),
             output_file)

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Batch BOLD-to-T2w registration')
    parser.add_argument('--study-root', type=Path, default=Path('/mnt/arborea/bpa-rat'))
    parser.add_argument('--n-cores', type=int, default=4, help='Cores for ANTs registration')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--force', action='store_true', help='Re-run even if transform exists')
    args = parser.parse_args()

    print("=" * 70)
    print("Batch BOLD-to-T2w Registration")
    print("=" * 70)
    print(f"Study root: {args.study_root}")
    print(f"Cores: {args.n_cores}")
    print()

    # Find all pairs
    pairs = find_registration_pairs(args.study_root)
    print(f"Found {len(pairs)} subject/sessions with mcf BOLD + brain mask + T2w")

    # Filter based on existing transforms
    if args.force:
        to_process = pairs
    else:
        to_process = [p for p in pairs if not p['transform_exists']]
        already_done = len(pairs) - len(to_process)
        if already_done > 0:
            print(f"  Already registered: {already_done} (use --force to redo)")

    print(f"  To process: {len(to_process)}")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for p in to_process:
            print(f"  {p['subject']}/{p['session']}")
        return

    if not to_process:
        print("\nNothing to do.")
        return

    # Process
    print(f"\nStarting registration of {len(to_process)} subjects...")
    results = {'success': 0, 'error': 0, 'errors': []}
    start_time = datetime.now()

    for i, p in enumerate(to_process, 1):
        subject = p['subject']
        session = p['session']
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(to_process)}] {subject}/{session}")

        work_dir = args.study_root / 'work' / subject / session / 'bold_registration'
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Compute temporal mean of motion-corrected BOLD
            mean_bold_file = work_dir / f'{subject}_{session}_mean_mcf_brain.nii.gz'
            if not mean_bold_file.exists():
                print(f"  Computing temporal mean of motion-corrected BOLD...")
                compute_mean_bold(p['mcf_file'], p['brain_mask'], mean_bold_file)
                print(f"  Mean BOLD: {mean_bold_file.name}")
            else:
                print(f"  Using existing mean BOLD: {mean_bold_file.name}")

            # Register mean BOLD to T2w
            registration_results = register_bold_to_t2w(
                bold_ref_file=mean_bold_file,
                t2w_file=p['t2w_file'],
                output_dir=args.study_root,
                subject=subject,
                session=session,
                work_dir=work_dir,
                n_cores=args.n_cores
            )

            # Save metadata JSON
            derivatives_dir = args.study_root / 'derivatives' / subject / session / 'func'
            derivatives_dir.mkdir(parents=True, exist_ok=True)
            reg_metadata_file = derivatives_dir / f'{subject}_{session}_BOLD_to_T2w_registration.json'
            reg_metadata = {
                'bold_ref_file': str(mean_bold_file),
                'mcf_source': str(p['mcf_file']),
                'brain_mask': str(p['brain_mask']),
                't2w_file': str(registration_results['t2w_file']),
                'affine_transform': str(registration_results['affine_transform']),
                'warped_bold': str(registration_results['warped_bold']) if registration_results.get('warped_bold') else None,
                'bold_shape': list(registration_results['bold_shape']),
                't2w_shape': list(registration_results['t2w_shape']),
                'timestamp': datetime.now().isoformat(),
            }
            with open(reg_metadata_file, 'w') as f:
                json.dump(reg_metadata, f, indent=2)

            results['success'] += 1

        except Exception as e:
            print(f"\n  FAILED: {e}")
            traceback.print_exc()
            results['error'] += 1
            results['errors'].append({'subject': subject, 'session': session, 'error': str(e)})

    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'=' * 70}")
    print("Registration Complete")
    print(f"{'=' * 70}")
    print(f"  Success: {results['success']}")
    print(f"  Errors:  {results['error']}")
    print(f"  Elapsed: {elapsed}")

    if results['errors']:
        print("\nFailed subjects:")
        for err in results['errors']:
            print(f"  {err['subject']}/{err['session']}: {err['error']}")


if __name__ == '__main__':
    main()
