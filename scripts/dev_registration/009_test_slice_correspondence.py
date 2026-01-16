#!/usr/bin/env python3
"""
009_test_slice_correspondence.py

Test the slice correspondence system on real data and generate QC figures.

This script demonstrates how the dual-approach (intensity + landmark) system
finds where partial-coverage modalities (DWI, fMRI) align with full T2w.

Usage:
    python 009_test_slice_correspondence.py /path/to/bpa-rat sub-Rat1 ses-p60

Output:
    QC figures in qc/{subject}/{session}/registration/
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neurofaune.registration import (
    find_slice_correspondence,
    SliceCorrespondenceFinder,
    plot_slice_correspondence,
    plot_slice_correspondence_detailed,
    plot_registration_quality,
)


def test_dwi_correspondence(
    study_root: Path,
    subject: str,
    session: str,
    output_dir: Path
) -> dict:
    """
    Test slice correspondence for DWI (FA) to T2w.

    DWI typically has 11 slices vs T2w's 41 slices.
    """
    print(f"\n{'='*70}")
    print(f"Testing DWI → T2w Slice Correspondence")
    print(f"{'='*70}")

    # Locate files
    derivatives = study_root / 'derivatives' / subject / session
    t2w_file = derivatives / 'anat' / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
    fa_file = derivatives / 'dwi' / f'{subject}_{session}_FA.nii.gz'

    if not t2w_file.exists():
        print(f"ERROR: T2w not found: {t2w_file}")
        return None

    if not fa_file.exists():
        print(f"ERROR: FA not found: {fa_file}")
        return None

    # Load data as NIfTI images to get header info
    t2w_img = nib.load(t2w_file)
    fa_img = nib.load(fa_file)

    t2w_data = t2w_img.get_fdata()
    fa_data = fa_img.get_fdata()

    # Get voxel sizes (slice thickness is the 3rd dimension)
    t2w_zooms = t2w_img.header.get_zooms()
    fa_zooms = fa_img.header.get_zooms()

    print(f"\nInput files:")
    print(f"  T2w: {t2w_file.name} - shape {t2w_data.shape}, voxels {t2w_zooms[:3]} mm")
    print(f"  FA:  {fa_file.name} - shape {fa_data.shape}, voxels {fa_zooms[:3]} mm")

    # Run slice correspondence (passing NIfTI images for automatic header extraction)
    finder = SliceCorrespondenceFinder(
        intensity_weight=0.6,
        landmark_weight=0.4,
        min_confidence=0.3
    )

    result = finder.find_correspondence(
        partial_image=fa_img,  # Pass NIfTI image to extract voxel sizes
        full_image=t2w_img,    # Pass NIfTI image to extract voxel sizes
        modality='dwi',
        partial_is_t2w=False,  # FA has different contrast
        full_is_t2w=True
    )

    # Generate QC figures
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Summary view
    fig1_path = output_dir / f'{subject}_{session}_dwi_slice_correspondence.png'
    plot_slice_correspondence(
        partial_data=fa_data,
        full_data=t2w_data,
        result=result,
        output_file=fig1_path,
        title=f'{subject} {session} DWI Slice Correspondence'
    )
    plt.close()

    # Figure 2: Detailed slice-by-slice view
    fig2_path = output_dir / f'{subject}_{session}_dwi_slice_detail.png'
    plot_slice_correspondence_detailed(
        partial_data=fa_data,
        full_data=t2w_data,
        result=result,
        output_file=fig2_path,
        title=f'{subject} {session} DWI Slice Detail'
    )
    plt.close()

    # Print summary
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"  FA slices 0-{fa_data.shape[2]-1} → T2w slices {result.start_slice}-{result.end_slice}")
    print(f"  Method used: {result.method_used}")
    print(f"  Intensity confidence: {result.intensity_confidence:.3f}")
    print(f"  Landmark confidence: {result.landmark_confidence:.3f}")
    print(f"  Combined confidence: {result.combined_confidence:.3f}")
    print(f"  Mean correlation: {np.mean(result.intensity_correlations):.3f}")

    # Physical coordinate info
    if result.partial_slice_thickness is not None:
        print(f"\nPhysical coordinates:")
        print(f"  Partial slice thickness: {result.partial_slice_thickness:.2f} mm")
        print(f"  Full slice thickness: {result.full_slice_thickness:.2f} mm")
        print(f"  Physical offset from full volume start: {result.physical_offset:.1f} mm")
        partial_coverage = fa_data.shape[2] * result.partial_slice_thickness
        full_coverage = t2w_data.shape[2] * result.full_slice_thickness
        print(f"  Partial coverage: {partial_coverage:.1f} mm / Full coverage: {full_coverage:.1f} mm")

    if result.landmarks_found:
        print(f"\nLandmarks detected: {result.landmarks_found}")

    print(f"\nQC figures saved to: {output_dir}")

    return {
        'result': result,
        'figures': [fig1_path, fig2_path]
    }


def test_multiple_subjects(
    study_root: Path,
    cohort: str = 'p60',
    n_subjects: int = 5,
    output_dir: Path = None
) -> dict:
    """
    Test slice correspondence across multiple subjects.
    """
    print(f"\n{'='*70}")
    print(f"Testing Slice Correspondence Across {n_subjects} Subjects")
    print(f"{'='*70}")

    # Find subjects with both T2w and DWI
    derivatives = study_root / 'derivatives'
    session = f'ses-{cohort}'

    subjects_tested = []
    results = []

    for subject_dir in sorted(derivatives.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith('sub-'):
            continue

        subject = subject_dir.name
        t2w = subject_dir / session / 'anat' / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
        fa = subject_dir / session / 'dwi' / f'{subject}_{session}_FA.nii.gz'

        if t2w.exists() and fa.exists():
            subjects_tested.append(subject)

            if len(subjects_tested) <= n_subjects:
                subj_output = output_dir / subject / session / 'registration'
                result = test_dwi_correspondence(study_root, subject, session, subj_output)
                if result:
                    results.append({
                        'subject': subject,
                        'session': session,
                        **result
                    })

    print(f"\n{'='*70}")
    print(f"SUMMARY: Tested {len(results)} subjects")
    print(f"{'='*70}")

    if results:
        confidences = [r['result'].combined_confidence for r in results]
        correlations = [np.mean(r['result'].intensity_correlations) for r in results]

        print(f"\nConfidence scores:")
        print(f"  Mean: {np.mean(confidences):.3f}")
        print(f"  Min:  {np.min(confidences):.3f}")
        print(f"  Max:  {np.max(confidences):.3f}")

        print(f"\nCorrelation scores:")
        print(f"  Mean: {np.mean(correlations):.3f}")
        print(f"  Min:  {np.min(correlations):.3f}")
        print(f"  Max:  {np.max(correlations):.3f}")

    return {'subjects_available': subjects_tested, 'results': results}


def create_summary_figure(results: list, output_file: Path):
    """
    Create summary figure showing correspondence across subjects.
    """
    if not results:
        return

    n_subjects = len(results)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Confidence scores
    subjects = [r['subject'].replace('sub-', '') for r in results]
    confidences = [r['result'].combined_confidence for r in results]
    colors = ['green' if c > 0.6 else 'orange' if c > 0.4 else 'red' for c in confidences]

    axes[0].bar(range(n_subjects), confidences, color=colors, alpha=0.7, edgecolor='black')
    axes[0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair')
    axes[0].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good')
    axes[0].set_xticks(range(n_subjects))
    axes[0].set_xticklabels(subjects, rotation=45, ha='right')
    axes[0].set_ylabel('Combined Confidence')
    axes[0].set_title('Slice Correspondence Confidence by Subject')
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # Plot 2: Slice positions found
    start_slices = [r['result'].start_slice for r in results]
    end_slices = [r['result'].end_slice for r in results]

    axes[1].bar(range(n_subjects), start_slices, label='Start slice', alpha=0.7)
    axes[1].bar(range(n_subjects), [e - s for s, e in zip(start_slices, end_slices)],
                bottom=start_slices, label='Range', alpha=0.7)
    axes[1].set_xticks(range(n_subjects))
    axes[1].set_xticklabels(subjects, rotation=45, ha='right')
    axes[1].set_ylabel('T2w Slice Index')
    axes[1].set_title('DWI Position in T2w Volume')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Summary figure saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Test slice correspondence system on real data'
    )
    parser.add_argument('study_root', type=Path, help='Path to study root')
    parser.add_argument('subject', type=str, nargs='?', default=None,
                        help='Subject ID (e.g., sub-Rat1). If not provided, tests multiple subjects.')
    parser.add_argument('session', type=str, nargs='?', default='ses-p60',
                        help='Session ID (default: ses-p60)')
    parser.add_argument('--n-subjects', type=int, default=5,
                        help='Number of subjects to test (default: 5)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory for QC figures (default: {study_root}/qc)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.study_root / 'qc'

    print("="*70)
    print("Slice Correspondence Testing")
    print("="*70)
    print(f"Study root: {args.study_root}")
    print(f"Output dir: {args.output_dir}")

    if args.subject:
        # Test single subject
        output_subdir = args.output_dir / args.subject / args.session / 'registration'
        result = test_dwi_correspondence(
            study_root=args.study_root,
            subject=args.subject,
            session=args.session,
            output_dir=output_subdir
        )
    else:
        # Test multiple subjects
        cohort = args.session.replace('ses-', '')
        multi_results = test_multiple_subjects(
            study_root=args.study_root,
            cohort=cohort,
            n_subjects=args.n_subjects,
            output_dir=args.output_dir
        )

        # Create summary figure
        if multi_results['results']:
            summary_file = args.output_dir / f'slice_correspondence_summary_{cohort}.png'
            create_summary_figure(multi_results['results'], summary_file)

    print("\nDone!")


if __name__ == '__main__':
    main()
