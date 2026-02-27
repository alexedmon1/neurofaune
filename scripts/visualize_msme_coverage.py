#!/usr/bin/env python3
"""
Visualize MSME registration coverage across cohorts.

Creates a mosaic figure showing where each subject's MSME slab lands
in template space, with per-cohort histograms and coverage heatmaps.

Usage:
    uv run python scripts/visualize_msme_coverage.py \
        --study-root /mnt/arborea/bpa-rat \
        --output /mnt/arborea/bpa-rat/qc/msme_coverage_mosaic.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.preprocess.workflows.msme_preprocess import detect_z_registration_outliers


def get_subject_coverage(warped_file: Path) -> tuple:
    """Return (z_start, z_end) of signal in a warped MSME file."""
    img = nib.load(warped_file)
    data = img.get_fdata()
    data_max = data.max()
    if data_max == 0:
        return None, None

    threshold = data_max * 0.05
    slice_max = np.array([data[:, :, z].max() for z in range(data.shape[2])])
    signal_slices = np.where(slice_max > threshold)[0]

    if len(signal_slices) == 0:
        return None, None

    return int(signal_slices[0]), int(signal_slices[-1])


def main():
    parser = argparse.ArgumentParser(
        description='Visualize MSME registration coverage across cohorts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None,
                        help='Output PNG path (default: {study_root}/qc/msme_coverage_mosaic.png)')
    parser.add_argument('--outlier-threshold', type=float, default=3.0)

    args = parser.parse_args()

    if args.output is None:
        args.output = args.study_root / 'qc' / 'msme_coverage_mosaic.png'
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cohorts = ['p30', 'p60', 'p90']
    transforms_root = args.study_root / 'transforms'
    templates_root = args.study_root / 'templates'

    # Collect data per cohort
    cohort_data = {}
    for cohort in cohorts:
        warped_files = sorted(transforms_root.glob(
            f'sub-*/ses-{cohort}/MSME_to_template_Warped.nii.gz'
        ))

        # Get template shape for reference
        tpl_path = templates_root / 'anat' / cohort / f'tpl-BPARat_{cohort}_T2w.nii.gz'
        if tpl_path.exists():
            tpl_nz = nib.load(tpl_path).shape[2]
        else:
            tpl_nz = 41

        subjects = []
        for wf in warped_files:
            parts = wf.parts
            sub = [p for p in parts if p.startswith('sub-')][0]
            z_start, z_end = get_subject_coverage(wf)
            if z_start is not None:
                subjects.append((sub, z_start, z_end))

        # Outlier detection
        outlier_result = detect_z_registration_outliers(
            args.study_root, cohort, threshold=args.outlier_threshold
        )
        outlier_subs = {s[0] for s in outlier_result['outliers']}

        cohort_data[cohort] = {
            'subjects': subjects,
            'tpl_nz': tpl_nz,
            'outlier_result': outlier_result,
            'outlier_subs': outlier_subs,
        }

    # Create figure: 2 rows x 3 columns
    # Row 1: coverage swim-lane (each subject as a horizontal bar)
    # Row 2: z_start histogram
    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                              gridspec_kw={'height_ratios': [3, 1]})

    for col, cohort in enumerate(cohorts):
        cd = cohort_data[cohort]
        subjects = cd['subjects']
        outlier_subs = cd['outlier_subs']
        tpl_nz = cd['tpl_nz']
        outlier_result = cd['outlier_result']

        # Sort by z_start
        subjects_sorted = sorted(subjects, key=lambda x: x[1])

        # --- Row 1: Swim-lane plot ---
        ax = axes[0, col]
        for i, (sub, z_start, z_end) in enumerate(subjects_sorted):
            is_outlier = sub in outlier_subs
            color = '#e74c3c' if is_outlier else '#3498db'
            alpha = 0.9 if is_outlier else 0.6
            ax.barh(i, z_end - z_start + 1, left=z_start, height=0.8,
                    color=color, alpha=alpha, edgecolor='none')

        # Median line
        if outlier_result['median_z'] is not None:
            med = outlier_result['median_z']
            ax.axvline(med, color='#2ecc71', linewidth=2, linestyle='--',
                       label=f'Median z={med}')

        ax.set_xlim(0, tpl_nz)
        ax.set_ylim(-0.5, len(subjects_sorted) - 0.5)
        ax.set_xlabel('Template slice (z)')
        ax.set_ylabel('Subject (sorted by z_start)')
        n_out = len([s for s in subjects if s[0] in outlier_subs])
        ax.set_title(f'{cohort}  (n={len(subjects)}, {n_out} outliers)',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_yticks([])

        # --- Row 2: Histogram ---
        ax2 = axes[1, col]
        z_starts = [s[1] for s in subjects]
        z_starts_ok = [s[1] for s in subjects if s[0] not in outlier_subs]
        z_starts_out = [s[1] for s in subjects if s[0] in outlier_subs]

        bins = np.arange(0, tpl_nz + 1) - 0.5
        ax2.hist(z_starts_ok, bins=bins, color='#3498db', alpha=0.7, label='OK')
        if z_starts_out:
            ax2.hist(z_starts_out, bins=bins, color='#e74c3c', alpha=0.7, label='Outlier')
        ax2.set_xlabel('z_start (template slice)')
        ax2.set_ylabel('Count')
        ax2.set_xlim(0, tpl_nz)

        # Stats annotation
        if outlier_result['median_z'] is not None:
            stats_text = (f"median={outlier_result['median_z']}, "
                          f"MAD={outlier_result['mad']:.1f}")
            ax2.text(0.98, 0.95, stats_text, transform=ax2.transAxes,
                     ha='right', va='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
        ax2.legend(fontsize=9)

    fig.suptitle('MSME Registration Coverage (z_range=13-22)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Coverage mosaic saved to: {args.output}")

    # Print summary
    for cohort in cohorts:
        cd = cohort_data[cohort]
        outlier_result = cd['outlier_result']
        n_out = len(outlier_result['outliers'])
        print(f"  {cohort}: n={len(cd['subjects'])}, "
              f"median_z={outlier_result['median_z']}, "
              f"MAD={outlier_result['mad']}, "
              f"outliers={n_out}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
