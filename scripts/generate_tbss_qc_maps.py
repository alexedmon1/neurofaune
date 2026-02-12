#!/usr/bin/env python3
"""
Generate mean/SD/coverage QC maps for TBSS analysis directories.

Works on any TBSS directory (DTI or MSME). For each metric, computes
within-mask statistics and generates montage PNGs for visual QC.

Usage:
    uv run python scripts/generate_tbss_qc_maps.py \
        --tbss-dir /mnt/arborea/bpa-rat/analysis/tbss --metrics FA MD AD RD

    uv run python scripts/generate_tbss_qc_maps.py \
        --tbss-dir /mnt/arborea/bpa-rat/analysis/tbss_msme --metrics MWF IWF CSFF T2
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def compute_stats(tbss_dir: Path, metric: str, overwrite: bool = False):
    """Compute mean, SD, and coverage maps for a single metric.

    Operates on masked voxels only to keep memory usage low.

    Returns dict with paths to output files and summary statistics,
    or None if the 4D input doesn't exist.
    """
    stats_dir = tbss_dir / 'stats'
    all_file = stats_dir / f'all_{metric}.nii.gz'
    mask_file = stats_dir / 'analysis_mask.nii.gz'

    if not all_file.exists():
        # Try alternate mask name
        if not mask_file.exists():
            mask_file = stats_dir / 'mean_FA_mask.nii.gz'
        print(f"  Skipping {metric}: {all_file} not found")
        return None

    if not mask_file.exists():
        print(f"  Skipping {metric}: no analysis mask found in {stats_dir}")
        return None

    mean_file = stats_dir / f'mean_{metric}.nii.gz'
    std_file = stats_dir / f'std_{metric}.nii.gz'
    coverage_file = stats_dir / f'coverage_{metric}.nii.gz'

    # Check if all outputs exist already
    if not overwrite and mean_file.exists() and std_file.exists() and coverage_file.exists():
        print(f"  {metric}: all stat maps exist, loading for summary")
        mean_img = nib.load(mean_file)
        std_img = nib.load(std_file)
        coverage_img = nib.load(coverage_file)
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata() > 0
        mean_data = mean_img.get_fdata()
        std_data = std_img.get_fdata()
        coverage_data = coverage_img.get_fdata()

        return {
            'metric': metric,
            'mean_file': mean_file,
            'std_file': std_file,
            'coverage_file': coverage_file,
            'mean_of_mean': float(np.mean(mean_data[mask_data])),
            'max_std': float(np.max(std_data[mask_data])),
            'min_coverage': int(np.min(coverage_data[mask_data])),
            'max_coverage': int(np.max(coverage_data[mask_data])),
            'n_subjects': int(np.max(coverage_data)),
            'n_mask_voxels': int(np.sum(mask_data)),
        }

    # Load mask
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata() > 0
    n_mask_voxels = int(np.sum(mask_data))

    # Load 4D volume header to get n_subjects without loading all data
    all_img = nib.load(all_file)
    n_subjects = all_img.shape[3]
    print(f"  {metric}: {n_subjects} subjects, {n_mask_voxels} mask voxels "
          f"({n_mask_voxels * n_subjects * 4 / 1e6:.0f} MB in memory)")

    # Extract masked voxels only: shape (n_mask_voxels, n_subjects)
    all_data = all_img.get_fdata()
    masked_4d = all_data[mask_data, :]  # (n_voxels, n_subjects)

    # Compute statistics
    voxel_mean = np.mean(masked_4d, axis=1)
    voxel_std = np.std(masked_4d, axis=1, ddof=1)
    voxel_coverage = np.sum(masked_4d != 0, axis=1)

    # Free the big array
    del all_data, masked_4d

    # Write back to 3D volumes
    mean_vol = np.zeros(mask_data.shape, dtype=np.float32)
    std_vol = np.zeros(mask_data.shape, dtype=np.float32)
    coverage_vol = np.zeros(mask_data.shape, dtype=np.float32)

    mean_vol[mask_data] = voxel_mean
    std_vol[mask_data] = voxel_std
    coverage_vol[mask_data] = voxel_coverage

    # Save NIfTI outputs with same affine/header as mask
    for vol, path in [(mean_vol, mean_file), (std_vol, std_file), (coverage_vol, coverage_file)]:
        img = nib.Nifti1Image(vol, mask_img.affine, mask_img.header)
        nib.save(img, path)
        print(f"    Saved: {path}")

    return {
        'metric': metric,
        'mean_file': mean_file,
        'std_file': std_file,
        'coverage_file': coverage_file,
        'mean_of_mean': float(np.mean(voxel_mean)),
        'max_std': float(np.max(voxel_std)),
        'min_coverage': int(np.min(voxel_coverage)),
        'max_coverage': int(np.max(voxel_coverage)),
        'n_subjects': n_subjects,
        'n_mask_voxels': n_mask_voxels,
    }


def generate_montage(stats: dict, output_file: Path, template_data=None,
                     n_slices: int = 15):
    """Generate a 3-row montage (mean/SD/coverage) across axial slices.

    If template_data is provided, it is shown as a grayscale background
    behind each overlay so the anatomical context is visible.
    """
    mean_img = nib.load(stats['mean_file'])
    std_img = nib.load(stats['std_file'])
    coverage_img = nib.load(stats['coverage_file'])

    mean_data = mean_img.get_fdata()
    std_data = std_img.get_fdata()
    coverage_data = coverage_img.get_fdata()

    metric = stats['metric']
    n_subjects = stats['n_subjects']

    # Normalise template for display once
    if template_data is not None:
        tpl_nz = template_data[template_data > 0]
        if len(tpl_nz) > 0:
            tpl_vmin, tpl_vmax = np.percentile(tpl_nz, [2, 98])
        else:
            tpl_vmin, tpl_vmax = 0, 1

    # Select axial slices with data
    total_slices = mean_data.shape[2]
    slice_has_data = [z for z in range(total_slices) if mean_data[:, :, z].max() > 0]

    if not slice_has_data:
        print(f"  Warning: no non-zero slices for {metric}, skipping montage")
        return

    # Evenly sample slices
    if len(slice_has_data) <= n_slices:
        selected = slice_has_data
    else:
        indices = np.linspace(0, len(slice_has_data) - 1, n_slices, dtype=int)
        selected = [slice_has_data[i] for i in indices]

    n_cols = len(selected)
    fig, axes = plt.subplots(3, n_cols, figsize=(2.5 * n_cols, 7.5))
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    # Compute percentile ranges from non-zero values
    mean_nonzero = mean_data[mean_data > 0]
    std_nonzero = std_data[std_data > 0]

    if len(mean_nonzero) > 0:
        mean_vmin, mean_vmax = np.percentile(mean_nonzero, [2, 98])
    else:
        mean_vmin, mean_vmax = 0, 1

    if len(std_nonzero) > 0:
        std_vmin, std_vmax = np.percentile(std_nonzero, [2, 98])
    else:
        std_vmin, std_vmax = 0, 1

    row_labels = ['Mean', 'SD', 'Coverage']
    cmaps = ['hot', 'viridis', 'YlOrRd']
    vmins = [mean_vmin, std_vmin, 0]
    vmaxs = [mean_vmax, std_vmax, n_subjects]
    datasets = [mean_data, std_data, coverage_data]
    # Semi-transparent overlay so template anatomy shows through
    alphas = [0.7, 0.7, 0.7]

    for row_idx, (data, cmap, vmin, vmax, label, alpha) in enumerate(
            zip(datasets, cmaps, vmins, vmaxs, row_labels, alphas)):
        for col_idx, z in enumerate(selected):
            ax = axes[row_idx, col_idx]

            # Template background
            if template_data is not None:
                tpl_slc = np.rot90(template_data[:, :, z])
                ax.imshow(tpl_slc, cmap='gray', vmin=tpl_vmin, vmax=tpl_vmax,
                          interpolation='nearest')

            slc = np.rot90(data[:, :, z])

            # Mask out zeros so template shows through outside the data
            masked_slc = np.ma.masked_where(slc == 0, slc)
            im = ax.imshow(masked_slc, cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation='nearest',
                           alpha=alpha if template_data is not None else 1.0)
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(f'z={z}', fontsize=8)

        # Add row label
        axes[row_idx, 0].set_ylabel(label, fontsize=11, fontweight='bold',
                                     rotation=0, labelpad=40, va='center')
        axes[row_idx, 0].yaxis.set_label_position('left')

        # Colorbar
        cbar = fig.colorbar(im, ax=axes[row_idx, :].tolist(), shrink=0.8,
                            pad=0.02, aspect=15)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f'TBSS QC: {metric}  (N={n_subjects})',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved montage: {output_file}")


def subject_qc(tbss_dir: Path, metric: str, subject_names: list,
               template_data=None, qc_dir: Path = None, n_worst: int = 10):
    """Per-subject spatial QC using coverage and Dice overlap.

    For each subject, computes:
    - coverage: fraction of analysis mask voxels where subject has non-zero data
    - dice: Dice coefficient between subject's non-zero footprint and the
      group consensus footprint (voxels where >50% of subjects have data)

    Misregistered subjects show low coverage (data shifted outside mask)
    and low Dice (spatial footprint doesn't match the group).

    Outputs:
    - CSV with per-subject metrics
    - Bar plot sorted by coverage (worst on left)
    - Montage of the N worst subjects showing their spatial footprint
    """
    stats_dir = tbss_dir / 'stats'
    all_file = stats_dir / f'all_{metric}.nii.gz'
    mask_file = stats_dir / 'analysis_mask.nii.gz'

    if not all_file.exists() or not mask_file.exists():
        print(f"  Skipping subject QC for {metric}: files not found")
        return

    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata() > 0
    n_mask_voxels = int(mask_data.sum())
    all_img = nib.load(all_file)
    all_data = all_img.get_fdata()
    n_subjects = all_data.shape[3]

    if len(subject_names) != n_subjects:
        print(f"  Warning: subject_list ({len(subject_names)}) != 4D vol ({n_subjects})")
        subject_names = subject_names[:n_subjects]
        while len(subject_names) < n_subjects:
            subject_names.append(f'subject_{len(subject_names)}')

    # Extract masked voxels: (n_voxels, n_subjects)
    masked_4d = all_data[mask_data, :]

    # Group consensus: voxels where >50% of subjects have non-zero data
    nonzero_count = np.sum(masked_4d != 0, axis=1)  # (n_voxels,)
    consensus = nonzero_count > (n_subjects * 0.5)
    n_consensus = int(consensus.sum())

    # Per-subject metrics
    coverages = []
    dices = []
    for i in range(n_subjects):
        subj_nz = masked_4d[:, i] != 0
        cov = subj_nz.sum() / n_mask_voxels
        coverages.append(cov)
        # Dice = 2*|A∩B| / (|A| + |B|)
        intersection = (subj_nz & consensus).sum()
        denom = subj_nz.sum() + n_consensus
        dice = 2 * intersection / denom if denom > 0 else 0.0
        dices.append(dice)

    coverages = np.array(coverages)
    dices = np.array(dices)

    # Sort by coverage (ascending = worst first)
    sorted_idx = np.argsort(coverages)

    # Identify outliers using IQR on coverage
    q1, q3 = np.percentile(coverages, [25, 75])
    iqr = q3 - q1
    outlier_thresh = q1 - 1.5 * iqr

    # --- CSV ---
    csv_file = qc_dir / f'subject_qc_{metric}.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject', 'coverage', 'dice', 'outlier', 'rank'])
        for rank, idx in enumerate(sorted_idx):
            is_outlier = coverages[idx] < outlier_thresh
            writer.writerow([subject_names[idx], f'{coverages[idx]:.4f}',
                             f'{dices[idx]:.4f}', is_outlier, rank + 1])
    print(f"    Saved: {csv_file}")

    # --- Bar plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, n_subjects * 0.12), 8),
                                    sharex=True)
    # Coverage
    colors_cov = ['red' if coverages[i] < outlier_thresh else 'steelblue'
                  for i in sorted_idx]
    ax1.bar(range(n_subjects), coverages[sorted_idx], color=colors_cov,
            width=1.0, edgecolor='none')
    ax1.axhline(outlier_thresh, color='red', linestyle='--', alpha=0.5,
                label=f'Outlier threshold ({outlier_thresh:.2f})')
    ax1.axhline(np.median(coverages), color='green', linestyle='-', alpha=0.5,
                label=f'Median ({np.median(coverages):.2f})')
    ax1.set_ylabel('Coverage (frac mask voxels)')
    ax1.set_title(f'Subject QC: {metric} — coverage & Dice with group consensus')
    ax1.legend(fontsize=7)
    ax1.set_ylim(0, 1.05)

    # Dice
    colors_dice = ['red' if coverages[i] < outlier_thresh else 'steelblue'
                   for i in sorted_idx]
    ax2.bar(range(n_subjects), dices[sorted_idx], color=colors_dice,
            width=1.0, edgecolor='none')
    ax2.set_ylabel('Dice with consensus')
    ax2.set_xlabel(f'Subjects (sorted by coverage, n={n_subjects})')
    ax2.set_xlim(-0.5, n_subjects - 0.5)
    ax2.set_ylim(0, 1.05)

    # Label worst N
    for rank in range(min(n_worst, n_subjects)):
        idx = sorted_idx[rank]
        short = subject_names[idx].replace('sub-', '').replace('_ses-', '/')
        ax1.text(rank, coverages[idx] + 0.02, short,
                 rotation=90, fontsize=5, ha='center', va='bottom')

    plt.tight_layout()
    barplot_file = qc_dir / f'subject_qc_{metric}_barplot.png'
    plt.savefig(barplot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {barplot_file}")

    # --- Worst-N montage ---
    n_show = min(n_worst, n_subjects)
    worst_indices = sorted_idx[:n_show]

    # Build group mean volume for reference
    group_mean = np.mean(masked_4d, axis=1)
    mean_vol = np.zeros(mask_data.shape, dtype=np.float32)
    mean_vol[mask_data] = group_mean
    slice_has_data = [z for z in range(mean_vol.shape[2])
                      if mean_vol[:, :, z].max() > 0]
    if not slice_has_data:
        del all_data, masked_4d
        return

    # Pick 3 representative slices
    nd = len(slice_has_data)
    rep_slices = [slice_has_data[nd // 4],
                  slice_has_data[nd // 2],
                  slice_has_data[3 * nd // 4]]

    n_cols = len(rep_slices)
    n_rows = 1 + n_show
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    # Template normalization
    if template_data is not None:
        tpl_nz = template_data[template_data > 0]
        tpl_vmin, tpl_vmax = (np.percentile(tpl_nz, [2, 98]) if len(tpl_nz) > 0
                              else (0, 1))

    # Data range from group mean
    mean_nz = group_mean[group_mean > 0]
    data_vmin, data_vmax = (np.percentile(mean_nz, [2, 98]) if len(mean_nz) > 0
                            else (0, 1))

    # Row 0: group mean
    for col_idx, z in enumerate(rep_slices):
        ax = axes[0, col_idx]
        if template_data is not None:
            ax.imshow(np.rot90(template_data[:, :, z]), cmap='gray',
                      vmin=tpl_vmin, vmax=tpl_vmax)
        slc = np.rot90(mean_vol[:, :, z])
        masked_slc = np.ma.masked_where(slc == 0, slc)
        ax.imshow(masked_slc, cmap='hot', vmin=data_vmin, vmax=data_vmax,
                  alpha=0.7)
        ax.axis('off')
        if col_idx == 0:
            ax.set_ylabel('Group mean', fontsize=9, fontweight='bold',
                          rotation=0, labelpad=70, va='center')
        ax.set_title(f'z={z}', fontsize=8)

    # Rows 1..N: worst subjects
    for row_offset, subj_idx in enumerate(worst_indices):
        row = row_offset + 1
        short = subject_names[subj_idx].replace('sub-', '').replace('_ses-', '/')
        cov = coverages[subj_idx]
        dice = dices[subj_idx]

        for col_idx, z in enumerate(rep_slices):
            ax = axes[row, col_idx]
            if template_data is not None:
                ax.imshow(np.rot90(template_data[:, :, z]), cmap='gray',
                          vmin=tpl_vmin, vmax=tpl_vmax)
            subj_slc = np.rot90(all_data[:, :, z, subj_idx])
            masked_slc = np.ma.masked_where(subj_slc == 0, subj_slc)
            ax.imshow(masked_slc, cmap='hot', vmin=data_vmin, vmax=data_vmax,
                      alpha=0.7)
            ax.axis('off')
            if col_idx == 0:
                color = 'red' if cov < outlier_thresh else 'black'
                ax.set_ylabel(f'{short}\ncov={cov:.2f} d={dice:.2f}',
                              fontsize=7, fontweight='bold', rotation=0,
                              labelpad=70, va='center', color=color)

    fig.suptitle(f'Subject QC: {metric} — {n_show} lowest-coverage subjects vs group mean',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0.1, 0, 1, 0.96])
    montage_file = qc_dir / f'subject_qc_{metric}_worst{n_show}.png'
    plt.savefig(montage_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {montage_file}")

    # Print summary
    n_outliers = int(np.sum(coverages < outlier_thresh))
    print(f"    {metric}: median coverage={np.median(coverages):.3f}, "
          f"outlier threshold={outlier_thresh:.3f}, "
          f"outliers={n_outliers}/{n_subjects}")
    if n_outliers > 0:
        print(f"    Outlier subjects (coverage < {outlier_thresh:.3f}):")
        for rank in range(min(n_outliers, 15)):
            idx = sorted_idx[rank]
            print(f"      {subject_names[idx]}: cov={coverages[idx]:.3f}, "
                  f"dice={dices[idx]:.3f}")

    del all_data, masked_4d


def main():
    parser = argparse.ArgumentParser(
        description='Generate mean/SD/coverage QC maps for TBSS directories')
    parser.add_argument('--tbss-dir', type=Path, required=True,
                        help='TBSS directory containing stats/')
    parser.add_argument('--metrics', nargs='+', required=True,
                        help='Metric names (e.g. FA MD AD RD or MWF IWF CSFF T2)')
    parser.add_argument('--template', type=Path, default=None,
                        help='Template NIfTI to show as anatomical background in montages')
    parser.add_argument('--subject-qc', action='store_true',
                        help='Run per-subject QC: correlate each subject with group mean')
    parser.add_argument('--n-worst', type=int, default=10,
                        help='Number of worst subjects to show in QC montage (default: 10)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Recompute even if output files exist')
    parser.add_argument('--n-slices', type=int, default=15,
                        help='Number of axial slices in montage (default: 15)')
    args = parser.parse_args()

    tbss_dir = args.tbss_dir
    stats_dir = tbss_dir / 'stats'

    if not stats_dir.exists():
        print(f"Error: stats directory not found: {stats_dir}")
        sys.exit(1)

    # Load template background if provided
    template_data = None
    if args.template:
        if not args.template.exists():
            print(f"Error: template not found: {args.template}")
            sys.exit(1)
        template_data = nib.load(args.template).get_fdata()
        print(f"  Template: {args.template.name}")

    print(f"TBSS QC Maps")
    print(f"  Directory: {tbss_dir}")
    print(f"  Metrics: {', '.join(args.metrics)}")
    print()

    qc_dir = tbss_dir / 'qc'
    qc_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for metric in args.metrics:
        print(f"Processing {metric}...")
        stats = compute_stats(tbss_dir, metric, overwrite=args.overwrite)
        if stats is None:
            continue

        # Generate montage
        montage_file = qc_dir / f'qc_maps_{metric}.png'
        generate_montage(stats, montage_file, template_data=template_data,
                         n_slices=args.n_slices)
        all_stats.append(stats)

    # Per-subject QC
    if args.subject_qc:
        subject_list_file = tbss_dir / 'subject_list.txt'
        if not subject_list_file.exists():
            print(f"\nWarning: subject_list.txt not found in {tbss_dir}, skipping subject QC")
        else:
            subject_names = subject_list_file.read_text().strip().split('\n')
            print(f"\nSubject-level QC ({len(subject_names)} subjects)")
            for metric in args.metrics:
                print(f"\n  {metric}:")
                subject_qc(tbss_dir, metric, subject_names,
                           template_data=template_data, qc_dir=qc_dir,
                           n_worst=args.n_worst)

    # Print summary table
    if all_stats:
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"{'Metric':<8} {'N_subj':>6} {'N_voxels':>10} {'Mean(mean)':>12} "
              f"{'Max(SD)':>10} {'Min_cov':>8} {'Max_cov':>8}")
        print('-' * 70)
        for s in all_stats:
            print(f"{s['metric']:<8} {s['n_subjects']:>6} {s['n_mask_voxels']:>10} "
                  f"{s['mean_of_mean']:>12.4f} {s['max_std']:>10.4f} "
                  f"{s['min_coverage']:>8} {s['max_coverage']:>8}")

        # Coverage warnings
        for s in all_stats:
            if s['min_coverage'] < s['n_subjects']:
                pct = 100 * s['min_coverage'] / s['n_subjects']
                print(f"\n  Warning: {s['metric']} has voxels with only "
                      f"{s['min_coverage']}/{s['n_subjects']} subjects "
                      f"({pct:.0f}% coverage)")
    else:
        print("\nNo metrics were processed.")


if __name__ == '__main__':
    main()
