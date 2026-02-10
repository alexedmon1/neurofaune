# TBSS Analysis Guide

This guide covers the full pipeline for running Tract-Based Spatial Statistics (TBSS) in Neurofaune, from DTI preprocessing through voxel-wise group analysis. For general pipeline documentation, see the main [README](../README.md).

---

## Overview

Neurofaune's TBSS pipeline performs **WM-masked voxel-wise analysis** of DTI metrics (FA, MD, AD, RD) using FSL randomise with permutation testing and TFCE multiple comparison correction.

Unlike standard FSL TBSS, Neurofaune does **not** use skeleton projection. Rodent WM tracts are only 1-3 voxels wide, making skeletonization inappropriate. Instead, an interior WM mask is constructed from tissue probability maps and FA thresholds.

### Pipeline Summary

```
1. DTI preprocessing           → FA, MD, AD, RD per subject
2. FA → Template registration  → affine alignment to cohort template
3. Warp metrics to SIGMA       → all subjects in common atlas space
4. Prepare TBSS data           → WM mask, 4D metric volumes
5. Prepare design matrices     → statistical models (dose, sex, age)
6. Run FSL randomise           → permutation testing with TFCE
7. Extract clusters            → significant regions labeled with SIGMA atlas
```

---

## Prerequisites

Before starting TBSS, the following must be complete:

- **Anatomical preprocessing** (Step 3 in [README](../README.md)) — produces cohort templates and Template-to-SIGMA transforms
- **DTI preprocessing** — produces per-subject FA, MD, AD, RD maps
- **SIGMA atlas in study space** — reoriented atlas from `init_study.py`
- **Study tracker CSV** — phenotype data (dose, sex, etc.) for design matrix creation

---

## Step 1: DTI Preprocessing

Run the standard DTI preprocessing pipeline on all subjects:

```bash
uv run python scripts/batch_preprocess_dwi.py \
    --bids-root /path/to/study/raw/bids \
    --output-root /path/to/study \
    --config /path/to/study/config.yaml
```

This performs eddy correction, tensor fitting, and produces per-subject DTI metrics:

```
derivatives/sub-Rat001/ses-p60/dwi/
├── sub-Rat001_ses-p60_FA.nii.gz
├── sub-Rat001_ses-p60_MD.nii.gz
├── sub-Rat001_ses-p60_AD.nii.gz
└── sub-Rat001_ses-p60_RD.nii.gz
```

### Quality Control

Check the DTI QC reports for outliers before proceeding:

```bash
uv run python scripts/generate_batch_qc.py /path/to/study --modality dwi --slice-qc
```

Subjects with failed preprocessing or bad slice counts should be excluded. Create an exclusion file (one `sub-ID_ses-ID` per line) for use in later steps:

```
# exclude_bad_dti.txt
sub-Rat42_ses-p30
sub-Rat99_ses-p60
```

---

## Step 2: Register FA to Cohort Templates

Register each subject's FA map directly to their age-matched cohort template using ANTs affine:

```bash
uv run python scripts/batch_register_fa_to_template.py
uv run python scripts/batch_register_fa_to_template.py --dry-run   # preview
uv run python scripts/batch_register_fa_to_template.py --n-cores 8  # parallel
```

This produces per-subject affine transforms:

```
transforms/sub-Rat001/ses-p60/
└── sub-Rat001_ses-p60_FA_to_template_0GenericAffine.mat
```

**Note:** This is the **direct** FA-to-Template pipeline, not the older FA-to-T2w-to-Template chain. Direct registration produces better atlas overlap, especially for subjects with 3D T2w acquisitions.

---

## Step 3: Warp DTI Metrics to SIGMA Space

Apply the two-stage transform chain (FA-to-Template + Template-to-SIGMA) to warp all DTI metrics into SIGMA atlas space:

```bash
uv run python scripts/batch_register_dwi.py \
    --study-root /path/to/study \
    --n-cores 4

# Preview what would be processed
uv run python scripts/batch_register_dwi.py \
    --study-root /path/to/study \
    --dry-run

# Skip subjects already warped
uv run python scripts/batch_register_dwi.py \
    --study-root /path/to/study \
    --skip-existing
```

This produces SIGMA-space metric maps for each subject:

```
derivatives/sub-Rat001/ses-p60/dwi/
├── sub-Rat001_ses-p60_space-SIGMA_FA.nii.gz
├── sub-Rat001_ses-p60_space-SIGMA_MD.nii.gz
├── sub-Rat001_ses-p60_space-SIGMA_AD.nii.gz
└── sub-Rat001_ses-p60_space-SIGMA_RD.nii.gz
```

The transform chain applied is:

```
Subject FA → Cohort Template → SIGMA Atlas
            (affine)          (affine + SyN warp)
```

---

## Step 4: Prepare TBSS Data

Collect all SIGMA-space metrics, create the WM analysis mask, and build 4D volumes:

```bash
uv run python -m neurofaune.analysis.tbss.prepare_tbss \
    --config /path/to/study/config.yaml \
    --output-dir /path/to/study/analysis/tbss

# With subject exclusions
uv run python -m neurofaune.analysis.tbss.prepare_tbss \
    --config config.yaml \
    --output-dir /study/analysis/tbss \
    --exclude-file /study/analysis/tbss/exclude_bad_dti.txt

# Dry run (discover subjects only)
uv run python -m neurofaune.analysis.tbss.prepare_tbss \
    --config config.yaml \
    --output-dir /study/analysis/tbss \
    --dry-run
```

### What This Does

The preparation runs 6 internal phases:

| Phase | Description |
|-------|-------------|
| 1. Subject discovery | Finds `space-SIGMA_*.nii.gz` files, applies exclusions |
| 2. Collect metrics | Links per-subject metric files into `{tbss_dir}/{metric}/` |
| 3. Mean FA + WM mask | Averages all FA maps; builds tissue-informed WM mask |
| 4. Analysis mask | Thresholds mean FA within WM mask |
| 5. 4D volumes | Stacks each metric across subjects into `all_{metric}.nii.gz` |
| 6. Manifest | Writes `subject_list.txt` and `subject_manifest.json` |

### WM Mask Construction

The WM analysis mask is built conservatively to avoid partial-volume contamination:

1. Load SIGMA WM probability template
2. Threshold: `WM_prob > 0.3` AND `mean_FA > 0.2`
3. Erode 2 voxels from brain boundary (removes exterior WM artifacts)
4. Remove small isolated clusters (<50 voxels)
5. Intersect with `mean_FA >= 0.3` (the analysis threshold)

### Tunable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fa-threshold` | 0.2 | FA threshold for initial WM mask |
| `--analysis-threshold` | 0.3 | FA threshold for final analysis mask |
| `--wm-prob-threshold` | 0.3 | WM tissue probability threshold |
| `--erosion-voxels` | 2 | Boundary erosion distance |

### Output

```
analysis/tbss/
├── subject_manifest.json       # Included/excluded subjects with reasons
├── subject_list.txt            # Included subjects in processing order
├── FA/, MD/, AD/, RD/          # Per-subject metric symlinks
│   ├── sub-Rat1_ses-p30_FA_sigma.nii.gz
│   └── ...
├── wm_mask/
│   ├── interior_wm_mask.nii.gz
│   └── wm_probability_sigma.nii.gz
└── stats/
    ├── mean_FA.nii.gz          # Mean FA across all subjects
    ├── analysis_mask.nii.gz    # Final WM mask for analysis
    ├── all_FA.nii.gz           # 4D: (x, y, z, n_subjects)
    ├── all_MD.nii.gz
    ├── all_AD.nii.gz
    └── all_RD.nii.gz
```

---

## Step 5: Prepare Design Matrices

Create FSL-format design matrices and contrast files for statistical testing:

```bash
uv run python scripts/prepare_tbss_designs.py \
    --study-tracker /path/to/study_tracker.csv \
    --tbss-dir /path/to/study/analysis/tbss \
    --output-dir /path/to/study/analysis/tbss/designs
```

### Design Structure

This creates 4 analysis designs:

| Design | Subjects | Factors | Contrasts |
|--------|----------|---------|-----------|
| `per_pnd_p30` | P30 only | dose (dummy, ref=C) + sex (effect) | 6: each dose vs Control, both directions |
| `per_pnd_p60` | P60 only | same | 6 |
| `per_pnd_p90` | P90 only | same | 6 |
| `pooled` | All ages | dose + PND (effect) + sex + dose x PND | 18: 6 main + 12 interaction |

**Per-PND contrasts** (6 per design):

| Contrast | Tests |
|----------|-------|
| `H_gt_C` / `C_gt_H` | High dose vs Control |
| `L_gt_C` / `C_gt_L` | Low dose vs Control |
| `M_gt_C` / `C_gt_M` | Medium dose vs Control |

**Pooled interaction contrasts** (12 additional): test whether dose effects differ across developmental timepoints (e.g., `dose_H x PND_P60` — does the High dose effect at P60 differ from its effect at P30?).

### Coding Scheme

- **Dose**: Dummy coding (reference = Control). Each dose column is 1 for that dose, 0 otherwise. Contrasts directly test dose vs Control.
- **Sex**: Effect coding (reference = Female). Removes sex variance without dedicating a contrast.
- **PND** (pooled only): Effect coding (reference = P30).

### Output per Design

```
designs/per_pnd_p60/
├── design.mat              # FSL format design matrix
├── design.con              # FSL format contrast file
├── design_summary.json     # Human-readable: column names, contrast names
└── subject_order.txt       # Subjects in design matrix row order
```

### Dependencies

The design preparation uses `DesignHelper` from the [neuroaider](https://github.com/alexedmon1/neuroaider) package, which handles FSL-format matrix generation, coding scheme application, and rank-deficiency validation.

---

## Step 6: Run Voxel-Wise Analysis

Run FSL randomise with TFCE correction for each design and metric:

```bash
# Run all 4 analyses
PYTHONUNBUFFERED=1 uv run python scripts/run_tbss_analysis.py \
    --tbss-dir /path/to/study/analysis/tbss \
    --config /path/to/study/config.yaml \
    --n-permutations 5000

# Run a single analysis
uv run python scripts/run_tbss_analysis.py \
    --tbss-dir /path/to/study/analysis/tbss \
    --config config.yaml \
    --analyses per_pnd_p60 \
    --n-permutations 5000

# Quick test (fewer permutations)
uv run python scripts/run_tbss_analysis.py \
    --tbss-dir /path/to/study/analysis/tbss \
    --config config.yaml \
    --analyses per_pnd_p30 \
    --n-permutations 100 --seed 42
```

**Tip:** Use `PYTHONUNBUFFERED=1` when redirecting output to a log file to avoid buffering delays on NFS filesystems.

### Running Analyses in Parallel

Each analysis is independent and can be run simultaneously:

```bash
for analysis in per_pnd_p30 per_pnd_p60 per_pnd_p90 pooled; do
    PYTHONUNBUFFERED=1 uv run python scripts/run_tbss_analysis.py \
        --tbss-dir /study/analysis/tbss \
        --config config.yaml \
        --analyses $analysis \
        --n-permutations 5000 \
        2>&1 | tee /study/analysis/tbss/logs/randomise_${analysis}.log &
done
```

### What This Does

For each analysis and metric:

1. **Subset 4D volumes** — extracts the design's subjects from the master 4D volume, reordering to match `subject_order.txt`
2. **Run FSL `randomise`** — permutation testing with 2D TFCE (`--T2`)
3. **Extract clusters** — threshold TFCE-corrected p-values at p < 0.05, identify connected clusters
4. **Label clusters** — map significant voxels to SIGMA atlas regions
5. **Generate reports** — HTML cluster reports with region names and voxel counts

### Permutation Count

- **100**: Quick test, not for publication
- **5000**: Standard for most analyses
- **10000**: For marginal effects or final publication-quality results

### Runtime

Runtime scales with subject count, number of contrasts, and permutations. Approximate per-metric times with 5000 permutations on a single core:

| Analysis | Subjects | Contrasts | Approx. time per metric |
|----------|----------|-----------|------------------------|
| per_pnd_p30 | ~45 | 6 | ~40 min |
| per_pnd_p60 | ~41 | 6 | ~35 min |
| per_pnd_p90 | ~61 | 6 | ~45 min |
| pooled | ~147 | 18 | ~2 hours |

Each analysis runs 4 metrics (FA, MD, AD, RD), so total time per analysis is roughly 4x the per-metric time.

### Output

```
randomise/per_pnd_p60/
├── data/
│   ├── all_FA.nii.gz                         # Subsetted 4D (design subjects only)
│   ├── all_MD.nii.gz
│   ├── all_AD.nii.gz
│   └── all_RD.nii.gz
├── randomise_FA/
│   ├── randomise_tfce_corrp_tstat1.nii.gz    # TFCE-corrected p-values, contrast 1
│   ├── randomise_tfce_corrp_tstat2.nii.gz    # contrast 2
│   └── ...                                    # one per contrast
├── randomise_MD/
│   └── ...
├── cluster_reports_FA/
│   ├── contrast_1_H_gt_C.html               # Cluster report with SIGMA labels
│   └── ...
└── analysis_summary.json                      # Structured results summary
```

### Interpreting Results

The key output files are `randomise_tfce_corrp_tstat{N}.nii.gz`:

- Values range 0-1; voxels with **value > 0.95** are significant at **p < 0.05** (corrected)
- Contrast numbering matches the order in `design_summary.json`
- Odd-numbered contrasts test the positive direction (e.g., H > C), even-numbered test the negative (C > H)

---

## Troubleshooting

### Subject count mismatch between design and 4D volumes

The TBSS prep script builds 4D volumes from the `subject_list.txt` generated during preparation. If you re-run prep with different exclusions, you must also regenerate designs to ensure the subject lists match.

```bash
# After changing exclusions, re-run both:
uv run python -m neurofaune.analysis.tbss.prepare_tbss --config config.yaml --output-dir /study/analysis/tbss --exclude-file exclude.txt
uv run python scripts/prepare_tbss_designs.py --study-tracker tracker.csv --tbss-dir /study/analysis/tbss --output-dir /study/analysis/tbss/designs
```

### No significant results

- Check sample size — small N means less statistical power
- Increase permutations (5000 minimum, 10000 for marginal effects)
- Verify WM mask coverage — view `analysis_mask.nii.gz` overlaid on `mean_FA.nii.gz`
- Check for registration failures — bimodal FA distributions in `mean_FA.nii.gz` suggest some subjects have failed registration

### Physiologically implausible results (e.g., FA and MD both increasing)

FA and MD typically move in opposite directions. If both increase in the same group, suspect a registration confound:

1. Check whether the affected group has systematically different scan parameters (e.g., 5-slice vs 11-slice DTI)
2. Look for bimodal distributions in the control or treatment groups
3. Exclude subjects with failed registration and re-run

### Stale files from previous runs

The prep script cleans stale metric files automatically (as of `b86fde0`). If you encounter issues with old runs, delete the `stats/` and metric directories and re-run preparation.

---

## Complete Example

End-to-end TBSS for a BPA-Rat study:

```bash
# 1. DTI preprocessing (if not already done)
uv run python scripts/batch_preprocess_dwi.py \
    --bids-root /mnt/arborea/bpa-rat/raw/bids \
    --output-root /mnt/arborea/bpa-rat \
    --config /mnt/arborea/bpa-rat/config.yaml

# 2. Register FA to templates
uv run python scripts/batch_register_fa_to_template.py

# 3. Warp all metrics to SIGMA
uv run python scripts/batch_register_dwi.py \
    --study-root /mnt/arborea/bpa-rat \
    --n-cores 4

# 4. Prepare TBSS data (with exclusions)
uv run python -m neurofaune.analysis.tbss.prepare_tbss \
    --config configs/bpa_rat_example.yaml \
    --output-dir /mnt/arborea/bpa-rat/analysis/tbss \
    --exclude-file /mnt/arborea/bpa-rat/analysis/tbss/exclude_bad_dti.txt

# 5. Create design matrices
uv run python scripts/prepare_tbss_designs.py \
    --study-tracker /mnt/arborea/bpa-rat/study_tracker_combined_250916.csv \
    --tbss-dir /mnt/arborea/bpa-rat/analysis/tbss \
    --output-dir /mnt/arborea/bpa-rat/analysis/tbss/designs

# 6. Run all analyses (parallel)
for analysis in per_pnd_p30 per_pnd_p60 per_pnd_p90 pooled; do
    PYTHONUNBUFFERED=1 uv run python scripts/run_tbss_analysis.py \
        --tbss-dir /mnt/arborea/bpa-rat/analysis/tbss \
        --config configs/bpa_rat_example.yaml \
        --analyses $analysis \
        --n-permutations 5000 \
        2>&1 | tee /mnt/arborea/bpa-rat/analysis/tbss/logs/randomise_${analysis}.log &
done
```

---

## Related Documentation

- [README](../README.md) — Full pipeline overview (Steps 1-5)
- [Architecture](ARCHITECTURE.md) — Codebase structure and design decisions
- [Atlas Guide](ATLAS_GUIDE.md) — SIGMA atlas setup and management
- [TBSS Implementation Plan](plans/TBSS_IMPLEMENTATION_PLAN.md) — Original design document
