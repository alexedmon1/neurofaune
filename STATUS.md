# Neurofaune Development Status

**Last Updated:** 2026-03-02

---

## Current Phase: Post-Phase 18 — Analysis & Reporting Infrastructure

All core preprocessing pipelines (Phases 1-8) are complete. Recent development has focused on group-level analysis modules and cross-modality integration tools.

---

## Recent Development

### 2026-03-02: Progress Tracking & MCCA Enhancements

- **Analysis progress tracking** — `neurofaune/analysis/progress.py` (NEW)
  - `AnalysisProgress` class: writes `_progress.json` with PID, timestamps, task/phase, completion counts
  - Atomic POSIX writes (tmp + rename), marks completed on finish
  - Integrated into all 8 runner scripts (classification, regression, covnet, mcca, tbss, vbm, fmri voxelwise, mvpa)

- **MCCA confound residualization** — `neurofaune/network/mcca.py`
  - `--confounds` parameter for OLS residualization of arbitrary metadata columns
  - Categorical confounds auto-encoded via `pd.get_dummies(drop_first=True)`
  - Applied after subject intersection, before z-scoring

- **MCCA sex difference testing** — `neurofaune/network/mcca.py`
  - `test_sex_differences()`: PERMANOVA on MCCA score space + per-CV Cohen's d with permutation p-values
  - Automatic post-hoc when sex column present with 2 levels
  - `plot_scores_by_sex()` visualization added to `mcca_visualization.py`

### 2026-02-23: CovNet Cross-Timepoint & Functional Classification

- **Cross-timepoint CovNet comparisons** — 12 new pairwise PND comparisons within each dose
- **Consolidated `default_dose_comparisons()`** in `matrices.py`
- **Functional classification/regression** extended to fALFF, ReHo, ALFF metrics

### 2026-02-18: Whole-Brain Voxelwise fMRI Analysis

- **`prepare_fmri_voxelwise.py`** — Discovers SIGMA-space fALFF/ReHo, stacks into masked 4D volumes
- **3D TFCE support** in `randomise_wrapper.py` — `tfce_2d` parameter for `-T` vs `--T2`
- **`run_voxelwise_fmri_analysis.py`** — Runner for whole-brain fMRI with 3D TFCE

### 2026-02-17: Resting-State Pipeline Split & OOM Fix

- Split monolithic resting-state script into `batch_falff_analysis.py`, `batch_reho_analysis.py`, `batch_fc_analysis.py`
- **Fixed OOM in BOLD-to-SIGMA warping** — volume-by-volume warping with fslmerge (peak 1.3 GB vs 10+ GB)
- Temp file location fix (use study work dir, not `/tmp` tmpfs)

### 2026-02-13: Config-Driven Workflows & fMRI Fixes

- Replaced ~35 hardcoded parameters across 4 workflows with `get_config_value()` calls
- Per-modality config validators in `config_validator.py`
- fMRI fixes: tissue mask resampling, FD scaling (10x), nuisance regression, bandpass ordering

### 2026-02-11: ROI-Level Analysis Modules

- **Classification module** — `neurofaune/analysis/classification/` (data_prep, omnibus, pca, lda, classifiers, visualization)
- **Regression module** — `neurofaune/analysis/regression/dose_response.py` (SVR, Ridge, PLS with LOOCV)
- **MSME TBSS** — `prepare_modality_tbss.py` generalized for non-DTI modalities
- **Provenance safety chain** — SHA256 hash validation from preparation through design to randomise

### Earlier Phases (2026-01 through 2026-02)

- **Phase 8:** Template-based registration (subject -> template -> SIGMA)
- **Phase 7:** fMRI preprocessing (ICA denoising, nuisance regression, bandpass filtering)
- **Phase 6:** MSME T2 mapping preprocessing
- **Phase 5:** Template building (ANTs, 3 cohorts)
- **Phase 4:** DTI preprocessing (eddy, tensor fitting)
- **Phase 3:** Anatomical T2w preprocessing (skull strip, registration)
- **Phase 2:** SIGMA atlas management and slice extraction
- **Phase 1:** Foundation (config system, BIDS discovery, study initialization)

---

## Module Architecture

### Preprocessing Pipelines
- `preprocess/workflows/` — anat, dwi, func, msme (all function-based, config-driven)
- `preprocess/qc/` — per-modality QC integrated into workflows
- `preprocess/utils/` — skull stripping (adaptive/atropos_bet), ICA, aCompCor

### Registration & Templates
- `registration/` — slice correspondence (partial-to-full), QC visualization
- `templates/` — builder, registration (subject->template->atlas), slice_registration

### Group Analysis
- `analysis/classification/` — PERMANOVA, PCA, LDA, SVM/logistic LOOCV
- `analysis/regression/` — SVR, Ridge, PLS dose-response
- `analysis/stats/` — randomise wrapper, cluster reporting
- `analysis/mvpa/` — whole-brain decoding, searchlight
- `analysis/progress.py` — lightweight progress tracking for runner scripts

### Network Analysis
- `network/mcca.py` — regularized MCCA with permutation testing, dose association, sex post-hoc
- `network/mcca_visualization.py` — canonical correlations, loadings, score plots
- `connectome/` — covariance network analysis (NBS, graph metrics, whole-network tests)

### Infrastructure
- `config.py` — YAML config with variable substitution (iterative resolution)
- `utils/transforms.py` — transform registry (centralized, avoids redundant computation)
- `utils/exclusion.py` — subject exclusion tracking
- `reporting/` — unified dashboard (register, discover, render, index generation)

---

## fMRI Preprocessing Pipeline

```
1.  Image validation
2.  Discard initial volumes (5 for T1 equilibration)
2.5 Slice timing correction (custom order)
3.  Brain extraction (adaptive per-slice BET)
4.  Motion correction (MCFLIRT, middle reference)
5.  ICA denoising (MELODIC -> classify -> remove noise)
6.  Spatial smoothing (0.5mm FWHM)
7.  Extract 24 motion confound regressors
8.  aCompCor extraction (5 CSF + 5 WM components)
9.  Nuisance regression (34 regressors in single OLS pass)
10. Temporal bandpass filtering (0.01-0.1 Hz, AFTER regression)
11. Save outputs and metadata
12. QC reports
13. BOLD -> template registration (rigid + NCC Z-offset initialization)
```

---

## Skull Stripping System

| Method | Slice Threshold | Used For | Description |
|--------|-----------------|----------|-------------|
| `adaptive` | <10 slices | BOLD (9), MSME (5) | Per-slice BET with iterative frac optimization |
| `atropos_bet` | >=10 slices | T2w (41, 110), DTI (11) | Two-pass: Atropos segmentation + BET refinement |

---

## Workflow Integration

| Workflow | Preprocessing | Cross-modal Registration | Skull Strip | 3D Support | Config-Driven |
|----------|--------------|--------------------------|-------------|------------|---------------|
| Anatomical | Complete | T2w->Template (SyN) | atropos_bet | Yes (resample) | Yes (~20 params) |
| DTI | Complete | FA->T2w (Affine) | atropos_bet | --exclude-3d | Yes (~4 params) |
| Functional | Complete | BOLD->Template (Rigid) | adaptive | --exclude-3d | Yes (FD threshold) |
| MSME | Complete | MSME->T2w (Rigid) | adaptive | --exclude-3d | Yes (~8 params) |
