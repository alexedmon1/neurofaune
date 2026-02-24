# Project Status

**Last Updated:** 2026-02-23

---

## Current Phase: 17 - Cross-Timepoint CovNet & Results Review

### Session Summary (2026-02-23)

**Completed this session:**

1. **Added cross-timepoint comparisons to CovNet** (`a10087c`)
   - New `cross_timepoint_comparisons()` in `matrices.py`: pairwise PND comparisons within each dose (12 new comparisons: 4 doses × 3 PND pairs)
   - Consolidated duplicate `_default_comparisons()` into shared `default_dose_comparisons()` in `matrices.py`
   - Updated `run_covnet_analysis.py`: NBS, territory Fisher z, and whole-network tests now run 21 comparisons (9 dose-vs-control + 12 cross-timepoint)
   - `--skip-cross-timepoint` flag to disable if needed
   - Fixed missing `import networkx` in `visualization.py` (crashed NBS network plots)

2. **Reviewed all completed voxel-based analyses** — see Group Analysis Status below

3. **Fixed ReHo voxelwise failure** — orchestrator referenced removed `configs/bpa_rat_example.yaml` (commit `9e9b4d0`). Updated to `/mnt/arborea/bpa-rat/config.yaml`, relaunched all 8 ReHo designs.

**Running in background:**

| Analysis | PID | Output | Monitor |
|----------|-----|--------|---------|
| CovNet DTI (FA/MD/AD/RD, 21 comparisons, 5000 perms) | 541319 | `analysis/covnet_dti/` | `tail -f .../logs/covnet_dti_full.log` |
| ReHo categorical (p30/p60/p90/pooled, 5000 perms) | 543627 | `analysis/voxelwise_fmri/randomise/` | `tail -f .../voxelwise_fmri/logs/orchestrator.log` |
| ReHo dose-response (p30/p60/p90/pooled) | — | queued after categorical | same orchestrator log |

### Session Summary (2026-02-18)

**Completed this session:**

1. **Created `scripts/prepare_fmri_voxelwise.py`** — Discovers SIGMA-space fALFF/ReHo maps, merges with study tracker, stacks into masked 4D volumes using SIGMA brain mask (whole-brain, not WM skeleton)
   - 133 sessions found, 131 matched with tracker (sub-Rat158/sub-Rat228 not in tracker)
   - Distribution: P30=37, P60=33, P90=61
   - 4D volumes: `all_fALFF.nii.gz` and `all_ReHo.nii.gz` (128×128×218×131)
   - Analysis mask: SIGMA brain mask, 751,262 voxels (vs 92,435 for WM skeleton)

2. **Added 3D TFCE support to `randomise_wrapper.py`** — New `tfce_2d` parameter (default `True` for backward compat). When `False`, uses `-T` (3D TFCE) instead of `--T2` (2D skeleton TFCE).

3. **Created `scripts/run_voxelwise_fmri_analysis.py`** — Runner for whole-brain analysis, calls `run_randomise(..., tfce_2d=False)`. Handles all 8 designs × 2 metrics with subsetting, cluster reporting, and provenance validation.

4. **Generated all design matrices** — 8 design directories (4 categorical + 4 dose-response), all full-rank, all with provenance tracking. Design scripts are modality-agnostic — just need `subject_list.txt`.

5. **Launched randomise** — 4 batches of 4 analyses each (fALFF categorical, fALFF dose-response, ReHo categorical, ReHo dose-response), running sequentially via orchestrator script.

**Running in background:**

| Batch | Analyses | Metric | Status |
|-------|----------|--------|--------|
| 1 | per_pnd_p30, p60, p90, pooled | fALFF | Running |
| 2 | dose_response_p30, p60, p90, pooled | fALFF | Queued |
| 3 | per_pnd_p30, p60, p90, pooled | ReHo | Queued |
| 4 | dose_response_p30, p60, p90, pooled | ReHo | Queued |

Monitor: `tail -f /mnt/arborea/bpa-rat/analysis/voxelwise_fmri/logs/orchestrator.log`

### Session Summary (2026-02-17b)

**Completed this session:**

1. **Fixed OOM in BOLD-to-SIGMA warping** — `neurofaune/templates/registration.py`
   - Root cause: `warp_bold_to_sigma()` passed full 4D to `antsApplyTransforms -e 3`, loading entire timeseries into RAM. SIGMA-space output is 128×128×218×355 float64 = **10.1 GB per session**. With 6 batch workers, this exceeded 32 GB and OOM-killed processes.
   - Fix: Volume-by-volume warping with `fslmerge` concatenation
     - Added `_warp_single_volume()` helper — extracts one 3D volume, runs `antsApplyTransforms -d 3`, masks, saves to temp file (~14 MB each)
     - Added `_warp_4d_volumewise()` — dispatches volumes to `ThreadPoolExecutor(max_workers=n_threads)` for parallel warping, then `fslmerge -t` concatenates from disk
     - Peak memory per subject: ~6 × (14 MB output + 196 MB ANTs) ≈ 1.3 GB (vs 10-13 GB before)
   - `warp_bold_to_sigma()` gains `low_memory=True` (default) and `n_threads=6` params
   - 3D maps (fALFF, ReHo) still use single-call path unchanged
   - Iteration: tried np.zeros pre-allocation (still 5 GB RSS), then np.memmap (nib.save pulled it all back through gzip), then fslmerge (works)

2. **Cleaned 4 corrupt SIGMA BOLD files** from prior OOM crashes
   - sub-Rat102/ses-p60 (1307 MB), sub-Rat110/ses-p90 (699 MB), sub-Rat111/ses-p90 (17 MB), sub-Rat115/ses-p60 (337 MB)
   - Truncated .nii.gz from processes killed mid-write

3. **Verified volume-by-volume warping works** — sub-Rat102 sequential test reached 150/355 volumes with Python at 304 MB RSS before being stopped to test parallel version

4. **Temp file location fix** — `/tmp` is often tmpfs (RAM-backed), so temp files there consumed RAM and defeated the purpose. Added `work_dir` parameter to `_warp_4d_volumewise` and `warp_bold_to_sigma`, defaulting to `{study_root}/work/`. Updated `batch_fc_analysis.py` to pass it.

5. **End-to-end test passed** — sub-Rat102/ses-p60
   - 355 volumes warped with 6 parallel threads, merged with fslmerge
   - FC matrix computed: 182 ROIs × 182 ROIs (Pearson r → Fisher z)
   - System memory: 2.7 GB used (26 GB free) during warping; fslmerge peaked at ~5 GB
   - Python process RSS: ~320 MB

**Commits this session:**
- `7bc5610` — Fix OOM in BOLD-to-SIGMA warping by splitting 4D into volume-by-volume
- `22f981d` — Use np.memmap for 4D output in volume-by-volume SIGMA warping
- `3836657` — Parallelize volume-by-volume SIGMA warping and use fslmerge
- `6901a5a` — Use output dir for temp files instead of /tmp
- `c51224c` — Use study work dir for warp temp files, not /tmp or derivatives

**Running in background (nohup) — check next session:**

| Analysis | PID | Log file | Status |
|----------|-----|----------|--------|
| FC batch (163 sessions, 1 worker) | 34759 | `fc_batch_overnight.log` | Running |

Monitor with: `tail -f fc_batch_overnight.log`

### Session Summary (2026-02-17)

**Completed this session:**

1. **Split monolithic resting-state script into standalone batch scripts**
   - Extracted `batch_resting_state_analysis.py` (fALFF + ReHo + FC in one script) into three focused scripts:
   - `scripts/batch_falff_analysis.py` — fALFF/ALFF computation + z-scoring + SIGMA warping
   - `scripts/batch_reho_analysis.py` — ReHo computation + z-scoring + SIGMA warping
   - `scripts/batch_fc_analysis.py` — ROI-to-ROI functional connectivity in SIGMA space
   - All three handle BOLD-to-template registration on-the-fly if transform is missing
   - Each saves its own metadata JSON (`desc-falff_analysis.json`, `desc-reho_analysis.json`, `desc-fc_analysis.json`)

2. **Common CLI interface across all three scripts**
   - `--study-root`, `--config`, `--n-workers`, `--force`, `--dry-run`, `--subjects`
   - `--skip-sigma` on fALFF and ReHo (FC requires SIGMA space, so no skip option)
   - Dry-run shows per-session data availability and SIGMA transform status

3. **Key differences between scripts**
   - fALFF: operates on unfiltered regressed BOLD (reconstructs from smooth + confounds if needed)
   - ReHo: operates on bandpass-filtered preproc_bold (simpler, no reconstruction needed)
   - FC: requires SIGMA-space BOLD, warps preproc_bold to SIGMA on-the-fly if missing

### Session Summary (2026-02-13b)

**Completed this session:**

1. **fMRI preprocessing — single subject test (sub-Rat49/ses-p90)**
   - Full pipeline run: 80×80×9×360 BOLD, TR=0.5s, run-14
   - All 13 steps completed including ICA denoising (31 components, 5 noise removed) and template registration

2. **Bug fix: tissue mask space mismatch** — `func_preprocess.py`
   - CSF/WM masks from anat preprocessing are in T2w space (256×256×41) but ICA classification and aCompCor operate in BOLD space (80×80×9)
   - Added `nibabel.processing.resample_from_to` to resample masks to BOLD space before use
   - Affects both ICA component CSF overlap calculation and aCompCor tissue timeseries extraction

3. **Bug fix: FD inflated by 10× voxel scaling** — `motion_qc.py`
   - MCFLIRT estimates motion in 10×-scaled voxel space, so FD and translation summaries were 10× too large
   - Added `voxel_scale=10.0` parameter to `calculate_framewise_displacement()`, divides FD by scale factor
   - Translation summaries also corrected
   - Changed default `fd_threshold` from 0.5mm (human) to 0.05mm (rodent real space) in `default.yaml`
   - Threshold now read from config (`functional.motion_qc.fd_threshold`) instead of hardcoded

4. **New: nuisance regression step** — `func_preprocess.py`
   - 24 motion regressors + 10 aCompCor regressors (34 total) regressed from BOLD via OLS in a single pass
   - Previously confounds were extracted but never applied
   - 38.8% variance reduction on test subject

5. **Pipeline reordering: regression before bandpass**
   - Old: smooth → bandpass → extract confounds (unused) → aCompCor (unused) → save
   - New: smooth → extract confounds → aCompCor → **regress (34 params)** → **bandpass** → save
   - Prevents bandpass from reintroducing frequencies removed by regression

6. **MSME batch reprocessing complete** — 189/189 sessions, 0 failures
   - All sessions reprocessed with registration to SIGMA space
   - Output: MWF, IWF, CSFF, T2 maps + SIGMA-space versions

7. **Committed and pushed** (`b949e19`) — "Fix func preprocessing: nuisance regression, FD scaling, mask resampling"

**Running in background (nohup) — check next session:**

| Analysis | PID | Log file | Status |
|----------|-----|----------|--------|
| DTI Regression (FA/MD/AD/RD, 1000 perms) | 432913 | `analysis/regression_run.log` | Running |
| MSME TBSS categorical (p60/p90/pooled, 5000 perms) | 433751 | `analysis/logs/tbss_msme_categorical.log` | Running |
| MSME Classification (MWF/T2/IWF/CSFF, 1000 perms) | 433774 | `analysis/logs/classification_msme.log` | Running |
| MSME Regression (MWF/T2/IWF/CSFF, 1000 perms) | 433807 | `analysis/logs/regression_msme.log` | Running |
| MSME TBSS per_pnd_p30 CSFF (5000 perms) | 429130 | (from earlier run) | randomise running |
| DTI TBSS pooled RD (5000 perms) | 394666 | (from earlier run) | randomise running |

### Session Summary (2026-02-13)

**Completed this session:**

1. **Config-driven preprocessing workflows** — Replaced ~35 hardcoded numeric parameters across 4 workflows with `get_config_value()` calls that read from YAML config
   - `anat_preprocess.py`: ~20 params (skull strip, N4, registration, intensity normalization)
   - `msme_preprocess.py`: ~8 params (skull strip method/classes, T2 fitting spectrum)
   - `dwi_preprocess.py`: ~4 params (eddy PE direction, readout time, repol, data_is_shelled)
   - `func_preprocess.py`: motion QC threshold
   - All changes preserve existing defaults — behavior unchanged unless config overridden

2. **Per-modality config validators** — `neurofaune/config_validator.py` (new file)
   - Ported from neurovrai's validation pattern
   - `AnatomicalConfigValidator`, `DWIConfigValidator`, `FunctionalConfigValidator`, `MSMEConfigValidator`
   - `validate_all_workflows(config)` returns per-workflow validation status
   - Integrated into `validate_config()` with `validate_workflows=True` flag

3. **Enhanced `configs/default.yaml`** — Added ~30 previously-hardcoded parameters
   - Skull strip: method, n_classes, atropos_iterations, convergence, MRF params, tissue threshold
   - N4 bias correction: iterations, shrink_factor, convergence_threshold
   - Registration: smoothing_sigmas, shrink_factors, iterations, SyN params, metric_bins
   - MSME T2 fitting: n_components, t2_range, lambda_reg, myelin_water_cutoff
   - Eddy: phase_encoding_direction, readout_time, data_is_shelled

4. **Enhanced `generate_config()` in study_initialization.py**
   - Loads package `default.yaml` as base, merges study-specific overrides
   - Auto-detects modalities from BIDS data
   - Writes comprehensive config with header comments to `{study_root}/config.yaml`

5. **Skull strip param passthrough** — `skull_strip.py` now accepts configurable Atropos params
   - `atropos_iterations`, `atropos_convergence`, `mrf_smoothing_factor`, `mrf_radius`
   - Merged with upstream improvements (improved class selection strategy, initial foreground mask)

6. **CLI enhancements** — `scripts/init_study.py`
   - `--validate-workflows` flag runs per-modality validation after config generation
   - `--modalities` flag to specify which modalities to include in generated config

7. **Updated `configs/bpa_rat_example.yaml`** with all new config keys as examples

### Session Summary (2026-02-11b)

**Completed this session:**

1. **Multivariate Classification module** — `neurofaune/analysis/classification/`
   - `data_prep.py`: Loads ROI CSVs (reuses covnet data loading), cohort filtering, bilateral/territory feature sets, StandardScaler
   - `omnibus.py`: PERMANOVA (custom implementation, ~50 lines, Euclidean distances + permutation) + optional MANOVA (statsmodels)
   - `pca.py`: Unsupervised PCA — scatter with 95% confidence ellipses, scree plot, feature loadings
   - `lda.py`: Supervised LDA — LD1 vs LD2 scatter, structure correlations, variance bar chart
   - `classifiers.py`: LOOCV with linear SVM + logistic regression, permutation p-values, confusion matrices
   - `visualization.py`: Shared plotting (scatter + ellipses, confusion heatmap, permutation null histogram, scree, loadings bar)

2. **Classification runner script** — `scripts/run_classification_analysis.py`
   - Loops: metric × cohort (None/p30/p60/p90) × feature_set (bilateral/territory)
   - Writes `classification_summary.json` + per-combo `summary.json`
   - Design description, analysis config saved
   - Registered with unified reporting system

3. **Reporting integration** — classification added to dashboard
   - `index_generator.py`: Added to `_TYPE_ORDER` and `_TYPE_LABELS`
   - `section_renderers.py`: `render_classification()` with stat cards (subjects, metrics, sig. PERMANOVA, best accuracy)
   - `discover.py`: `_discover_classification()` for backfill

4. **Regression module** — `neurofaune/analysis/classification/regression.py`
   - `run_regression()`: LOOCV with SVR (linear), Ridge, and PLS regression
   - Dose as ordinal (C=0, L=1, M=2, H=3) for dose-response testing
   - Reports R², MAE, Spearman ρ, permutation p-value per regressor
   - Predicted-vs-actual scatter plots with identity line, jitter, regression fit
   - `--skip-regression` flag in runner, Phase 6 in pipeline
   - Reporting dashboard updated with best R² stat card

5. **README updated** — CovNet, Classification, and Regression sections added to Group Analysis, architecture tree updated

### Session Summary (2026-02-11)

**Completed this session:**

1. **MSME TBSS pipeline** — Created `scripts/prepare_msme_tbss.py`
   - Discovers MSME SIGMA-space maps (MWF, IWF, CSFF, T2) from derivatives
   - Merges with study tracker: 183 subjects found, 181 matched (2 not in tracker)
   - Stacks into masked 4D volumes (128×128×218×181) using DTI-derived WM analysis mask (92,435 voxels)
   - Writes `tbss_config.json`, `subject_list.txt`, `subject_manifest.json`

2. **Provenance safety chain** — Prevents subject/design mismatches across modalities
   - `prepare_*_tbss.py` → writes `subject_list.txt` + SHA256 hash in `tbss_config.json`
   - `prepare_tbss_*designs.py` → writes `provenance.json` per design directory (records hash)
   - `run_tbss_analysis.py` → validates hash at runtime before launching randomise
   - Backwards-compatible: old designs without `provenance.json` produce a warning

3. **`run_tbss_analysis.py` generalized** — Removed `choices=DTI_METRICS` restriction from `--metrics` argparse, now accepts arbitrary metric names (e.g., MWF, IWF, CSFF, T2)

4. **Generated MSME TBSS designs** — 8 design directories (4 categorical + 4 dose-response), all with provenance, all full-rank

5. **DTI categorical TBSS results (completed)**

   | Analysis | FA | MD | AD | RD |
   |----------|-----|-----|-----|-----|
   | per_pnd_p30 (n=45) | ns | ns | ns | ns |
   | per_pnd_p60 (n=41) | ns | ns | ns | ns |
   | per_pnd_p90 (n=61) | ns | **C>H 1388 vox** | ns | **C>H 3484 vox** |
   | pooled (n=147) | ns (18 contrasts) | running | — | — |

   **Notable P90 finding**: High-dose BPA shows lower MD and RD vs control at P90 (TFCE-corrected p<0.05). Lower RD suggests increased myelination — aligns with the rationale for running MSME MWF analysis.

6. **MSME randomise running** — dose-response designs (4 analyses × 4 metrics = 16 randomise calls, 5000 permutations each)

### Session Summary (2026-02-05)

**Completed this session:**
1. **Registered all 63 3D T2w sessions to cohort templates**
   - 62 new ANTs SyN registrations (1 was already done)
   - Registration correlations: min 0.626, max 0.907, avg 0.720
   - 0 failures across all cohorts

2. **Created template manifest files**
   - Generated `template_manifest.json` for p30, p60, p90 cohorts
   - Enables `--phase 2 --skip-template-build` workflow

3. **Fixed config variable substitution**
   - `substitute_variables()` now iterates until stable (handles chained refs like A→B→C)
   - Fixes `${atlas.study_space.parcellation}` resolving correctly

4. **All 183 subjects now have complete T2w→Template→SIGMA registration**
   - 120 2D subjects (registered previously)
   - 63 3D subjects (registered this session)
   - Only 6 `ses-unknown` subjects skipped (no cohort template)

### Session Summary (2026-02-03)

**Completed this session:**
1. **3D T2w resampling integration** - Implemented automatic detection and resampling of 3D isotropic T2w to standard 2D multi-slice geometry
   - Added `_is_3d_acquisition()` detection (checks `acq-3D` filename tag + geometry heuristics)
   - Added `_resample_to_2d_geometry()` using ANTs identity transform with cohort template as reference
   - Inserted as Step 4b in `anat_preprocess.py` (after tissue segmentation, before intensity normalization)
   - All outputs resampled consistently: brain, mask, segmentation, GM/WM/CSF probabilities
   - 3D originals preserved in work directory

2. **Batch processed all 63 3D sessions** - 63/63 success, 0 failures
   - All outputs have correct geometry: 256x256x41 at 1.25x1.25x8mm
   - QC reports generated for all subjects

3. **`--exclude-3d` flag** added to all batch scripts
   - `batch_preprocess_anat.py`, `batch_preprocess_dwi.py`, `batch_preprocess_func.py`, `batch_preprocess_msme.py`
   - Subjects tagged with `is_3d_only` during discovery

4. **New utilities**
   - `is_3d_only_subject()` in `neurofaune/utils/select_anatomical.py`
   - `scripts/list_3d_subjects.py` - lists 3D-only subjects (text or JSON output)

5. **Other improvements**
   - QC directory structure simplified: `qc/{subject}/` instead of `qc/sub/{subject}/`
   - Skull strip morphological cleanup: largest connected component + opening instead of closing
   - Updated README with 3D resampling documentation

6. **Committed and pushed** (`cf8cc7f`) to GitHub

### Session Summary (2026-02-02)

**Completed previously:**
1. Tested 3D T2w preprocessing on `sub-Rat8/ses-p60`
   - Created `scripts/test_3d_t2w_preprocess.py` for single-subject testing
   - Preprocessing workflow successfully handles 3D isotropic data (140x256x110 at 2mm)
   - Geometry comparison: resampling 3D to 2D-like gives r=0.679 (vs r=0.842 for real 2D)

2. Geometry comparison 3D vs 2D T2w:
   | Property | 2D T2w (standard) | 3D T2w |
   |----------|-------------------|--------|
   | Shape | 256x256x41 | 140x256x110 |
   | Voxel (scaled) | 1.25x1.25x8.0mm | 2.0x2.0x2.0mm |
   | Type | Anisotropic | Isotropic |

### Session Summary (2026-01-30)

**Completed previously:**
1. Created unified skull stripping dispatcher (`neurofaune/preprocess/utils/skull_strip.py`)
   - Auto-selects method based on slice count: <10 slices -> adaptive, >=10 -> atropos_bet
   - Updated all 4 preprocessing workflows (anat, dwi, func, msme) to use unified interface
2. Identified and converted 3D isotropic T2w acquisitions
   - Found 31 subjects (63 sessions) with 3D RARE instead of 2D multi-slice T2w
   - Created `scripts/convert_3d_rare_to_bids.py` to convert and add to BIDS
   - **All 141 subjects now have T2w data** (637 2D + 63 3D files)
3. Updated README with skull stripping documentation and 3D RARE handling

---

## Next Session TODOs

### High Priority — Review Running Analyses

1. **Check CovNet DTI results** — `analysis/covnet_dti/`
   - FA/MD/AD/RD × 21 comparisons (9 dose-vs-control + 12 cross-timepoint)
   - NBS, territory Fisher z, graph metrics, whole-network (5000 perms each)
   - Log: `analysis/logs/covnet_dti_full.log`

2. **Check ReHo voxelwise results** — `analysis/voxelwise_fmri/randomise/`
   - 8 designs (4 categorical + 4 dose-response), 5000 perms, 3D TFCE
   - Log: `analysis/voxelwise_fmri/logs/orchestrator.log`

3. **Review DTI/MSME classification & regression results** — completed in earlier sessions, not yet reviewed
   - DTI classification: `analysis/classification/` (complete)
   - DTI regression: `analysis/regression/` (complete)
   - MSME classification: `analysis/classification_msme/` (complete)
   - MSME regression: `analysis/regression_msme/` (complete)

### Medium Priority

4. **Consolidate all results into cross-modality summary**
   - DTI TBSS + MSME TBSS + voxelwise fMRI + CovNet
   - Key pattern: no effects at p30, emerging at p60, strongest at p90
   - Effect size: fALFF >> MSME >> DTI

5. **Run BOLD preprocessing batch** — Only 1/294 BOLD scans fully preprocessed with new pipeline
   - New pipeline includes nuisance regression + corrected FD
   - Consider running batch for p90 cohort first (matches MSME/DTI findings)

6. **Fix MSME analysis_summary.json overwrite bug** — each summary only records last metric processed, not all 4

7. **Re-run CovNet graph metrics with optimized comparisons** — Graph metric permutation testing (`compare_metrics`) currently does all C(12,2)=66 pairwise comparisons instead of the 21 meaningful ones (9 dose-vs-control + 12 cross-timepoint). Each pair takes ~2.75h (5000 perms × 4 densities × networkx ops on 93-node graphs). Need to either pass explicit comparison list or optimize the inner loop. Skipped for now (`--skip-graph`) to unblock whole-network tests.

### Low Priority

7. **Fix 3 subjects with unscaled BOLD headers** (Rat209/ses-p60, Rat228/ses-p60, Rat41/ses-p30)
8. **Generate batch QC summary for 3D subjects**
9. **Refactor scripts into importable functions/classes** — Many `scripts/` are monolithic `main()` scripts with logic embedded in the argument parsing block. Audit all scripts and redesign into reusable functions (or classes where appropriate) that can be imported and composed, with thin CLI wrappers. CovNet scripts already follow this pattern after the prepare/test refactor.

---

## fMRI Preprocessing Pipeline (Updated 2026-02-13)

Pipeline order after fixes:

```
1.  Image validation
2.  Discard initial volumes (5 for T1 equilibration)
2.5 Slice timing correction (custom order, before motion correction)
3.  Brain extraction (adaptive per-slice BET, 9 slices)
4.  Motion correction (MCFLIRT, middle reference)
5.  ICA denoising (MELODIC → classify → remove noise components)
6.  Spatial smoothing (0.5mm FWHM)
7.  Extract 24 motion confound regressors
8.  aCompCor extraction (5 CSF + 5 WM components)
9.  Nuisance regression (34 regressors in single OLS pass)
10. Temporal bandpass filtering (0.01-0.1 Hz, AFTER regression)
11. Save outputs and metadata
12. QC reports (motion, confounds, skull strip, ICA, aCompCor)
13. BOLD → template registration (rigid + NCC Z-offset initialization)
```

Key fixes applied:
- Tissue masks (CSF/WM) resampled from T2w to BOLD space before ICA/aCompCor
- FD divided by 10× voxel scale factor (real mean FD ~0.065mm, not 0.645mm)
- FD threshold from config (default 0.05mm for rodents)
- Nuisance regression applied (was previously extract-only)
- Regression before bandpass (prevents frequency reintroduction)

---

## Template Building Status

| Cohort | T2w Template | SIGMA Registration | Correlation | Tissue Templates | Subjects Used |
|--------|--------------|-------------------|-------------|------------------|---------------|
| p30 | Complete | Complete | r=0.68 | Complete | 10 |
| p60 | Complete | Complete | r=0.72 | Complete | 10 |
| p90 | Complete | Complete | r=0.70 | Complete | 10 |

**Template locations:**
- `templates/anat/p30/tpl-BPARat_p30_T2w.nii.gz`
- `templates/anat/p60/tpl-BPARat_p60_T2w.nii.gz`
- `templates/anat/p90/tpl-BPARat_p90_T2w.nii.gz`

**Study-space SIGMA atlas:**
- `atlas/SIGMA_study_space/` - SIGMA reoriented to match study acquisition orientation

---

## Transform Chain Status

All modalities share a common registration chain through T2w:

```
Subject FA/BOLD/MSME -> Subject T2w -> Cohort Template -> SIGMA Atlas
       Affine/Rigid           SyN                  SyN
```

| Component | Status | Notes |
|-----------|--------|-------|
| T2w -> Template | Complete (183 subjects) | ANTs SyN, integrated in anat_preprocess.py |
| T2w -> Template (3D) | Complete (63 sessions) | r=0.63-0.91, avg 0.72 |
| Template -> SIGMA | Complete (3 cohorts) | ANTs SyN, via study-space atlas |
| FA -> T2w | Complete (118 subjects) | ANTs affine, integrated in dwi_preprocess.py |
| BOLD -> Template | 1 subject | ANTs rigid + NCC Z-init |
| MSME -> T2w | Complete (189 sessions) | ANTs rigid + NCC Z-init |

---

## Preprocessed Data Inventory

**Location:** `/mnt/arborea/bpa-rat/derivatives/`

| Modality | Available Raw | Preprocessed | Notes |
|----------|---------------|--------------|-------|
| Anatomical T2w (2D) | 637 files | 126 subjects | Standard multi-slice |
| Anatomical T2w (3D) | 63 sessions | 63 sessions | **Complete** - resampled to 2D geometry |
| T2w -> Template transforms | - | 183 subjects | **Complete** - all cohorts |
| DTI (FA, MD, AD, RD) | 181 sessions | 181 sessions | Complete |
| FA -> T2w transforms | - | 118 subjects | |
| Functional BOLD | 294 sessions | 166 sessions | Pipeline complete, resting-state scripts ready |
| BOLD -> Template transforms | - | 166 sessions | |
| BOLD -> SIGMA warping | - | ~62 sessions | Volume-by-volume warping implemented, ~104 remaining |
| MSME T2 mapping | 189 sessions | **189 sessions** | **Complete** - all reprocessed 2026-02-13 |

---

## Group Analysis Status

### DTI TBSS (`/analysis/tbss/`)
- **Subjects:** 148 (with complete DTI + transforms)
- **Metrics:** FA, MD, AD, RD
- **Analysis mask:** 92,435 WM voxels (FA≥0.3, tissue-informed, eroded)

| Design Type | Analyses | Status | Significant Results |
|-------------|----------|--------|---------------------|
| Categorical (per-PND) | p30(45), p60(41), p90(61) | **Complete** | P90: MD C>H (1388 vox), RD C>H (3484 vox) |
| Categorical (pooled) | pooled (147) | **Complete** | MD/AD/RD: L×P60 interaction (28k-52k vox) |
| Dose-response (per-PND) | p30, p60, p90 | **Complete** | P90 RD dose_neg (53 vox, marginal) |
| Dose-response (pooled) | pooled (147) | **Complete** | FA dose_pos (3333 vox) |

DTI effects are the weakest across modalities. Sparse findings concentrated at p90.

### MSME TBSS (`/analysis/tbss_msme/`)
- **Subjects:** 181
- **Metrics:** MWF, IWF, CSFF, T2
- **Analysis mask:** Same DTI-derived mask (92,435 voxels)

| Design Type | Analyses | Status | Significant Results |
|-------------|----------|--------|---------------------|
| Categorical (per-PND) | p30(54) | **Complete** | ns (all 4 metrics) |
| Categorical (per-PND) | p60(49) | **Complete** | T2/IWF/CSFF: H>C, L>C, M>C; MWF: H>C, M>C |
| Categorical (per-PND) | p90(78) | **Complete** | All 4 metrics: H>C; T2/IWF/CSFF: M>C |
| Categorical (pooled) | pooled (181) | **Complete** | All 4 metrics: H>C; T2/IWF/CSFF: M>C; CSFF L×P60 interaction |
| Dose-response (per-PND) | p30(54) | **Complete** | ns |
| Dose-response (per-PND) | p60(49) | **Complete** | All 4 metrics: dose_pos |
| Dose-response (per-PND) | p90(78) | **Complete** | All 4 metrics: dose_pos; MWF also dose_neg (non-monotonic) |
| Dose-response (pooled) | pooled (181) | **Complete** | All 4 metrics: dose_pos |

Strong dose-related effects at p60 and p90 across all MSME metrics. No effects at p30.

### Voxelwise fMRI (`/analysis/voxelwise_fmri/`)
- **Subjects:** 131 (with complete fALFF + ReHo in SIGMA space)
- **Metrics:** fALFF, ReHo
- **Analysis mask:** 751,262 whole-brain voxels (SIGMA brain mask)
- **TFCE:** 3D (`-T`), not 2D skeleton (`--T2`)

| Design Type | Metric | Status | Significant Results |
|-------------|--------|--------|---------------------|
| Categorical (per-PND) | fALFF | **Complete** | P60/P90/pooled: H>C, L>C, M>C (55k-321k vox). P30: ns |
| Dose-response (per-PND) | fALFF | **Complete** | All PNDs: dose_pos (12k-315k vox, increasing with age) |
| Dose-response (pooled) | fALFF | **Complete** | dose_pos (358k vox) |
| Categorical (per-PND) | ReHo | **Running** | — |
| Categorical (pooled) | ReHo | **Running** | — |
| Dose-response (per-PND) | ReHo | Queued | — |
| Dose-response (pooled) | ReHo | Queued | — |

fALFF shows the most widespread effects of any modality. Positive dose-response at all ages.

### CovNet (`/analysis/covnet_dti/`)
- **Subjects:** 148
- **Metrics:** FA, MD, AD, RD
- **ROIs:** 93 bilateral + 11 territory
- **Comparisons:** 9 dose-vs-control + 12 cross-timepoint = 21 per metric

| Phase | Status |
|-------|--------|
| Matrices + heatmaps | **Running** (FA in progress) |
| NBS (5000 perms) | **Running** (FA comparison 5/21) |
| Territory Fisher z | Queued |
| Graph metrics (5000 perms) | Queued |
| Whole-network (5000 perms) | Queued |

### DTI Classification (`/analysis/classification/`)
- **Complete:** FA, MD, AD, RD × 4 cohorts × 2 feature sets = 32 combos
- 1000 permutations, PERMANOVA + PCA + LDA + SVM + logistic LOOCV

### DTI Regression (`/analysis/regression/`)
- **Complete:** FA, MD, AD, RD × 4 cohorts × 2 feature sets
- SVR, Ridge, PLS with LOOCV + 1000 permutations

### MSME Classification (`/analysis/classification_msme/`)
- **Complete:** MWF, T2, IWF, CSFF × 4 cohorts × 2 feature sets = 32 combos
- Same pipeline as DTI classification

### MSME Regression (`/analysis/regression_msme/`)
- **Complete:** MWF, T2, IWF, CSFF × 4 cohorts × 2 feature sets
- Same pipeline as DTI regression

### Cross-Modality Summary

Consistent pattern across all modalities:
- **P30:** No significant dose effects in any modality
- **P60:** Strong effects in fALFF and MSME; absent in DTI
- **P90:** Effects in all modalities (fALFF > MSME > DTI)
- **Effect size hierarchy:** Voxelwise fMRI (fALFF) >> MSME TBSS >> DTI TBSS
- **Direction:** All dose groups show elevated values vs controls (not just high dose)

---

## 3D T2w Handling

31 subjects (63 sessions) have only 3D isotropic T2w (no 2D available). These are automatically detected and resampled during preprocessing.

**Detection:** `acq-3D` BIDS tag or geometry heuristics (isotropic voxels + >60 slices)

**Resampling:** ANTs identity transform with cohort template as reference grid
- Input: 140x256x110 at 2.0x2.0x2.0mm (isotropic)
- Output: 256x256x41 at 1.25x1.25x8.0mm (matches standard 2D)

**Exclusion:** All batch scripts support `--exclude-3d` flag

**Listing:** `uv run python scripts/list_3d_subjects.py /path/to/bids [--json] [--cohort p60]`

---

## Skull Stripping System

Unified dispatcher with automatic method selection based on image geometry:

| Method | Slice Threshold | Used For | Description |
|--------|-----------------|----------|-------------|
| `adaptive` | <10 slices | BOLD (9), MSME (5) | Per-slice BET with iterative frac optimization |
| `atropos_bet` | >=10 slices | T2w (41, 110), DTI (11) | Two-pass: Atropos segmentation + BET refinement |

**Configuration:** `neurofaune/preprocess/utils/skull_strip.py`
- `SLICE_THRESHOLD = 10`
- Adaptive: target ~15% brain extraction per slice
- Atropos+BET: returns posteriors for tissue segmentation

---

## Workflow Integration Status

| Workflow | Preprocessing | Cross-modal Registration | Skull Strip | 3D Support | Config-Driven | Integrated |
|----------|--------------|--------------------------|-------------|------------|---------------|------------|
| Anatomical (`anat_preprocess.py`) | Complete | T2w->Template (SyN) | atropos_bet | Yes (resample) | Yes (~20 params) | Yes |
| DTI (`dwi_preprocess.py`) | Complete | FA->T2w (Affine) | atropos_bet | --exclude-3d | Yes (~4 params) | Yes |
| Functional (`func_preprocess.py`) | Complete | BOLD->Template (Rigid) | adaptive | --exclude-3d | Yes (FD threshold) | Yes |
| MSME (`msme_preprocess.py`) | Complete | MSME->T2w (Rigid) | adaptive | --exclude-3d | Yes (~8 params) | Yes |

### Configuration System

- **Package defaults:** `configs/default.yaml` — ships with neurofaune, never edited per-study
- **Study config:** `{study_root}/config.yaml` — generated by `init_study.py`, study-specific overrides
- **Runtime merge:** `load_config()` merges defaults + study config, substitutes `${variable}` references
- **Validation:** `validate_all_workflows(config)` checks required/optional keys per modality
- **Per-modality validators:** `neurofaune/config_validator.py` (Anatomical, DWI, Functional, MSME)

---

## Batch QC System

| Modality | Subjects | Outliers | Notes |
|----------|----------|----------|-------|
| DWI | 181 | 40 (22%) | 26 subjects with bad slices |
| Anatomical (2D) | 126 | 7 (6%) | Skull stripping/segmentation metrics |
| Anatomical (3D) | 63 | TBD | Needs batch QC review |
| Functional | 1 | TBD | Pipeline updated, needs batch run |
| MSME | 189 | TBD | All reprocessed, needs QC review |

---

## Known Issues

1. **3 subjects have unscaled BOLD headers** (Rat209/ses-p60, Rat228/ses-p60, Rat41/ses-p30) — need header fix
2. **3D T2w registration quality** — resampled 3D correlates at r=0.679 with template (vs r=0.842 for real 2D). Cross-modal registration for these subjects may need verification.
3. **FC analysis incomplete** — 166 sessions preprocessed, ~62 warped to SIGMA, ~104 need SIGMA warping before FC can run. Volume-by-volume parallel warping implemented but not yet batch-tested.

---

## File Locations Reference

```
/mnt/arborea/bpa-rat/
├── raw/bids/                       # Input BIDS data (141 subjects, 100% T2w coverage)
├── derivatives/                    # Preprocessed outputs
├── templates/anat/{cohort}/        # Age-specific T2w templates
├── atlas/SIGMA_study_space/        # Study-space SIGMA atlas (reoriented)
├── transforms/{subject}/{session}/ # Subject transforms (FA->T2w, BOLD->T2w, T2w->Template)
├── analysis/                       # Group analysis outputs
│   ├── tbss/                       #   DTI voxel-wise TBSS
│   ├── tbss_msme/                  #   MSME voxel-wise TBSS
│   ├── roi/                        #   ROI extractions (DTI + MSME CSVs)
│   ├── classification/             #   DTI classification (complete)
│   ├── classification_msme/        #   MSME classification (running)
│   ├── regression/                 #   DTI regression (running)
│   ├── regression_msme/            #   MSME regression (running)
│   ├── voxelwise_fmri/            #   Whole-brain fALFF/ReHo (3D TFCE)
│   └── logs/                       #   Analysis log files
├── qc/                             # Quality control reports
└── work/                           # Temporary files

/mnt/arborea/atlases/SIGMA_scaled/  # Original SIGMA atlas (scaled 10x)
/mnt/arborea/bruker/                # Raw Bruker data (Cohort1-8)
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/batch_preprocess_anat.py` | Anatomical T2w preprocessing (supports `--exclude-3d`) |
| `scripts/batch_preprocess_dwi.py` | DTI preprocessing (supports `--exclude-3d`) |
| `scripts/batch_preprocess_func.py` | Functional BOLD preprocessing (supports `--exclude-3d`) |
| `scripts/batch_preprocess_msme.py` | MSME T2 mapping preprocessing (supports `--exclude-3d`) |
| `scripts/init_study.py` | Initialize study: directory structure, config generation, BIDS discovery |
| `scripts/convert_3d_rare_to_bids.py` | Convert 3D isotropic RARE to BIDS T2w |
| `scripts/list_3d_subjects.py` | List 3D-only subjects (text or JSON output) |
| `scripts/prepare_msme_tbss.py` | Prepare MSME metrics for TBSS (4D volumes + provenance) |
| `scripts/prepare_tbss_designs.py` | Create categorical design matrices with provenance |
| `scripts/prepare_tbss_dose_response_designs.py` | Create ordinal dose-response designs with provenance |
| `scripts/run_tbss_analysis.py` | Run FSL randomise with 2D TFCE for TBSS (DTI + MSME) |
| `scripts/prepare_fmri_voxelwise.py` | Prepare fALFF/ReHo for whole-brain voxelwise analysis |
| `scripts/run_voxelwise_fmri_analysis.py` | Run FSL randomise with 3D TFCE for whole-brain fMRI |
| `scripts/covnet_prepare.py` | CovNet data preparation: load, bilateral average, matrices, heatmaps |
| `scripts/covnet_nbs.py` | CovNet NBS: Network-Based Statistic with permutation testing |
| `scripts/covnet_territory.py` | CovNet territory: Fisher z-tests with FDR on territory-level edges |
| `scripts/covnet_graph_metrics.py` | CovNet graph metrics: efficiency, clustering, modularity + permutations |
| `scripts/covnet_whole_network.py` | CovNet whole-network: Mantel, Frobenius, spectral divergence tests |
| `scripts/run_covnet_analysis.py` | CovNet full pipeline wrapper (calls prepare + all test scripts) |
| `scripts/run_classification_analysis.py` | ROI-level multivariate classification (PERMANOVA, PCA, LDA, SVM) |
| `scripts/run_regression_analysis.py` | ROI-level dose-response regression (SVR, Ridge, PLS) |
| `scripts/batch_falff_analysis.py` | Batch fALFF/ALFF computation + SIGMA warping |
| `scripts/batch_reho_analysis.py` | Batch ReHo computation + SIGMA warping |
| `scripts/batch_fc_analysis.py` | Batch ROI-to-ROI functional connectivity in SIGMA space |
| `scripts/test_3d_t2w_preprocess.py` | Test 3D T2w preprocessing on single subject |
| `scripts/visualize_msme_skull_strip.py` | MSME skull stripping QC visualization |
