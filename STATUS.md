# Project Status

**Last Updated:** 2026-02-13

---

## Current Phase: 13 - Config-Driven Preprocessing

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
   - SIGMA atlas propagated to all subjects

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

### High Priority
1. **Review MSME TBSS randomise results** (running in background)
   - 16 randomise jobs (4 dose-response analyses × 4 metrics)
   - Check significance, generate cluster reports
   - Cross-reference with DTI P90 MD/RD findings

2. **Run MSME categorical TBSS** (per_pnd + pooled designs already generated)
   - `run_tbss_analysis.py --analyses per_pnd_p30 per_pnd_p60 per_pnd_p90 pooled`

3. **Review DTI dose-response and pooled results** (still running)
   - DTI dose-response: P30+P60+P90+pooled × FA/MD/AD/RD
   - DTI categorical pooled: MD still running, AD+RD queued

### Medium Priority
4. **Run BOLD preprocessing batch**
   - Only 33/294 BOLD scans preprocessed
   - Large backlog needs processing

5. **Fix unscaled BOLD headers** for 3 subjects
   - Rat209/ses-p60, Rat228/ses-p60, Rat41/ses-p30

6. **Test config-driven workflow end-to-end**
   - Run single-subject anat preprocessing with custom config overrides
   - Verify parameter changes (e.g., skull strip n_classes=3) propagate correctly

### Low Priority
7. **Generate batch QC summary for 3D subjects**
8. **Update exclusion lists based on QC**

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
| BOLD -> T2w | In progress (33 subjects) | ANTs rigid + NCC Z-init, integrated in func_preprocess.py |
| MSME -> T2w | In progress (~24/189) | ANTs rigid + NCC Z-init, adaptive slice-wise BET |

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
| Functional BOLD | 294 sessions | 33 sessions | **Backlog** |
| BOLD -> T2w transforms | - | 33 subjects | |
| MSME T2 mapping | 189 sessions | 183 subjects | **Complete** (SIGMA-space maps for all) |

---

## TBSS Analysis Status

### DTI TBSS (`/analysis/tbss/`)
- **Subjects:** 148 (with complete DTI + transforms)
- **Metrics:** FA, MD, AD, RD
- **Analysis mask:** 92,435 WM voxels (FA≥0.3, tissue-informed, eroded)

| Design Type | Analyses | Status | Significant Results |
|-------------|----------|--------|---------------------|
| Categorical (per-PND) | p30(45), p60(41), p90(61) | **Complete** | P90: MD C>H (1388 vox), RD C>H (3484 vox) |
| Categorical (pooled) | pooled (147) | Running | FA: ns (all 18 contrasts) |
| Dose-response (per-PND) | p30(45), p60(41), p90(61) | Running | FA: ns for p30, p60 |
| Dose-response (pooled) | pooled (147) | Queued | — |

### MSME TBSS (`/analysis/tbss_msme/`)
- **Subjects:** 181 (more than DTI due to broader MSME coverage)
- **Metrics:** MWF, IWF, CSFF, T2
- **Analysis mask:** Same DTI-derived mask (92,435 voxels)

| Design Type | Analyses | Status |
|-------------|----------|--------|
| Categorical (per-PND) | p30(54), p60(49), p90(78) | Designs ready |
| Categorical (pooled) | pooled (181) | Designs ready |
| Dose-response (per-PND) | p30(54), p60(49), p90(78) | **Running** (5000 perms) |
| Dose-response (pooled) | pooled (181) | Running |

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
| Functional (`func_preprocess.py`) | Complete | BOLD->T2w (Rigid) | adaptive | --exclude-3d | Yes (1 param) | Yes |
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
| Functional | 33 | TBD | Needs QC generation |
| MSME | ~24 | TBD | Needs QC generation |

---

## Known Issues

1. **Slice timing correction disabled** in functional workflow due to acquisition artifacts
2. **3 subjects have unscaled BOLD headers** (Rat209/ses-p60, Rat228/ses-p60, Rat41/ses-p30) -- need header fix
3. **3D T2w registration quality** -- resampled 3D correlates at r=0.679 with template (vs r=0.842 for real 2D). Cross-modal registration for these subjects may need verification.

---

## File Locations Reference

```
/mnt/arborea/bpa-rat/
├── raw/bids/                       # Input BIDS data (141 subjects, 100% T2w coverage)
├── derivatives/                    # Preprocessed outputs
├── templates/anat/{cohort}/        # Age-specific T2w templates
├── atlas/SIGMA_study_space/        # Study-space SIGMA atlas (reoriented)
├── transforms/{subject}/{session}/ # Subject transforms (FA->T2w, BOLD->T2w, T2w->Template)
├── analysis/                       # Group analysis outputs (TBSS, connectivity)
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
| `scripts/run_tbss_analysis.py` | Run FSL randomise with provenance validation (DTI + MSME) |
| `scripts/test_3d_t2w_preprocess.py` | Test 3D T2w preprocessing on single subject |
| `scripts/visualize_msme_skull_strip.py` | MSME skull stripping QC visualization |
