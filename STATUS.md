# Project Status

**Last Updated:** 2026-02-05

---

## Current Phase: 10 - Multi-modal Registration & Preprocessing

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
1. **~~Register 3D T2w subjects to cohort templates~~** COMPLETE
   - 62 registrations completed, correlations 0.63-0.91 (avg 0.72)

2. **Complete MSME batch preprocessing**
   - Currently ~24/189 complete
   - Check results and verify registration quality
   - Run remaining subjects if needed

3. **Run BOLD preprocessing batch**
   - Only 33/294 BOLD scans preprocessed
   - Large backlog needs processing

### Medium Priority
4. **Fix unscaled BOLD headers** for 3 subjects
   - Rat209/ses-p60, Rat228/ses-p60, Rat41/ses-p30

5. **Generate batch QC summary for 3D subjects**
   - Verify skull stripping quality across all 63 sessions
   - Check for systematic issues with 3D -> 2D resampling

### Low Priority
6. **Update exclusion lists based on QC**
7. **Assess cross-modal registration for 3D subjects** (DTI/BOLD/MSME -> T2w)

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
| MSME T2 mapping | 189 sessions | ~24 subjects | Batch in progress |

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

| Workflow | Preprocessing | Cross-modal Registration | Skull Strip | 3D Support | Integrated |
|----------|--------------|--------------------------|-------------|------------|------------|
| Anatomical (`anat_preprocess.py`) | Complete | T2w->Template (SyN) | atropos_bet | Yes (resample) | Yes |
| DTI (`dwi_preprocess.py`) | Complete | FA->T2w (Affine) | atropos_bet | --exclude-3d | Yes |
| Functional (`func_preprocess.py`) | Complete | BOLD->T2w (Rigid) | adaptive | --exclude-3d | Yes |
| MSME (`msme_preprocess.py`) | Complete | MSME->T2w (Rigid) | adaptive | --exclude-3d | Yes |

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
| `scripts/convert_3d_rare_to_bids.py` | Convert 3D isotropic RARE to BIDS T2w |
| `scripts/list_3d_subjects.py` | List 3D-only subjects (text or JSON output) |
| `scripts/test_3d_t2w_preprocess.py` | Test 3D T2w preprocessing on single subject |
| `scripts/visualize_msme_skull_strip.py` | MSME skull stripping QC visualization |
