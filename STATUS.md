# Project Status

**Last Updated:** 2026-02-02

---

## Current Phase: 10 - Multi-modal Registration & Preprocessing

### Session Summary (2026-02-02)

**Completed this session:**
1. Tested 3D T2w preprocessing on `sub-Rat8/ses-p60`
   - Created `scripts/test_3d_t2w_preprocess.py` for single-subject testing
   - Preprocessing workflow successfully handles 3D isotropic data (140×256×110 at 2mm)
   - Auto-selected Atropos+BET method (110 slices ≥ 10 threshold)
   - Adaptive BET frac: 0.390, extraction ratio: 8.2%
   - QC flagged "potential_over_stripping" - may need visual inspection
   - All outputs generated: preprocessed T2w, brain mask, tissue segmentation

2. Geometry comparison 3D vs 2D T2w:
   | Property | 2D T2w (standard) | 3D T2w |
   |----------|-------------------|--------|
   | Shape | 256×256×41 | 140×256×110 |
   | Voxel (scaled) | 1.25×1.25×8.0mm | 2.0×2.0×2.0mm |
   | Type | Anisotropic | Isotropic |

### Session Summary (2026-01-30)

**Completed previously:**
1. Created unified skull stripping dispatcher (`neurofaune/preprocess/utils/skull_strip.py`)
   - Auto-selects method based on slice count: <10 slices → adaptive, ≥10 → atropos_bet
   - Updated all 4 preprocessing workflows (anat, dwi, func, msme) to use unified interface
2. Identified and converted 3D isotropic T2w acquisitions
   - Found 31 subjects (63 sessions) with 3D RARE instead of 2D multi-slice T2w
   - Created `scripts/convert_3d_rare_to_bids.py` to convert and add to BIDS
   - **All 141 subjects now have T2w data** (637 2D + 63 3D files)
3. Updated README with skull stripping documentation and 3D RARE handling

---

## Next Session TODOs

### High Priority
1. **Run full batch preprocessing on 3D T2w subjects**
   - 31 subjects (63 sessions) with 3D T2w need preprocessing
   - Test subject (sub-Rat8/ses-p60) completed successfully
   - Review QC for potential over-stripping before full batch

2. **Complete MSME batch preprocessing**
   - Currently 24/189 complete (batch was running)
   - Check results and verify registration quality
   - Run remaining subjects if needed

3. **Run BOLD preprocessing batch**
   - Only 33/294 BOLD scans preprocessed
   - Large backlog needs processing

### Medium Priority
4. **Verify 3D T2w → Template registration quality**
   - 3D T2w has different FOV than 2D
   - May need registration parameter tuning for 3D subjects

5. **Fix unscaled BOLD headers** for 3 subjects
   - Rat209/ses-p60, Rat228/ses-p60, Rat41/ses-p30

### Low Priority
6. **Generate QC for newly preprocessed subjects**
7. **Update exclusion lists based on QC**

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
Subject FA/BOLD/MSME → Subject T2w → Cohort Template → SIGMA Atlas
       Affine/Rigid           SyN                  SyN
```

| Component | Status | Notes |
|-----------|--------|-------|
| T2w → Template | Complete (121 subjects) | ANTs SyN, integrated in anat_preprocess.py |
| Template → SIGMA | Complete (3 cohorts) | ANTs SyN, via study-space atlas |
| FA → T2w | Complete (118 subjects) | ANTs affine, integrated in dwi_preprocess.py |
| BOLD → T2w | In progress (33 subjects) | ANTs rigid + NCC Z-init, integrated in func_preprocess.py |
| MSME → T2w | In progress (~24/189) | ANTs rigid + NCC Z-init, adaptive slice-wise BET |

---

## Preprocessed Data Inventory

**Location:** `/mnt/arborea/bpa-rat/derivatives/`

| Modality | Available Raw | Preprocessed | Notes |
|----------|---------------|--------------|-------|
| Anatomical T2w (2D) | 637 files | 126 subjects | Standard multi-slice |
| Anatomical T2w (3D) | 63 sessions | 1 session | Test run complete (sub-Rat8/ses-p60) |
| T2w → Template transforms | - | 121 subjects | |
| DTI (FA, MD, AD, RD) | 181 sessions | 181 sessions | Complete |
| FA → T2w transforms | - | 118 subjects | |
| Functional BOLD | 294 sessions | 33 sessions | **Backlog** |
| BOLD → T2w transforms | - | 33 subjects | |
| MSME T2 mapping | 189 sessions | ~24 subjects | Batch in progress |

---

## Skull Stripping System

Unified dispatcher with automatic method selection based on image geometry:

| Method | Slice Threshold | Used For | Description |
|--------|-----------------|----------|-------------|
| `adaptive` | <10 slices | BOLD (9), MSME (5) | Per-slice BET with iterative frac optimization |
| `atropos_bet` | ≥10 slices | T2w (41), DTI (11) | Two-pass: Atropos segmentation + BET refinement |

**Configuration:** `neurofaune/preprocess/utils/skull_strip.py`
- `SLICE_THRESHOLD = 10`
- Adaptive: target ~15% brain extraction per slice
- Atropos+BET: returns posteriors for tissue segmentation

---

## Workflow Integration Status

| Workflow | Preprocessing | Cross-modal Registration | Skull Strip | Integrated |
|----------|--------------|--------------------------|-------------|------------|
| Anatomical (`anat_preprocess.py`) | Complete | T2w→Template (SyN) | atropos_bet | Yes |
| DTI (`dwi_preprocess.py`) | Complete | FA→T2w (Affine) | atropos_bet | Yes |
| Functional (`func_preprocess.py`) | Complete | BOLD→T2w (Rigid) | adaptive | Yes |
| MSME (`msme_preprocess.py`) | Complete | MSME→T2w (Rigid) | adaptive | Yes |

---

## Batch QC System

| Modality | Subjects | Outliers | Notes |
|----------|----------|----------|-------|
| DWI | 181 | 40 (22%) | 26 subjects with bad slices |
| Anatomical | 126 | 7 (6%) | Skull stripping/segmentation metrics |
| Functional | 33 | TBD | Needs QC generation |
| MSME | ~24 | TBD | Needs QC generation |

---

## Known Issues

1. **Slice timing correction disabled** in functional workflow due to acquisition artifacts
2. **3 subjects have unscaled BOLD headers** (Rat209/ses-p60, Rat228/ses-p60, Rat41/ses-p30) — need header fix
3. **3D T2w preprocessing tested** — QC flagged potential over-stripping, visual review recommended before full batch

---

## File Locations Reference

```
/mnt/arborea/bpa-rat/
├── raw/bids/                       # Input BIDS data (141 subjects, 100% T2w coverage)
├── derivatives/                    # Preprocessed outputs
├── templates/anat/{cohort}/        # Age-specific T2w templates
├── atlas/SIGMA_study_space/        # Study-space SIGMA atlas (reoriented)
├── transforms/{subject}/{session}/ # Subject transforms (FA→T2w, BOLD→T2w, T2w→Template)
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
| `scripts/batch_preprocess_anat.py` | Anatomical T2w preprocessing |
| `scripts/batch_preprocess_dwi.py` | DTI preprocessing |
| `scripts/batch_preprocess_func.py` | Functional BOLD preprocessing |
| `scripts/batch_preprocess_msme.py` | MSME T2 mapping preprocessing |
| `scripts/convert_3d_rare_to_bids.py` | Convert 3D isotropic RARE to BIDS T2w |
| `scripts/test_3d_t2w_preprocess.py` | Test 3D T2w preprocessing on single subject |
| `scripts/visualize_msme_skull_strip.py` | MSME skull stripping QC visualization |
