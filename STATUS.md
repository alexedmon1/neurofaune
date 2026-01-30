# Project Status

**Last Updated:** 2026-01-30

---

## Current Phase: 10 - MSME Registration

### Template Building Status

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

### Transform Chain Status

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
| BOLD → T2w | Complete (6 subjects) | ANTs rigid + NCC Z-init, integrated in func_preprocess.py |
| MSME → T2w | In progress (16/189 complete) | ANTs rigid + NCC Z-init, adaptive slice-wise BET |

### DTI Analysis Pipeline

**Status:** Complete

- WM-masked VBA (not skeleton-based TBSS — rodent WM tracts too thin)
- Analysis mask: interior WM + FA >= 0.3
- Warps subject FA/MD/AD/RD to SIGMA space for group analysis
- FSL randomise with TFCE, cluster labeling via SIGMA atlas
- Validated on P90 cohort (47 subjects, 92,798 analysis voxels)

### Functional (BOLD) Normalization

**Status:** Complete (registration) — 6 subjects tested

BOLD-to-T2w registration uses:
- NCC-based Z initialization (partial-coverage BOLD has no header positioning info)
- Rigid-only registration (6 DOF — affine over-fits on 9-slice data)
- Conservative shrink factors (4x2x1x1 — avoids losing Z info at coarse resolution)
- Integrated into `func_preprocess.py` Step 12

BOLD-to-SIGMA warping:
- `scripts/batch_warp_bold_to_sigma.py` chains BOLD→T2w→Template→SIGMA
- Uses `antsApplyTransforms -e 3` for 4D timeseries
- Produces `space-SIGMA_bold.nii.gz` for group analysis

**3 subjects skipped** (Rat209, Rat228, Rat41) due to unscaled BOLD headers (1.0mm voxels instead of 10x-scaled).

### Preprocessed Data Inventory

**Location:** `/mnt/arborea/bpa-rat/derivatives/`

| Modality | p30 | p60 | p90 | Total |
|----------|-----|-----|-----|-------|
| Anatomical T2w | 38 | 34 | 48 | 126 |
| T2w → Template transforms | 38 | 35 | 48 | 121 |
| DTI (FA, MD, AD, RD) | - | - | - | 181 sessions |
| FA → T2w transforms | - | - | - | 118 subjects |
| Functional BOLD | ~98 | ~98 | ~98 | 294 |
| BOLD → T2w transforms | - | - | - | 6 subjects |
| MSME T2 mapping | - | - | - | 16 subjects (batch in progress) |

---

## Workflow Integration Status

| Workflow | Preprocessing | Cross-modal Registration | Integrated |
|----------|--------------|--------------------------|------------|
| Anatomical (`anat_preprocess.py`) | Complete | T2w→Template (SyN) | Yes |
| DTI (`dwi_preprocess.py`) | Complete | FA→T2w (Affine) | Yes (Step 7) |
| Functional (`func_preprocess.py`) | Complete | BOLD→T2w (Rigid) | Yes (Step 12) |
| MSME (`msme_preprocess.py`) | Complete | MSME→T2w (Rigid) | Yes (Step 4), skull stripping WIP |

---

## Batch QC System

| Modality | Subjects | Outliers | Notes |
|----------|----------|----------|-------|
| DWI | 181 | 40 (22%) | 26 subjects with bad slices |
| Anatomical | 126 | 7 (6%) | Skull stripping/segmentation metrics |
| Functional | 2 | 0 | Only 2 subjects fully preprocessed |

---

### MSME Preprocessing

**Status:** Batch processing in progress — 16/189 subjects complete (0 failures)

MSME-to-T2w registration uses:
- First echo extraction (highest SNR) as registration reference
- MSME data layout: (X, Y, echoes, slices) — echoes in dim 2, spatial slices in dim 3
- NCC-based Z initialization (same as BOLD — origins at 0,0,0)
- Rigid-only registration (6 DOF — only 5 slices, affine over-fits)
- Conservative shrink factors (2x1x1 — even fewer slices than BOLD)
- Integrated into `msme_preprocess.py` Step 4

**Skull stripping solution:** Adaptive slice-wise BET
- Per-slice BET with iterative frac optimization targeting ~15% brain extraction
- COG offset config (`cog_offset_x=0, cog_offset_y=-40`) for brain positioned lower in coronal FOV
- Consistent 14-16% extraction across p30, p60, p90 cohorts
- NCC 0.64-0.74 for MSME-to-T2w registration

**Batch processing:**
- Script: `scripts/batch_preprocess_msme.py`
- 189 MSME scans total, 126 with T2w available for registration
- Parallel processing with 4 workers
- Outputs: T2 maps, MWF maps, MSME→T2w transforms

**QC visualization:** `scripts/visualize_msme_skull_strip.py`
- Brain mask outline overlay on first echo
- Registration result with MSME edges on T2w background

---

## Known Issues

1. **Slice timing correction disabled** in functional workflow due to acquisition artifacts
2. **3 subjects have unscaled BOLD headers** (Rat209/ses-p60, Rat228/ses-p60, Rat41/ses-p30) — need header fix

---

## File Locations Reference

```
/mnt/arborea/bpa-rat/
├── raw/bids/                       # Input BIDS data (141 subjects)
├── derivatives/                    # Preprocessed outputs
├── templates/anat/{cohort}/        # Age-specific T2w templates
├── atlas/SIGMA_study_space/        # Study-space SIGMA atlas (reoriented)
├── transforms/{subject}/{session}/ # Subject transforms (FA→T2w, BOLD→T2w, T2w→Template)
├── analysis/                       # Group analysis outputs (TBSS, connectivity)
├── qc/                             # Quality control reports
└── work/                           # Temporary files

/mnt/arborea/atlases/SIGMA_scaled/  # Original SIGMA atlas (scaled 10x)
```
