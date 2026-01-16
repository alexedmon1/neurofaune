# Project Status

**Last Updated:** 2026-01-16

This file tracks the current state of the neurofaune project. Update this file after important milestones or before ending a session.

---

## Current Phase: 8 - Template-Based Registration

### Template Building Status

| Cohort | T2w Template | SIGMA Registration | Correlation | Tissue Templates | Subjects Used |
|--------|--------------|-------------------|-------------|------------------|---------------|
| p30 | ✅ Complete | ✅ Complete | r=0.68 | ❌ Not started | 10 |
| p60 | ✅ Complete | ✅ Complete | r=0.72 | ❌ Not started | 10 |
| p90 | ✅ Complete | ✅ Complete | r=0.70 | ❌ Not started | 10 |

> **✅ SIGMA Registration Fixed (2026-01-16):** The registration issue was resolved by creating
> a **study-space SIGMA atlas** - reorienting SIGMA to match the study's native acquisition
> orientation. This avoids the need for slice-wise registration and allows standard 3D SyN
> registration to work correctly.

**Template locations:**
- `templates/anat/p30/tpl-BPARat_p30_T2w.nii.gz`
- `templates/anat/p60/tpl-BPARat_p60_T2w.nii.gz`
- `templates/anat/p90/tpl-BPARat_p90_T2w.nii.gz`

**Transform locations:**
- `templates/anat/{cohort}/transforms/tpl-to-SIGMA_0GenericAffine.mat`
- `templates/anat/{cohort}/transforms/tpl-to-SIGMA_1Warp.nii.gz`
- `templates/anat/{cohort}/transforms/tpl-to-SIGMA_1InverseWarp.nii.gz`
- `templates/anat/{cohort}/transforms/registration_mosaic_{cohort}.png` (QC visualization)

**Study-space SIGMA atlas location:**
- `atlas/SIGMA_study_space/` - SIGMA reoriented to match study acquisition orientation
- Contains: template, mask, tissue priors (GM/WM/CSF), and 183-region parcellation

### DTI Registration Pipeline Status

**Status:** ✅ Pipeline validated and integrated

The DTI-to-atlas registration chain is now working:
```
Subject FA (128×128×11) → Subject T2w → Cohort Template → SIGMA Atlas
```

| Component | Status | Script |
|-----------|--------|--------|
| DTI preprocessing | ✅ Working | `dwi_preprocess.py` (bvec NaN fix applied) |
| FA → T2w registration | ✅ Working | `003_register_dwi_to_t2w.py` |
| T2w → Template registration | ✅ Working | `007_register_subject_to_template.py` |
| Template → SIGMA registration | ✅ Fixed | `008_register_template_to_sigma.py` |
| Atlas propagation to FA | ✅ Working | `006_propagate_atlas_to_dwi.py` |

**Test result (sub-Rat1/ses-p60):**
- 96 unique atlas labels in FA space
- 65% coverage of FA brain voxels
- Per-slice coverage: 35-110%

**Development scripts:** `scripts/dev_registration/`

### Workflow Integration Status

| Workflow | Preprocessing | Registration to Template | Registration to SIGMA |
|----------|--------------|-------------------------|----------------------|
| Anatomical (`anat_preprocess.py`) | ✅ Complete | ❌ Not integrated | ❌ Not integrated |
| Functional (`func_preprocess.py`) | ✅ Complete | ❌ Not integrated | ❌ Not integrated |
| DTI (`dwi_preprocess.py`) | ✅ Complete | ✅ Validated | ✅ Atlas propagation added |
| MSME (`msme_preprocess.py`) | ✅ Complete | ❌ Not integrated | ❌ Not integrated |

### Registration Utilities Status

| Module | Status | Notes |
|--------|--------|-------|
| `templates/builder.py` | ✅ Complete | Template construction, SIGMA registration |
| `templates/registration.py` | ✅ Complete | Subject-to-template, apply_transforms, label propagation, DTI atlas propagation |
| Integration into workflows | 🔄 In progress | DTI done, anat/func pending |

---

## Preprocessed Data Inventory

**Location:** `/mnt/arborea/bpa-rat/derivatives/`

| Modality | p30 | p60 | p90 | Total |
|----------|-----|-----|-----|-------|
| Anatomical T2w | 38 | 34 | 47 | 119 |
| Functional BOLD | ~98 | ~98 | ~98 | 294 |
| DTI | TBD | TBD | TBD | TBD |
| MSME | TBD | TBD | TBD | TBD |

---

## Immediate Next Steps

1. ~~**Implement 2D slice-wise template-to-SIGMA registration**~~ ✅ RESOLVED via study-space atlas
2. **Generate tissue probability templates** - Average GM/WM/CSF maps for each cohort
3. **Integrate registration into anatomical workflow** - Add subject→template→SIGMA registration
4. **Integrate registration into functional workflow** - Add func→anat→template→SIGMA chain
5. **Batch process DTI data** - Run full pipeline on all subjects

---

## ~~Planned Work: 2D Slice-Wise Registration~~ (RESOLVED)

> **This section is historical.** The slice-wise approach was not needed after implementing
> the study-space atlas solution.

### Original Problem (Now Resolved)
The cohort templates (p30, p60, p90) were built from thick coronal slices (8mm) resulting in
highly anisotropic volumes (256×256×41 @ 1.25×1.25×8mm). SIGMA atlas is isotropic (128×128×218
@ 1mm). Standard 3D non-linear registration initially failed due to orientation mismatch.

### Solution Implemented: Study-Space SIGMA Atlas (2026-01-16)

Instead of slice-wise registration, we reoriented the SIGMA atlas to match the study's native
acquisition orientation:

1. **Created `setup_study_atlas()` function** (`neurofaune/templates/slice_registration.py`)
   - Reorients SIGMA: transpose(0,2,1) + flip(axis=0) + flip(axis=1)
   - Saves to `{study_root}/atlas/SIGMA_study_space/`
   - Updates config file with atlas paths

2. **Standard 3D SyN registration now works**
   - Templates and SIGMA are now in the same orientation
   - Correlations: p30=0.68, p60=0.72, p90=0.70
   - QC mosaics generated for visual verification

### Implementation Location
- `neurofaune/templates/slice_registration.py` - `setup_study_atlas()`, `reorient_sigma_to_study()`
- `scripts/dev_registration/008_register_template_to_sigma.py` - Updated to use study-space atlas

---

## Recent Changes

### 2026-01-16 - Study-Space SIGMA Atlas & Template Registration

**SIGMA Registration Issue Resolved!**

- Created **study-space SIGMA atlas** by reorienting SIGMA to match study acquisition orientation
- Added `setup_study_atlas()` function to automate atlas setup and config updates
- Updated `008_register_template_to_sigma.py` to use study-space atlas
- Successfully registered all three templates (p30, p60, p90) to SIGMA with SyN
- Registration quality: p30=0.68, p60=0.72, p90=0.70 correlation
- Generated QC mosaic visualizations for each cohort

**Key insight:** The original problem was orientation mismatch, not anisotropy. Reorienting
SIGMA once (instead of reorienting every image) is simpler and avoids resampling artifacts.

**Files created:**
- `atlas/SIGMA_study_space/` - Reoriented SIGMA atlas files
- `templates/anat/{cohort}/transforms/registration_mosaic_{cohort}.png` - QC visualizations

### 2026-01-15 (Session 2) - Template-SIGMA Registration Investigation

**Problem Identified:** The p60 template-to-SIGMA registration produces high Dice scores (0.94+) but
incorrect anatomical correspondence. The SyN non-linear warping is distorting the brain to match
boundaries while misaligning anterior-posterior anatomy.

**Root Cause:** The p60 template was acquired in **coronal orientation** (256×256×41 @ 1.25×1.25×8mm)
while SIGMA is **isotropic** (128×128×218 @ 1mm). The 8mm thick slices in the A-P direction cannot
be meaningfully warped to match 1mm isotropic data without severe distortion.

**Approaches Tried (All Failed):**
1. **Y-flip before registration** - Fixed apparent orientation but SyN still warped anterior↔posterior
2. **XY-flip** - Same issue, Dice high (0.941) but anatomy misaligned
3. **YZ-flip** - Similar results
4. **ZXY axis permutation** - Improved Dice to 0.957 but introduced angular distortion
5. **Center-of-mass initialization** - Helped initial alignment but didn't fix the fundamental issue

**Key Insight:** 3D non-linear registration between thick-slice coronal (8mm) and isotropic (1mm)
data is fundamentally flawed. The registration minimizes intensity differences but cannot preserve
anatomical correspondence due to the extreme anisotropy mismatch.

**Created:** `neurofaune/utils/orientation.py` - Orientation verification/correction utilities
(useful for future work, but doesn't solve the thick-slice problem)

**Files generated in `/mnt/arborea/bpa-rat/templates/anat/p60/`:**
- Various test templates: `*_yflipped.nii.gz`, `*_xyflipped.nii.gz`, `*_ZXY.nii.gz`
- Diagnostic figures in `transforms/` directory

### 2026-01-15 (Session 1)
- **Fixed SIGMA orientation issue**: Re-registered all templates (p30, p60, p90) to STANDARD SIGMA
  (128×218×128) instead of coronal (128×128×218) - atlas labels are in standard orientation
- **DTI registration pipeline validated**: Full chain FA→T2w→Template→SIGMA working
- Added `propagate_atlas_to_dwi()` to `templates/registration.py`
- Created development scripts `scripts/dev_registration/001-008`
- Fixed bvec NaN issue in `dwi_preprocess.py` (eddy rotation produces NaN for b0 volumes)

### 2026-01-08
- Verified all T2w templates complete (p30, p60, p90)
- Verified all SIGMA registrations complete
- Created STATUS.md for project tracking
- Streamlined CLAUDE.md

### 2024-12-15 (from ROADMAP.md)
- p30 T2w template built with SIGMA registration
- p60, p90 T2w templates completed
- Fixed builder.py to handle ANTs separate transform files

---

## Known Issues

1. ~~**Template-to-SIGMA registration broken**~~ ✅ RESOLVED (2026-01-16) - Fixed via study-space atlas
2. **Slice timing correction disabled** in functional workflow due to acquisition artifacts
3. **Tissue probability templates** not yet generated for any cohort
4. **Registration not integrated** into preprocessing workflows (anat, func)

---

## File Locations Reference

```
/mnt/arborea/bpa-rat/
├── raw/bids/                    # Input BIDS data (141 subjects)
├── derivatives/                 # Preprocessed outputs
├── templates/anat/{cohort}/     # Age-specific templates
├── atlas/SIGMA_study_space/     # Study-space SIGMA atlas (reoriented)
├── transforms/                  # Subject transform registry (empty until integration)
├── qc/                          # Quality control reports
└── work/                        # Temporary files

/mnt/arborea/atlases/SIGMA_scaled/  # Original SIGMA atlas (scaled 10x, standard orientation)
```
