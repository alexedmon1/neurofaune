# Project Status

**Last Updated:** 2026-01-16 (Session 2)

This file tracks the current state of the neurofaune project. Update this file after important milestones or before ending a session.

---

## Current Phase: 8 - Template-Based Registration

### Template Building Status

| Cohort | T2w Template | SIGMA Registration | Correlation | Tissue Templates | Subjects Used |
|--------|--------------|-------------------|-------------|------------------|---------------|
| p30 | ✅ Complete | ✅ Complete | r=0.68 | ✅ Complete | 10 |
| p60 | ✅ Complete | ✅ Complete | r=0.72 | ✅ Complete | 10 |
| p90 | ✅ Complete | ✅ Complete | r=0.70 | ✅ Complete | 10 |

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
| `registration/slice_correspondence.py` | ✅ Complete | Dual-approach slice matching (intensity + landmarks) for partial-coverage modalities |
| `registration/qc_visualization.py` | ✅ Complete | QC figure generation (checkerboard, edge overlay, slice correspondence) |
| Integration into workflows | 🔄 In progress | DTI done, anat/func pending |

### Slice Correspondence System (NEW)

**Status:** ✅ Complete with unit tests (21 tests passing)

For registering partial-coverage modalities (DWI: 11 slices, fMRI: 9 slices) to full T2w (41 slices),
a dual-approach system combines:

1. **Intensity-based matching**: Correlates 2D slices using normalized cross-correlation + gradient matching
2. **Landmark detection**: Identifies ventricles to validate/refine alignment

**Key classes:**
- `SliceCorrespondenceFinder` - Main interface combining both approaches
- `IntensityMatcher` - Handles intensity-based slice correlation
- `LandmarkDetector` - Detects anatomical landmarks (ventricles)
- `SliceCorrespondenceResult` - Detailed result with confidence metrics

**Usage:**
```python
from neurofaune.registration import find_slice_correspondence

result = find_slice_correspondence(
    partial_image='sub-001_dwi_b0.nii.gz',
    full_image='sub-001_T2w.nii.gz',
    modality='dwi'
)
print(f"DWI slices 0-10 -> T2w slices {result.start_slice}-{result.end_slice}")
print(f"Confidence: {result.combined_confidence:.2f}")
```

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
2. ~~**Generate tissue probability templates**~~ ✅ Complete (GM/WM/CSF for all cohorts)
3. ~~**Slice correspondence for partial-coverage modalities**~~ ✅ Complete (intensity + landmark detection)
4. **Integrate registration into anatomical workflow** - Add subject→template→SIGMA registration
5. **Integrate registration into functional workflow** - Add func→anat→template→SIGMA chain (use slice correspondence)
6. **Batch process DTI data** - Run full pipeline on all subjects

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

### 2026-01-16 - Slice Correspondence System for Partial-Coverage Modalities

Implemented robust dual-approach slice correspondence system to handle registration of
partial-coverage modalities (DWI, fMRI) to full T2w anatomical images.

**Challenge solved:** All modalities have affine origins at [0,0,0] with no header
information about slice positioning. Need to determine which T2w slices correspond
to the partial-coverage volume.

**Solution:** Combined approach for robustness:
1. **Intensity matching** - Correlates slices using normalized cross-correlation with gradient enhancement
   - Uses absolute correlation to handle contrast-inverted modalities (FA vs T2w)
   - Uses physical coordinate-based matching to handle different slice thicknesses
2. **Landmark detection** - Identifies ventricle peaks to validate/refine the intensity-based match
3. **Confidence scoring** - Weighted combination; when methods disagree, uses higher-confidence method

**Physical coordinate support:**
- Extracts slice thickness from NIfTI headers automatically
- Computes physical positions (mm) for each slice center
- Matches based on physical position, not slice index (handles different thicknesses)
- Reports physical offset from full volume start

**Test results (sub-Rat1 DWI → T2w):**
- FA slices 0-10 mapped to T2w slices 12-22
- Landmark confidence: 0.374, Intensity confidence: 0.203
- Ventricle peak detected at partial slice 1 → T2w slice 13
- Physical offset: 96mm from T2w start (12 slices × 8mm)
- Coverage: 88mm partial / 328mm full

**Files created:**
- `neurofaune/registration/slice_correspondence.py` - Core implementation
- `neurofaune/registration/qc_visualization.py` - QC figure generation
- `neurofaune/registration/__init__.py` - Module exports
- `tests/unit/test_slice_correspondence.py` - 27 unit tests (all passing)
- `scripts/dev_registration/009_test_slice_correspondence.py` - Real data testing

**QC figures generated:**
- `qc/{subject}/{session}/registration/*_slice_correspondence.png` - Summary view
- `qc/{subject}/{session}/registration/*_slice_detail.png` - Slice-by-slice detail

### 2026-01-16 - Tissue Probability Templates

- Built GM, WM, CSF probability templates for all cohorts (p30, p60, p90)
- Used existing subject→template transforms to warp tissue maps
- Created `scripts/build_tissue_templates.py` for automated generation
- QC visualizations generated for each cohort

**Files created:**
- `templates/anat/{cohort}/tpl-BPARat_{cohort}_label-{GM,WM,CSF}_probseg.nii.gz`
- `templates/anat/{cohort}/tissue_templates_qc_{cohort}.png`

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
3. ~~**Tissue probability templates**~~ ✅ COMPLETE (2026-01-16) - GM/WM/CSF for all cohorts
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
