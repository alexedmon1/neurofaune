# Project Status

**Last Updated:** 2026-01-15

This file tracks the current state of the neurofaune project. Update this file after important milestones or before ending a session.

---

## Current Phase: 8 - Template-Based Registration

### Template Building Status

| Cohort | T2w Template | SIGMA Registration | Tissue Templates | Subjects Used |
|--------|--------------|-------------------|------------------|---------------|
| p30 | ✅ Complete | ⚠️ Needs slice-wise | ❌ Not started | 10 |
| p60 | ✅ Complete | ⚠️ Needs slice-wise | ❌ Not started | 10 |
| p90 | ✅ Complete | ⚠️ Needs slice-wise | ❌ Not started | 10 |

> **⚠️ SIGMA Registration Issue (2026-01-15):** 3D non-linear registration fails due to
> anisotropy mismatch (8mm coronal slices vs 1mm isotropic). See "Planned Work" section
> for the slice-wise registration approach.

**Template locations:**
- `templates/anat/p30/tpl-BPARat_p30_T2w.nii.gz`
- `templates/anat/p60/tpl-BPARat_p60_T2w.nii.gz`
- `templates/anat/p90/tpl-BPARat_p90_T2w.nii.gz`

**Transform locations:**
- `templates/anat/{cohort}/transforms/tpl-to-SIGMA_0GenericAffine.mat`
- `templates/anat/{cohort}/transforms/tpl-to-SIGMA_1Warp.nii.gz`
- `templates/anat/{cohort}/transforms/tpl-to-SIGMA_1InverseWarp.nii.gz`

> **Note:** SIGMA registration was re-done on 2026-01-15 to use the STANDARD SIGMA template
> (128×218×128 @ 1.5mm) instead of the coronal template (128×128×218 @ 1.0mm).
> This is critical because the atlas labels are defined in the standard orientation.

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

1. **Implement 2D slice-wise template-to-SIGMA registration** (HIGH PRIORITY)
2. Generate tissue probability templates - Average GM/WM/CSF maps for each cohort
3. Integrate registration into anatomical workflow - Add subject→template→SIGMA registration
4. Visual QC of templates - Verify template quality and SIGMA alignment

---

## Planned Work: 2D Slice-Wise Registration

### Problem
The cohort templates (p30, p60, p90) were built from thick coronal slices (8mm) resulting in
highly anisotropic volumes (256×256×41 @ 1.25×1.25×8mm). SIGMA atlas is isotropic (128×128×218
@ 1mm). Standard 3D non-linear registration (ANTs SyN) produces anatomically incorrect results
because:
- The 8mm slice thickness cannot capture the fine detail present in 1mm isotropic data
- SyN warping distorts the brain geometry trying to match boundaries
- High Dice scores are misleading - boundaries overlap but internal anatomy is misaligned

### Proposed Solution: 2D Slice-Wise Registration

For each of the 41 coronal slices in the template:

1. **Extract corresponding SIGMA slice**
   - Compute the A-P coordinate of each template slice
   - Extract/average the corresponding SIGMA coronal slice(s) covering that 8mm region

2. **2D affine registration per slice**
   - Register each template coronal slice to its SIGMA counterpart
   - Use rigid or affine (no non-linear) to preserve brain geometry
   - This respects the native resolution in each plane

3. **Build slice-wise transform mapping**
   - Store per-slice transforms
   - For atlas label propagation: apply per-slice inverse transforms to bring SIGMA labels
     into template space

4. **Interpolate labels in 3D**
   - After per-slice label assignment, apply 3D smoothing/interpolation
   - This provides consistent labeling between slices

### Alternative Approach: Resample Template to SIGMA Grid

If slice-wise registration proves too complex:
1. Resample template to SIGMA's grid (will have poor A-P resolution but correct geometry)
2. Do 3D affine-only registration (no SyN)
3. Accept lower precision in the A-P direction

### Implementation Location
- New module: `neurofaune/templates/slice_registration.py`
- Development script: `scripts/dev_registration/009_slice_wise_sigma_registration.py`

---

## Recent Changes

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

1. **Template-to-SIGMA registration broken** - 3D non-linear registration produces anatomically
   incorrect results due to 8mm vs 1mm anisotropy. Requires slice-wise approach (see Planned Work).
2. **Slice timing correction disabled** in functional workflow due to acquisition artifacts
3. **Tissue probability templates** not yet generated for any cohort
4. **Registration not integrated** into preprocessing workflows

---

## File Locations Reference

```
/mnt/arborea/bpa-rat/
├── raw/bids/                    # Input BIDS data (141 subjects)
├── derivatives/                 # Preprocessed outputs
├── templates/anat/{cohort}/     # Age-specific templates
├── transforms/                  # Subject transform registry (empty until integration)
├── qc/                          # Quality control reports
└── work/                        # Temporary files

/mnt/arborea/atlases/SIGMA_scaled/  # SIGMA atlas (scaled 10x)
```
