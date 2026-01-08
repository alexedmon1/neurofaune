# Project Status

**Last Updated:** 2026-01-08

This file tracks the current state of the neurofaune project. Update this file after important milestones or before ending a session.

---

## Current Phase: 8 - Template-Based Registration

### Template Building Status

| Cohort | T2w Template | SIGMA Registration | Tissue Templates | Subjects Used |
|--------|--------------|-------------------|------------------|---------------|
| p30 | ✅ Complete | ✅ Complete | ❌ Not started | 10 |
| p60 | ✅ Complete | ✅ Complete | ❌ Not started | 10 |
| p90 | ✅ Complete | ✅ Complete | ❌ Not started | 10 |

**Template locations:**
- `templates/anat/p30/tpl-BPARat_p30_T2w.nii.gz`
- `templates/anat/p60/tpl-BPARat_p60_T2w.nii.gz`
- `templates/anat/p90/tpl-BPARat_p90_T2w.nii.gz`

**Transform locations:**
- `templates/anat/{cohort}/transforms/tpl-to-SIGMA_0GenericAffine.mat`
- `templates/anat/{cohort}/transforms/tpl-to-SIGMA_1Warp.nii.gz`
- `templates/anat/{cohort}/transforms/tpl-to-SIGMA_1InverseWarp.nii.gz`

### Workflow Integration Status

| Workflow | Preprocessing | Registration to Template | Registration to SIGMA |
|----------|--------------|-------------------------|----------------------|
| Anatomical (`anat_preprocess.py`) | ✅ Complete | ❌ Not integrated | ❌ Not integrated |
| Functional (`func_preprocess.py`) | ✅ Complete | ❌ Not integrated | ❌ Not integrated |
| DTI (`dwi_preprocess.py`) | ✅ Complete | ❌ Not integrated | ❌ Not integrated |
| MSME (`msme_preprocess.py`) | ✅ Complete | ❌ Not integrated | ❌ Not integrated |

### Registration Utilities Status

| Module | Status | Notes |
|--------|--------|-------|
| `templates/builder.py` | ✅ Complete | Template construction, SIGMA registration |
| `templates/registration.py` | ✅ Complete | Subject-to-template, apply_transforms, label propagation |
| Integration into workflows | ❌ Not started | Next priority |

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

1. **Generate tissue probability templates** - Average GM/WM/CSF maps for each cohort
2. **Integrate registration into anatomical workflow** - Add subject→template→SIGMA registration
3. **Visual QC of templates** - Verify template quality and SIGMA alignment

---

## Recent Changes

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

1. **Slice timing correction disabled** in functional workflow due to acquisition artifacts
2. **Tissue probability templates** not yet generated for any cohort
3. **Registration not integrated** into preprocessing workflows

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
