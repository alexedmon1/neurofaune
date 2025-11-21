# Multi-Modal Template Architecture

## Design Decision: Independent Spaces Approach

**Date**: 2025-11-21
**Decision**: Use independent template spaces for T2w and DTI modalities
**Rationale**: Different spatial coverage (whole brain vs. partial) and analysis requirements

---

## Overview

Neurofaune implements a **dual-template approach** for multi-modal rodent MRI preprocessing:

1. **T2w Template** (whole brain): Anatomical reference, registered to SIGMA atlas
2. **FA Template** (partial coverage): DTI analysis in native space

This architecture optimally supports both:
- **Voxel-wise analyses** (TBSS-like): All subjects → FA template space
- **ROI-based analyses**: SIGMA parcellation propagated to subject FA space

---

## Spatial Coverage

### T2w Anatomical
- **Coverage**: Whole brain
- **Purpose**: Anatomical reference, SIGMA parcellation
- **Template registration**: T2w template → SIGMA (automatic)

### DTI/FA
- **Coverage**: Partial (11 slices: hippocampus → frontal cortex)
- **Purpose**: DTI analysis in native resolution
- **Template registration**: FA template stays in FA space (independent)

### Challenge
T2w and FA templates have **different geometries** and **different coverage**:
- T2w: Full brain (e.g., 128×128×128)
- FA: Partial (e.g., 160×160×11)

**Solution**: Keep templates in independent spaces, use within-subject registration for label propagation.

---

## Architecture

### Age-Specific Templates

**Three cohorts** (developmental study):
```
p30/ (30 days postnatal)
├── tpl-BPARatp30_T2w.nii.gz          # T2w template (whole brain)
├── tpl-BPARatp30_FA.nii.gz           # FA template (11 slices)
├── tpl-BPARatp30_space-SIGMA_T2w.nii.gz  # T2w in SIGMA space
└── transforms/
    ├── tpl-to-SIGMA_composite.h5     # T2w template → SIGMA
    └── SIGMA-to-tpl_composite.h5     # SIGMA → T2w template (inverse)

p60/ (60 days postnatal)
├── tpl-BPARatp60_T2w.nii.gz
├── tpl-BPARatp60_FA.nii.gz
├── tpl-BPARatp60_space-SIGMA_T2w.nii.gz
└── transforms/ ...

p90/ (90 days postnatal)
├── tpl-BPARatp90_T2w.nii.gz
├── tpl-BPARatp90_FA.nii.gz
├── tpl-BPARatp90_space-SIGMA_T2w.nii.gz
└── transforms/ ...
```

---

## Registration Chains

### Subject-Level Registration

**Anatomical (T2w)**:
```
subject_T2w → T2w_template → SIGMA
```
- Provides anatomical normalization
- Enables SIGMA parcellation access

**DTI (FA)**:
```
subject_FA → FA_template
```
- Native FA space analysis
- No SIGMA registration (different coverage)

**Within-Subject**:
```
subject_T2w ↔ subject_FA
```
- Links anatomical and DTI spaces
- Enables label propagation

---

## Analysis Workflows

### 1. Voxel-Wise DTI Analysis (TBSS-like)

**Setup**:
```
All subjects' FA → FA_template
```

**Analysis**:
- Voxel-wise statistics in FA template space
- Group comparisons (p30 vs p60 vs p90)
- Tract-based spatial statistics

**ROI Definition** (optional):
```
SIGMA → median_subject_T2w → median_subject_FA → FA_template
```
- Use a representative subject to propagate SIGMA labels to FA template
- Define ROIs for mean extraction

---

### 2. ROI-Based DTI Analysis

**For each subject**:
```
SIGMA_labels
  → T2w_template⁻¹ (inverse transform)
  → subject_T2w
  → subject_FA (within-subject registration)
  → Extract ROI statistics
```

**Implementation**:
```python
from neurofaune.utils.labels import propagate_sigma_to_fa

# Propagate SIGMA parcellation to subject FA space
fa_labels = propagate_sigma_to_fa(
    subject_id='sub-Rat207',
    cohort='p60',
    sigma_atlas_labels=sigma_labels,
    template_dir=template_dir,
    derivatives_dir=derivatives_dir
)

# Extract ROI statistics
roi_stats = extract_roi_statistics(
    fa_file=subject_fa,
    labels_file=fa_labels,
    rois=['hippocampus', 'corpus_callosum']
)
```

---

## Transform Composition

### Anatomical Chain (T2w → SIGMA)
```
subject_T2w → T2w_template → SIGMA
```
**Stored transforms**:
- `{subject}/T2w_to_template_Composite.h5`
- `{template}/tpl-to-SIGMA_Composite.h5`

**Composed**:
```bash
antsApplyTransforms \
  -d 3 \
  -i subject_T2w.nii.gz \
  -r SIGMA_template.nii.gz \
  -o subject_T2w_in_SIGMA.nii.gz \
  -t tpl-to-SIGMA_Composite.h5 \
  -t T2w_to_template_Composite.h5
```

### Label Propagation (SIGMA → subject FA)
```
SIGMA → T2w_template → subject_T2w → subject_FA
```
**Stored transforms**:
- `{template}/SIGMA-to-tpl_Composite.h5` (inverse of above)
- `{subject}/template_to_T2w_Composite.h5` (inverse)
- `{subject}/T2w_to_FA_Composite.h5` (within-subject)

**Composed**:
```bash
antsApplyTransforms \
  -d 3 \
  -i SIGMA_labels.nii.gz \
  -r subject_FA.nii.gz \
  -o SIGMA_labels_in_FA.nii.gz \
  -n NearestNeighbor \  # Labels need NN interpolation
  -t T2w_to_FA_Composite.h5 \
  -t template_to_T2w_Composite.h5 \
  -t SIGMA-to-tpl_Composite.h5
```

---

## Template Building

### Quality Filtering

Use **top 25% of subjects** per cohort based on QC metrics:
- T2w: SNR, motion artifacts, skull stripping quality
- FA: SNR, eddy QC, registration quality

**BPA-Rat cohort sizes**:
- p30: 54 subjects → use ~14 best
- p60: 50 subjects → use ~13 best
- p90: 79 subjects → use ~20 best

### ANTs Template Construction

**T2w Template** (whole brain):
```bash
antsMultivariateTemplateConstruction.sh \
  -d 3 \
  -o tpl-BPARatp60_ \
  -n 0 \  # Use all CPUs
  -i 4 \  # 4 iterations
  -g 0.2 \  # Gradient step
  -c 2 \  # Use parallel processing
  -j 8 \  # 8 cores
  subject1_T2w.nii.gz \
  subject2_T2w.nii.gz \
  ...
```

**FA Template** (11 slices):
```bash
antsMultivariateTemplateConstruction.sh \
  -d 3 \
  -o tpl-BPARatp60_FA_ \
  -n 0 \
  -i 4 \
  -g 0.2 \
  -c 2 \
  -j 8 \
  subject1_FA.nii.gz \
  subject2_FA.nii.gz \
  ...
```

### Automatic SIGMA Registration

After T2w template creation:
```bash
antsRegistrationSyN.sh \
  -d 3 \
  -f SIGMA_InVivo_Brain.nii.gz \  # SIGMA as fixed
  -m tpl-BPARatp60_T2w.nii.gz \   # Template as moving
  -o tpl-to-SIGMA_ \
  -n 8
```

**Outputs**:
- `tpl-to-SIGMA_Composite.h5` (template → SIGMA)
- `tpl-to-SIGMA_InverseComposite.h5` (SIGMA → template)
- `tpl-to-SIGMA_Warped.nii.gz` (template in SIGMA space)

---

## Directory Structure

```
{study_root}/
├── derivatives/
│   └── sub-{subject}/
│       ├── {session}/
│       │   ├── anat/
│       │   │   ├── sub-{subject}_{session}_desc-preproc_T2w.nii.gz
│       │   │   ├── sub-{subject}_{session}_space-template_T2w.nii.gz
│       │   │   └── sub-{subject}_{session}_space-SIGMA_T2w.nii.gz
│       │   └── dwi/
│       │       ├── sub-{subject}_{session}_FA.nii.gz
│       │       ├── sub-{subject}_{session}_space-template_FA.nii.gz
│       │       └── sub-{subject}_{session}_space-SIGMA_labels.nii.gz
│       └── transforms/
│           ├── T2w_to_template_Composite.h5
│           ├── template_to_T2w_Composite.h5  # Inverse
│           ├── T2w_to_FA_Composite.h5        # Within-subject
│           └── FA_to_T2w_Composite.h5        # Inverse
├── templates/
│   ├── p30/
│   │   ├── tpl-BPARatp30_T2w.nii.gz
│   │   ├── tpl-BPARatp30_FA.nii.gz
│   │   ├── tpl-BPARatp30_space-SIGMA_T2w.nii.gz
│   │   └── transforms/
│   │       ├── tpl-to-SIGMA_Composite.h5
│   │       └── SIGMA-to-tpl_Composite.h5
│   ├── p60/ ...
│   └── p90/ ...
└── qc/
    ├── templates/
    │   ├── p30_template_qc.html
    │   ├── p60_template_qc.html
    │   └── p90_template_qc.html
    └── subjects/ ...
```

---

## Advantages of This Approach

### 1. **Preserves Native Resolution**
- DTI analysis in native FA resolution (no unnecessary resampling)
- T2w provides anatomical reference at its native resolution

### 2. **Handles Different Coverage**
- FA template: 11 slices (hippocampus → frontal cortex)
- T2w template: Whole brain
- No forced geometric matching required

### 3. **Flexible Analysis**
- **Voxel-wise**: FA template space (TBSS-like)
- **ROI-based**: SIGMA labels in subject FA space
- **Both** approaches supported

### 4. **Age-Specific Templates**
- Developmental changes properly modeled
- p30, p60, p90 cohorts have appropriate reference anatomy
- Cannot do cross-age comparisons (by design - developmentally different)

### 5. **Modular**
- Each modality independently preprocessed
- Templates built separately
- SIGMA integration only where needed (T2w)

---

## Limitations and Considerations

### 1. **No Cross-Age DTI Comparisons**
- FA templates are age-specific
- Cannot directly compare p30 vs p60 in FA template space
- Solution: Use ROI-based analyses with SIGMA parcellation (common across ages)

### 2. **Within-Subject Registration Required**
- T2w ↔ FA registration needed for label propagation
- Adds computational cost
- Quality depends on image contrast between modalities

### 3. **Label Propagation Chain**
- Multiple transforms: SIGMA → template → T2w → FA
- Each step introduces interpolation error
- Use NearestNeighbor for labels to minimize errors

### 4. **Template Quality Depends on Input**
- Need sufficient subjects (≥25% of cohort)
- Quality filtering critical
- May need iteration if initial template poor

---

## Implementation Notes

### Template Building Workflow
1. **Preprocess all subjects** (skull strip, bias correct, normalize)
2. **QC filtering** (select top 25% per cohort)
3. **Build T2w template** (ANTs multivariate construction)
4. **Auto-register T2w template → SIGMA**
5. **Build FA template** (independent, stays in FA space)
6. **Generate QC reports** for templates

### Subject Registration Workflow
1. **T2w → T2w template** (age-matched)
2. **FA → FA template** (age-matched)
3. **T2w ↔ FA** (within-subject, for label propagation)
4. **Compose transforms** as needed for analysis

### Label Propagation Workflow
1. **Load SIGMA parcellation**
2. **Apply inverse transforms**: SIGMA → template → subject_T2w → subject_FA
3. **Save labels in subject FA space**
4. **Extract ROI statistics**

---

## References

This approach is based on:
- **TBSS** (Tract-Based Spatial Statistics): Smith et al., 2006
- **ANTs template construction**: Avants et al., 2011
- **Multi-modal rodent imaging**: Barrière et al., 2019 (SIGMA atlas)
- **Developmental template approach**: Calabrese et al., 2013

---

## Summary

**Key Principle**: Keep modalities in their native spaces when coverage differs. Use within-subject registration and transform composition for cross-modal integration.

**Pipeline**:
```
Preprocessing → Template Building → Template-to-SIGMA → Subject Registration → Analysis
     ↓                  ↓                    ↓                     ↓                ↓
  Individual      Age-specific         Automatic           T2w, FA,        Voxel-wise
   subjects         T2w, FA             T2w only         within-subject      or ROI
```

This architecture provides maximum flexibility while preserving data quality and supporting both voxel-wise and ROI-based analyses for developmental rodent DTI studies.
