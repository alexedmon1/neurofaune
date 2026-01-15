# DTI Registration Development Scripts

This directory contains development scripts for implementing slice-aware DTI registration in neurofaune.

## Problem Statement

DTI acquisitions in rodent MRI often cover only a portion of the brain (e.g., 11 hippocampal slices with 8mm thickness), while T2w anatomical images cover the full brain (41 slices). Direct registration of DTI to the SIGMA atlas fails due to this geometry mismatch.

## Solution: Multi-Stage Transform Chain

```
Subject FA (11 slices)
    ↓ [Within-subject affine]
Subject T2w (matching 11 slices extracted)
    ↓ [Subject-to-template SyN]
T2w Template (age-specific)
    ↓ [Template-to-SIGMA SyN, already computed]
SIGMA Atlas
```

For atlas propagation (labels to FA space), inverse transforms are applied.

## Key Findings from Geometry Exploration

### File: 001_explore_geometry.py, 002_detailed_geometry.py

1. **T2w geometry**: 256×256×41 voxels at 1.25×1.25×8.0mm (scaled 10x)
2. **DWI geometry**: 128×128×11 voxels at ~2.0×3.5×8.0mm (when headers are correct)
3. **Slice alignment**: DWI covers T2w slices 0-10 (same z-origin, same slice thickness)
4. **Header issues**: ~15% of DWI files have identity affine (incorrect headers from Bruker conversion)

### Voxel Size Issue

Some DWI files have incorrect 1×1×1mm voxel sizes in headers. Use `neurofaune/utils/fix_bruker_voxel_sizes.py` to correct these before registration.

## Scripts

### 001_explore_geometry.py
Quick survey of T2w/DWI geometry across multiple subjects.
```bash
uv run python 001_explore_geometry.py /mnt/arborea/bpa-rat --max-subjects 10
```

### 002_detailed_geometry.py
Detailed examination of a single subject's T2w/DWI relationship.
```bash
uv run python 002_detailed_geometry.py /mnt/arborea/bpa-rat sub-Rat1 ses-p60
```

### 003_register_dwi_to_t2w.py
Register FA/b0 to corresponding T2w slices (within-subject).
```bash
uv run python 003_register_dwi_to_t2w.py /mnt/arborea/bpa-rat sub-Rat1 ses-p60
```
**Requires**: Preprocessed FA map and T2w

### 004_batch_preprocess_dwi.py
Batch DTI preprocessing to generate FA maps.
```bash
uv run python 004_batch_preprocess_dwi.py /mnt/arborea/bpa-rat --max-subjects 10 --dry-run
uv run python 004_batch_preprocess_dwi.py /mnt/arborea/bpa-rat --cohort p60
```

### 005_build_fa_templates.py
Build age-specific FA templates from preprocessed DTI.
```bash
uv run python 005_build_fa_templates.py /mnt/arborea/bpa-rat --cohort p60 --dry-run
uv run python 005_build_fa_templates.py /mnt/arborea/bpa-rat --cohort all
```

### 006_propagate_atlas_to_dwi.py
Propagate SIGMA atlas labels to FA space through the transform chain.
```bash
uv run python 006_propagate_atlas_to_dwi.py /mnt/arborea/bpa-rat sub-Rat1 ses-p60
```
**Requires**: All transforms in the chain

### 007_register_subject_to_template.py
Register subject T2w to age-matched cohort template.
```bash
uv run python 007_register_subject_to_template.py /mnt/arborea/bpa-rat sub-Rat1 ses-p60
```
**Requires**: Preprocessed T2w, cohort template

### 008_register_template_to_sigma.py
Register cohort template to STANDARD SIGMA atlas space.
```bash
uv run python 008_register_template_to_sigma.py /mnt/arborea/bpa-rat p60 --n-cores 8
```
**Important**: Must use STANDARD SIGMA (128×218×128), not coronal (128×128×218).

## Workflow Order

1. **Fix headers** (if needed): `fix_bruker_voxel_sizes.py`
2. **DTI preprocessing**: `004_batch_preprocess_dwi.py` → FA maps
3. **Within-subject registration**: `003_register_dwi_to_t2w.py` → FA→T2w transforms
4. **Subject-to-template**: `007_register_subject_to_template.py` → T2w→Template transforms
5. **Template-to-SIGMA**: `008_register_template_to_sigma.py` → Template→SIGMA transforms (per cohort)
6. **FA templates** (optional): `005_build_fa_templates.py` → cohort FA templates
7. **Atlas propagation**: `006_propagate_atlas_to_dwi.py` → labels in FA space

## Critical Issue Fixed: SIGMA Orientation

The SIGMA atlas comes in two orientations:
- **Standard**: 128×218×128 @ 1.5mm - **Use this one** (atlas labels are in this space)
- **Coronal**: 128×128×218 @ 1.0mm - Do NOT use (different coordinate system)

The original template-to-SIGMA registration used the coronal template, causing atlas propagation
to fail completely. Script 008 fixes this by registering to the standard SIGMA template.

## Integration Plan

After validation, these scripts will be integrated into the neurofaune architecture:

- **003**: → `neurofaune/preprocess/workflows/dwi_preprocess.py` (add within-subject registration)
- **005**: → `neurofaune/templates/builder.py` (add FA template building)
- **006**: → `neurofaune/templates/registration.py` (add atlas propagation)

## Notes

- FA templates may not be necessary if T2w space is sufficient for analysis
- The thick slices (8mm) mean partial volume effects are significant
- Consider weighted ROI averaging when extracting DTI metrics from atlas regions
