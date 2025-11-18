# SIGMA Atlas Management Guide

This guide explains how to use the SIGMA rat brain atlas in neurofaune for slice-specific registration and ROI-based analyses.

---

## Overview

The SIGMA (Sprague Dawley Imaging Protocol) rat brain atlas is integrated into neurofaune through the `AtlasManager` class. This provides:

- **Brain templates** (T2w-weighted, InVivo and ExVivo)
- **Tissue probability maps** (GM, WM, CSF)
- **Anatomical parcellations** with ROI labels
- **Modality-specific slice extraction** for targeted registration
- **ROI masks** for region-based analyses

---

## Quick Start

```python
from neurofaune.config import load_config
from neurofaune.atlas import AtlasManager

# Load configuration
config = load_config('config.yaml')

# Initialize atlas manager
atlas = AtlasManager(config, atlas_type='invivo')

# Get full brain template
template = atlas.get_template()

# Get hippocampal slices for DTI
dwi_template = atlas.get_template(modality='dwi')

# Get tissue masks
gm_mask = atlas.get_tissue_mask('gm')
wm_prob = atlas.get_tissue_mask('wm', probability=True)

# Get ROI mask
hippocampus = atlas.get_roi_mask('Hippocampus')
```

---

## Atlas Types

### InVivo Atlas (Default)
High-resolution T2w template from living rat brain imaging:
- **Resolution**: ~150 μm isotropic
- **Use cases**: Standard anatomical reference, registration target
- **Tissue maps**: GM, WM, CSF probability maps and binary masks

### ExVivo Atlas
Ultra-high resolution template from ex vivo imaging:
- **Resolution**: ~60 μm isotropic
- **Use cases**: High-detail anatomical studies, histology alignment
- **Tissue maps**: GM, WM, CSF segmentations

```python
# Use ExVivo atlas
atlas = AtlasManager(config, atlas_type='exvivo')
template = atlas.get_template()
```

---

## Modality-Specific Slice Extraction

Neurofaune supports extracting specific anatomical slices for different imaging modalities. This improves registration accuracy when acquisitions only cover partial brain regions.

### Configuration

Define slice ranges in `config.yaml`:

```yaml
atlas:
  name: "SIGMA"
  base_path: "/path/to/SIGMA"

  slice_definitions:
    # DTI: Hippocampus-centered (11 slices)
    dwi:
      start: 15
      end: 25
      description: "Hippocampal region for DTI"

    # fMRI: Broader cortical coverage
    func:
      start: 10
      end: 35
      description: "Cortical and subcortical regions"

    # Anatomical: Full atlas
    anat:
      start: 0
      end: -1  # -1 means all slices
      description: "Complete brain"
```

### Usage

```python
# Get modality-specific template
dwi_template = atlas.get_template(modality='dwi')  # Slices 15-25 only
func_template = atlas.get_template(modality='func')  # Slices 10-35

# Tissue masks with same slice extraction
wm_dwi = atlas.get_tissue_mask('wm', modality='dwi')  # Matches DTI slices

# Parcellation with slice extraction
parcellation_dwi = atlas.get_parcellation(modality='dwi')
```

### Why Slice-Specific Registration?

Many rodent imaging protocols only acquire partial brain coverage:

- **DTI**: Often limited to 11 hippocampal slices (scan time constraints)
- **High-res fMRI**: May cover cortex but not cerebellum
- **Spectroscopy**: Single-slice acquisitions

By extracting only the relevant atlas slices, you get:
- ✅ **Better registration accuracy** (no interference from distant anatomy)
- ✅ **Faster computation** (smaller images to register)
- ✅ **Appropriate quality metrics** (QC focused on acquired region)

---

## Working with Templates

### Brain Templates

```python
# Masked (skull-stripped) template
template_masked = atlas.get_template(masked=True)

# Full template with skull
template_full = atlas.get_template(masked=False)

# Coronal orientation (for registration)
template_coronal = atlas.get_template(coronal=True)

# EPI template for functional data
epi_template = atlas.get_template(modality='func')
```

### Specialized Templates

For DTI, neurofaune automatically uses the hippocampus-focused template:

```python
# This loads SIGMA_InVivo_Brain_Template_Masked_Coronal_Hippocampus.nii.gz
dwi_hipp_template = atlas.get_template(modality='dwi', masked=True)
```

---

## Tissue Segmentation

### Binary Masks

```python
# Binary tissue masks (values: 0 or 1)
gm_mask = atlas.get_tissue_mask('gm', probability=False)
wm_mask = atlas.get_tissue_mask('wm', probability=False)
csf_mask = atlas.get_tissue_mask('csf', probability=False)

# Brain mask (all brain tissue)
brain_mask = atlas.get_brain_mask()
```

### Probability Maps

```python
# Tissue probability maps (values: 0.0 to 1.0)
gm_prob = atlas.get_tissue_mask('gm', probability=True)
wm_prob = atlas.get_tissue_mask('wm', probability=True)
csf_prob = atlas.get_tissue_mask('csf', probability=True)
```

### With Slice Extraction

```python
# Tissue masks for specific modalities
wm_dwi = atlas.get_tissue_mask('wm', modality='dwi')  # Hippocampal slices only
csf_func = atlas.get_tissue_mask('csf', modality='func', probability=True)
```

---

## ROI Parcellation

### Loading Parcellations

```python
# Anatomical parcellation (Waxholm + Tohoku combined)
anat_parcellation = atlas.get_parcellation(atlas_type='anatomical')

# Functional parcellation
func_parcellation = atlas.get_parcellation(atlas_type='functional')
```

### ROI Labels

```python
# Get all ROI definitions
roi_df = atlas.get_roi_labels()

# ROI dataframe columns:
# - Labels: Integer label value
# - Hemisphere: L/R/M (left/right/midline)
# - Matter: Grey Matter / White Matter
# - Territories: Anatomical territory
# - System: Brain system (e.g., Hippocampus Formation)
# - Region of interest: Full ROI name

# Filter by region
hippocampus_rois = roi_df[roi_df['System'] == 'Hippocampus Fomation']
cortical_rois = roi_df[roi_df['Territories'] == 'Cortex']
```

### Creating ROI Masks

```python
# Get mask for specific ROI (partial name matching)
ca1_mask = atlas.get_roi_mask('Cornu.Ammonis.1')  # CA1 region
hipp_mask = atlas.get_roi_mask('Hippocampus')  # All hippocampal subregions

# With modality-specific slices
ca1_dwi = atlas.get_roi_mask('Cornu.Ammonis.1', modality='dwi')

# Using functional parcellation
motor_cortex = atlas.get_roi_mask('Motor', atlas_type='functional')
```

### Example: Extract Mean Values from ROIs

```python
import nibabel as nib
import numpy as np

# Load your DTI metrics
fa_img = nib.load('subject_FA.nii.gz')
fa_data = fa_img.get_fdata()

# Get hippocampus mask
hipp_mask = atlas.get_roi_mask('Hippocampus')
hipp_data = hipp_mask.get_fdata()

# Calculate mean FA in hippocampus
mean_fa = np.mean(fa_data[hipp_data > 0])
print(f"Mean FA in hippocampus: {mean_fa:.3f}")

# Or loop through all ROIs
roi_df = atlas.get_roi_labels()
hippocampal_rois = roi_df[roi_df['System'] == 'Hippocampus Fomation']

for idx, roi in hippocampal_rois.iterrows():
    roi_mask = atlas.get_roi_mask(roi['Region of interest'])
    mask_data = roi_mask.get_fdata()
    mean_val = np.mean(fa_data[mask_data > 0])
    print(f"{roi['Region of interest']}: FA = {mean_val:.3f}")
```

---

## Slice Extraction Utilities

### Manual Slice Extraction

```python
from neurofaune.atlas import extract_slices
import nibabel as nib

# Load image
img = nib.load('subject_T2w.nii.gz')

# Extract slices 15-25 (hippocampus)
extracted = extract_slices(img, slice_start=15, slice_end=25, axis=2)

# Extract from slice 10 to end
partial = extract_slices(img, slice_start=10, slice_end=-1, axis=2)

# Extract along different axis (sagittal)
sagittal = extract_slices(img, slice_start=0, slice_end=50, axis=0)
```

### Get Slice Range from Config

```python
from neurofaune.atlas import get_slice_range

# Get configured slice range for DTI
start, end = get_slice_range(img, 'dwi', config)
# Returns: (15, 25)

# Extract using config
extracted = extract_slices(img, start, end)
```

### Match Slice Geometry

For registration, ensure source and target have matching slice ranges:

```python
from neurofaune.atlas.slice_extraction import match_slice_geometry

# Load subject and atlas images
subject_fa = nib.load('subject_FA.nii.gz')
atlas_template = atlas.get_template()

# Match geometries for DTI
fa_matched, template_matched = match_slice_geometry(
    subject_fa,
    atlas_template,
    modality='dwi',
    config=config
)

# Now both images have same z-dimension (11 slices)
print(fa_matched.shape)      # (128, 128, 10)
print(template_matched.shape)  # (128, 218, 10)  # May differ in x,y but z matches
```

### Slice Metadata for Transform Registry

```python
from neurofaune.atlas.slice_extraction import get_slice_metadata

# Create metadata for transform registry
metadata = get_slice_metadata(img, slice_start=15, slice_end=25)

# Metadata includes:
# - slice_range: (15, 25)
# - axis: 2
# - original_shape: (128, 218, 40)
# - extracted_shape: (128, 218, 10)
# - n_slices: 10
# - description: "Slices 15:25 along axis 2"

# Save with transform
registry.save_ants_composite_transform(
    composite_file=transform_file,
    source_space='FA',
    target_space='SIGMA',
    modality='dwi',
    slice_metadata=metadata
)
```

---

## Performance and Caching

The `AtlasManager` caches loaded images to avoid redundant I/O:

```python
# First load reads from disk
template1 = atlas.get_template()  # Loads from disk

# Second load uses cache
template2 = atlas.get_template()  # Returns cached image
assert template1 is template2  # Same object

# Clear cache to free memory
atlas.clear_cache()

# Next load reads from disk again
template3 = atlas.get_template()  # Loads from disk
```

---

## Integration with Preprocessing Workflows

### Example: Anatomical Registration

```python
from neurofaune.config import load_config
from neurofaune.atlas import AtlasManager
from neurofaune.utils.transforms import create_transform_registry

# Setup
config = load_config('config.yaml')
atlas = AtlasManager(config)
registry = create_transform_registry(config, subject='sub-001', cohort='p60')

# Get atlas reference for registration
template = atlas.get_template(masked=True, coronal=True)
brain_mask = atlas.get_brain_mask()

# Run ANTs registration (pseudocode)
# ants_registration(
#     moving=subject_t2w,
#     fixed=template,
#     fixed_mask=brain_mask,
#     output=output_dir
# )

# Save transform to registry
# registry.save_ants_composite_transform(...)
```

### Example: DTI with Slice-Specific Atlas

```python
# Get hippocampus-focused atlas for DTI
dwi_template = atlas.get_template(modality='dwi', masked=True)
wm_mask = atlas.get_tissue_mask('wm', modality='dwi')

# Register subject FA to atlas
# ants_registration(
#     moving=subject_fa,
#     fixed=dwi_template,
#     fixed_mask=wm_mask,
#     ...
# )

# Save with slice metadata
slice_metadata = get_slice_metadata(dwi_template, 15, 25)
# registry.save_ants_composite_transform(..., slice_metadata=slice_metadata)
```

---

## Atlas File Structure

### SIGMA Directory Layout

```
/mnt/arborea/atlases/SIGMA/
├── SIGMA_Rat_Anatomical_Imaging/
│   ├── SIGMA_Rat_Anatomical_InVivo_Template/
│   │   ├── SIGMA_InVivo_Brain_Template.nii
│   │   ├── SIGMA_InVivo_Brain_Template_Masked.nii
│   │   ├── SIGMA_InVivo_Brain_Template_Masked_Coronal.nii
│   │   ├── SIGMA_InVivo_Brain_Template_Masked_Coronal_Hippocampus.nii.gz
│   │   ├── SIGMA_InVivo_Brain_Mask.nii
│   │   ├── SIGMA_InVivo_GM.nii / GM_mask.nii
│   │   ├── SIGMA_InVivo_WM.nii / WM_mask.nii
│   │   └── SIGMA_InVivo_CSF.nii / CSF_mask.nii
│   └── SIGMA_Rat_Anatomical_ExVivo_Template/
│       └── (similar structure for ExVivo)
│
├── SIGMA_Rat_Brain_Atlases/
│   ├── SIGMA_Anatomical_Atlas/
│   │   ├── InVivo_Atlas/
│   │   │   ├── SIGMA_InVivo_Anatomical_Brain_Atlas.nii
│   │   │   └── SIGMA_InVivo_Anatomical_Brain_Atlas_ListOfStructures.csv
│   │   └── ExVivo_Atlas/
│   │       └── SIGMA_ExVivo_Anatomical_Brain_Atlas.nii
│   └── SIGMA_Functional_Atlas/
│       ├── SIGMA_Functional_Brain_Atlas_InVivo_Anatomical_Template.nii
│       └── SIGMA_Functional_Brain_Atlas_ListOfStructures.csv
│
├── SIGMA_Rat_Functional_Imaging/
│   └── SIGMA_EPI_Brain_Template_Masked_Coronal.nii.gz
│
└── SIGMA_InVivo_Anatomical_Brain_Atlas_Labels.csv
```

---

## API Reference

### AtlasManager

```python
class AtlasManager(config, atlas_type='invivo')
```

**Methods:**
- `get_template(modality=None, masked=True, coronal=False)` - Get brain template
- `get_tissue_mask(tissue_type, modality=None, probability=False)` - Get tissue mask
- `get_brain_mask(modality=None)` - Get binary brain mask
- `get_parcellation(atlas_type='anatomical', modality=None)` - Get parcellation
- `get_roi_labels()` - Get ROI definitions DataFrame
- `get_roi_mask(roi_name, modality=None, atlas_type='anatomical')` - Create ROI mask
- `get_slice_range(modality)` - Get configured slice range
- `clear_cache()` - Clear image cache

### Slice Extraction Functions

```python
extract_slices(img, slice_start, slice_end, axis=2)
get_slice_range(img, modality, config, axis=2)
extract_modality_slices(img, modality, config, axis=2)
match_slice_geometry(source_img, target_img, modality=None, config=None)
get_slice_metadata(img, slice_start, slice_end, axis=2)
```

---

## Citation

If you use the SIGMA atlas in your research, please cite:

> Barrière, D. A., Magalhães, R., Novais, A., Marques, P., Selingue, E., Geffroy, F., ... & Sousa, N. (2019). **The SIGMA rat brain templates and atlases for multimodal MRI data analysis and visualization.** *Nature Communications*, 10(1), 1-13. https://doi.org/10.1038/s41467-019-13575-7

---

## Troubleshooting

### Atlas not found error

```python
FileNotFoundError: Atlas directory not found: /path/to/SIGMA
```

**Solution**: Update `atlas.base_path` in your `config.yaml` to the correct SIGMA atlas location.

### Slice range mismatch

```python
ValueError: slice_start (15) exceeds image size (10)
```

**Solution**: Your acquired image has fewer slices than the configured range. Either:
1. Adjust `slice_definitions` in config.yaml
2. Acquire more slices in your protocol
3. Use geometry matching: `match_slice_geometry()` will auto-crop

### ROI not found

```python
ValueError: No ROIs found matching 'CA4'
```

**Solution**: Check ROI names in the atlas:

```python
roi_df = atlas.get_roi_labels()
print(roi_df['Region of interest'].unique())
```

SIGMA uses specific naming conventions (e.g., "Cornu.Ammonis.1" not "CA1").

---

## Next Steps

- **Registration workflows**: See `docs/TRANSFORMS_GUIDE.md` for transform registry integration
- **Preprocessing pipelines**: See workflow documentation for complete preprocessing examples
- **ROI-based analysis**: See analysis module documentation for group-level ROI statistics
