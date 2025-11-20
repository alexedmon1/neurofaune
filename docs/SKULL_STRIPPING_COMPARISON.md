# Skull Stripping Method Comparison for Rodent Brains

## Problem Statement

Skull stripping rodent MRI data is challenging due to the cylindrical shape of rodent heads (vs spherical human heads). The goal was to find a method that properly separates brain tissue from skull/head tissues.

## Test Data

- **Subject**: sub-Rat207, session p60
- **Image**: T2w RARE axial, 1.25x1.25x8mm resolution
- **Original volume**: 2,686,966 voxels (including skull, head, background)

## Methods Tested

### 1. FSL BET (Brain Extraction Tool)

**Configuration**:
- Fractional intensity threshold: 0.6 (for P60 cohort)
- Radius: 125mm

**Results**:
- **Volume retained**: 545,591 voxels (20.3%)
- **Status**: ❌ FAILED - Too aggressive but still kept non-brain tissue
- **Issue**: BET is finicky with cylindrical rodent brains. Even though it appeared to remove too much tissue, visual inspection showed brain remained mixed with skull/head tissues

### 2. ANTs BrainExtraction

**Configuration**:
- Template-based using SIGMA rat brain atlas template
- Atlas brain probability mask as prior

**Results**:
- **Volume retained**: 2,686,976 voxels (100%)
- **Status**: ❌ FAILED - Not aggressive enough, didn't remove skull
- **Issue**: Method retained essentially all original voxels, providing no skull stripping benefit

### 3. ANTs Atropos 5-Component Segmentation ✅

**Configuration**:
- 5 tissue classes: Background, CSF, Gray Matter, White Matter, Skull/Other
- K-means initialization
- MRF smoothing (factor=0.1, radius=[1,1,1])
- Foreground mask created from subject's own T2w image using Otsu thresholding
- Brain mask = posteriors 2-4 (CSF + GM + WM, excluding Background and Skull/Other)

**Results**:
- **Volume retained**: 294,256 voxels (10.95%)
- **Status**: ✅ SUCCESS
- **Key insight**: Must use subject-space foreground mask, NOT atlas mask

**Implementation**: `neurofaune/preprocess/workflows/anat_preprocess.py:210-260`

## Critical Lesson Learned

**DO NOT use atlas masks during skull stripping!**

Initial Atropos implementation attempted to use the SIGMA atlas brain mask, which failed because:
1. Skull stripping happens in **subject space** before atlas registration
2. Atlas and subject have different origins and voxel spacing
3. Even if resampled, using an atlas mask presupposes alignment that hasn't happened yet

**Correct approach**:
1. Create foreground mask from subject's own image (Otsu thresholding at 0.3× threshold)
2. This masks only background air, keeping entire head (brain + skull)
3. Let Atropos segment tissues within that foreground
4. Combine tissue posteriors (CSF + GM + WM) to create brain mask

## Workflow Integration

The final skull stripping function signature:

```python
def skull_strip_rodent(
    input_file: Path,
    output_file: Path,
    cohort: str = 'p60',
    method: str = 'atropos',
    template_file: Optional[Path] = None,  # Only for 'ants' method
    template_mask: Optional[Path] = None   # Only for 'ants' method
) -> Tuple[Path, Path]:
    """Skull strip rodent brain with age-specific parameters."""
```

## Recommendations

1. **Use Atropos 5-component segmentation** for rodent skull stripping
2. **Create subject-space foreground masks** using intensity thresholding
3. **Keep atlas operations separate** from skull stripping
4. **Atlas registration should use the skull-stripped brain** as input

## Files Modified

- `/home/edm9fd/sandbox/neurofaune/neurofaune/preprocess/workflows/anat_preprocess.py`
  - Lines 146-260: `skull_strip_rodent()` function
  - Lines 689-697: Workflow integration
