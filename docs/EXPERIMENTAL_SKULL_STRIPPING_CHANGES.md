# Experimental Skull Stripping Changes - NOT RECOMMENDED

**Date:** December 2025
**Status:** EXPERIMENTAL - Not validated, not recommended for production use
**File:** `neurofaune/preprocess/workflows/anat_preprocess.py`

## Overview

During troubleshooting of skull stripping issues, several experimental modifications were made to the Atropos-based skull stripping method. These changes were **NOT successful** and have been reverted.

This document serves as a reference for what was tried and why it didn't work well.

## Experimental Changes Made

### 1. BET + Atropos Mask Intersection

**What was tried:**
- Run BET on the unmasked N4-corrected image
- Intersect BET mask with Atropos tissue mask
- Use combined mask for brain extraction

**Rationale:**
- BET provides good brain boundary detection
- Atropos provides tissue-based constraints
- Intersection should give "best of both worlds"

**Why it didn't work well:**
- Added complexity without clear benefit
- Intersection often too conservative (removed valid brain tissue)
- BET parameters still difficult to optimize for rodent brains
- Original Atropos-only method performed comparably

### 2. Enhanced Morphological Operations

**What was tried:**
```python
# Closing: fill small holes
combined_mask = ndimage.binary_closing(combined_mask, iterations=2)

# Opening: remove small islands
combined_mask = ndimage.binary_opening(combined_mask, iterations=1)
```

**Rationale:**
- Clean up mask boundaries
- Remove isolated voxels
- Fill small holes in brain mask

**Why it didn't work well:**
- Over-smoothing of brain boundaries
- Loss of fine anatomical detail
- Arbitrary iteration counts not validated
- Original masks already reasonably clean

### 3. Largest Connected Component Selection

**What was tried:**
```python
# Keep only the largest connected component
labeled_mask, num_features = ndimage.label(combined_mask)
component_sizes = np.bincount(labeled_mask.ravel())
largest_component = component_sizes.argmax()
combined_mask = labeled_mask == largest_component
```

**Rationale:**
- Remove scattered voxels outside brain
- Focus on main brain mass
- Eliminate isolated tissue regions

**Why it didn't work well:**
- Too aggressive in some cases
- Could remove valid brain regions if mask was fragmented
- Not necessary if initial segmentation is good
- Added computational overhead

### 4. Adaptive BET Frac Parameter

**What was tried:**
- Calculate image contrast (CNR) between tissue classes
- Adaptively adjust BET frac parameter based on CNR
- Higher contrast → more aggressive skull stripping

**Why it didn't work well:**
- CNR calculation unstable across subjects
- Relationship between CNR and optimal frac parameter not linear
- Required careful tuning per dataset
- Original fixed parameters worked comparably

### 5. bet4animal Integration

**What was added:**
- New `method='bet4animal'` option
- Wrapper for bet4animal tool
- CNR-based parameter optimization

**Status:**
- Implemented but not validated
- May be useful for future work
- Not recommended for production use yet
- Requires bet4animal installation

## Complete Patch File

The full experimental changes are saved in:
`/tmp/experimental_anat_preprocess_changes.patch`

To review: `cat /tmp/experimental_anat_preprocess_changes.patch`

## What Actually Worked

After all this experimentation, the **root causes** of skull stripping failures were:

### 1. Wrong Anatomical Scan Selection ✓ FIXED
- **Problem:** Localizers/scouts being selected instead of high-resolution anatomical scans
- **Fix:** Updated `neurofaune/utils/select_anatomical.py` scoring:
  - Added strong penalty (-5.0) for scans with <10 slices
  - Increased rewards for high slice counts (≥40 slices → +5.0)
- **Result:** All subjects now correctly select run-6 (256×256×41) scans

### 2. Incorrect NIfTI Voxel Sizes ✓ FIXED
- **Problem:** NIfTI headers had wrong voxel dimensions (not scaled from Bruker DICOM)
- **Fix:** Created header correction script applying 10× scaling:
  - Raw: 0.125 × 0.125 × 0.8 mm
  - Scaled: 1.25 × 1.25 × 8.0 mm (FSL/ANTs compatible)
- **Result:** All 141 subjects' T2w scans corrected

### 3. Original Atropos Method is Fine ✓ VALIDATED
- **Finding:** Original 5-class Atropos tissue segmentation works well (~75% success)
- **With fixes:** Success rate expected to improve to near 100%
- **Conclusion:** The algorithm was fine; data preparation was the issue

## Lessons Learned

1. **Fix data quality issues first** - Don't modify algorithms to compensate for bad input data
2. **Simple is better** - Complex multi-step approaches add failure points
3. **Validate incrementally** - Test each change independently before combining
4. **Original pipeline wisdom** - The established method had good reasons behind it
5. **Measure success objectively** - Need QC metrics, not just visual inspection

## Recommendations

### For Production Use
1. ✓ Use original Atropos method (`method='atropos'`)
2. ✓ Ensure correct anatomical scan selection
3. ✓ Verify NIfTI headers have correct voxel sizes
4. ✗ **DO NOT** use experimental mask intersection approach
5. ✗ **DO NOT** use arbitrary morphological operations
6. ✗ **DO NOT** use untested adaptive parameters

### For Future Research
- bet4animal may be worth exploring after proper validation
- Machine learning-based skull stripping (e.g., SynthStrip)
- Atlas-based approaches for specific cohorts
- But always validate against ground truth!

## Code Status

**Current (Production):**
- `neurofaune/utils/select_anatomical.py`: Fixed scoring ✓
- `neurofaune/preprocess/workflows/anat_preprocess.py`: Original Atropos method ✓

**Experimental (Reverted):**
- All changes to anat_preprocess.py reverted
- Patch file saved for reference only
- Not recommended for use

---

**Last Updated:** 2025-12-01
**Author:** Troubleshooting session
**Status:** EXPERIMENTAL CHANGES REVERTED - Use original pipeline
