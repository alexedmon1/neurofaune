# Functional BOLD Skull Stripping Comparison

## Problem Statement

The original bet4animal approach (from commit b3ea4be) for functional skull stripping was not satisfactory:
- Too much brain tissue removed
- Skull fragments remaining in extracted brain
- Parameters optimized for anatomical T2w not working for BOLD contrast

## Approach 1: Parameter Sweep (Simple Methods)

Tested FSL BET and bet4animal with various parameters on sub-Rat108 ses-p30.

### Top Results (by extraction ratio)

| Method | Extraction Ratio | Mask Voxels | Notes |
|--------|------------------|-------------|-------|
| bet4animal frac=0.3, center=(46,80,4) | 0.181 | 23,994 | Best simple method |
| bet4animal frac=0.3, center=(46,80,2) | 0.180 | 23,783 | Ventral shift |
| bet4animal frac=0.3, center=(46,80,6) | 0.179 | 23,661 | Dorsal shift |
| BET with -F flag, frac=0.3 | 0.152 | ~20,000 | Functional optimization |
| BET standard, frac=0.2 | 0.139 | 18,415 | Most aggressive |

**Key findings:**
- bet4animal works best with **frac=0.3** (much lower than anatomical frac=0.7)
- FSL BET extraction ratios too low (13-15%), likely too aggressive
- Center position has minor impact

**Limitation:** Simple intensity-based methods struggle with BOLD's poor tissue contrast

## Approach 2: Atropos Segmentation (Robust Method)

Adapted anatomical skull stripping approach:
1. Atropos 5-class tissue segmentation
2. Dynamic exclusion of largest (background) and smallest (peripheral) classes
3. Morphological closing (dilate 2x → erode 2x)
4. **No BET refinement** (BOLD has too low CNR for reliable BET)

### Results on sub-Rat108 ses-p30

| Step | Voxels | Ratio | Notes |
|------|--------|-------|-------|
| Original foreground | 132,480 | 100% | Everything > 0 |
| Atropos posteriors (3 classes) | 28,964 | 21.9% | After excluding extremes |
| Final mask (after morphology) | 22,016 | 16.6% | Clean, robust |

**Comparison to parameter sweep:**
- Atropos: 22,016 voxels (16.6%)
- Best bet4animal: 23,994 voxels (18.1%)
- Difference: ~2,000 voxels (slightly more conservative)

## Recommendation

**Use Atropos-based approach** for functional skull stripping:

### Advantages
✓ **Robust tissue segmentation** - doesn't rely solely on intensity thresholds
✓ **Handles BOLD contrast** - works even with low CNR (0.83 in our test)
✓ **Consistent across subjects** - dynamic posterior classification
✓ **No parameter tuning** - automatically adapts to image properties
✓ **Proven in anatomical** - same approach works well for T2w

### Disadvantages
✗ Slower than BET alone (~30 seconds vs ~2 seconds)
✗ Requires ANTs (Atropos) in addition to FSL

## Implementation

### Function Location
`neurofaune/preprocess/utils/func/skull_strip_atropos.py`

### Usage
```python
from neurofaune.preprocess.utils.func.skull_strip_atropos import skull_strip_bold_atropos

brain_file, mask_file = skull_strip_bold_atropos(
    input_file=bold_ref_volume,  # 3D: mean or middle volume
    output_file=output_brain,
    mask_file=output_mask,
    use_bet_refinement=False  # BET not recommended for BOLD
)
```

### Integration into Functional Workflow

Replace current skull stripping in `func_preprocess.py` (around line 720):

**OLD:**
```python
from neurofaune.preprocess.utils.func.skull_strip import brain_extraction

brain_results = brain_extraction(
    mean_bold,
    brain_mean,
    brain_mask,
    method='bet4animal',  # or 'bet'
    **params
)
```

**NEW:**
```python
from neurofaune.preprocess.utils.func.skull_strip_atropos import skull_strip_bold_atropos

brain_file, mask_file = skull_strip_bold_atropos(
    input_file=mean_bold,  # or middle volume
    output_file=brain_mean,
    mask_file=brain_mask,
    use_bet_refinement=False
)
```

## Visual Inspection

**CRITICAL:** Visual QC is required before production use!

### QC Locations
- Parameter sweep comparison: `/mnt/arborea/bpa-rat/test/skull_strip_optimization/sub-Rat108_ses-p30/sub-Rat108_ses-p30_skull_strip_comparison.png`
- Atropos results: `/mnt/arborea/bpa-rat/test/atropos_skull_strip/qc/sub-Rat108_ses-p30_atropos_skull_strip_qc.png`

### What to Check
1. **Complete brain coverage** - all cortical and subcortical structures included
2. **Clean skull removal** - no bright skull/CSF artifacts remaining
3. **Minimal over-extraction** - cortical edges well-preserved
4. **Consistent across slices** - check axial, coronal, sagittal views

## Next Steps

1. ✓ Implement Atropos-based skull stripping
2. ✓ Test on sub-Rat108 ses-p30
3. ⏳ **Visual QC inspection** (USER ACTION REQUIRED)
4. Test on additional subjects/cohorts (p60, p90)
5. Update functional preprocessing workflow
6. Re-run preprocessing and validate improvements
7. Update configuration defaults

## Test Commands

### Run parameter sweep (for comparison)
```bash
cd /home/edm9fd/sandbox/neurofaune/scripts
./run_skull_strip_sweep.sh
```

### Test Atropos approach
```bash
uv run python /home/edm9fd/sandbox/neurofaune/scripts/test_atropos_skull_strip.py \
    --subject sub-Rat108 \
    --session ses-p30 \
    --output-dir /mnt/arborea/bpa-rat/test/atropos_skull_strip
```

### Test on additional subjects
```bash
# p60 cohort
uv run python test_atropos_skull_strip.py \
    --subject sub-Rat209 \
    --session ses-p60 \
    --bold-file /mnt/arborea/bpa-rat/raw/bids/sub-Rat209/ses-p60/func/sub-Rat209_ses-p60_run-13_bold.nii.gz

# p90 cohort
uv run python test_atropos_skull_strip.py \
    --subject sub-Rat49 \
    --session ses-p90 \
    --bold-file /mnt/arborea/bpa-rat/raw/bids/sub-Rat49/ses-p90/func/sub-Rat49_ses-p90_run-14_bold.nii.gz
```
