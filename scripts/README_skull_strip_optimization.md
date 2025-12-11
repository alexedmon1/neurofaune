# Functional Skull Stripping Parameter Optimization

## Overview

This directory contains scripts for optimizing skull stripping parameters for functional BOLD data in rodents. The goal is to find optimal parameters that:

1. **Remove skull and non-brain tissue** effectively
2. **Preserve brain tissue** without over-extraction
3. **Work consistently** across different subjects and age cohorts

## Problem Statement

The current skull stripping for functional data is not satisfactory (as documented in commit b3ea4be):

- **bet4animal** with anatomical parameters (frac=0.7) removes too much brain tissue
- **Skull fragments** remain in the extracted brain
- **BOLD contrast** is different from T2w anatomical, requiring different parameters

## Parameter Sweep Approach

The `optimize_func_skull_strip.py` script tests multiple parameter combinations:

### FSL BET Tests

1. **Standard BET** with frac values: 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5
2. **Functional BET** (with -F flag) with frac values: 0.3, 0.35, 0.4

The `-F` flag applies functional-specific optimizations in BET.

### bet4animal Tests

Tests combinations of:
- **frac values**: 0.3, 0.4, 0.5, 0.6 (lower than anatomical default of 0.7)
- **center coordinates**: Default center ± 2 voxels in z-direction (dorsal/ventral shifts)
- **radius**: 125 voxels (fixed)
- **scale**: (1, 1, 1.5) for anisotropic voxels (fixed)
- **width**: 2.5 smoothness parameter (fixed)

## Usage

### Quick Start

Run the parameter sweep on a test subject:

```bash
cd /home/edm9fd/sandbox/neurofaune/scripts
./run_skull_strip_sweep.sh
```

This will:
1. Extract middle volume from 4D BOLD timeseries (or use temporal mean if configured)
2. Test all parameter combinations on sub-Rat108 ses-p30
3. Generate quality metrics for each test
4. Create a comprehensive comparison visualization
5. Save results to `/mnt/arborea/bpa-rat/test/skull_strip_optimization/`

### Reference Volume Methods

The script supports three methods for extracting a reference volume for skull stripping:

1. **"middle"** (default, recommended): Extract the middle volume from the timeseries
   - Fastest method (no temporal averaging)
   - Similar to using a 1-TR BOLD acquisition
   - Good SNR if motion is minimal

2. **"mean"**: Compute temporal average of all volumes
   - Better SNR than single volume
   - More computationally expensive
   - May blur if significant motion present

3. **Integer index**: Extract a specific volume (e.g., "65" for volume 65)
   - Useful for selecting a known good volume
   - Can avoid motion-corrupted volumes

To change the method, edit `REF_METHOD` in `run_skull_strip_sweep.sh` or use the `--ref-method` flag.

### Custom Subject

To test on a different subject:

```bash
uv run python optimize_func_skull_strip.py \
    --subject sub-Rat209 \
    --session ses-p60 \
    --bold-file /path/to/bold.nii.gz \
    --output-dir /output/path \
    --ref-method middle
```

## Output Files

The script creates:

1. **`results.json`**: Complete results with metrics for all parameter combinations
2. **`{subject}_{session}_skull_strip_comparison.png`**: Visual comparison of all methods
3. **`bet_standard/`**: BET results without -F flag
4. **`bet_functional/`**: BET results with -F flag
5. **`bet4animal/`**: bet4animal results
6. **`work/`**: Temporary files (mean BOLD image)

## Quality Metrics

For each parameter combination, the script calculates:

- **Extraction ratio**: mask_voxels / original_nonzero_voxels
  - Higher ratio = more tissue retained
  - Lower ratio = more aggressive extraction
- **Mask voxels**: Number of voxels in brain mask
- **Brain intensity statistics**: Mean, std, min, max of extracted brain

## Visual Inspection

**IMPORTANT**: Quality metrics alone are insufficient! Visual inspection is critical:

1. **Check mask overlay** on three orthogonal views (axial, coronal, sagittal)
2. **Verify brain coverage**: All cortical and subcortical structures included
3. **Check for skull fragments**: No bright non-brain tissue remaining
4. **Assess over-extraction**: No excessive removal of cortical tissue

## Selecting Optimal Parameters

After running the sweep:

1. **Review the comparison figure** (`*_skull_strip_comparison.png`)
2. **Check top 5 results** by extraction ratio (printed in summary)
3. **Visually inspect** each method's mask overlay
4. **Select the method** that best balances:
   - Complete brain coverage
   - Clean skull removal
   - Minimal over-extraction

5. **Update the config** (`configs/default.yaml`) with optimal parameters:

```yaml
functional:
  bet:
    method: "bet"  # or "bet4animal"
    frac: 0.35     # Optimal frac value
    functional: true  # Use -F flag if method="bet"
```

## Testing Across Cohorts

Since brain size/position varies by age, test on multiple cohorts:

```bash
# Test p30 cohort
./run_skull_strip_sweep.sh  # (default: sub-Rat108 ses-p30)

# Test p60 cohort
uv run python optimize_func_skull_strip.py \
    --subject sub-Rat209 \
    --session ses-p60 \
    --bold-file /mnt/arborea/bpa-rat/raw/bids/sub-Rat209/ses-p60/func/sub-Rat209_ses-p60_run-13_bold.nii.gz \
    --output-dir /mnt/arborea/bpa-rat/test/skull_strip_optimization/sub-Rat209_ses-p60

# Test p90 cohort
uv run python optimize_func_skull_strip.py \
    --subject sub-Rat49 \
    --session ses-p90 \
    --bold-file /mnt/arborea/bpa-rat/raw/bids/sub-Rat49/ses-p90/func/sub-Rat49_ses-p90_run-14_bold.nii.gz \
    --output-dir /mnt/arborea/bpa-rat/test/skull_strip_optimization/sub-Rat49_ses-p90
```

## Expected Results

Based on preliminary analysis:

- **Standard BET (frac=0.3)** often works well for BOLD data
- **BET with -F flag** may provide better functional-specific optimization
- **bet4animal** with lower frac (0.3-0.5) may work better than anatomical settings (0.7)
- **Center adjustments** may be needed for younger animals (p30) vs adults (p90)

## Troubleshooting

### All tests fail

Check:
- FSL and bet4animal are installed and in PATH
- BOLD file exists and is readable
- Sufficient disk space in output directory

### Poor results across all parameters

Consider:
- **BOLD image quality**: Check for motion artifacts, low SNR
- **Preprocessing**: May need bias correction before skull stripping
- **Custom centers**: Manually estimate brain center from image

### bet4animal not found

Install bet4animal or use BET-only mode by commenting out bet4animal tests in the script.

## Next Steps

After identifying optimal parameters:

1. Update `neurofaune/preprocess/utils/func/skull_strip.py` with best parameters
2. Update `configs/default.yaml` with recommended settings
3. Add age-specific parameters if needed (p30 vs p60/p90)
4. Re-run functional preprocessing workflow to verify
5. Generate QC reports to validate improvement

## References

- FSL BET documentation: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET
- bet4animal: Rodent-specific brain extraction tool
- Previous discussion: See commit b3ea4be for original problem description
