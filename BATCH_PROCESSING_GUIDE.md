# Batch Processing Guide

## Overview

This guide covers batch anatomical preprocessing and template building for all subjects in the BPA-Rat study.

## Scripts

### 1. `batch_anatomical_preprocessing.py`

Runs anatomical preprocessing on all subjects, automatically:
- Finds all subjects in BIDS directory
- Selects best T2w scan (avoids 3D scans with -10.0 penalty)
- Runs preprocessing (skull stripping, bias correction, normalization)
- Creates exclusion markers for failed subjects
- Logs all processing

**Usage:**
```bash
python batch_anatomical_preprocessing.py <bids_dir> <output_dir> [--force]
```

**Example:**
```bash
python batch_anatomical_preprocessing.py \
    /mnt/arborea/bpa-rat/raw/bids \
    /mnt/arborea/bpa-rat
```

**Options:**
- `--force` : Reprocess subjects even if already completed

**Outputs:**
- `derivatives/{subject}/{session}/anat/` : Preprocessed images
  - `{subject}_{session}_desc-preproc_T2w.nii.gz` : Preprocessed brain
  - `{subject}_{session}_desc-brain_mask.nii.gz` : Brain mask
  - `{subject}_{session}_label-{GM,WM,CSF}_probseg.nii.gz` : Tissue segmentations
- `logs/batch_anatomical_preprocessing_*.log` : Processing log
- `logs/preprocessing_summary_*.json` : JSON summary of results
- `derivatives/{subject}/{session}/{subject}_{session}_EXCLUDE.txt` : Exclusion markers (if failed)

### 2. `build_templates.py`

Creates age-specific templates (p30, p60, p90) from preprocessed anatomical data using ANTs.

**Usage:**
```bash
python build_templates.py <study_root> [options]
```

**Example:**
```bash
python build_templates.py /mnt/arborea/bpa-rat \
    --cohorts p30,p60,p90 \
    --iterations 4 \
    --cores 6
```

**Options:**
- `--cohorts p30,p60,p90` : Specify cohorts (default: all)
- `--iterations 4` : Number of ANTs iterations (default: 4)
- `--cores 6` : Number of CPU cores (default: 6)

**Requirements:**
- ANTs must be installed and `buildtemplateparallel.sh` in PATH
- At least 3 successfully preprocessed subjects per cohort

**Outputs:**
- `templates/template_{cohort}_template.nii.gz` : Final template for each cohort
- `templates/template_building_results.json` : JSON summary

## Workflow

### Step 1: Run Batch Preprocessing

```bash
cd /home/edm9fd/sandbox/neurofaune

# Run batch preprocessing
python batch_anatomical_preprocessing.py \
    /mnt/arborea/bpa-rat/raw/bids \
    /mnt/arborea/bpa-rat
```

**What happens:**
1. Finds all subjects with T2w anatomical data
2. For each subject/session:
   - Automatically selects best T2w scan (excludes 3D scans)
   - Runs skull stripping, bias correction, normalization
   - Tissue segmentation (GM, WM, CSF)
   - Saves preprocessed outputs
3. Creates exclusion markers for:
   - Subjects with only 3D T2w scans
   - Preprocessing failures
   - Poor quality data

**Monitor progress:**
```bash
# Follow log in real-time
tail -f /mnt/arborea/bpa-rat/logs/batch_anatomical_preprocessing_*.log

# Check summary after completion
cat /mnt/arborea/bpa-rat/logs/preprocessing_summary_*.json
```

### Step 2: Build Templates

After preprocessing completes:

```bash
# Build templates for all cohorts
python build_templates.py /mnt/arborea/bpa-rat
```

**What happens:**
1. Finds all successfully preprocessed subjects for each cohort
2. Excludes subjects with EXCLUDE markers
3. Builds template using ANTs for each cohort:
   - p30: Postnatal day 30 (juvenile)
   - p60: Postnatal day 60 (young adult)
   - p90: Postnatal day 90 (adult)

**Note:** Template building is computationally intensive and may take several hours per cohort.

### Step 3: Review Results

```bash
# Check which subjects were processed
ls /mnt/arborea/bpa-rat/derivatives/

# Check which subjects were excluded
find /mnt/arborea/bpa-rat/derivatives -name "*_EXCLUDE.txt"

# View exclusion reasons
cat /mnt/arborea/bpa-rat/derivatives/sub-*/ses-*/*_EXCLUDE.txt

# Check templates
ls -lh /mnt/arborea/bpa-rat/templates/
```

## Exclusion Criteria

Subjects are automatically excluded if:

1. **3D T2w scans only** (scoring penalty: -10.0)
   - 3D TurboRARE scans from Cohorts 1-2
   - Known to have skull stripping issues
   - Prefer 2D RARE scans

2. **Preprocessing failures**
   - Skull stripping failure
   - Registration failure
   - Data quality issues

3. **Insufficient coverage**
   - Too few slices (< 10)
   - Poor brain coverage

## Troubleshooting

### Issue: No subjects found
- Check BIDS directory structure
- Ensure T2w files exist in `{subject}/{session}/anat/`

### Issue: All subjects excluded
- Check log file for specific reasons
- Review T2w scan names for "3D" or "TurboRARE_3D"
- Verify 2D RARE scans are available

### Issue: Template building fails
- Ensure ANTs is installed: `which buildtemplateparallel.sh`
- Check that at least 3 subjects passed preprocessing
- Review template building log for specific errors

### Issue: Out of memory
- Reduce `--cores` for template building
- Process cohorts separately
- Use a machine with more RAM

## Next Steps

After successful preprocessing and template building:

1. **Run DWI preprocessing** on subjects with DTI data
2. **Run MSME preprocessing** on subjects with multi-echo T2 data
3. **Register subjects to templates** for normalized analyses
4. **Perform group analyses** on template space

See `test_dwi_workflow.py` and `test_msme_workflow.py` for testing additional modalities.
