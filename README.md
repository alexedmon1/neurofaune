# Neurofaune

**Rodent-specific MRI preprocessing, analysis, and connectivity package**

Neurofaune is a comprehensive neuroimaging pipeline designed specifically for rodent (rat and mouse) brain MRI data. Built on the architecture of [neurovrai](https://github.com/yourusername/neurovrai), it provides Bruker data conversion and advanced preprocessing for multiple MRI modalities.

**Current test study**: BPA-Rat (Bisphenol A rat cohort study) - 141 subjects, 189 sessions across 7 cohorts

---

## Current Implementation Status

### ✅ **Completed Workflows**

1. **Anatomical T2w Preprocessing**
   - Automatic T2w scan selection (excludes problematic 3D scans)
   - Adaptive skull stripping (Atropos 5-component + adaptive BET)
     - Subject-specific BET frac calculation based on contrast-to-noise ratio
     - Conservative parameters (frac 0.1-0.4) to minimize over-stripping
   - N4 bias field correction
   - Tissue segmentation (GM, WM, CSF)
   - Batch processing support with --force flag for reprocessing

2. **DWI/DTI Preprocessing**
   - 5D→4D conversion for Bruker data
   - GPU-accelerated eddy correction with **slice padding** to prevent edge slice loss
   - FSL dtifit for tensor fitting (FA, MD, AD, RD)
   - Comprehensive QC (motion, eddy, DTI metrics)
   - Batch processing with header validation and optional Bruker-based fixing
   - Multi-shell ready (currently tested on 6-direction DTI)

3. **MSME T2 Mapping**
   - Multi-echo T2 mapping
   - Myelin Water Fraction (MWF) calculation via NNLS
   - T2 compartment analysis (myelin, intra/extra-cellular, CSF)
   - QC with T2 decay curves and NNLS spectra visualization

4. **Functional fMRI Preprocessing**
   - **Adaptive skull stripping** (per-slice BET with -R flag, protocol-robust 0.8% variability)
     - N4 bias correction + intensity normalization on reference volume
     - Slice-wise adaptive frac optimization targeting 15% brain extraction
     - BET -R flag for robust center estimation across acquisition geometries
     - Tested across 6 distinct acquisition protocols (voxel sizes 1.2-5.9mm)
   - **Motion correction** (MCFLIRT with middle volume reference)
   - **ICA denoising** (MELODIC with automatic dimensionality estimation)
     - Rodent-optimized automated component classification (score-based)
     - Motion, edge, CSF, and frequency-based noise detection
     - Typical signal retention: 75-77% of components
   - **Spatial smoothing** (6mm FWHM)
   - **Temporal filtering** (0.01-0.1 Hz bandpass for resting-state)
   - **Confound extraction** (24 extended motion regressors + aCompCor when tissue masks available)
   - **Comprehensive QC** (motion FD/DVARS, ICA classification, confounds, registration overlays)
   - **Note:** Some acquisitions contain zebra stripe artifacts inherent to the scan (not introduced by preprocessing)

### ✅ **Template-Based Registration (January 2025)**

- **Age-specific T2w templates:** Complete for p30, p60, p90
- **Template → SIGMA registration:** Complete with study-space atlas
- **Subject → Template registration:** ANTs SyN registration
- **Atlas propagation:** SIGMA labels propagated to each subject's T2w space
- **Registration QC:** Dice coefficient, correlation metrics, edge overlay figures

---

## Workflow Overview

Neurofaune uses a **two-phase workflow** that separates one-time study setup from per-subject preprocessing.

### Phase 1: Initialize (One-time per study)

Run these steps once when setting up a new study:

```bash
# Step 1: Initialize study structure and create config
uv run python scripts/init_study.py /path/to/study \
    --name "My Study" --code mystudy \
    --bids-root /path/to/bids \
    --sigma-atlas /path/to/SIGMA_scaled

# Step 2: Preprocess a subset of subjects for template building
uv run python scripts/batch_preprocess_for_templates.py \
    --bids-root /path/to/bids \
    --output-root /path/to/study \
    --cohorts p30 p60 p90 \
    --fraction 0.20

# Step 3: Build age-specific templates and register to SIGMA
uv run python scripts/build_templates.py \
    --config /path/to/study/config.yaml \
    --cohorts p30 p60 p90
```

**What this creates:**
- `{study_root}/config.yaml` - Study configuration
- `{study_root}/atlas/SIGMA_study_space/` - Reoriented atlas files
- `{study_root}/templates/anat/{cohort}/` - Age-specific T2w templates
- Template → SIGMA transforms for atlas propagation

### Phase 2: Preprocessing (All subjects)

Run preprocessing on all subjects using the templates:

```bash
# Anatomical preprocessing with registration and atlas propagation
uv run python scripts/batch_preprocess_anat.py \
    --config /path/to/study/config.yaml

# DTI preprocessing
uv run python scripts/batch_preprocess_dwi.py \
    --bids-root /path/to/bids \
    --output-root /path/to/study \
    --config /path/to/study/config.yaml

# Functional preprocessing
uv run python scripts/batch_preprocess_func.py \
    --bids-root /path/to/bids \
    --output-root /path/to/study
```

**What anatomical preprocessing does:**
1. Bias field correction (N4)
2. Skull stripping (Atropos + BET)
3. Tissue segmentation (GM/WM/CSF)
4. Intensity normalization
5. **Register to age-specific template** (ANTs SyN)
6. **Propagate SIGMA atlas to subject space** (using study-space atlas)

---

## Features

### Current Status

**✅ Phase 1 (Foundation) - Complete**
- Configuration System: YAML-based configuration with variable substitution
- Transform Registry: Centralized storage for spatial transformations with slice-specific support
- Neurovrai-compatible Output: Same directory hierarchy (derivatives/, transforms/, qc/)
- Rodent-optimized Parameters: Atlas slice extraction, age cohort support

**✅ Phase 2 (Atlas Management) - Complete**
- AtlasManager: Unified interface for SIGMA rat brain atlas
- Slice Extraction: Modality-specific slice extraction utilities
- Template Access: InVivo/ExVivo templates, tissue masks, parcellations
- ROI Operations: Label loading, ROI mask creation, metadata access

**✅ Phase 3 (Anatomical Preprocessing) - Complete**
- Image Validation: Pre-pipeline validation (voxel size, orientation, dimensions, data type)
- Orientation Matching: Automatic orientation detection and correction between images
- Robust Skull Stripping: Two-pass approach with dynamic posterior classification
  - ANTs Atropos 5-component segmentation for rough brain mask
  - **Dynamic posterior identification**: Volume-based exclusion of largest (background/CSF) and smallest (peripheral skull/eyes) posteriors
  - Handles non-deterministic Atropos KMeans initialization (no hardcoded posterior indices)
  - Morphological closing (dilate→erode) to remove mask speckles and fill holes
  - FSL BET refinement with adaptive frac calculation (CNR-based, range 0.1-0.4)
  - Conservative parameters to minimize over-stripping across variable image quality
- Tissue Segmentation: Reuses Atropos posteriors from skull stripping (30-40% faster, no redundant execution)
  - Extracts GM, WM, CSF probability maps from skull stripping posteriors
  - Applies refined BET mask for accurate tissue classification
  - Creates hard segmentation (dseg) via argmax with confidence thresholding
  - Minimum probability threshold (0.35) reduces tissue classification noise
- N4 Bias Correction: Intensity inhomogeneity correction
- Intensity Normalization: Scale-invariant preprocessing
- Automatic T2w Scan Selection: Score-based selection with 3D scan penalty
- Exclusion Marker System: Automatic tracking of preprocessing failures
- Preprocessing Only: SIGMA registration removed for template-based approach
- Batch Processing: Support for --force flag to reprocess all subjects
- Tested on BPA-Rat data: Validated on 189 subject/session combinations across all cohorts

**✅ Phase 4 (DTI Preprocessing Foundation) - Complete**
- 5D→4D Conversion: Handle Bruker multi-average DTI acquisitions
- Gradient Table Validation: Verify bval/bvec consistency with automatic normalization
- GPU-Accelerated Eddy: FSL eddy_cuda support (with CPU fallback)
- **Slice Padding for Edge Protection**: Mirror-pads DWI data before eddy to prevent edge slice loss from motion correction interpolation (critical for thin-slice acquisitions like 5-11 slice hippocampal DTI)
- Brain Masking: BET-based masking from b0 volume
- DTI Fitting: Compute FA, MD, AD, RD maps using FSL dtifit
- Comprehensive QC: Motion/eddy QC with framewise displacement, DTI metric QC with histograms and montages
- **Batch Processing**: Production-ready script with header validation, Bruker-based fixing, and parallel execution
- Preprocessing Only: SIGMA registration removed for template-based approach

**✅ Phase 5 (Multi-Modal Template Architecture) - Complete**
- Architecture Design: Independent T2w and FA template spaces (documented)
- Workflow Refactoring: Removed direct SIGMA registration from preprocessing pipelines
- Template Building Module: ANTs multivariate template construction wrapper
- Subject Selection: QC-based selection of top 25% subjects per cohort
- Tissue-Specific Templates: GM, WM, CSF probability templates for each cohort
- Automatic T2w Template → SIGMA Registration
- Subject-to-Template Registration: Register individuals to age-matched templates
- Within-Subject Registration: Cross-modal registration (T2w ↔ FA, T2w ↔ BOLD)
- Label Propagation: Transform composition utilities for SIGMA → subject space
- Naming Convention: Underscore-separated cohorts (e.g., `tpl-BPARat_p60_T2w.nii.gz`)
- CLI: `scripts/build_templates.py` for end-to-end template building
- Example Config: `configs/bpa_rat_example.yaml` with all parameters
- Documentation: See [`docs/MULTIMODAL_TEMPLATE_ARCHITECTURE.md`](docs/MULTIMODAL_TEMPLATE_ARCHITECTURE.md)

**✅ Phase 6 (MSME T2 Mapping) - Complete**
- Multi-echo T2 relaxometry mapping
- Myelin Water Fraction (MWF) calculation via NNLS with regularization
- T2 compartment analysis (myelin <25ms, intra/extra-cellular 25-40ms, CSF 41-2000ms)
- Comprehensive QC with T2 decay curves and NNLS spectra visualization
- Template-based normalization ready

**✅ Phase 7 (Functional fMRI Preprocessing) - Complete**
- **Slice timing correction**: Corrects temporal differences in multi-slice acquisition (Bruker interleaved)
- Volume discarding for T1 equilibration (configurable)
- MCFLIRT motion correction with selectable reference volume
- Brain extraction optimized for BOLD contrast
- Spatial smoothing with rodent-appropriate kernels (0.5mm FWHM)
- Temporal filtering (bandpass 0.01-0.1 Hz for resting-state)
- **Rodent-specific ICA denoising**: Automated noise component classification (26/30 signal retained)
- **aCompCor**: Physiological noise extraction from CSF/WM (5 components per tissue)
- Extended confound extraction (24 regressors: motion + derivatives + squares)
- **Comprehensive QC**: Motion (FD/DVARS), confounds, ICA classification, aCompCor

**Note**: Registration to anatomical/atlas spaces is under development and currently disabled.

**✅ Phase 8 (Template-Based Registration) - Complete**
- Age-specific T2w templates built (p30, p60, p90) with SIGMA registration
- Tissue probability templates (GM, WM, CSF) for all cohorts
- **Slice Correspondence System** for partial-coverage modalities (see below)
- **Subject → Template registration** with ANTs SyN
- **SIGMA atlas propagation** to each subject's T2w space using study-space atlas
- Registration QC with Dice coefficient, correlation metrics, edge overlays

**🚧 Future Work**
- MTR: Magnetization transfer ratio calculation
- Multi-echo fMRI support
- DWI/fMRI atlas propagation via T2w alignment

### Slice Correspondence for Partial-Coverage Modalities

When registering partial-coverage modalities (DWI: 11 slices, fMRI: 9 slices) to full T2w (41 slices), the slice correspondence system determines which T2w slices correspond to the partial volume.

**Challenge**: All modalities have affine origins at [0,0,0] with no header information about slice positioning.

**Solution**: Dual-approach matching for robustness:
1. **Intensity-based matching** - Correlates 2D slices using normalized cross-correlation with gradient enhancement
2. **Landmark detection** - Identifies ventricle peaks to validate/refine alignment
3. **Physical coordinate support** - Handles different slice thicknesses between modalities

```python
from neurofaune.registration import find_slice_correspondence

result = find_slice_correspondence(
    partial_image='sub-001_dwi_b0.nii.gz',
    full_image='sub-001_T2w.nii.gz',
    modality='dwi'
)
print(f"DWI slices 0-10 -> T2w slices {result.start_slice}-{result.end_slice}")
print(f"Confidence: {result.combined_confidence:.2f}")
print(f"Physical offset: {result.physical_offset:.1f} mm")
```

**QC Visualization**:
```python
from neurofaune.registration import plot_slice_correspondence

plot_slice_correspondence(
    partial_data=dwi_data,
    full_data=t2w_data,
    result=result,
    output_file='qc/slice_correspondence.png'
)
```

**✅ Bruker to BIDS Conversion - Complete**
- 141 subjects converted from 7 cohorts (Cohorts 1-5, 7-8)
- 189 sessions (54 p30, 50 p60, 79 p90, 6 unknown)
- 2,256 NIfTI files with JSON metadata sidecars
- Modalities: T2w anatomical, DTI/DWI, fMRI, MSME, MTR, FLASH, field maps
- See [`docs/BRUKER_CONVERSION_SUMMARY.md`](docs/BRUKER_CONVERSION_SUMMARY.md) for details

### Planned Preprocessing Capabilities

#### Core Modalities
- **Anatomical (T2w)**: N4 bias correction, skull stripping, ANTs registration to SIGMA atlas, tissue segmentation
- **Diffusion (DWI/DTI)**: Eddy correction, DTI fitting, advanced models (DKI, NODDI), spatial normalization
- **Functional (fMRI)**: Rodent-specific ICA denoising, aCompCor, motion correction, SIGMA registration
- **Spectroscopy (MRS)**: FSL-MRS integration, metabolite quantification, regional analysis

#### Advanced Features
- **Slice-specific Atlas Registration**: Register to relevant anatomical slices only (e.g., 11 hippocampal slices for DTI)
- **SIGMA Atlas Integration**: Rat brain atlas with modality-specific slice extraction
- **Age Cohort Support**: Developmental studies (p30, p60, p90)
- **Transform Reuse**: Compute once, reuse across modalities
- **Comprehensive QC**: Automated quality control reports for all preprocessing steps

---

## BPA-Rat Study Data

### Study Organization
The BPA-Rat study folder serves as a test dataset for neurofaune development:

- **Study root**: `/mnt/arborea/bpa-rat/`
- **Source Bruker data**: `/mnt/arborea/bruker/` (permanent archive)
- **Converted BIDS data**: `/mnt/arborea/bpa-rat/raw/bids/`
  - 141 subjects: `sub-Rat001` through `sub-Rat298`
  - Sessions: `ses-p30`, `ses-p60`, `ses-p90`
  - Modalities: `anat/`, `dwi/`, `func/`, `msme/`, `mtr/`, `flash/`, `fmap/`

### Atlas Location
- **SIGMA rat brain atlas (original)**: `/mnt/arborea/atlases/SIGMA_scaled/`
- **Study-space SIGMA atlas**: `/mnt/arborea/bpa-rat/atlas/SIGMA_study_space/`

**Note**: Bruker data in temporary Cohort folders has been deleted after successful conversion. The original data is preserved in `/mnt/arborea/bruker/`.

See [`docs/BRUKER_CONVERSION_SUMMARY.md`](docs/BRUKER_CONVERSION_SUMMARY.md) for complete conversion details.

---

## Installation

### Prerequisites

- Python 3.10+
- FSL 6.0+
- ANTs 2.3+
- (Optional) CUDA for GPU acceleration

### Install Neurofaune

```bash
# Clone repository
git clone https://github.com/yourusername/neurofaune.git
cd neurofaune

# Install with pip
pip install -e .

# Or install with optional dependencies
pip install -e ".[all]"  # Includes FSL-MRS, Bruker conversion tools
```

---

## Study Initialization

Neurofaune provides a comprehensive study initialization system that discovers raw data, creates directory structures, and generates configuration files.

### Quick Start - Initialize a New Study

```bash
# Initialize with both Bruker and BIDS data
uv run python scripts/init_study.py /path/to/study \
    --name "My Study" \
    --code mystudy \
    --bruker-root /path/to/bruker \
    --bids-root /path/to/bids \
    --sigma-atlas /path/to/SIGMA

# Initialize with just BIDS data
uv run python scripts/init_study.py /path/to/study \
    --name "My Study" \
    --code mystudy \
    --bids-root /path/to/bids
```

### What Study Initialization Does

1. **Creates directory structure**: `raw/`, `derivatives/`, `templates/`, `atlas/`, `transforms/`, `qc/`, `work/`
2. **Discovers raw Bruker data**: Scans for sessions and categorizes scans by type (RARE, DtiEpi, EPI, MSME, etc.)
3. **Discovers BIDS data**: Inventories subjects, sessions, cohorts, and modalities
4. **Generates configuration**: Creates `config.yaml` with proper paths and parameters
5. **Sets up SIGMA atlas**: Reorients atlas to match study acquisition orientation
6. **Provides next steps**: Suggests appropriate preprocessing commands based on available data

### Discover Data Only (No Directories Created)

```bash
# Discover both Bruker and BIDS data
uv run python scripts/init_study.py --discover-only /path/to/data

# Output to JSON for programmatic use
uv run python scripts/init_study.py --discover-only /path/to/data --output-json manifest.json
```

### View Study Summary

```bash
# Print summary of initialized study
uv run python scripts/init_study.py /path/to/study --summary

# List subjects by cohort
uv run python scripts/init_study.py /path/to/study --list-subjects --cohort p60
```

### Example Output

```
======================================================================
Initializing Study: BPA-Rat Study
Study Root: /mnt/arborea/bpa-rat
======================================================================

[1/6] Creating directory structure...
  Created 18 directories

[2/6] Discovering raw Bruker data...
  Found 45 sessions, 686 scans
  Scan types: {'Bruker:RARE': 230, 'Bruker:DtiEpi': 47, 'Bruker:EPI': 88, ...}

[3/6] Discovering BIDS data...
  Found 141 subjects, 189 sessions
  Cohorts: {'p30': 54, 'p60': 50, 'p90': 79}
  Modalities: {'anat': 126, 'dwi': 187, 'func': 140}

[4/6] Setting up configuration...
  Generated config: /mnt/arborea/bpa-rat/config.yaml

[5/6] Setting up study-space atlas...
  Created study-space atlas: /mnt/arborea/bpa-rat/atlas/SIGMA_study_space

[6/6] Saving study manifest...
  Saved manifest: /mnt/arborea/bpa-rat/study_manifest.json

Status: SUCCESS

Next Steps:
  1. Run anatomical preprocessing (126 sessions)
  2. Run DTI preprocessing (187 sessions)
  3. Run functional preprocessing (140 sessions)
  4. Build age-specific templates
```

---

## Study-Space SIGMA Atlas Setup

Before running any preprocessing, you must create a **study-space SIGMA atlas**. This reorients the SIGMA atlas to match your study's native acquisition orientation.

### Why is this necessary?

MRI data is acquired with scanner-specific orientations. The BPA-Rat study uses thick coronal slices (8mm) with axes:
- X = Left-Right
- Y = Inferior-Superior
- Z = Anterior-Posterior (thick coronal direction)

The SIGMA atlas uses standard neuroimaging orientation:
- X = Left-Right
- Y = Anterior-Posterior
- Z = Inferior-Superior

Instead of reorienting every image to match SIGMA (which causes interpolation artifacts in thick-slice data), we reorient SIGMA once to match the study's native space.

### Create Study-Space SIGMA Atlas

```python
from neurofaune.templates.slice_registration import reorient_sigma_to_study
from pathlib import Path

# Source SIGMA files (10x scaled version)
sigma_dir = Path('/mnt/arborea/atlases/SIGMA_scaled')
output_dir = Path('/mnt/arborea/bpa-rat/atlas/SIGMA_study_space')
output_dir.mkdir(parents=True, exist_ok=True)

# Files to reorient
files = [
    'SIGMA_Rat_Anatomical_Imaging/SIGMA_Rat_Anatomical_InVivo_Template/SIGMA_InVivo_Brain_Template_Masked.nii',
    'SIGMA_Rat_Anatomical_Imaging/SIGMA_Rat_Anatomical_InVivo_Template/SIGMA_InVivo_Brain_Mask.nii',
    'SIGMA_Rat_Anatomical_Imaging/SIGMA_Rat_Anatomical_InVivo_Template/SIGMA_InVivo_GM.nii',
    'SIGMA_Rat_Anatomical_Imaging/SIGMA_Rat_Anatomical_InVivo_Template/SIGMA_InVivo_WM.nii',
    'SIGMA_Rat_Anatomical_Imaging/SIGMA_Rat_Anatomical_InVivo_Template/SIGMA_InVivo_CSF.nii',
    'SIGMA_Rat_Brain_Atlases/SIGMA_Anatomical_Atlas/InVivo_Atlas/SIGMA_InVivo_Anatomical_Brain_Atlas.nii',
]

for f in files:
    input_path = sigma_dir / f
    output_name = Path(f).name.replace('.nii', '.nii.gz')
    is_labels = 'Atlas' in f

    reorient_sigma_to_study(
        sigma_path=input_path,
        output_path=output_dir / output_name,
        is_labels=is_labels
    )
```

### Transformation Details

The transformation applied (SIGMA → study space):
```python
data = np.transpose(data, (0, 2, 1))  # Swap Y↔Z axes
data = np.flip(data, axis=0)           # Flip X axis
data = np.flip(data, axis=1)           # Flip Y axis
```

### Study-Space Atlas Files

After running the setup, your atlas directory should contain:
```
{study_root}/atlas/SIGMA_study_space/
├── SIGMA_InVivo_Brain_Template_Masked.nii.gz  # Brain template
├── SIGMA_InVivo_Brain_Mask.nii.gz             # Brain mask
├── SIGMA_InVivo_GM.nii.gz                     # Gray matter probability
├── SIGMA_InVivo_WM.nii.gz                     # White matter probability
├── SIGMA_InVivo_CSF.nii.gz                    # CSF probability
├── SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz # Region labels
└── atlas_metadata.json                         # Transformation documentation
```

---

## Batch Processing

Process all subjects with automatic scan validation and exclusion:

```bash
# Initialize study (discovers data, creates config)
uv run python scripts/init_study.py /path/to/study \
    --name "My Study" --code mystudy \
    --bids-root /path/to/bids

# Anatomical preprocessing for all subjects
uv run python scripts/batch_preprocess_anat.py \
    --config /path/to/study/config.yaml

# DTI preprocessing for all subjects (with header validation)
uv run python scripts/batch_preprocess_dwi.py \
    --bids-root /path/to/bids \
    --output-root /path/to/study \
    --config /path/to/study/config.yaml

# DTI preprocessing with header fixing from Bruker source
uv run python scripts/batch_preprocess_dwi.py \
    --bids-root /path/to/bids \
    --output-root /path/to/study \
    --fix-headers --bruker-root /path/to/bruker

# Force reprocessing of all subjects
uv run python scripts/batch_preprocess_dwi.py \
    --bids-root /path/to/bids \
    --output-root /path/to/study \
    --force

# Build age-specific templates
uv run python scripts/build_templates.py --config /path/to/study/config.yaml
```

### DTI Batch Processing Features

- **Header validation**: Automatically detects and skips files with incorrect voxel sizes (identity affine)
- **Bruker-based fixing**: Can fix headers using original Bruker source data
- **Slice padding**: Protects edge slices from eddy correction artifacts
- **GPU support**: Uses eddy_cuda when available (falls back to CPU)
- **Dry run mode**: `--dry-run` to check files without processing

See [BATCH_PROCESSING_GUIDE.md](BATCH_PROCESSING_GUIDE.md) for details.

## Quick Start

### 1. Create Configuration

Create a `config.yaml` file for your study:

```yaml
# Study information
study:
  name: "My Rodent Study"
  code: "STUDY001"
  species: "rat"

# Directory structure
paths:
  study_root: "/path/to/study"
  derivatives: "${paths.study_root}/derivatives"
  transforms: "${paths.study_root}/transforms"
  qc: "${paths.study_root}/qc"

# Atlas configuration
atlas:
  name: "SIGMA"
  base_path: "/path/to/SIGMA"
  slice_definitions:
    dwi:
      start: 15
      end: 25  # 11 hippocampal slices

# Processing parameters
execution:
  n_procs: 6

anatomical:
  bet:
    frac: 0.3  # Rodent-optimized

diffusion:
  eddy:
    use_cuda: true
```

### 2. Run Preprocessing

```bash
# Anatomical preprocessing
neurofaune run anatomical --config config.yaml --subject sub-001

# Diffusion preprocessing
neurofaune run dwi --config config.yaml --subject sub-001

# Functional preprocessing
neurofaune run functional --config config.yaml --subject sub-001

# All modalities
neurofaune run all --config config.yaml --subject sub-001
```

### 3. Quality Control

```bash
# Generate QC reports
neurofaune qc generate --config config.yaml --subject sub-001

# View QC in browser
open qc/sub-001/anat/skull_strip_qc.html
```

---

## Directory Structure

Neurofaune uses a standardized directory hierarchy compatible with neurovrai:

```
study_root/
├── raw/
│   ├── bruker/              # Raw Bruker data
│   └── bids/                # Converted BIDS data
│       └── sub-001/
│           ├── anat/
│           ├── dwi/
│           └── func/
├── atlas/                   # Study-space atlas (REQUIRED - see Study Setup)
│   └── SIGMA_study_space/
│       ├── SIGMA_InVivo_Brain_Template_Masked.nii.gz
│       ├── SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz
│       └── ...
├── derivatives/             # Preprocessed outputs
│   └── sub-001/
│       ├── anat/
│       │   ├── sub-001_desc-preproc_T2w.nii.gz
│       │   ├── sub-001_desc-brain_mask.nii.gz
│       │   ├── sub-001_dseg.nii.gz                    # Tissue segmentation
│       │   ├── sub-001_label-GM_probseg.nii.gz       # GM probability
│       │   ├── sub-001_label-WM_probseg.nii.gz       # WM probability
│       │   ├── sub-001_label-CSF_probseg.nii.gz      # CSF probability
│       │   └── sub-001_space-SIGMA_T2w.nii.gz
│       ├── dwi/
│       │   ├── sub-001_desc-preproc_dwi.nii.gz
│       │   ├── sub-001_FA.nii.gz
│       │   └── sub-001_space-SIGMA_FA.nii.gz
│       └── func/
│           └── sub-001_desc-preproc_bold.nii.gz
├── templates/               # Age-specific templates (organized by modality)
│   ├── anat/                # T2w anatomical templates
│   │   ├── p30/
│   │   │   ├── tpl-BPARat_p30_T2w.nii.gz
│   │   │   ├── tpl-BPARat_p30_space-SIGMA_T2w.nii.gz
│   │   │   ├── tpl-BPARat_p30_label-GM_probseg.nii.gz
│   │   │   ├── tpl-BPARat_p30_label-WM_probseg.nii.gz
│   │   │   ├── tpl-BPARat_p30_label-CSF_probseg.nii.gz
│   │   │   └── transforms/
│   │   │       ├── tpl-to-SIGMA_Composite.h5
│   │   │       └── SIGMA-to-tpl_Composite.h5
│   │   ├── p60/ (same structure)
│   │   └── p90/ (same structure)
│   ├── dwi/                 # FA templates (future)
│   └── func/                # BOLD templates (future)
├── transforms/              # Transform registry
│   └── sub-001/
│       ├── T2w_to_template_Composite.h5
│       ├── FA_to_template_Composite.h5
│       └── transforms.json
├── qc/                      # Quality control reports
│   └── sub-001/
│       ├── anat/
│       ├── dwi/
│       └── func/
└── work/                    # Temporary files
    └── sub-001/
```

---

## Architecture

Neurofaune follows a modular, four-part architecture:

### 1. Preprocessing (`neurofaune.preprocess`)
- Subject-level preprocessing for all modalities
- Workflows: `anatomical`, `diffusion`, `functional`, `spectroscopy`
- Utilities: Registration, normalization, file discovery
- QC: Automated quality control for each modality

### 2. Atlas Management (`neurofaune.atlas`)
- SIGMA rat brain atlas integration
- Slice-specific extraction for different modalities
- Atlas-aware registration with ANTs
- Registration quality metrics

### 3. Analysis (Planned)
- Group-level statistical analyses
- ROI-based analyses
- Connectivity metrics

### 4. Connectome (Planned)
- Network neuroscience
- Structural and functional connectivity
- Graph theory metrics

---

## Key Differences from Neurovrai

| Feature | Neurovrai (Human) | Neurofaune (Rodent) |
|---------|-------------------|---------------------|
| **Primary modality** | T1w | T2w |
| **Atlas** | MNI152 | SIGMA (rat brain) |
| **Atlas strategy** | Full template | Slice-specific extraction |
| **Registration** | ANTs | ANTs (slice-aware) |
| **Skull stripping** | BET (default) | BET (rodent-optimized) |
| **Input format** | DICOM | Bruker / DICOM |
| **Age cohorts** | Single | Developmental (p30/p60/p90) |
| **Spectroscopy** | Not included | FSL-MRS integration |

---

## Configuration

Neurofaune uses YAML configuration files with:
- **Variable substitution**: `${paths.study_root}` references other config values
- **Environment variables**: `${HOME}` expands to environment variables
- **Nested parameters**: Hierarchical organization of preprocessing parameters
- **Default values**: Sensible defaults for rodent brains

See `configs/default.yaml` for all available parameters.

---

## Transform Registry

The transform registry avoids redundant computation by storing all spatial transformations:

```python
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.config import load_config

config = load_config("config.yaml")
registry = create_transform_registry(config, "sub-001", cohort="p60")

# Save transform after computing
registry.save_ants_composite_transform(
    composite_file=Path("T2w_to_SIGMA_Composite.h5"),
    source_space="T2w",
    target_space="SIGMA"
)

# Reuse in other workflows
transform = registry.get_ants_composite_transform("T2w", "SIGMA")
if transform:
    # Apply to functional data without recomputing
    pass
```

---

## Contributing

Contributions are welcome! Please see `CLAUDE.md` for development guidelines.

---

## Citation

If you use Neurofaune in your research, please cite:

```
[Citation to be added]
```

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- Built on the neurovrai architecture
- Preprocessing patterns adapted from rat-mri-preprocess
- SIGMA rat brain atlas ([Barrière et al., 2019](https://doi.org/10.1016/j.neuroimage.2019.06.063))
- FSL-MRS for spectroscopy ([Clarke et al., 2021](https://doi.org/10.1002/mrm.28630))

---

## Contact

For questions or issues, please open an issue on GitHub.
