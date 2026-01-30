# Neurofaune

Rodent-specific MRI preprocessing and analysis pipeline built on ANTs and FSL. Designed for multi-modal rat brain imaging with age-cohort support and standardized normalization to the SIGMA rat brain atlas.

---

## Prerequisites

- Python 3.10+
- FSL 6.0+
- ANTs 2.3+
- CUDA (optional, for GPU-accelerated eddy correction)

```bash
git clone https://github.com/yourusername/neurofaune.git
cd neurofaune
uv pip install -e ".[dev]"
```

---

## Workflow

Neurofaune processing follows a strict order. Anatomical preprocessing must complete first because all other modalities register through T2w to reach SIGMA atlas space.

```
1. Initialize Study
2. Bruker → BIDS Conversion
3. Anatomical Preprocessing (template building, then full preprocessing)
4. Other Modalities (DTI, fMRI, MSME — require T2w transforms)
5. Group Analysis (all subjects warped to SIGMA space)
```

### Normalization Strategy

All analysis is performed in SIGMA atlas space. Subject data is warped **to** SIGMA (not labels to subjects):

```
Subject native → Subject T2w → Cohort Template → SIGMA Atlas
                    Affine/Rigid      SyN              SyN
```

SIGMA atlas labels (183 regions) are applied directly in SIGMA space after warping. This avoids interpolation artifacts from warping discrete labels through multiple non-linear transforms.

---

## Step 1: Initialize Study

Creates directory structure, discovers data, generates configuration, and sets up the study-space SIGMA atlas.

```bash
uv run python scripts/init_study.py /path/to/study \
    --name "My Study" --code mystudy \
    --bids-root /path/to/bids \
    --sigma-atlas /path/to/SIGMA_scaled
```

This creates:
```
{study_root}/
├── config.yaml                  # Study configuration
├── atlas/SIGMA_study_space/     # SIGMA reoriented to study acquisition orientation
├── raw/bids/                    # Symlink or copy of BIDS data
├── derivatives/                 # Preprocessed outputs (per subject/session)
├── templates/                   # Age-specific templates
├── transforms/                  # Cross-modal transforms
├── qc/                          # Quality control reports
└── work/                        # Temporary files (deletable)
```

**Study-space SIGMA atlas**: The SIGMA atlas is reoriented once to match the study's native acquisition orientation (thick coronal slices). This avoids resampling every image to atlas orientation.

---

## Step 2: Bruker to BIDS Conversion

Convert raw Bruker ParaVision data to BIDS-formatted NIfTI:

```bash
uv run python scripts/convert_bruker_to_bids.py \
    --bruker-root /path/to/bruker \
    --output-root /path/to/study/raw/bids
```

Produces standard BIDS layout:
```
raw/bids/sub-Rat001/ses-p60/
├── anat/sub-Rat001_ses-p60_T2w.nii.gz
├── dwi/sub-Rat001_ses-p60_dwi.nii.gz
├── func/sub-Rat001_ses-p60_bold.nii.gz
└── msme/sub-Rat001_ses-p60_MSME.nii.gz
```

### Handling 3D Isotropic T2w Acquisitions

Some studies may have subjects with **3D isotropic RARE** acquisitions instead of the standard 2D multi-slice T2w. This is common in earlier cohorts where acquisition protocols evolved over time.

| Acquisition | Typical Geometry | Resolution |
|-------------|-----------------|------------|
| 2D multi-slice | 256×256×41 | 0.125×0.125×0.8mm |
| 3D isotropic | 256×140×110 | 0.2×0.2×0.2mm |

The standard BIDS converter may not recognize 3D RARE as T2w. Use the dedicated script to convert these:

```bash
# Dry run to see what would be converted
uv run python scripts/convert_3d_rare_to_bids.py \
    --bruker-root /path/to/bruker \
    --bids-root /path/to/study/raw/bids \
    --dry-run

# Run the conversion
uv run python scripts/convert_3d_rare_to_bids.py \
    --bruker-root /path/to/bruker \
    --bids-root /path/to/study/raw/bids
```

This creates T2w files with the `acq-3D` label to distinguish them:
```
raw/bids/sub-Rat001/ses-p60/anat/sub-Rat001_ses-p60_acq-3D_run-6_T2w.nii.gz
```

The 3D acquisitions can then be processed through the standard anatomical pipeline. ANTs registration handles the different voxel sizes when warping to the cohort template.

---

## Step 3: Anatomical Preprocessing

Anatomical T2w preprocessing must happen first because it produces the T2w-to-Template transforms that all other modalities chain through.

### 3a: Build Age-Specific Templates

Select a subset of subjects (top quality) and build cohort templates:

```bash
# Preprocess a subset for template building (20% of subjects)
uv run python scripts/batch_preprocess_for_templates.py \
    --bids-root /path/to/bids \
    --output-root /path/to/study \
    --cohorts p30 p60 p90 \
    --fraction 0.20

# Build templates and register to SIGMA
uv run python scripts/build_templates.py \
    --config /path/to/study/config.yaml \
    --cohorts p30 p60 p90
```

This creates age-specific T2w templates and computes Template-to-SIGMA transforms:
```
templates/anat/p90/
├── tpl-BPARat_p90_T2w.nii.gz              # Template
├── tpl-BPARat_p90_label-{GM,WM,CSF}_probseg.nii.gz  # Tissue priors
└── transforms/
    ├── tpl-to-SIGMA_0GenericAffine.mat     # Template → SIGMA affine
    └── tpl-to-SIGMA_1Warp.nii.gz           # Template → SIGMA warp
```

### 3b: Preprocess All Subjects

Run full anatomical preprocessing with registration to template:

```bash
uv run python scripts/batch_preprocess_anat.py \
    --config /path/to/study/config.yaml
```

Per-subject pipeline:
1. N4 bias field correction
2. Skull stripping (two-pass Atropos+BET for full-coverage T2w)
3. Tissue segmentation (GM, WM, CSF from Atropos posteriors)
4. Register T2w to age-matched template (ANTs SyN)

Outputs:
```
derivatives/sub-Rat001/ses-p60/anat/
├── sub-Rat001_ses-p60_desc-preproc_T2w.nii.gz
├── sub-Rat001_ses-p60_desc-brain_mask.nii.gz
├── sub-Rat001_ses-p60_label-{GM,WM,CSF}_probseg.nii.gz
└── sub-Rat001_ses-p60_dseg.nii.gz

transforms/sub-Rat001/ses-p60/
├── sub-Rat001_ses-p60_T2w_to_template_0GenericAffine.mat
└── sub-Rat001_ses-p60_T2w_to_template_1Warp.nii.gz
```

---

## Step 4: Other Modalities

These require completed anatomical preprocessing (T2w transforms must exist).

### DTI Preprocessing

```bash
uv run python scripts/batch_preprocess_dwi.py \
    --bids-root /path/to/bids \
    --output-root /path/to/study \
    --config /path/to/study/config.yaml
```

Pipeline:
1. 5D to 4D conversion (Bruker multi-average)
2. Intensity normalization (handles Bruker reconstruction differences)
3. Skull stripping (two-pass Atropos+BET for 11-slice coverage)
4. Eddy correction with slice padding (GPU when available)
5. DTI fitting (FA, MD, AD, RD)
6. QC reports
7. **FA to T2w registration** (ANTs affine)

The FA-to-T2w transform completes the chain to SIGMA:
```
FA → T2w (affine) → Template (SyN) → SIGMA (SyN)
```

### Functional (fMRI) Preprocessing

```bash
uv run python scripts/batch_preprocess_func.py \
    --bids-root /path/to/bids \
    --output-root /path/to/study
```

Pipeline:
1. Volume discarding (T1 equilibration)
2. Slice timing correction (optional)
3. Skull stripping (adaptive slice-wise BET for 9-slice partial coverage)
4. Motion correction (MCFLIRT)
5. ICA denoising (MELODIC with rodent-specific classification)
6. Spatial smoothing
7. Temporal filtering (0.01-0.1 Hz bandpass)
8. Confound extraction (24 motion regressors)
9. aCompCor (CSF/WM physiological noise, optional)
10. QC reports
11. **BOLD to T2w registration** (NCC Z-initialization + rigid)

BOLD registration handles partial-coverage data (9 slices out of 41 T2w slices):
- NCC scan finds optimal Z position (both images have origin at 0,0,0)
- Rigid-only transform (6 DOF — affine over-fits on 9 slices)
- Conservative shrink factors (4x2x1x1) to preserve Z information

### MSME T2 Mapping

```bash
uv run python scripts/batch_preprocess_msme.py \
    --bids-root /path/to/bids \
    --output-root /path/to/study
```

Pipeline:
1. Skull stripping (adaptive slice-wise BET for 5-slice partial coverage)
2. Multi-echo T2 fitting
3. Myelin Water Fraction (MWF) via NNLS
4. T2 compartment analysis (myelin, intra/extracellular, CSF)
5. **MSME to T2w registration** (NCC Z-initialization + rigid)
6. QC reports

---

## Step 5: Group Analysis

Once subjects are preprocessed and registered, warp data to SIGMA space for group-level analysis.

### DTI Voxel-Based Analysis

```bash
# Prepare: warp FA/MD/AD/RD to SIGMA, create WM analysis mask
uv run python -m neurofaune.analysis.tbss.prepare_tbss \
    --config config.yaml \
    --output-dir /study/analysis/tbss/ \
    --cohorts p90 \
    --analysis-threshold 0.3

# Run statistics (requires design matrices from neuroaider)
uv run python -m neurofaune.analysis.tbss.run_tbss_stats \
    --tbss-dir /study/analysis/tbss/ \
    --design-dir /study/designs/model1/ \
    --analysis-name dose_response
```

Uses WM-masked voxel-based analysis (not skeleton-based TBSS):
- Rodent WM tracts are only 1-3 voxels wide, making skeletonization inappropriate
- Analysis mask: interior WM (tissue probability + erosion) intersected with FA >= 0.3
- FSL randomise with TFCE correction
- Cluster labeling via SIGMA atlas parcellation

### Functional Normalization

```bash
# Warp preprocessed BOLD to SIGMA space
uv run python scripts/batch_warp_bold_to_sigma.py \
    --study-root /path/to/study
```

Chains BOLD→T2w→Template→SIGMA transforms using `antsApplyTransforms -e 3` for 4D timeseries. Produces `space-SIGMA_bold.nii.gz` for group ICA or connectivity analysis.

---

## Quality Control

QC is integrated into all preprocessing workflows and can also be run retroactively:

```bash
# Generate batch QC summary
uv run python scripts/generate_batch_qc.py /path/to/study --modality dwi --slice-qc
uv run python scripts/generate_batch_qc.py /path/to/study --modality anat
uv run python scripts/generate_batch_qc.py /path/to/study --modality func

# Retroactive QC for already-preprocessed data
uv run python scripts/generate_anat_qc_retroactive.py /path/to/study
uv run python scripts/generate_func_qc_retroactive.py /path/to/study
```

QC outputs:
```
qc/
├── sub/{subject}/{session}/{modality}/  # Per-subject metrics + figures
├── dwi_batch_summary/                   # Batch aggregation + exclusion lists
├── anat_batch_summary/
└── func_batch_summary/
```

Features:
- Subject-level exclusion lists (with categorized reasons)
- Slice-level QC for DTI (identifies bad slices per subject)
- Cohort-specific exclusion lists
- Visual heatmaps and distribution plots

---

## Configuration

YAML-based configuration with variable substitution:

```yaml
study:
  name: "BPA-Rat Study"
  species: "rat"

paths:
  study_root: "/mnt/arborea/bpa-rat"
  derivatives: "${paths.study_root}/derivatives"

atlas:
  name: "SIGMA"
  base_path: "/mnt/arborea/atlases/SIGMA_scaled"

execution:
  n_procs: 6

anatomical:
  bet:
    frac: 0.3

diffusion:
  eddy:
    use_cuda: true
  intensity_normalization:
    enabled: true
    target_max: 10000
```

See `configs/default.yaml` for all parameters and `configs/bpa_rat_example.yaml` for a complete example.

---

## Architecture

```
neurofaune/
├── config.py                        # YAML config with variable substitution
├── atlas/                           # SIGMA atlas management
├── preprocess/
│   ├── workflows/                   # Per-modality pipelines
│   │   ├── anat_preprocess.py       # T2w: N4, skull strip, segment, register
│   │   ├── dwi_preprocess.py        # DTI: eddy, tensor fit, FA→T2w
│   │   ├── func_preprocess.py       # fMRI: motion, ICA, filter, BOLD→T2w
│   │   └── msme_preprocess.py       # MSME: T2 mapping, MWF, MSME→T2w
│   ├── qc/                          # Quality control (per modality)
│   └── utils/
│       └── skull_strip.py           # Unified skull stripping dispatcher
├── templates/                       # Template building and registration
│   ├── builder.py                   # ANTs template construction
│   ├── registration.py              # Subject→template, atlas propagation
│   └── slice_registration.py        # Study-space atlas setup
├── analysis/                        # Group-level analysis
│   ├── tbss/                        # WM-masked voxel-based analysis
│   └── stats/                       # FSL randomise, cluster reports
├── registration/                    # Cross-modal registration utilities
└── utils/                           # Transforms, exclusions, orientation
```

Key design decisions:
- **T2w is the primary anatomical modality** (better rodent brain contrast than T1w)
- **ANTs for all registrations** (better quality than FSL for rodent brains)
- **10x voxel scaling** for FSL/ANTs compatibility (sub-mm rodent voxels)
- **Age cohorts** (p30, p60, p90) with cohort-specific templates
- **All normalization goes to SIGMA space** (subject data warped up, labels applied there)
- **Unified skull stripping** with automatic method selection based on image geometry

---

## Skull Stripping

Neurofaune provides a unified skull stripping interface that automatically selects the optimal method based on image geometry. This is critical for rodent MRI where different modalities have vastly different slice coverage.

### The Challenge

Rodent MRI acquisitions vary significantly in slice coverage:

| Modality | Typical Slices | Coverage |
|----------|---------------|----------|
| T2w anatomical | 41 | Full brain |
| DTI diffusion | 11 | Hippocampus-focused |
| BOLD functional | 9 | Partial brain |
| MSME T2 mapping | 5 | Very partial |

Standard 3D skull stripping (BET, ANTs) assumes a roughly spherical brain geometry. This fails catastrophically on partial-coverage data like BOLD (9 slices) or MSME (5 slices) where the volume is essentially a flat slab.

### Unified Dispatcher

Neurofaune's `skull_strip()` function automatically selects the appropriate method:

```python
from neurofaune.preprocess.utils.skull_strip import skull_strip

# Auto-selects method based on slice count
brain, mask, info = skull_strip(
    input_file=image_path,
    output_file=brain_path,
    mask_file=mask_path,
    work_dir=work_dir,
    method='auto',  # <10 slices → adaptive, ≥10 → atropos_bet
)
```

### Methods

**Adaptive Slice-wise BET** (for <10 slices: BOLD, MSME)
- Processes each 2D slice independently
- Iterative frac optimization to achieve target ~15% brain extraction per slice
- Configurable center-of-gravity offset for off-center brain positioning
- Handles the flat-slab geometry where 3D methods fail

**Two-pass Atropos+BET** (for ≥10 slices: T2w, DTI)
- Pass 1: Atropos 5-component segmentation provides rough brain mask and tissue posteriors
- Pass 2: BET refinement using Atropos center-of-gravity for initialization
- Adaptive frac calculation based on image contrast
- Returns posteriors for tissue segmentation reuse (GM, WM, CSF)

### Configuration

Skull stripping parameters can be configured per modality in `config.yaml`:

```yaml
functional:
  skull_strip_adaptive:
    target_ratio: 0.15      # Target brain extraction per slice
    frac_range: [0.30, 0.90]
    frac_step: 0.05

msme:
  skull_strip:
    target_ratio: 0.15
    frac_min: 0.30
    frac_max: 0.80
    cog_offset_x: 0         # X offset for COG estimation
    cog_offset_y: -40       # Y offset (negative = inferior)
```

### Why This Matters

Without geometry-aware skull stripping:
- 3D BET on 5-slice MSME data produces empty or nonsensical masks
- Registration fails because the brain boundary is undefined
- Downstream analysis (T2 mapping, connectivity) becomes unreliable

With the unified dispatcher:
- MSME (5 slices) → adaptive method → 14-16% extraction → successful registration
- BOLD (9 slices) → adaptive method → consistent masks across protocols
- DTI (11 slices) → atropos_bet → robust extraction with tissue posteriors
- T2w (41 slices) → atropos_bet → high-quality masks for template building

---

## Testing

```bash
uv run pytest                                    # All tests
uv run pytest tests/unit/ -v                     # Unit tests
uv run pytest -m "not slow"                      # Skip slow tests
uv run pytest --cov=neurofaune --cov-report=term-missing  # Coverage
```

Tests use synthetic data generation (no external data required). Integration tests (`@pytest.mark.integration`) require FSL/ANTs installed.

---

## Acknowledgments

- SIGMA rat brain atlas ([Barriere et al., 2019](https://doi.org/10.1016/j.neuroimage.2019.06.063))
- Built on ANTs, FSL, and Nipype
- Architecture adapted from neurovrai
