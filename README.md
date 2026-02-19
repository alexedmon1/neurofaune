# Neurofaune

Rodent MRI preprocessing and analysis pipeline built on ANTs and FSL. Handles multi-modal rat brain imaging with age-cohort templates and standardized normalization to the SIGMA rat brain atlas.

## Prerequisites

- Python 3.10+
- [FSL 6.0+](https://fsl.fmrib.ox.ac.uk/fsl/)
- [ANTs 2.3+](https://github.com/ANTsX/ANTs)
- CUDA (optional, for GPU-accelerated eddy correction)

```bash
git clone https://github.com/alexedmon1/neurofaune.git
cd neurofaune
uv pip install -e ".[dev]"
```

## Workflow Overview

Processing follows a strict order. Anatomical preprocessing must complete first because all other modalities register through T2w to reach SIGMA atlas space.

```
1. Initialize Study        → directory structure, config, study-space atlas
2. Bruker → BIDS           → convert raw scanner data to standard format
3. Anatomical (T2w)        → N4, skull strip, segment, build templates, register
4. Other Modalities        → DTI, fMRI, MSME (all require T2w transforms)
5. Group Analysis          → TBSS, ROI extraction, classification, connectivity
```

### Normalization Strategy

All analysis is performed in SIGMA atlas space. Subject data is warped **to** SIGMA (not labels to subjects):

```
Subject native → Subject T2w → Cohort Template → SIGMA Atlas
                    Affine/Rigid      SyN              SyN
```

## Quick Start

### 1. Initialize a Study

```bash
uv run python scripts/init_study.py /path/to/study \
    --name "My Study" --code mystudy \
    --bids-root /path/to/bids \
    --sigma-atlas /path/to/SIGMA_scaled \
    --validate-workflows
```

This creates the directory structure, generates `config.yaml` with all preprocessing parameters, sets up the study-space SIGMA atlas, and discovers available BIDS data:

```
{study_root}/
├── config.yaml                  # Study configuration (all preprocessing params)
├── atlas/SIGMA_study_space/     # SIGMA reoriented to study acquisition orientation
├── raw/bids/                    # BIDS data
├── derivatives/                 # Preprocessed outputs (per subject/session)
├── templates/                   # Age-specific templates
├── transforms/                  # Cross-modal transforms
├── qc/                          # Quality control reports
└── work/                        # Temporary files (deletable)
```

### 2. Convert Bruker Data

```bash
uv run python scripts/convert_bruker_to_bids.py \
    --bruker-root /path/to/bruker \
    --output-root /path/to/study/raw/bids
```

### 3. Build Templates and Preprocess

```bash
# Build age-specific templates (subset of best subjects)
uv run python scripts/batch_preprocess_for_templates.py \
    --bids-root /path/to/bids --output-root /path/to/study --cohorts p30 p60 p90

uv run python scripts/build_templates.py --config config.yaml --cohorts p30 p60 p90

# Preprocess all subjects
uv run python scripts/batch_preprocess_anat.py --config config.yaml
uv run python scripts/batch_preprocess_dwi.py --config config.yaml --bids-root /path/to/bids --output-root /path/to/study
uv run python scripts/batch_preprocess_func.py /path/to/bids /path/to/study
uv run python scripts/batch_preprocess_msme.py /path/to/bids /path/to/study
```

### 4. Resting-State Analysis

Individual focused scripts for each resting-state metric (run after functional preprocessing):

```bash
# fALFF (fractional ALFF) — from unfiltered regressed BOLD
uv run python scripts/batch_falff_analysis.py --config config.yaml --n-workers 6

# ReHo (Regional Homogeneity) — from bandpass-filtered BOLD
uv run python scripts/batch_reho_analysis.py --config config.yaml --n-workers 6

# Functional Connectivity — ROI-to-ROI in SIGMA space
uv run python scripts/batch_fc_analysis.py --config config.yaml --n-workers 6
```

All three support `--dry-run`, `--subjects sub-Rat49 sub-Rat50`, `--force`, and `--skip-sigma` (fALFF/ReHo only). Each script handles BOLD-to-template registration on-the-fly if the transform is missing.

### 5. Group Analysis

```bash
# TBSS voxel-wise analysis (WM skeleton, 2D TFCE)
uv run python -m neurofaune.analysis.tbss.prepare_tbss --config config.yaml --output-dir /study/analysis/tbss/
uv run python scripts/run_tbss_analysis.py --tbss-dir /study/analysis/tbss --config config.yaml

# Whole-brain voxelwise fMRI analysis (fALFF/ReHo, 3D TFCE)
uv run python scripts/prepare_fmri_voxelwise.py \
    --study-root /path/to/study \
    --output-dir /path/to/study/analysis/voxelwise_fmri
uv run python scripts/prepare_tbss_designs.py \
    --study-tracker /path/to/tracker.csv \
    --tbss-dir /path/to/study/analysis/voxelwise_fmri \
    --output-dir /path/to/study/analysis/voxelwise_fmri/designs
uv run python scripts/prepare_tbss_dose_response_designs.py \
    --study-tracker /path/to/tracker.csv \
    --tbss-dir /path/to/study/analysis/voxelwise_fmri \
    --output-dir /path/to/study/analysis/voxelwise_fmri/designs
uv run python scripts/run_voxelwise_fmri_analysis.py \
    --analysis-dir /path/to/study/analysis/voxelwise_fmri \
    --config config.yaml

# ROI extraction, classification, connectivity — see scripts/ directory
```

Both design scripts (`prepare_tbss_designs.py` and `prepare_tbss_dose_response_designs.py`) automatically pre-create per-analysis 4D subset volumes after generating designs. This avoids large memory spikes at randomise runtime by loading each master 4D volume once and subsetting sequentially (~2.6 GB peak instead of ~25 GB when parallelizing). Use `--skip-subset` to skip this step if subsets already exist.

## Configuration System

Neurofaune uses a two-layer YAML configuration system:

- **`configs/default.yaml`** — Package defaults shipped with neurofaune (never edit per-study)
- **`{study_root}/config.yaml`** — Study-specific overrides generated by `init_study.py`

At runtime, `load_config()` merges defaults with study overrides and resolves `${variable}` references:

```python
from neurofaune.config import load_config, get_config_value

config = load_config(Path('config.yaml'))
n_classes = get_config_value(config, 'anatomical.skull_strip.n_classes', default=5)
```

### Variable Substitution

Config values can reference other config keys using `${section.key}` syntax, with chained references resolved automatically:

```yaml
paths:
  study_root: "/mnt/data/my-study"
  derivatives: "${paths.study_root}/derivatives"

atlas:
  study_space:
    base_path: "${paths.study_root}/atlas/SIGMA_study_space"
    template: "${atlas.study_space.base_path}/SIGMA_InVivo_Brain_Template.nii.gz"
```

### Per-Modality Configuration

All preprocessing parameters are configurable. Workflows read from config with sensible defaults — existing behavior is unchanged unless you override a value.

**Anatomical T2w** (`anatomical.*`):

```yaml
anatomical:
  skull_strip:
    method: "atropos_bet"          # 'atropos_bet', 'atropos', 'bet', or 'auto'
    n_classes: 5                   # Atropos tissue classes
    atropos_iterations: 5
    atropos_convergence: 0.0
    mrf_smoothing_factor: 0.1
    mrf_radius: [1, 1, 1]
    tissue_confidence_threshold: 0.35
    adaptive_bet:
      cnr_thresholds: [1.5, 3.0]
      frac_mapping: [0.20, 0.28, 0.38]
  n4:
    iterations: [50, 50, 30, 20]
    shrink_factor: 3
    convergence_threshold: 1.0e-6
  intensity_normalization:
    factor: 1000.0
  registration:
    smoothing_sigmas: [[3, 2, 1, 0], [2, 1, 0]]
    shrink_factors: [[8, 4, 2, 1], [4, 2, 1]]
    iterations: [[1000, 500, 250, 100], [100, 70, 50, 20]]
    syn_params: [0.1, 3.0, 0.0]
    metric_bins: 32
```

**Diffusion DWI** (`diffusion.*`):

```yaml
diffusion:
  skull_strip:
    method: "atropos_bet"
    n_classes: 3                   # 3-class: brightest = brain (for b0 images)
  eddy:
    phase_encoding_direction: "0 -1 0"
    readout_time: 0.05
    repol: true
    data_is_shelled: true
    slice_padding: 2
```

**MSME T2 Mapping** (`msme.*`):

```yaml
msme:
  skull_strip:
    method: "atropos_bet"
    n_classes: 3
  t2_fitting:
    n_components: 120              # NNLS spectrum components
    t2_range: [10, 2000]           # T2 distribution range (ms)
    lambda_reg: 0.5                # Tikhonov regularization
    myelin_water_cutoff: 25        # T2 cutoff for myelin water (ms)
    intra_extra_cutoff: 40         # Intra/extra-cellular boundary (ms)
```

**Functional fMRI** (`functional.*`):

```yaml
functional:
  skull_strip_adaptive:
    target_ratio: 0.15
    frac_range: [0.30, 0.90]
    frac_step: 0.05
  motion_qc:
    fd_threshold: 0.5
```

See `configs/default.yaml` for all parameters and `configs/bpa_rat_example.yaml` for a complete study example.

### Config Validation

Validate that all required parameters are present before running preprocessing:

```bash
# Validate during initialization
uv run python scripts/init_study.py /path/to/study --name "My Study" --code mystudy --validate-workflows

# Validate programmatically
uv run python -c "
from neurofaune.config import load_config
from neurofaune.config_validator import validate_all_workflows
config = load_config('config.yaml')
validate_all_workflows(config)
"
```

## Skull Stripping

Neurofaune automatically selects the skull stripping method based on image geometry. This is critical because rodent MRI modalities have vastly different slice coverage:

| Modality | Slices | Method | Strategy |
|----------|--------|--------|----------|
| T2w anatomical | 41 | `atropos_bet` (5-class) | Middle 3 classes by volume = brain |
| DTI diffusion | 11 | `atropos_bet` (3-class) | Brightest class = brain |
| BOLD functional | 9 | `adaptive` | Per-slice BET with iterative frac |
| MSME T2 mapping | 5 | `atropos_bet` (3-class) | Brightest class = brain |

The threshold between methods is 10 slices. Standard 3D BET fails on partial-coverage data (BOLD, MSME) where the volume is essentially a flat slab.

```python
from neurofaune.preprocess.utils.skull_strip import skull_strip

brain, mask, info = skull_strip(
    input_file=image_path,
    output_file=brain_path,
    mask_file=mask_path,
    work_dir=work_dir,
    method='auto',  # selects based on slice count
)
```

All skull stripping parameters are configurable per modality in `config.yaml`.

## Preprocessing Pipelines

### Anatomical (T2w)

N4 bias correction, two-pass Atropos+BET skull stripping, tissue segmentation (GM/WM/CSF), optional 3D-to-2D resampling, registration to age-matched template (ANTs SyN). 3D isotropic acquisitions are automatically detected and resampled to standard 2D geometry.

### DTI

5D-to-4D conversion, intensity normalization, skull stripping, GPU-accelerated eddy correction with slice padding, DTI tensor fitting (FA, MD, AD, RD), FA-to-T2w registration (ANTs affine).

### Functional (fMRI)

Volume discarding, adaptive skull stripping, motion correction (MCFLIRT), ICA denoising (MELODIC), spatial smoothing, temporal bandpass filtering, confound extraction (24 motion + aCompCor), BOLD-to-T2w registration (NCC Z-init + rigid).

### MSME T2 Mapping

Skull stripping, NNLS-based T2 fitting, Myelin Water Fraction (MWF) and compartment analysis, MSME-to-T2w registration.

## Group Analysis

Neurofaune includes several group-level analysis tools, all operating in SIGMA atlas space:

- **TBSS** — WM-skeleton voxel-wise analysis for DTI and MSME metrics (FSL randomise + 2D TFCE)
- **Voxelwise fMRI** — Whole-brain voxel-wise analysis for fALFF and ReHo (FSL randomise + 3D TFCE)
- **ROI Extraction** — Mean metrics per SIGMA atlas region (234 regions, 11 territories)
- **CovNet** — Covariance network analysis (correlation matrices, NBS, graph metrics)
- **Classification** — PERMANOVA, PCA, LDA, SVM/logistic regression with LOOCV
- **Regression** — Dose-response with SVR, Ridge, PLS
- **MVPA** — Whole-brain decoding and searchlight mapping

See `scripts/` for runner scripts and `neurofaune/analysis/` for library code.

## Architecture

```
neurofaune/
├── config.py                        # YAML config with variable substitution
├── config_validator.py              # Per-modality config validation
├── study_initialization.py          # Study setup, BIDS discovery, config generation
├── atlas/                           # SIGMA atlas management + slice extraction
├── preprocess/
│   ├── workflows/                   # Per-modality pipelines
│   │   ├── anat_preprocess.py       # T2w: N4, skull strip, segment, register
│   │   ├── dwi_preprocess.py        # DTI: eddy, tensor fit, FA→T2w
│   │   ├── func_preprocess.py       # fMRI: motion, ICA, filter, BOLD→T2w
│   │   ├── msme_preprocess.py       # MSME: T2 mapping, MWF, MSME→T2w
│   │   └── bruker_session.py        # Single-session orchestrator
│   ├── qc/                          # Quality control (per modality)
│   └── utils/
│       └── skull_strip.py           # Unified skull stripping dispatcher
├── templates/                       # Template building and registration
├── analysis/                        # Group-level analysis (TBSS, ROI, CovNet, etc.)
├── registration/                    # Cross-modal registration utilities
└── utils/                           # Transforms, exclusions, orientation
```

Key design decisions:
- **T2w is the primary anatomical modality** (better rodent brain contrast than T1w)
- **ANTs for all registrations** (better quality than FSL for rodent brains)
- **10x voxel scaling** for FSL/ANTs compatibility (sub-mm rodent voxels)
- **Age cohorts** (p30, p60, p90) with cohort-specific templates
- **Config-driven** — all preprocessing parameters configurable via YAML, validated per modality

## Testing

```bash
uv run pytest                                    # All tests
uv run pytest tests/unit/ -v                     # Unit tests
uv run pytest --cov=neurofaune --cov-report=term-missing  # Coverage
```

Tests use synthetic data generation (no external data required). Integration tests (`@pytest.mark.integration`) require FSL/ANTs.

## Acknowledgments

- [SIGMA rat brain atlas](https://doi.org/10.1016/j.neuroimage.2019.06.063) (Barriere et al., 2019)
- Built on [ANTs](https://github.com/ANTsX/ANTs), [FSL](https://fsl.fmrib.ox.ac.uk/fsl/), and [Nipype](https://nipype.readthedocs.io/)
