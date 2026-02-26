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
5. Connectome              → ROI extraction, FC matrices, covariance networks
6. Analysis                → TBSS, classification, regression, MVPA
7. Reporting               → Unified dashboard across all analysis types
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

---

## Preprocess

All preprocessing scripts live in `scripts/` and use library code from `neurofaune/preprocess/`. Each modality has a batch script that discovers BIDS data and processes all subjects.

### Template Building

Age-specific templates are built from a subset of subjects and used for group-level normalization:

```bash
# Select and preprocess template subjects
uv run python scripts/batch_preprocess_for_templates.py \
    --bids-root /path/to/bids --output-root /path/to/study --cohorts p30 p60 p90

# Build ANTs templates
uv run python scripts/build_templates.py --config config.yaml --cohorts p30 p60 p90
```

### Anatomical (T2w)

N4 bias correction, two-pass Atropos+BET skull stripping, tissue segmentation (GM/WM/CSF), optional 3D-to-2D resampling, registration to age-matched template (ANTs SyN). 3D isotropic acquisitions are automatically detected and resampled to standard 2D geometry.

```bash
uv run python scripts/batch_preprocess_anat.py --config config.yaml
```

### Diffusion (DTI)

5D-to-4D conversion, intensity normalization, skull stripping, GPU-accelerated eddy correction with slice padding, DTI tensor fitting (FA, MD, AD, RD), FA-to-T2w registration (ANTs affine).

```bash
uv run python scripts/batch_preprocess_dwi.py --config config.yaml \
    --bids-root /path/to/bids --output-root /path/to/study
```

### Functional (fMRI)

Volume discarding, adaptive skull stripping, motion correction (MCFLIRT), ICA denoising (MELODIC), spatial smoothing, temporal bandpass filtering, confound extraction (24 motion + aCompCor), BOLD-to-T2w registration.

```bash
uv run python scripts/batch_preprocess_func.py /path/to/bids /path/to/study
```

### MSME T2 Mapping

Skull stripping, NNLS-based T2 fitting, Myelin Water Fraction (MWF) and compartment analysis, MSME-to-T2w registration.

```bash
uv run python scripts/batch_preprocess_msme.py /path/to/bids /path/to/study
```

### Resting-State Metrics

Individual scripts for each resting-state metric (run after functional preprocessing):

```bash
# fALFF (fractional ALFF) — from unfiltered regressed BOLD
uv run python scripts/batch_falff_analysis.py --config config.yaml --n-workers 6

# ReHo (Regional Homogeneity) — from bandpass-filtered BOLD
uv run python scripts/batch_reho_analysis.py --config config.yaml --n-workers 6
```

All support `--dry-run`, `--subjects sub-Rat49 sub-Rat50`, `--force`, and `--skip-sigma`.

### Cross-Modal Registration

Standalone registration scripts for individual modality-to-template steps:

```bash
uv run python scripts/batch_register_fa_to_t2w.py --config config.yaml
uv run python scripts/batch_register_fa_to_template.py --config config.yaml
uv run python scripts/batch_register_bold_to_t2w.py --config config.yaml
uv run python scripts/batch_register_bold_to_template.py --config config.yaml
uv run python scripts/batch_register_msme.py --config config.yaml
uv run python scripts/batch_warp_bold_to_sigma.py --config config.yaml
```

### Quality Control

```bash
# Batch QC summary with outlier detection
uv run python scripts/generate_batch_qc.py --study-root /path/to/study --modalities anat dwi func

# Skull stripping QC montages
uv run python scripts/batch_skull_strip_qc.py --config config.yaml
```

---

## Connectome

The `neurofaune/connectome/` module provides ROI extraction, connectivity matrices, and network analysis tools. All operate in SIGMA atlas space.

### ROI Extraction

Extract mean metric values (FA, MD, T2, etc.) per SIGMA atlas region across all subjects:

```bash
uv run python scripts/extract_roi_means.py \
    --derivatives-dir /path/to/study/derivatives \
    --parcellation /path/to/study/atlas/SIGMA_study_space/SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz \
    --labels-csv /path/to/atlases/SIGMA/SIGMA_InVivo_Anatomical_Brain_Atlas_Labels.csv \
    --study-tracker /path/to/tracker.csv \
    --modality dwi --metrics FA MD AD RD \
    --output-dir /path/to/study/analysis/roi
```

Produces wide and long CSVs with per-region and per-territory means (234 regions, 11 territories).

```python
from neurofaune.connectome.roi_extraction import load_parcellation, extract_all_subjects

parcellation, labels = load_parcellation(parcellation_path, labels_csv_path)
wide_df, long_df = extract_all_subjects(derivatives_dir, parcellation, labels, modality="dwi")
```

### Functional Connectivity

ROI-to-ROI Pearson correlation from SIGMA-space BOLD timeseries with Fisher z-transform:

```bash
uv run python scripts/batch_fc_analysis.py --config config.yaml --n-workers 6
```

```python
from neurofaune.connectome.functional import extract_roi_timeseries, compute_fc_matrix

timeseries, labels = extract_roi_timeseries(bold_4d, atlas, mask=brain_mask)
fc_matrix = compute_fc_matrix(timeseries)  # Pearson r → Fisher z
```

### Covariance Network Analysis (CovNet)

Builds Spearman correlation matrices per experimental group and compares them using NBS, graph metrics, and whole-network tests. Supports bilateral ROI averaging and territory-level analysis.

**All-in-one** (prepare + all tests):

```bash
uv run python scripts/run_covnet_analysis.py \
    --roi-dir /path/to/analysis/roi \
    --exclusion-csv /path/to/exclusions.csv \
    --output-dir /path/to/analysis/covnet \
    --metrics FA MD AD RD \
    --n-permutations 5000 --seed 42
```

**Step-by-step** (prepare once, run tests independently):

```bash
# Step 1: Prepare data and correlation matrices
uv run python scripts/covnet_prepare.py \
    --roi-dir /path/to/analysis/roi \
    --output-dir /path/to/analysis/covnet \
    --metrics FA MD AD RD

# Step 2: Network-Based Statistic (edge-level permutation testing)
uv run python scripts/covnet_nbs.py \
    --prep-dir /path/to/analysis/covnet \
    --metrics FA MD --comparisons dose cross-timepoint \
    --n-permutations 5000 --nbs-threshold 3.0 --n-workers 8

# Step 3: Territory-level Fisher z-tests with FDR correction
uv run python scripts/covnet_territory.py \
    --prep-dir /path/to/analysis/covnet \
    --metrics FA MD --comparisons dose cross-timepoint

# Step 4: Graph metrics (efficiency, clustering, modularity, small-worldness)
uv run python scripts/covnet_graph_metrics.py \
    --prep-dir /path/to/analysis/covnet \
    --metrics FA MD --densities 0.10 0.15 0.20 0.25 --n-permutations 5000

# Step 5: Whole-network similarity (Mantel, Frobenius, spectral divergence)
uv run python scripts/covnet_whole_network.py \
    --prep-dir /path/to/analysis/covnet \
    --metrics FA MD --comparisons dose cross-timepoint --n-workers 8
```

```python
from neurofaune.connectome import CovNetAnalysis

analysis = CovNetAnalysis.prepare(roi_dir, exclusion_csv, output_dir, "FA")
analysis.save()

analysis = CovNetAnalysis.load(output_dir, "FA")
analysis.run_nbs(comparisons=analysis.resolve_comparisons(["dose"]))
analysis.run_territory()
analysis.run_graph_metrics()
analysis.run_whole_network()
```

---

## Analysis

Group-level statistical analysis tools in `neurofaune/analysis/`. All operate on data already extracted/warped to SIGMA atlas space.

### TBSS (Tract-Based Spatial Statistics)

WM-skeleton voxel-wise analysis for DTI and MSME metrics using FSL randomise with 2D TFCE:

```bash
# Prepare TBSS skeleton
uv run python -m neurofaune.analysis.tbss.prepare_tbss --config config.yaml \
    --output-dir /path/to/analysis/tbss

# Prepare designs (group contrasts + dose-response)
uv run python scripts/prepare_tbss_designs.py \
    --study-tracker /path/to/tracker.csv \
    --tbss-dir /path/to/analysis/tbss \
    --output-dir /path/to/analysis/tbss/designs
uv run python scripts/prepare_tbss_dose_response_designs.py \
    --study-tracker /path/to/tracker.csv \
    --tbss-dir /path/to/analysis/tbss \
    --output-dir /path/to/analysis/tbss/designs

# Run randomise (permutation testing)
uv run python scripts/run_tbss_analysis.py \
    --tbss-dir /path/to/analysis/tbss --config config.yaml
```

### Voxelwise fMRI Analysis

Whole-brain voxel-wise analysis for fALFF and ReHo using FSL randomise with 3D TFCE:

```bash
uv run python scripts/prepare_fmri_voxelwise.py \
    --study-root /path/to/study \
    --output-dir /path/to/analysis/voxelwise_fmri

uv run python scripts/run_voxelwise_fmri_analysis.py \
    --analysis-dir /path/to/analysis/voxelwise_fmri --config config.yaml
```

### Classification

PERMANOVA, PCA, LDA, SVM/logistic regression with LOOCV:

```bash
uv run python scripts/run_classification_analysis.py \
    --roi-dir /path/to/analysis/roi \
    --output-dir /path/to/analysis/classification \
    --metrics FA MD AD RD --n-permutations 5000
```

### Regression

Dose-response regression with SVR, Ridge, and PLS:

```bash
uv run python scripts/run_regression_analysis.py \
    --roi-dir /path/to/analysis/roi \
    --output-dir /path/to/analysis/regression \
    --metrics FA MD AD RD --n-permutations 5000
```

### MVPA (Multi-Voxel Pattern Analysis)

Whole-brain decoding and searchlight mapping:

```bash
uv run python scripts/run_mvpa_analysis.py \
    --study-root /path/to/study \
    --output-dir /path/to/analysis/mvpa \
    --metrics FA --n-permutations 1000
```

---

## Reporting

The `neurofaune/reporting/` module provides a unified analysis dashboard. Every analysis script automatically registers its results; the index generator builds a self-contained HTML dashboard.

### Generating the Dashboard

```bash
# Backfill existing results and generate dashboard
uv run python scripts/generate_analysis_index.py \
    --analysis-root /path/to/analysis \
    --study-name "BPA Rat Study" \
    --backfill

# Regenerate from existing registry
uv run python scripts/generate_analysis_index.py \
    --analysis-root /path/to/analysis

# List registered entries
uv run python scripts/generate_analysis_index.py \
    --analysis-root /path/to/analysis --list
```

### Programmatic Usage

```python
from neurofaune.reporting import register, backfill_registry, generate_index_html

# Register an analysis result
register(
    analysis_root=Path("/study/analysis"),
    entry_id="tbss_per_pnd_p60",
    analysis_type="tbss",
    display_name="TBSS: PND60 Dose Response",
    output_dir="tbss/randomise/per_pnd_p60",
    summary_stats={"n_subjects": 49, "metrics": ["FA", "MD", "AD", "RD"]},
)

# Discover and register all existing results
n_added = backfill_registry(Path("/study/analysis"), study_name="BPA Rat Study")

# Regenerate the HTML dashboard
generate_index_html(Path("/study/analysis"))
```

Supported analysis types: `tbss`, `roi_extraction`, `covnet`, `connectome`, `classification`, `regression`, `mvpa`, `batch_qc`.

---

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
├── connectome/                      # ROI extraction, connectivity, network analysis
│   ├── roi_extraction.py            # Atlas-based ROI means and territory aggregation
│   ├── functional.py                # BOLD FC matrices (Pearson, Fisher z)
│   ├── matrices.py                  # Spearman correlation matrices per group
│   ├── nbs.py                       # Network-Based Statistic (permutation testing)
│   ├── graph_metrics.py             # Efficiency, clustering, modularity
│   ├── whole_network.py             # Mantel, Frobenius, spectral divergence
│   ├── visualization.py             # Heatmaps, network plots, comparison charts
│   └── pipeline.py                  # CovNetAnalysis orchestrator class
├── reporting/                       # Unified analysis dashboard
│   ├── registry.py                  # JSON registry (file-locked, NFS-safe)
│   ├── discover.py                  # Backfill existing results into registry
│   ├── section_renderers.py         # Per-type HTML section builders
│   └── index_generator.py           # Self-contained HTML dashboard generator
├── analysis/                        # Group-level statistical analysis
│   ├── func/                        # ReHo, fALFF (voxel-level resting-state)
│   ├── tbss/                        # Tract-Based Spatial Statistics
│   ├── stats/                       # FSL randomise wrapper, cluster reporting
│   ├── classification/              # PERMANOVA, PCA, LDA, SVM
│   ├── regression/                  # Dose-response regression
│   └── mvpa/                        # Multi-voxel pattern analysis
├── templates/                       # Template building and registration
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
