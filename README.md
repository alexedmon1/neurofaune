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

Processing follows a strict order. Anatomical preprocessing must complete first to build age-cohort templates; all modalities then register directly to template before warping to SIGMA atlas space.

```
1. Initialize Study        → directory structure, config, study-space atlas
2. Bruker → BIDS           → convert raw scanner data to standard format
3. Anatomical (T2w)        → N4, skull strip, segment, build templates, register
4. Other Modalities        → DTI, fMRI, MSME (each registers directly to template)
5. Analysis (voxelwise)    → TBSS, VBM, voxelwise fMRI (fALFF, ReHo), MVPA
6. Network (ROI-based)     → ROI extraction, CovNet, classification, regression, MCCA
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
├── analysis/                    # Voxelwise group analyses (TBSS, VBM, fMRI, MVPA)
├── network/                     # ROI-based analyses (covnet, classification, MCCA)
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

## Network

ROI-based analyses that operate in SIGMA atlas space. Outputs are organized under `{study_root}/network/` with subdirectories per analysis type and modality.

### AUC Lookup Preparation

Generate a session-matched AUC (area under the plasma concentration curve) lookup CSV from the ROI wide CSVs. This is used as a continuous pharmacokinetic exposure target across all applicable analyses (regression, MCCA, TBSS, VBM, MVPA, CovNet):

```bash
uv run python scripts/prepare_auc_lookup.py \
    --roi-csv /path/to/network/roi/roi_FA_wide.csv \
    --output /path/to/study/auc_lookup.csv
```

### ROI Extraction

Extract mean metric values (FA, MD, T2, etc.) per SIGMA atlas region across all subjects:

```bash
uv run python scripts/extract_roi_means.py \
    --derivatives-dir /path/to/study/derivatives \
    --parcellation /path/to/study/atlas/SIGMA_study_space/SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz \
    --labels-csv /path/to/atlases/SIGMA/SIGMA_InVivo_Anatomical_Brain_Atlas_Labels.csv \
    --study-tracker /path/to/tracker.csv \
    --modality dwi --metrics FA MD AD RD \
    --output-dir /path/to/study/network/roi
```

Produces wide and long CSVs with per-region and per-territory means (234 regions, 11 territories).

```python
from neurofaune.network.roi_extraction import load_parcellation, extract_all_subjects

parcellation, labels = load_parcellation(parcellation_path, labels_csv_path)
wide_df, long_df = extract_all_subjects(derivatives_dir, parcellation, labels, modality="dwi")
```

### Functional Connectivity

ROI-to-ROI Pearson correlation from SIGMA-space BOLD timeseries with Fisher z-transform:

```bash
uv run python scripts/batch_fc_analysis.py --config config.yaml --n-workers 6
```

```python
from neurofaune.network.functional import extract_roi_timeseries, compute_fc_matrix

timeseries, labels = extract_roi_timeseries(bold_4d, atlas, mask=brain_mask)
fc_matrix = compute_fc_matrix(timeseries)  # Pearson r → Fisher z
```

### Covariance Network Analysis (CovNet)

Builds Spearman correlation matrices per experimental group and compares them using network distance tests (absolute and relative), NBS with post-hoc characterization, graph theory, and territory-level analysis.

**Primary interface** is the Python API. Paths are derived from `config.yaml`:

```python
from pathlib import Path
from neurofaune.network.covnet import CovNetAnalysis

# Prepare and run a single metric
analysis = CovNetAnalysis.prepare(
    config_path=Path("/path/to/config.yaml"),
    modality="dwi", metric="FA",
    sex="M",       # optional: sex-stratified analysis
    force=True,    # overwrite existing results
)
analysis.save()
analysis.run_abs_distance(n_perm=1000)      # Mantel, Frobenius, spectral
analysis.run_rel_distance(n_perm=5000)      # shift relative to reference
analysis.run_nbs(n_perm=1000, posthoc=True) # NBS + edge direction + characterization
analysis.run_graph_metrics(n_perm=1000)     # clustering, centrality, small-worldness
analysis.run_territory()                    # Fisher z + FDR at territory level
```

Each `run_*()` method checks for existing results and errors unless `force=True`, preventing accidental overwrites or ambiguous mixed results.

**Example CLI scripts** in `scripts/` demonstrate usage but are not the primary interface. Each study should create its own wrapper scripts:

```bash
# Example: run absolute distance for all DTI metrics
uv run python scripts/run_covnet_abs_distance.py \
    --config /path/to/config.yaml \
    --modality dwi --metrics FA MD AD RD \
    --n-permutations 1000 --n-workers 4 --force
```

**Config requirements** — add network paths to your `config.yaml`:
```yaml
paths:
  network:
    roi: ${paths.study_root}/network/roi
    covnet: ${paths.study_root}/network/covnet
```

### Edge Regression

Edge-level regression testing whether pairwise ROI co-variation scales with a continuous covariate (e.g. log-AUC). Uses NBS-style component extraction with permutation FWER correction. This is appropriate **only for continuous targets** — for categorical group comparisons, use NBS instead. Results are saved under `network/edge_regression/`, separate from CovNet.

```bash
uv run python scripts/run_edge_regression.py \
    --roi-dir /path/to/network/roi \
    --output-dir /path/to/network/edge_regression \
    --modality dwi --metrics FA MD AD RD \
    --exclusion-csv /path/to/exclusions.csv \
    --target log_auc --auc-csv /path/to/auc_lookup.csv \
    --n-permutations 1000 --seed 42
```

### Classification

PERMANOVA, PCA, LDA, SVM/logistic regression with LOOCV. The default `all` feature set uses all individual L/R ROIs (~234 features) with PCA dimensionality reduction (95% variance, fit inside each LOOCV fold to avoid data leakage). Model weights are mapped back to ROI space via weight inversion (`coef_ @ pca.components_`) and visualized grouped by atlas territory.

```bash
uv run python scripts/run_classification_analysis.py \
    --roi-dir /path/to/network/roi \
    --output-dir /path/to/network/classification/dwi \
    --metrics FA MD AD RD \
    --feature-sets all \
    --atlas-labels /path/to/SIGMA_Labels.csv \
    --n-permutations 5000
```

Feature sets: `all` (default, all L/R ROIs + PCA), `bilateral` (bilateral-averaged ~50 features), `territory` (coarse aggregates ~15 features).

### Regression

Dose-response regression with SVR, Ridge, and PLS. Same PCA-in-LOOCV pattern and weight inversion as classification. Supports both ordinal dose groups (`--target dose`, default) and continuous pharmacokinetic exposure (`--target auc`) as the target variable. AUC values are session-matched from the ROI wide CSVs.

```bash
# Ordinal dose-response (default)
uv run python scripts/run_regression_analysis.py \
    --roi-dir /path/to/network/roi \
    --output-dir /path/to/network/regression/dwi \
    --metrics FA MD AD RD \
    --feature-sets all \
    --atlas-labels /path/to/SIGMA_Labels.csv \
    --n-permutations 5000

# Continuous AUC target
uv run python scripts/run_regression_analysis.py \
    --roi-dir /path/to/network/roi \
    --output-dir /path/to/network/regression_auc/dwi \
    --metrics FA MD AD RD \
    --feature-sets all --target auc \
    --atlas-labels /path/to/SIGMA_Labels.csv \
    --n-permutations 5000
```

### MCCA (Multiset Canonical Correlation Analysis)

Cross-modality integration that finds linear combinations of ROI features maximizing correlation across modality views. Uses regularized generalized eigenvalue decomposition with Ledoit-Wolf shrinkage and PCA dimensionality reduction for fast permutation testing. Supports `--target auc` for continuous AUC-based dose-response association.

```bash
uv run python scripts/run_mcca_analysis.py \
    --roi-dir /path/to/network/roi \
    --output-dir /path/to/network/mcca \
    --views dwi:FA,MD,AD,RD msme:MWF,IWF,CSFF,T2 func:fALFF,ReHo,ALFF \
    --feature-set bilateral \
    --n-components 5 \
    --regs lw \
    --n-permutations 5000 --seed 42

# With continuous AUC target
uv run python scripts/run_mcca_analysis.py \
    --roi-dir /path/to/network/roi \
    --output-dir /path/to/network/mcca_auc \
    --views dwi:FA,MD,AD,RD msme:MWF,IWF,CSFF,T2 func:fALFF,ReHo,ALFF \
    --feature-set bilateral --target auc \
    --n-components 5 --regs lw --n-permutations 5000
```

Per cohort (pooled, p30, p60, p90), the pipeline runs:
1. Load and intersect subjects across all views (bilateral ROIs, z-scored per view)
2. Fit regularized MCCA via generalized eigenvalue problem
3. Permutation test for significance of canonical correlations (5000 perms)
4. Dose-response association (Spearman correlation with ordinal dose or continuous AUC per canonical variate)
5. PERMANOVA on MCCA score space (group separability)
6. Generate visualizations (canonical correlations, score scatter plots, loading heatmaps, null distributions)

```python
from neurofaune.network.mcca import load_multiview_data, run_mcca, permutation_test_mcca

Xs, view_names, metadata = load_multiview_data(
    roi_dir, views={"dwi": ["FA", "MD"], "msme": ["MWF", "T2"]},
    feature_set="bilateral",
)
result = run_mcca(Xs, n_components=5, regs="lw")
perm = permutation_test_mcca(Xs, result.canonical_correlations, n_permutations=5000)
```

---

## Analysis

Voxel-wise group-level statistical analysis tools in `neurofaune/analysis/`. All operate on data already warped to SIGMA atlas space.

### TBSS (Tract-Based Spatial Statistics)

WM-skeleton voxel-wise analysis for DTI and MSME metrics using FSL randomise with 2D TFCE:

```bash
# Prepare TBSS skeleton (DTI)
uv run python -m neurofaune.analysis.tbss.prepare_tbss --config config.yaml \
    --output-dir /path/to/analysis/tbss/dwi

# Prepare designs (group contrasts + dose-response)
uv run python scripts/prepare_tbss_designs.py \
    --study-tracker /path/to/tracker.csv \
    --tbss-dir /path/to/analysis/tbss/dwi \
    --output-dir /path/to/analysis/tbss/dwi/designs
uv run python scripts/prepare_tbss_dose_response_designs.py \
    --study-tracker /path/to/tracker.csv \
    --tbss-dir /path/to/analysis/tbss/dwi \
    --output-dir /path/to/analysis/tbss/dwi/designs

# AUC dose-response designs (continuous pharmacokinetic exposure)
uv run python scripts/prepare_tbss_dose_response_designs.py \
    --study-tracker /path/to/tracker.csv \
    --tbss-dir /path/to/analysis/tbss/dwi \
    --output-dir /path/to/analysis/tbss/dwi/designs \
    --target auc --auc-csv /path/to/auc_lookup.csv

# Run randomise (permutation testing)
uv run python scripts/run_tbss_analysis.py \
    --tbss-dir /path/to/analysis/tbss/dwi --config config.yaml
```

### Voxelwise fMRI Analysis

Whole-brain voxel-wise analysis for fALFF and ReHo using FSL randomise with 3D TFCE:

```bash
# Prepare and run ReHo
uv run python scripts/prepare_fmri_voxelwise.py \
    --study-root $STUDY_ROOT \
    --output-dir $STUDY_ROOT/analysis/reho --metrics ReHo

uv run python scripts/run_voxelwise_fmri_analysis.py \
    --analysis-dir $STUDY_ROOT/analysis/reho --metrics ReHo --config config.yaml
```

### VBM (Voxel-Based Morphometry)

Voxel-wise analysis of tissue density (GM, WM, CSF) using FSL randomise. Design scripts support both ordinal dose and continuous AUC targets:

```bash
uv run python scripts/prepare_vbm_designs.py \
    --study-tracker /path/to/tracker.csv \
    --vbm-dir /path/to/analysis/vbm \
    --output-dir /path/to/analysis/vbm/designs

# AUC designs
uv run python scripts/prepare_vbm_designs.py \
    --study-tracker /path/to/tracker.csv \
    --vbm-dir /path/to/analysis/vbm \
    --output-dir /path/to/analysis/vbm/designs \
    --target auc --auc-csv /path/to/auc_lookup.csv

uv run python scripts/run_vbm_analysis.py \
    --vbm-dir /path/to/analysis/vbm \
    --analyses auc_response_p30 auc_response_p60 auc_response_p90 \
    --n-permutations 5000
```

### MVPA (Multi-Voxel Pattern Analysis)

Whole-brain decoding and searchlight mapping. Supports both categorical group designs and continuous regression targets (ordinal dose or AUC):

```bash
uv run python scripts/run_mvpa_analysis.py \
    --study-root /path/to/study \
    --output-dir /path/to/analysis/mvpa \
    --metrics FA --n-permutations 1000

# Prepare AUC regression designs
uv run python scripts/prepare_mvpa_designs.py \
    --study-tracker /path/to/tracker.csv \
    --derivatives-root /path/to/derivatives \
    --output-dir /path/to/analysis/mvpa/designs \
    --metrics FA MD AD RD \
    --target auc --auc-csv /path/to/auc_lookup.csv
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
    output_dir="tbss/dwi/randomise/per_pnd_p60",
    summary_stats={"n_subjects": 49, "metrics": ["FA", "MD", "AD", "RD"]},
)

# Discover and register all existing results
n_added = backfill_registry(Path("/study/analysis"), study_name="BPA Rat Study")

# Regenerate the HTML dashboard
generate_index_html(Path("/study/analysis"))
```

Supported analysis types: `tbss`, `roi_extraction`, `covnet`, `connectome`, `classification`, `regression`, `mcca`, `mvpa`, `batch_qc`.

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
├── connectome/                      # Deprecated shims → neurofaune.network
├── network/                         # ROI-based analyses
│   ├── matrices.py                  # Spearman correlation matrices per group
│   ├── roi_extraction.py            # Atlas-based ROI means and territory aggregation
│   ├── functional.py                # BOLD FC matrices (Pearson, Fisher z)
│   ├── covnet/                      # Covariance network analysis (CovNet)
│   │   ├── pipeline.py              # CovNetAnalysis orchestrator class
│   │   ├── nbs.py                   # Network-Based Statistic (permutation testing)
│   │   ├── graph_metrics.py         # Efficiency, clustering, modularity
│   │   ├── whole_network.py         # Mantel, Frobenius, spectral divergence
│   │   └── visualization.py         # Heatmaps, network plots, comparison charts
│   ├── classification/              # PERMANOVA, PCA, LDA, SVM + PCA weight inversion
│   ├── regression/                  # Dose-response regression (SVR, Ridge, PLS)
│   ├── mcca.py                      # MCCA: load, fit, permutation, dose, PERMANOVA
│   └── mcca_visualization.py        # Canonical correlations, scores, loadings plots
├── analysis/                        # Voxelwise group-level statistical analysis
│   ├── stats/                       # FSL randomise wrapper, cluster reporting
│   ├── mvpa/                        # Multi-voxel pattern analysis
│   ├── progress.py                  # Lightweight progress tracking for runner scripts
│   └── provenance.py                # Provenance chain for analysis reproducibility
├── reporting/                       # Unified analysis dashboard
│   ├── registry.py                  # JSON registry (file-locked, NFS-safe)
│   ├── discover.py                  # Backfill existing results into registry
│   ├── section_renderers.py         # Per-type HTML section builders
│   └── index_generator.py           # Self-contained HTML dashboard generator
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
