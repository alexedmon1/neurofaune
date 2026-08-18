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

## Developer Gate

Two tiers, both defined in the `Makefile`:

```bash
make check        # BLOCKING — unit + regression tests. Must pass before commit/tag.
make advisory     # INFORMATIONAL — ruff + mypy. Never blocks.
make integration  # SLOW — real ANTs/FSL end-to-end. Run before a release tag.
```

`make check` includes a gate test that fails if `CAPABILITIES.md` is stale.
Regenerate and commit it in the same change whenever you add or rename an entry
point **or a config key**:

```bash
make capabilities   # rewrites CAPABILITIES.md
```

`make advisory` currently reports several thousand ruff findings and ~1,180 mypy
errors, nearly all stylistic (`UP006`/`UP045` typing modernization) or `nibabel`
stub gaps. It is advisory by design — a clean run is not a precondition for
anything.

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
# Config-driven Bruker -> BIDS (raw/bids roots + session regex live in config.yaml;
# --raw/--bids override). Add --dry-run to preview discovered sessions first.
neurofaune bids --config config.yaml
```

---

## Preprocess

All preprocessing scripts live in `scripts/` and use library code from `neurofaune/preprocess/`. Each modality has a batch script that discovers BIDS data and processes all subjects.

### Template Building

Age-specific templates are built from a subset of subjects and used for group-level normalization:

```bash
# Select and preprocess template subjects.
# NOTE: takes no arguments — it reads paths from configs/bpa_rat_example.yaml,
# which no longer ships with the repo, and hardcodes cohorts p30/p60/p90 and a
# sub-Rat* subject glob. Point it at your own config before use.
uv run python scripts/batch_preprocess_for_templates.py

# Build ANTs templates — both --cohort (singular) and --modality are REQUIRED.
uv run python scripts/build_templates.py \
    --config config.yaml --cohort p60 --modality anat

# ...or every cohort in one invocation:
uv run python scripts/build_templates.py \
    --config config.yaml --cohort all --modality anat
```

### Anatomical (T2w)

N4 bias correction, two-pass Atropos+BET skull stripping, tissue segmentation (GM/WM/CSF), optional 3D-to-2D resampling, registration to age-matched template (ANTs SyN). 3D isotropic acquisitions are automatically detected and resampled to standard 2D geometry.

```bash
# bids_dir and output_dir are POSITIONAL and required, even with --config.
uv run python scripts/batch_preprocess_anat.py \
    /path/to/bids /path/to/study --config config.yaml
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
# Takes no positional arguments — paths are given as flags.
uv run python scripts/batch_preprocess_func.py \
    --bids-root /path/to/bids --output-root /path/to/study --config config.yaml
```

### MSME T2 Mapping

Skull stripping, NNLS-based T2 fitting, Myelin Water Fraction (MWF) and compartment analysis, MSME-to-T2w registration.

```bash
# Takes no positional arguments — paths are given as flags.
uv run python scripts/batch_preprocess_msme.py \
    --bids-root /path/to/bids --output-root /path/to/study --config config.yaml
```

### MR Spectroscopy (single-voxel PRESS)

Bruker PRESS to quantified metabolite concentrations via FSL-MRS: conversion,
coil combination and shot alignment, basis-set fitting, water-scaled
quantification, and QC.

```bash
uv run python scripts/batch_preprocess_mrs.py \
    /path/to/bruker /path/to/study/mrs \
    --config config.yaml \
    --derivatives /path/to/study/preprocessing/derivatives \
    --basis /path/to/basis/gamma_press_te20_7t_v1 --n-jobs 4
```

Spectroscopy reads the **raw Bruker tree**, not BIDS — `spec2nii` cannot read
ParaVision 360.3 SVS data, so it is never converted during BIDS-ification and
neurofaune ships its own reader. Outputs are likewise self-contained under the
given root rather than under `derivatives/`:

```
{study}/mrs/{sub}/{ses}/        NIfTI-MRS, voxel mask, preproc/, fit/, metabolites CSV
                                plus *_fit-curves.csv and *_fit-metabolites.csv
{study}/mrs/qc/{sub}/{ses}/     QC report, voxel-placement overlay, CRLB chart
{study}/mrs/logs/               batch summaries and failure tracebacks
{study}/mrs/mrs_metabolites_long.csv    combined table for group analysis
```

No conda environment is needed: FSL 6.0.7+ bundles `fsl_mrs` and the workflow
shells out to it, as the other modalities do for BET and ANTs. Point
`spectroscopy.fsl_bin` at another directory to use a different build.

`--derivatives` supplies the T2w segmentation used to measure the voxel's
GM/WM/CSF content for absolute quantification. Without it the workflow falls
back to assumed fractions, which affects water-scaled concentrations but not
ratios to creatine.

**Voxel localisation, and why it is checked automatically.** The voxel is
positioned from Bruker geometry parameters rather than from the NIfTI affine,
because the converter writes a scaled-identity affine with no scanner geometry.
Every axis assignment and sign in that reconstruction is a convention, and
three of them were initially wrong. All three failed the same way: the error is
proportional to some offset that is zero on most sessions, so the majority
looked right and the minority looked like operator error.

| Sign error | Invisible when | Sessions affected |
|---|---|---|
| `PVM_VoxArrPosition` is in the voxel's rotated frame, not magnet coords | the voxel is not rotated | up to 1.7 mm on the most angled |
| slice-axis direction | the voxel sits at isocentre | 36 of 52, by >2 slices |
| `PVM_SPackArrSliceOffset` sign | the slice package is not offset | 12 of 53, by up to 5 mm |

The systematic fix is to stop reconstructing the mapping at all. Bruker writes
the DICOM-equivalent geometry in `pdata/*/visu_pars` — `VisuCoreOrientation`,
`VisuCorePosition`, `VisuCoreExtent` — which defines the index-to-world affine
outright, signs and axis order included. Locating the voxel is then affine
composition, exactly as in human MRS. That is now the default path, and it is
both more principled and slightly better: 71.4% mean hippocampal overlap with
50 of 50 sessions on target, against 70.8% and 49 of 50 for the reconstructed
mapping, which remains only as a fallback when `visu_pars` is unusable.

One piece is irreducible. A PRESS scan's `visu_pars` has `VisuCoreDim = 1` and
no spatial fields, so the voxel exists only in gradient coordinates while
images are in subject coordinates. The two differ by a signed permutation set
by `VisuSubjectPosition`, and that cannot be recovered from the files: with a
square FOV and the package at isocentre — 47 of 50 sessions here — every
candidate reproduces the geometry equally well. It is therefore calibrated once
per subject position and validated (`Head_Supine` → `diag(1, -1, -1)`, 71.2%
against 32.7% for the next best), cross-checked per scan by requiring the
gradient axes to align with the image axes, and an unknown subject position
raises rather than guesses. One constant tied to a documented parameter, rather
than three scattered sign choices.

So the pipeline no longer relies on anyone eyeballing an overlay. Set
`spectroscopy.target_structure` (a substring matched against the atlas label
table, e.g. `hippocamp`) and every session is scored against the structure the
voxel was aimed at, with QC flagging anything below
`min_target_overlap`. On the cuprizone study that runs at 66–81% per session.

`read_anat_geometry` additionally warns when an acquisition departs from what
has been validated — non-axial slices, 3D acquisitions, multiple slice
packages, a non-zero phase offset, or an unvalidated `RECO_transposition`.
Those are unproven rather than known-wrong, but they mean the placement should
be verified against anatomy before the tissue fractions are trusted.

Start with `--dry-run` to see which PRESS scan is selected per session (sessions
typically hold unsuppressed shim prescans alongside the real acquisition) and
how many sessions have a segmentation available.

**Frequency referencing.** `fsl_mrs_preproc` shifts and phases the spectrum on
whatever is strongest in a hardcoded 2.9–3.1 ppm window, so total creatine has
to already be near 3.027 before it runs. Its own alignment step doesn't provide
that — it aligns the individual shots to each other, not to an absolute
chemical shift. So the converter references the spectrum itself, in two stages:
water to its true shift from the unsuppressed reference, then tCr onto 3.027.
Across 52 CPZ sessions that moved tCr from 2.939 ± 0.012 (worst case 0.019 ppm
from falling out of the window) to 3.0269 ± 0.0007.

The tCr search window is deliberately wide (2.7–3.4), which is safe because the
result is cross-checked against NAA: the two singlets are a fixed 1.019 ppm
apart, so a misidentified peak is caught and the session falls back to water
referencing. Measured separation across those sessions was 1.0212 ± 0.0010,
with no failures.

**Preprocessing chain.** `spectroscopy.preproc` selects between:

- `internal` (default) — coil combination, windowed alignment, outlier removal,
  averaging and eddy-current correction, driven step by step through
  `nifti_mrs_proc`.
- `fsl_mrs_preproc` — the stock FSL pipeline, for comparison.

They run the same steps and differ only in how they finish. The stock pipeline
ends with `shift_to_reference` and `phase_correct`, both of which take
`argmax(|spectrum|)` in that same hardcoded 2.9–3.1 ppm window and move it to
3.027. When the wrong point wins, the spectrum is displaced in ppm and given an
arbitrary global phase, and no metabolite can be fit afterwards. On cuprizone
data that cost 6–7 of 53 sessions, and the window is not adjustable from the
command line.

The `internal` chain drops `shift_to_reference` entirely — the converter has
already referenced the spectrum on tCr, over a wide window cross-checked against
NAA rather than a 0.2 ppm window with no validation.

It **keeps** zero-order phasing, because leaving the phase to `fsl_mrs` as a
free parameter costs about 30% of the fitted SNR. `spectroscopy.phase_method`
selects how:

- `search` (default) — scans zero-order phase over the full ±180° circle and
  scores the whole 0.5–4.2 ppm region for absorptive character: positive real
  signal, with a penalty on the negative lobes a wrong phase produces (coarse
  1° pass, then 0.05°). Scoring the band rather than one peak avoids inheriting
  that peak's noise. Covering the full circle is the point — `fsl_mrs` fits
  phase by local descent from zero with concentrations bounded non-negative, so
  a spectrum near 180° out cannot be recovered by the fit; the metabolites just
  go to zero.
- `tcr` — phases on the creatine peak alone, over a 2.95–3.10 ppm window. Safe
  here for the reason the stock version isn't: with tCr already at 3.027 ±
  0.001, the peak it lands on is the one intended, rather than whatever is
  tallest in a window the spectrum may have drifted out of.

Set `spectroscopy.phase_method: tcr` to choose the latter, or pass `--no-phase`
to `_fsl_preproc` to skip phasing entirely and leave it to `fsl_mrs`.

On a 13-session subset, comparing the stock chain against `internal` with
phasing off and with `tcr`:

| chain | sessions fit | median SNR |
|---|---|---|
| stock `fsl_mrs_preproc` | 7/13 | 18.6 |
| `internal`, `--no-phase` | 13/13 | 13.2 |
| `internal`, `phase_method: tcr` | 13/13 | 15.6 |

`search` was added after this benchmark and is not represented in it; it is the
default because it does not depend on any single peak being correctly placed.

Sessions the fitter still declines are reported as `unquantifiable` rather than
counted as failures, with their preprocessed data left on disk to inspect.

**Figures.** `fsl_mrs` writes an interactive HTML report and a summary PNG but
no fit curves as data, so building a custom or group-level figure would mean
scraping the HTML. `spectroscopy.export_curves` (on by default) additionally
writes per session:

```
{sub}_{ses}_fit-curves.csv       ppm, data, fit, baseline, residual
{sub}_{ses}_fit-metabolites.csv  ppm plus one column per basis metabolite
```

Real-valued spectra over the fit range, ready to plot directly.

**Fitter.** `spectroscopy.fitter` selects `fsl_mrs` (default) or `lcmodel`.
LCModel needs `spectroscopy.lcmodel.basis` — an LCModel `.basis` file, not the
JSON directory `spectroscopy.basis` points at; the two fitters take different
formats of the same basis.

LCModel is worth running as an independent check: same basis, same preprocessed
FID, different implementation, so agreement validates the whole chain rather
than just the fit. On CPZ sessions the two agreed closely on the major
metabolite ratios (NAA+NAAG 1.25 vs 1.28, Glu 1.25 vs 1.26, GPC+PCh 0.188 vs
0.188 against tCr) while LCModel reported much lower CRLBs (2–6% against
10–30%) and fit sessions `fsl_mrs` could not. It models the macromolecule
baseline internally, which the JSON basis conversion loses — the same
limitation noted in `mrs/basis/README.md` in the study tree (that path is
outside this repo).

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
# These take --study-root (not --config) and derive the rest from the study tree.
# All support --dry-run and --force; all but the last also take --n-cores.
uv run python scripts/batch_register_fa_to_t2w.py        --study-root /path/to/study
uv run python scripts/batch_register_fa_to_template.py   --study-root /path/to/study
uv run python scripts/batch_register_bold_to_t2w.py      --study-root /path/to/study
uv run python scripts/batch_register_bold_to_template.py --study-root /path/to/study
uv run python scripts/batch_register_msme.py             --study-root /path/to/study
uv run python scripts/batch_warp_bold_to_sigma.py        --study-root /path/to/study
```

### Quality Control

```bash
# Batch QC summary with outlier detection.
# study_root is POSITIONAL; --modality takes ONE of {dwi,anat,func,msme,all}.
uv run python scripts/generate_batch_qc.py /path/to/study --modality all

# Restrict to specific subjects, or change the outlier threshold:
uv run python scripts/generate_batch_qc.py /path/to/study \
    --modality dwi --subjects sub-Rat49 sub-Rat50 --z-threshold 3.0

# Skull stripping QC montages (study_root also positional)
uv run python scripts/batch_skull_strip_qc.py /path/to/study --modality anat
```

---

## Network

ROI-based analyses that operate in SIGMA atlas space. Outputs are organized under `{study_root}/network/` with subdirectories per analysis type and modality.

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
    --feature-sets bilateral \
    --n-components 5 \
    --regs lw \
    --n-permutations 5000 --seed 42

# With continuous AUC target
uv run python scripts/run_mcca_analysis.py \
    --roi-dir /path/to/network/roi \
    --output-dir /path/to/network/mcca_auc \
    --views dwi:FA,MD,AD,RD msme:MWF,IWF,CSFF,T2 func:fALFF,ReHo,ALFF \
    --feature-sets bilateral --target auc \
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
# Designs are written to {vbm-dir}/designs — there is no --output-dir.
uv run python scripts/prepare_vbm_designs.py \
    --study-tracker /path/to/tracker.csv \
    --vbm-dir /path/to/analysis/vbm

# AUC designs
uv run python scripts/prepare_vbm_designs.py \
    --study-tracker /path/to/tracker.csv \
    --vbm-dir /path/to/analysis/vbm \
    --target auc --auc-csv /path/to/auc_lookup.csv

uv run python scripts/run_vbm_analysis.py \
    --vbm-dir /path/to/analysis/vbm \
    --analyses auc_response_p30 auc_response_p60 auc_response_p90 \
    --n-permutations 5000
```

### MVPA (Multi-Voxel Pattern Analysis)

Whole-brain decoding and searchlight mapping. Supports both categorical group designs and continuous regression targets (ordinal dose or AUC):

```bash
# Paths come from --config, or individually via --derivatives-dir /
# --design-dir / --output-dir. There is no --study-root.
uv run python scripts/run_mvpa_analysis.py \
    --config config.yaml \
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

See `configs/default.yaml` for all parameters. A study-specific `config.yaml` is
generated in your study root by `scripts/init_study.py` and overrides these
defaults; `configs/` ships defaults only, with no per-study example.

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

## Script Status

`scripts/` holds **example CLI wrappers**, not the supported interface — the
library under `neurofaune/` is. Scripts vary in maturity:

| Tier | Scripts | Notes |
|---|---|---|
| Documented | the invocations shown above | verified against `--help` |
| Undocumented | ~25 others, incl. `prepare_vbm.py`, `run_covnet_nbs.py`, `run_melodic_clean.py`, `fit_multishell_models.py` | working, but no README coverage — run `--help` |
| Ad-hoc | `test_anat_registration.py`, `test_msme_multi_subject.py`, `test_msme_adaptive_skull_strip.py` | developer scratch scripts with absolute paths hardcoded to one machine; **not** part of the pytest suite despite the `test_` prefix |

Several scripts hardcode `/mnt/arborea/...` paths. Check the source before
running one that is not documented above.

### TBSS entry points

There are four overlapping TBSS drivers. The active pair is:

```bash
scripts/run_template_tbss_prepare.py   # called by run_template_tbss_pipeline.sh
scripts/run_tbss_analysis.py           # end-to-end driver
```

`run_tbss_prepare.py` and `run_tbss_stats.py` are thin wrappers that may be
superseded by `run_tbss_analysis.py` (see `docs/CLEANUP_TODO.md`). Prefer
`run_tbss_analysis.py` for new work.

## Acknowledgments

- [SIGMA rat brain atlas](https://doi.org/10.1016/j.neuroimage.2019.06.063) (Barriere et al., 2019)
- Built on [ANTs](https://github.com/ANTsX/ANTs), [FSL](https://fsl.fmrib.ox.ac.uk/fsl/), and [Nipype](https://nipype.readthedocs.io/)
