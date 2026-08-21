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
   MR Spectroscopy         → reads raw Bruker, not BIDS; self-contained under mrs/
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

Bruker PRESS to quantified metabolite concentrations: conversion, coil
combination, shot alignment, eddy-current correction, basis-set fitting,
water-scaled quantification with measured tissue fractions, and QC.

```bash
# see which PRESS scan is selected per session, and what anatomy is available
uv run python scripts/batch_preprocess_mrs.py \
    /path/to/bruker /path/to/study/mrs --config config.yaml --dry-run

uv run python scripts/batch_preprocess_mrs.py \
    /path/to/bruker /path/to/study/mrs \
    --config config.yaml \
    --derivatives /path/to/study/preprocessing/derivatives \
    --basis /path/to/basis/gamma_press_te20_7t_v1 --n-jobs 4
```

Two things differ from the other modalities. It reads the **raw Bruker tree
rather than BIDS**, because `spec2nii` cannot read ParaVision 360.3 SVS data —
PV-360 no longer writes the TopSpin-style `fid`, and `brukerapi` rejects the
`rawdata.job0` that replaced it — so spectroscopy is never converted during
BIDS-ification and neurofaune ships its own reader. And its outputs are
**self-contained** rather than living under `derivatives/`:

```
{study}/mrs/{sub}/{ses}/      NIfTI-MRS, voxel mask, tissue fractions, preproc/, fit/
                              {sub}_{ses}_metabolites.csv    tidy per-session results
                              {sub}_{ses}_fit-curves.csv      ppm, data, fit, baseline, residual
                              {sub}_{ses}_fit-metabolites.csv ppm + one column per metabolite
                              {sub}_{ses}_mm-areas.csv        upfield MM/lipid band areas
                              {sub}_{ses}_mm-envelope.csv     MM spline + its anchor line
{study}/mrs/qc/{sub}/{ses}/   QC report, voxel-placement overlay, CRLB chart
{study}/mrs/logs/             batch summaries, failure tracebacks
{study}/mrs/index.html        study QC index: sortable, links every report
{study}/mrs/mrs_metabolites_long.csv    combined table for group analysis
```

No conda environment is needed — FSL 6.0.7+ bundles `fsl_mrs`, and the workflow
shells out to it as the other modalities do for BET and ANTs. Point
`spectroscopy.fsl_bin` elsewhere to use a different build.

`--derivatives` supplies the T2w segmentation used to measure the voxel's
GM/WM/CSF content. Without it the workflow assumes fractions, which affects
water-scaled concentrations but not ratios to creatine.

#### Configuration

| key | default | |
|---|---|---|
| `basis` | — | FSL-MRS basis directory (required) |
| `fsl_bin` | `$FSLDIR/bin` | where to find `fsl_mrs` |
| `prefer_raw` | `true` | read `rawdata.job0`, keeping coils and averages |
| `preproc` | `internal` | or `fsl_mrs_preproc` for the stock chain |
| `align_window` | `32` | shots per alignment window; `0` disables |
| `remove_outliers` | `true` | drop averages unlike the rest |
| `remove_water` | `false` | HLSVD; also applied automatically as a retry |
| `phase_method` | `search` | or `tcr` |
| `fitter` | `fsl_mrs` | or `lcmodel` |
| `lcmodel.basis` / `.bin` / `.license` | — | LCModel `.basis` file, binary, licence |
| `export_curves` | `true` | write plottable CSVs alongside the HTML report |
| `ppmlim` | `[0.2, 4.2]` | fit range |
| `baseline` | `poly,4` | higher order compensates for a basis without MM |
| `metab_groups` | `["NAA"]` | separate lineshape groups |
| `free_shift` | `true` | let the fit find peak positions |
| `internal_ref` | `["Cr","PCr"]` | falls back to NAA if it fits to zero |
| `combine` | NAA+NAAG etc. | metabolites reported as sums |
| `target_structure` | `null` | e.g. `hippocamp` — enables the placement check |
| `atlas_labels` | SIGMA labels CSV | label table for that check |

#### Preprocessing

`spectroscopy.preproc` selects `internal` (default) or `fsl_mrs_preproc`. Both
do coil combination, windowed alignment, outlier removal, averaging and
eddy-current correction; they differ in how they finish.

The stock pipeline ends with `shift_to_reference` and `phase_correct`, which
both take `argmax(|spectrum|)` in a hardcoded 2.9–3.1 ppm window and move that
point to 3.027 ppm. When the wrong point wins, the spectrum is displaced and
arbitrarily phased and nothing can be fit. On cuprizone data that cost 6–7 of
53 sessions, and the window is not adjustable from the command line.

The `internal` chain drops `shift_to_reference` — the converter has already
referenced the spectrum (below) — but **keeps** zero-order phasing, since
leaving phase to `fsl_mrs` as a free parameter costs about 30% of the fitted
SNR. `phase_method` selects how:

- `search` (default) scans the full ±180° circle, scoring the whole 0.5–4.2 ppm
  band for absorptive character. Covering the whole circle matters: `fsl_mrs`
  fits phase by local descent from zero with concentrations bounded
  non-negative, so a spectrum near 180° out cannot be recovered — the
  metabolites simply go to zero. Against phasing on tCr alone it was equal or
  better on all 53 sessions, six improved by 16–107%.
- `tcr` phases on the creatine peak over 2.95–3.10 ppm.

A session the fitter declines is retried with HLSVD water removal, then
reported as `unquantifiable` rather than counted as a failure, with its
preprocessed data left on disk. HLSVD is not the default because it costs SNR
on sessions that do not need it.

#### Frequency referencing

`fsl_mrs_preproc`'s window search assumes tCr is already near 3.027 ppm, and
nothing upstream guarantees that — its alignment step aligns the shots to each
other, not to an absolute chemical shift. So the converter references the
spectrum itself: water to its true shift from the unsuppressed reference, then
tCr onto 3.027. Across 52 sessions that moved tCr from 2.939 ± 0.012 — worst
case 0.019 ppm from falling out of the window — to 3.0269 ± 0.0007.

The tCr search window is deliberately wide (2.7–3.4 ppm), which is safe because
the result is cross-checked against NAA: the two singlets are a fixed 1.019 ppm
apart, so a misidentified peak is caught and the session falls back to water
referencing. Measured separation was 1.0212 ± 0.0010 with no failures.

#### Voxel localisation

Locating the voxel on the T2w is affine composition, as in human MRS. Bruker
writes the DICOM-equivalent geometry in `pdata/*/visu_pars`
(`VisuCoreOrientation`, `VisuCorePosition`, `VisuCoreExtent`), which defines
the index-to-world affine outright — signs and axis order included. A
parameter-reconstructed mapping survives only as a fallback when `visu_pars` is
unusable.

One piece is irreducible. A PRESS scan's `visu_pars` has `VisuCoreDim = 1` and
no spatial fields, so the voxel exists only in gradient coordinates while
images are in subject coordinates. The two differ by a signed permutation set
by `VisuSubjectPosition`, and it cannot be recovered from the files: with a
square FOV and the slice package at isocentre — 47 of 50 cuprizone sessions —
every candidate rotation reproduces the geometry equally well. It is therefore
calibrated once per subject position and validated (`Head_Supine` →
`diag(1,-1,-1)`, 71.2% mean hippocampal overlap against 32.7% for the next best
candidate), cross-checked per scan by requiring the gradient axes to align with
the image axes, and an unknown subject position raises rather than guesses.

**Check placement automatically.** Set `spectroscopy.target_structure` (a
substring matched against the atlas label table, e.g. `hippocamp`) and every
session is scored against the structure the voxel was aimed at, using the
parcellation anatomical preprocessing warps into subject space. The score lands
in the QC metrics, gates `overall_pass`, and gets a column in the study index.

This is worth enabling on every study, because a wrong geometry convention is
otherwise silent: the spectrum still fits and the concentrations still look
physiological — only the tissue fractions are wrong. Three sign errors were
found this way during development, each invisible on the majority of sessions
because the error was proportional to an offset that is zero for most
acquisitions (an unrotated voxel, a voxel at isocentre, an un-offset slice
package). A spot check on one session cannot find those; a score across a study
can. `read_anat_geometry` additionally warns on acquisitions outside what has
been validated — non-axial slices, 3D acquisitions, multiple slice packages, a
non-zero phase offset, or an unvalidated `RECO_transposition`.

#### Fitting, figures, and the macromolecule caveat

`spectroscopy.fitter` selects `fsl_mrs` (default) or `lcmodel`. LCModel needs
`spectroscopy.lcmodel.basis` — a `.basis` file, not the JSON directory
`spectroscopy.basis` points at.

`fsl_mrs` writes an interactive HTML report and a summary PNG but no fit curves
as data, so `export_curves` additionally writes `*_fit-curves.csv` and
`*_fit-metabolites.csv`: real-valued spectra over the fit range, so a figure
with the fit, baseline, residual and individual metabolite traces is a plain
matplotlib call over two CSVs.

LCModel is worth running as an independent check — same basis, same
preprocessed FID, different implementation, so agreement validates the whole
chain rather than just the fit. On cuprizone data the two agreed closely on the
major ratios (NAA+NAAG 1.25 vs 1.28, Glu 1.25 vs 1.26, GPC+PCh 0.188 vs 0.188
against tCr) while LCModel reported lower CRLBs and fit sessions `fsl_mrs`
could not.

They are **not interchangeable within an analysis**, because they treat
macromolecules differently. LCModel simulates 13 MM/lipid components at analysis
time (`NSIMUL`) with priors on their shifts, widths and concentration ratios;
these are never in the `.basis` file, so this is not something the JSON
conversion loses. On cuprizone data it fits MM09 at 1.17 relative to tCr,
comparable to NAA. FSL-MRS has no equivalent and pushes that signal into the
polynomial baseline.

Both adding FSL-MRS's default MM peaks and transcribing LCModel's `CHSIMU`
parameters were tested and rejected — the first made agreement slightly worse,
the second produced degenerate fits, because in LCModel the peaks only work
alongside constraint machinery FSL-MRS cannot express. A correct MM basis has
to be *measured*, with a metabolite-nulled acquisition; it cannot be simulated,
since MM is not a spin system. Until then, treat metabolites under the MM
resonances with more caution than NAA, tCr, Glu and Ins. The evidence is written
up in `mrs/basis/README.md` in the study tree (outside this repo).

#### Macromolecule areas, measured after the fit

What *can* be recovered without an MM basis is measured post hoc, by
`quantify_mm` (on by default; needs `export_curves`). It runs after the
metabolite fit and never feeds back into it, which is precisely why it is safe
— the failures above all came from letting MM components compete with the
baseline *during* fitting.

Because `fit` is metabolites plus baseline, the metabolite-free spectrum is
`residual + baseline`. Adding the baseline back is the point, not an oversight:
with an MM-free basis, the polynomial is where the MM signal went. A cubic
spline through that recovers the upfield envelope, where the basis is nearly
empty (lactate and alanine are its only occupants and both are subtracted).

Areas are measured against a line through MM-poor flank windows, not against
zero. A pedestal of about −0.15 runs the full width of the spectrum — present
even with the baseline switched off entirely — and integrating from zero gave
negative areas and a mean CV of 34% across baseline orders. Anchoring fixes
that, at the cost of a stated convention: areas are relative to the 1.55–1.80
ppm level.

The criterion for whether a band means anything needs no second fitting
package. The area is measured partly *from* the polynomial, so the failure mode
is that it reports the polynomial — in which case changing the baseline order
changes the answer. Across four sessions and baseline orders `poly,2`–`poly,5`:

Measured on the full study (87 sessions with a sound reference); baseline CV
from 4 sessions fitted at `poly,2`–`poly,5`:

| band | window | median /tCr | baseline CV | between-session CV | negative | verdict |
|------|--------|-------------|-------------|--------------------|----------|---------|
| MM09 | 0.70–0.95 | 0.566 | 5.2% | 26.7% | 0% | measurable |
| MM12 | 1.10–1.40 | 0.216 | 44% | 36.5% | 0% | `provisional` |
| MM14 | — | −0.020 | 88% | 138% | 78% | **not reported** |
| MM17 | — | 0.006 | — | 92% | 14% | **not reported** |

Those are two different quantities. Baseline CV is stability of one session
under a changed fit; between-session CV is spread across animals, mixing real
biology with measurement error. On the same sessions the metabolites give
between-session CVs of 8.1% (Glu+Gln), 8.8% (NAA+NAAG), 18.9% (GSH), 24.3%
(Tau) — so MM09 sits alongside Tau: usable, but needing larger groups than NAA
to detect an effect of a given size.

Whether MM14 and MM17 are unreportable because of the *data* or because of
where the anchor sits was tested directly, on 87 sessions across four zero
references (both flanks sloped/flat, upfield flank only sloped/flat). Dropping
the upper flank does free MM17 to be non-zero — and it is still negative in 26%
of sessions with a 184% robust CV. MM14 is negative in 39–86% under every
variant. MM09 is nearly indifferent to the choice (median 0.566–0.577, never
negative). The default is kept.

Plotting the envelope (`plot_mm_envelope`, included in the QC report) turned up
something the numbers alone hid: a systematic negative trough at 0.95–1.10 ppm
in every session, just downfield of the MM09 peak. It is in the acquired data,
not introduced by the fit — the raw spectrum averages −0.10 to −0.21 there,
about 3 standard errors below zero, while the fitted polynomial stays smooth and
positive and the metabolite model has almost no amplitude there to
over-subtract. Zero-order phase was tested and ruled out: the best rotation
varies from 18° to 53° between sessions and removes only about a quarter of it,
where a dispersion artifact would rotate away almost entirely. The cause is
unresolved.

That is why MM09 ends at 0.95 rather than the conventional 1.10 — the usual
window integrates straight through the trough and subtracts real signal.
Narrowing took its CV from 7.6% to 5.2% and its area from 0.40 to 0.56 /tCr.
MM12 is noise-limited rather than trough-limited (narrowing makes it worse,
44% → 60% → 88%), so it keeps the conventional window and its flag.

Only MM09 is claimed. MM14 and MM17 are omitted rather than emitted as numbers
that look like measurements; MM17 also defines the upper anchor and would be
zero by construction.

These are **area ratios, not concentrations**: no MM relaxation correction is
applied, and at TE 20 ms an unknown fraction of the MM signal has already
decayed. Nor can this separate macromolecules from instrumental baseline roll —
the polynomial absorbed both. It measures the one resonance that survives
without a metabolite-nulled acquisition; it does not replace one.

**The creatine denominator is checked before dividing.** `reference_area`
integrates the real part of the modelled creatine, so a fit that puts creatine
into dispersion collapses or inverts it while `fsl_mrs`' own concentration
parameter is unaffected. Five of 92 sessions did exactly that — one with a
*negative* reference area — and all five passed SNR, linewidth and placement QC
with entirely normal reported concentrations. Unguarded, one returned −11.7
/tCr from an ordinary MM area of 0.18. `check_reference` now refuses a
non-positive reference, or one holding under 5% of the total modelled
metabolite area (median is 10.2%); those sessions get a finite `area` and a
`NaN` ratio plus `mm_reference_ok: false` in QC, never a plausible wrong number.

Outputs are `{sub}_{ses}_mm-areas.csv` (band, area, area_per_tcr, ppm limits,
provisional flag) and `{sub}_{ses}_mm-envelope.csv` (ppm, signal, envelope,
anchor). The QC report plots them, and validated band areas are added to the
per-session QC metrics as `mm_mm09_per_tcr`; provisional bands stay in the CSV
and the figure, where their flag travels with them.
Configured by `spectroscopy.quantify_mm`, `mm_range`, `mm_knot_spacing` and
`mm_flanks`.

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
