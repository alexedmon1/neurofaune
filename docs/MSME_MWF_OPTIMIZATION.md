# MSME Myelin Water Fraction — Optimization Notes

**Date:** 2026-02-25
**Data:** Dev_4 Scan 6, sub-PracticeRat4 / ses-dev4msme6
**Tool:** neurofaune (`~/sandbox/neurofaune`)

---

## Acquisition Parameters

| Parameter | Value |
|-----------|-------|
| Sequence | Bruker MSME |
| Matrix | 160 x 160 |
| Slices | 11 (0.8mm thick, 0.2mm gap, 1.0mm center-to-center) |
| FOV | 32 x 32 mm |
| In-plane resolution | 0.2 mm |
| TR | 3700 ms |
| Echoes | 32 at 10 ms spacing (TE = 10–320 ms) |
| Averages | 4 (scanner-averaged) |
| Repetitions | 1 |

---

## NNLS Method Overview

The pipeline uses regularized Non-Negative Least Squares (NNLS) following
Whittall & MacKay (1989) and Prasloski et al. (2012):

- **T2 basis:** 120 log-spaced components from 10–2000 ms
- **Regularization:** 2nd-order Tikhonov (minimum curvature)
- **Lambda selection:** chi-square discrepancy principle — lambda is chosen
  so that the fit residual chi² ≈ n_echoes × sigma² × chi2_factor
- **Compartment boundaries:** Myelin water < 25 ms, IEW 25–200 ms, CSF > 200 ms

---

## Iterations Summary

### Run 1 — Original (adaptive skull strip + global lambda, pooled noise)

**Skull strip:** adaptive slice-wise BET, target_ratio=0.15, cog_offset_y=-40
**Lambda estimation:** Global lambda from 200-voxel subset. Noise estimated by
pooling last-4-echo values from all subset voxels into one list and computing variance.

| Metric | Value |
|--------|-------|
| Brain voxels | 51,357 |
| Extraction ratio | 18.2% |
| SNR | 2.19 |
| Noise sigma² | 519,988 |
| Lambda | ~1,000,000 (ceiling) |
| MWF mean | 0.415 |
| MWF median | 0.430 |
| IWF median | 0.526 |
| CSF mean | 0.093 |
| T2 median | 57.6 ms |

**Problems identified:**
1. **Skull strip too loose.** Adaptive method forces 15% extraction per slice
   regardless of brain content. With 11 slices, edge slices with little brain
   get filled with skull/muscle.
2. **Noise massively overestimated.** Pooling last-4-echo values across 200
   voxels conflates inter-voxel signal variation (CSF voxel at TE=300ms ≈ 5000
   vs WM voxel ≈ 100) with actual noise. sigma² = 520k is ~25x too high.
3. **Lambda at ceiling (10^6).** Because the chi² target was enormous (17M),
   any lambda produced chi² below target, so bisection pushed lambda to the
   maximum. Over-regularization smeared the T2 spectrum, bleeding signal into
   short-T2 (myelin water) bins.
4. **MWF ~42% everywhere.** No WM/GM contrast. Entire brain saturated.

---

### Run 2 — atropos_bet skull strip (same NNLS)

**Change:** Switched `config.yaml` msme.skull_strip.method from `adaptive` to
`atropos_bet` (3-class Atropos + BET refinement). This is the proper 3D method
for data with ≥10 slices.

| Metric | Value | Change |
|--------|-------|--------|
| Brain voxels | 32,956 | -36% (tighter) |
| Extraction ratio | 11.7% | |
| SNR | 3.46 | +58% |
| MWF mean | 0.397 | similar (lambda unchanged) |
| CSF mean | 0.069 | lower |
| T2 mean | 73.1 ms | lower |

**Result:** Tighter mask, better SNR, but MWF unchanged because the NNLS
lambda bug was still present. The `potential_over_stripping` flag was set.

---

### Run 3 — Fixed noise estimation (per-voxel echo-to-echo differences)

**Change:** Replaced the noise estimator. Instead of pooling all last-4-echo
values, computes per-voxel noise from echo-to-echo differences in the tail
(last 6 echoes), using var(diff)/2 to correct for differencing. Takes median
across voxels.

| Metric | Value | Change |
|--------|-------|--------|
| Noise sigma² | 8,182 | was 520k |
| Chi² target | 267,044 | was 17M |
| Lambda | 0.0000 (floor) | was 10^6 |
| MWF mean | 0.023 | |

**Problem:** Lambda swung to the opposite extreme. With sigma²=8,182 the chi²
target was 267k, but for 99% of voxels the unregularized chi² was already
above this target. The noise estimate was too LOW because echo-to-echo
differences at the Rician noise floor only capture ~43% of true Gaussian
variance (Rician: var = sigma² × (2 - pi/2)).

**NNLS spectra:** All single delta spikes (no regularization = no smoothing).
MWF ≈ 0 for most voxels because the single spike landed in the IEW band.

---

### Run 4 — Per-voxel lambda bisection (proper Whittall & MacKay)

**Changes:**
1. Replaced global lambda with per-voxel lambda estimation in the main
   fitting loop. Each voxel gets its own lambda via 15-step bisection.
2. Voxels where unregularized chi² ≥ target use the unregularized solution.
3. Only voxels with chi² < target get regularized.

| Metric | Value |
|--------|-------|
| Noise sigma² | 7,937 (per-voxel tail, median) |
| Voxels regularized | 291 / 32,933 (0.9%) |
| Median lambda (regularized) | 137,327 |
| MWF mean | 0.023 |

**Problem:** Same noise underestimate. Only 0.9% of voxels had chi² below
target and got regularized. The rest (99.1%) used unregularized solutions.
Same result as Run 3.

---

### Run 5 — Background noise from all non-brain voxels

**Change:** Estimated sigma² from the variance of all background voxels
(outside brain mask) in the first echo, with Rayleigh→Gaussian correction.

| Metric | Value |
|--------|-------|
| Background variance | 13,292,507 |
| Gaussian sigma² | 30,970,162 (sigma = 5,565) |
| Lambda | ~1,000,000 (ceiling, all voxels) |
| MWF mean | 0.397 |

**Problem:** The "background" includes skull, muscle, and fat tissue with
real signal — not just air/noise. The variance was dominated by tissue
contrast, not noise. sigma = 5,565 is absurd when max signal = 32,766.
Lambda hit ceiling again, producing the same over-regularized MWF ≈ 40%.

---

### Run 6 — Air-only noise estimation (CURRENT)

**Change:** Instead of all background voxels, use only the lowest quartile
(≤ Q25) of background intensity in the first echo. These are the true
air/noise voxels. Estimate sigma from the Rayleigh mean:
sigma² = mean² × (2/pi).

| Metric | Value |
|--------|-------|
| Background voxels | 248,669 |
| Noise-floor voxels (≤ Q25=351) | 62,293 |
| Noise mean | 233.6 |
| **Gaussian sigma²** | **34,733 (sigma = 186)** |
| Chi² target | 1,133,688 |
| Voxels regularized | 24,334 / 32,931 (73.9%) |
| Voxels unregularized | 8,597 / 32,931 (26.1%) |
| **Median lambda (regularized)** | **2,685** |
| **MWF mean** | **0.101** |
| MWF median | 0.042 |
| MWF std | 0.133 |
| MWF range | [0.000, 0.983] |
| IWF mean | 0.842 |
| IWF median | 0.920 |
| CSF mean | 0.058 |
| CSF median | 0.013 |
| T2 mean | 73.0 ms |
| T2 median | 56.6 ms |
| Brain voxels | 32,931 |
| Skull strip method | atropos_bet (3-class) |

**Result:** First run with plausible MWF values showing WM/GM contrast.
White matter regions (corpus callosum, external capsule, fimbria) show
MWF ≈ 0.15–0.25. Gray matter ≈ 0.02–0.08. NNLS spectra show smooth,
multi-component distributions with distinct IEW peaks and small MW peaks.

26% of voxels were unregularized (chi² already above target) — these tend
to be lower-SNR voxels where the NNLS still produces single spikes. This
biases the MWF distribution toward zero for those voxels (MWF median = 0.04
vs mean = 0.10).

---

## Comparison: Original vs Current

| Metric | Original (Run 1) | Current (Run 6) | Notes |
|--------|-------------------|------------------|-------|
| Skull strip | adaptive (per-slice BET) | atropos_bet (3D) | 36% fewer voxels |
| Brain voxels | 51,357 | 32,931 | Tighter mask |
| SNR | 2.19 | 3.47 | Much better |
| Noise sigma² | 519,988 | 34,733 | 15x lower |
| Lambda | ~10^6 (global, ceiling) | 2,685 median (per-voxel) | Proper regularization |
| Lambda method | Global from subset | Per-voxel bisection | Whittall & MacKay proper |
| MWF mean | 0.415 | 0.101 | Was 4x too high |
| MWF median | 0.430 | 0.042 | |
| MWF spatial pattern | Uniform ~0.4 everywhere | WM/GM contrast visible | Key improvement |
| IWF median | 0.526 | 0.920 | Was suppressed by MW |
| CSF mean | 0.093 | 0.058 | Less contamination |
| T2 median | 57.6 ms | 56.6 ms | Unchanged (independent fit) |
| Runtime | ~1 min | ~40 min | Per-voxel bisection cost |

---

## Code Changes Made

All changes in `neurofaune/preprocess/workflows/msme_preprocess.py`:

### 1. Noise estimation
- **Removed:** `_estimate_nnls_lambda()` — pooled all tail echoes across voxels
- **Added:** `_estimate_noise_sigma2()` — per-voxel echo-to-echo differences
  with Rician correction fallback
- **Added:** Background noise estimation in MSME workflow — uses lowest
  quartile of non-brain first-echo voxels (air region), Rayleigh→Gaussian
  correction via `sigma² = mean² × 2/pi`

### 2. Per-voxel lambda
- **Removed:** Global lambda applied to all voxels
- **Added:** Per-voxel lambda bisection (15 iterations) in main fitting loop
- For each voxel: solve unregularized first; if chi² < target, bisect to
  find lambda where chi² rises to target; otherwise use unregularized solution
- Pre-allocated augmented matrix to reduce memory allocation overhead

### 3. Skull stripping
- **Changed:** `config.yaml` msme.skull_strip.method from `adaptive` to
  `atropos_bet`
- atropos_bet uses 3-class K-means segmentation + BET refinement
- Appropriate for 11-slice data (adaptive was designed for ≤5 slices)

### 4. Function signature
- Added `noise_sigma2` parameter to `calculate_mwf_nnls()` — allows
  pre-computed noise estimate to be passed in (preferred over in-function
  estimation from masked data)

---

## Remaining Concerns

1. **MWF median (0.04) vs mean (0.10).** Large gap suggests many voxels at
   MWF=0 (the 26% unregularized voxels). These produce delta-spike spectra
   with all signal in one bin. Could be improved by applying a small minimum
   lambda to all voxels.

2. **Runtime (~40 min).** Per-voxel bisection with 15 iterations × 33k voxels
   is expensive. Possible optimizations:
   - Reduce to 10 bisection steps (minimal accuracy loss)
   - Use subset-estimated lambda as starting point to narrow bisection range
   - Vectorize the NNLS (batch solver)

3. **potential_over_stripping flag.** The atropos_bet mask extraction ratio
   is 11.7%. The QC flags this as possible over-stripping. Visual inspection
   suggests the mask is reasonable but could be checked against a manual trace.

4. **Comparison to prior results not possible.** The user noted these MWF
   values are lower than previously obtained (in a session before this one).
   The old data is not currently accessible for direct comparison.

5. **chi2_factor = 1.02.** This is the Prasloski et al. recommendation but
   could be tuned. Higher values (1.05–1.10) would produce smoother spectra
   at the cost of losing fine spectral detail.

6. **T2 range starts at 10 ms.** The first TE is also 10 ms, meaning the
   shortest T2 components have already decayed by exp(-1) = 37% at the first
   measurement. Components with T2 < 10 ms are essentially invisible.
   Literature suggests myelin water T2 in rodent brain at 7T is ~8–15 ms,
   so we may be missing the shortest MW components.

---

## Literature Comparison

### Our Current Values (Run 6)

| Region | MWF | Notes |
|--------|-----|-------|
| Whole brain mean | 0.10 (10%) | |
| Whole brain median | 0.04 (4%) | Biased low by unregularized voxels |
| White matter (visual) | ~0.15–0.25 | CC, external capsule, fimbria |
| Gray matter (visual) | ~0.02–0.08 | Cortex, hippocampus |

### Rodent Studies

**Kozlowski et al. (2008)** — *High-resolution myelin water measurements in rat spinal cord*
[PubMed](https://pubmed.ncbi.nlm.nih.gov/18302247/) |
[Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.21527)
- **Species:** Rat spinal cord (ex vivo and in vivo)
- **Field:** 7T
- **Method:** Multi-echo SE, NNLS analysis
- **Resolution:** 61–117 µm in-plane
- **MWF in WM:** Correlated with Luxol Fast Blue stain (R²=0.95)
- **Notes:** Foundational rat MWI study. Showed that high-resolution MWF maps
  faithfully reflect myelin distribution.

**Dula et al. (2010)** — *Multiexponential T2, magnetization transfer, and
quantitative histology in white matter tracts of rat spinal cord*
[PubMed/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2852261/) |
[Wiley](https://onlinelibrary.wiley.com/doi/10.1002/mrm.22267)
- **Species:** Rat spinal cord (ex vivo)
- **Method:** Multi-echo SE, NNLS
- **MWF in WM:** 10–35% across four spinal cord WM tracts
- **Key finding:** Wide MWF variation (10–35%) in tracts with similar histological
  myelin content. Attributed to intercompartmental water exchange effects.

**Harkins et al. (2012)** — *Effect of intercompartmental water exchange on
the apparent myelin water fraction in multiexponential T2 measurements of
rat spinal cord*
[PubMed](https://pubmed.ncbi.nlm.nih.gov/21713984/) |
[Wiley](https://onlinelibrary.wiley.com/doi/10.1002/mrm.23053)
- **Species:** Rat spinal cord
- **Key finding:** Water exchange between myelin and non-myelin compartments
  causes underestimation of MWF. The apparent MWF can vary by ~2x between
  regions with similar true myelin content. Exchange rate is a confound
  for MWF quantification.

**Thiessen et al. (2013)** — *Quantitative MRI and ultrastructural examination
of the cuprizone mouse model of demyelination*
[PubMed](https://pubmed.ncbi.nlm.nih.gov/23943390/) |
[Wiley](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/nbm.2992)
- **Species:** Mouse brain (in vivo)
- **Field:** 7T (Bruker)
- **Method:** Multi-echo SE, 10 ms echo spacing
- **MWF:** No visible myelin water component detected with 10 ms echo spacing
- **Key finding:** At 7T with 10 ms minimum TE, the myelin water T2 component
  (~8–12 ms at 7T) may be too short to detect reliably. Only 1 echo samples
  the MW component. MTR was more sensitive to demyelination than T2-based MWF.
- **Relevance:** Our data also uses 10 ms echo spacing at 7T. This raises
  questions about whether we can reliably detect the myelin water component,
  or whether what we're measuring is partially an artifact of the fitting.

**Dula et al. (2016) / Soustelle et al. (2016)** — *Assessment of the myelin
water fraction in rodent spinal cord using T2-prepared ultrashort echo time MRI*
[PubMed](https://pubmed.ncbi.nlm.nih.gov/27394911/)
- **Species:** Mouse and rat spinal cord (ex vivo and in vivo)
- **Field:** 9.4T
- **Method:** T2-prepared UTE (ultrashort TE) to capture very short T2 components
- **MWF:** Multicomponent T2 detected in only 12% of voxels in rat spinal cord
  and 6% in mouse spinal cord
- **Key finding:** At high field, the myelin water T2 is so short that standard
  multi-echo SE cannot reliably detect it. UTE methods are needed.

**Canales-Rodríguez et al. (2025)** — *Validation of a data-driven
multicomponent T2 analysis for quantifying myelin content in the cuprizone
mouse model of multiple sclerosis*
[PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0323614)
- **Species:** Mouse brain (cuprizone model)
- **Method:** Data-driven mcT2 analysis
- **MWF:** Control vs cuprizone mice across 6 brain regions. Corpus callosum
  showed ~26.5% reduction in MWF. Strong correlation with histology (R²=0.61–0.64).
- **Key finding:** Data-driven approaches may be more robust than traditional
  NNLS for rodent MWF at high field.

### Human Studies (for reference)

**Laule et al. (2006, 2007, 2008)** — *Myelin water imaging in multiple
sclerosis* (review and validation studies)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/17263002/)
- **MWF cutoff:** T2 < 30 ms (broader than the 25 ms we use)
- **Corpus callosum:** ~12% MWF
- **Normal WM:** ~10–15% MWF
- **Gray matter:** ~2–5% MWF
- **Validation:** Strong correlation with histological myelin staining
  (R² = 0.67, range 0.45–0.92)

**Prasloski et al. (2012)** — *Applications of stimulated echo correction
to multicomponent T2 analysis*
[Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.24108)
- **Method reference:** Introduced the chi-square discrepancy principle for
  NNLS regularization (the method we implement). Also introduced stimulated
  echo correction for GRASE sequences. chi2_factor = 1.02 recommended.
- **Typical human WM MWF:** 10–18%

**Liu et al. (2020)** — *Myelin water imaging data analysis in less than
one minute*
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1053811920300380)
- **Method reference:** DECAES (Julia implementation). Fast per-voxel NNLS
  with chi-square regularization. Gold standard implementation.

**Multi-site studies:**
- **Human WM MWF:** Typically 10–18% (major WM tracts)
- **Human corpus callosum:** ~12% (splenium slightly higher than genu)
- **Human gray matter:** ~2–5%
- **Human whole brain WM mean:** ~11%

### Summary Comparison

| Measure | Our Data (7T, rat) | Human WM (3T) | Rat SC WM (7T) | Notes |
|---------|-------------------|----------------|-----------------|-------|
| WM MWF | ~15–25% | 10–18% | 10–35% | Higher at 7T? Exchange effects? |
| GM MWF | ~2–8% | 2–5% | — | Consistent |
| CC MWF | ~15–20% (est.) | ~12% | — | Slightly higher |
| Whole brain mean | 10% | ~11% | — | Comparable |

### Key Considerations for Our Data

1. **Echo spacing = minimum TE = 10 ms.** At 7T, myelin water T2 is estimated
   at ~8–15 ms. With TE starting at 10 ms, the first echo has already lost
   37–71% of the MW signal (exp(-10/15) to exp(-10/8)). This means we are
   partially blind to the shortest MW components. Thiessen et al. (2013)
   found no detectable MW component at 7T with 10 ms echo spacing.

2. **Intercompartmental water exchange.** Dula et al. (2010) and Harkins et al.
   (2012) showed that exchange between myelin and non-myelin water compartments
   causes the apparent MWF to vary by up to 2x, independent of true myelin
   content. This is a fundamental limitation of multi-echo T2-based MWF.

3. **Rician noise floor.** Magnitude data at the noise floor follows a Rician
   distribution, not Gaussian. The NNLS model assumes Gaussian residuals.
   Late echoes at the Rician floor introduce systematic bias. Our noise
   estimation accounts for this by estimating from the air region.

4. **4 averages.** The scanner-averaged data has reasonable SNR (sigma ≈ 186
   for first-echo signal max of ~32,766, SNR ≈ 176). This is adequate for
   NNLS decomposition.

5. **Our MWF values are plausible** but should be interpreted cautiously given
   the 10 ms TE floor at 7T. The WM/GM contrast is in the expected direction,
   and whole-brain mean MWF (10%) is consistent with literature. However,
   the true myelin water component may be underestimated due to T2 decay
   before the first echo.
