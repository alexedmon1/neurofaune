# Results specification — `neurofaune.results` 0.1.0

What an analysis writes so that **anyone can read its results without the package
that produced them**: a plain pandas script, a manuscript table, neuro-lightbox.
Producers are neurofaune, neurovrai and any study's own orchestration scripts;
`RESULTS_PRODUCERS.md` says how each one meets this. neurofaune owns the
specification, its JSON Schemas (`neurofaune/results/schemas/`) and the
conformance checker (`neurofaune results check`).

**Status: 0.1.0, draft.** It becomes 1.0.0 when neuro-lightbox's MRI view and the
first study read it end to end. Until then a minor version may change a field;
from 1.0.0 on, versions follow semver and a reader written for 1.x reads every 1.y.

## 1. Principles

- **Open formats only.** Tables are CSV or TSV, metadata is JSON, maps are NIfTI.
  No pickles, no `.npz` as the only copy, nothing that needs one package to read.
- **Self-describing.** Every table has a column dictionary beside it (BIDS-style
  sidecar). Nothing is to be inferred from a column name, a filename or a folder.
- **Complete.** Every test that was run has its row, significant or not, with its
  effect and uncertainty. Tables are never truncated.
- **Explicit.** The correction and what it is over, the effect measure, the
  direction a test asks, units and each table's role are written down.
- **Provenance included.** Which package, version and commit; when; on what; and
  whether the run finished.
- **Workflow-neutral.** No field belongs to a workflow tool. Anything a study wants
  attached (a pre-registration id, a finding id) goes in the opaque `references`.
- **The reporting contract is part of the format** (§6): a result is not reported
  until its magnitude, direction, extent and location are stated, and a test that
  found nothing is still a result. The checker enforces it.

## 2. The analysis folder

An **analysis** is one family of tests run together under one inference scheme —
a TBSS battery over several measures, one VBM model, one covariance-network
comparison. Each analysis is one folder:

```
<analysis>/
  analysis.json          what the analysis is (§3)
  provenance.json        what produced it (§4)
  <table>.csv|.tsv       results; each with <table>.json, its column dictionary (§5)
  ...                    maps (NIfTI) and figures, each listed in analysis.json
```

- Every file the analysis reports is **listed in `analysis.json`** with a path
  relative to the folder. Paths stay inside the folder (no absolute paths, no `..`).
- The layout inside the folder is free. `tables/`, `maps/`, `figures/` is
  recommended for new producers; existing output trees adopt the spec in place by
  adding `analysis.json`, `provenance.json` and the column dictionaries.
- A results root holds any number of analysis folders at any depth. A reader finds
  them by looking for `analysis.json` files whose `spec` is `"neurofaune.results"`.
  Analysis folders do not nest.

## 3. `analysis.json`

Fields marked * are required.

| field | content |
|---|---|
| `spec`* | `"neurofaune.results"` |
| `spec_version`* | `"0.1.0"` |
| `id`* | identifier, unique under the results root and stable across re-runs, e.g. `"tbss/dose_p60"` |
| `title`*, `description`* | a heading, and what the analysis asks in a sentence or two |
| `analysis_type`* | `tbss`, `vbm`, `tbm`, `voxelwise`, `roi`, `covariance_network`, `nbs`, `graph`, `connectome`, `fixel`, `other` |
| `modality` | `anat`, `dwi`, `func`, `asl`, `msme`, `mrs`, `multimodal`, … |
| `measures` | the measured quantities, in display order, e.g. `["FA", "MD", "RD"]` |
| `role`* | `confirmatory`, `exploratory`, `descriptive` or `diagnostic` |
| `design`* | `{n*, groups: {name: n}, test_kinds: [...], records: [paths to design.json]}` |
| `inference`* | `{method*, correction*, n_permutations, notes}`; `correction` = `{p_kind*, family*, statement*, alpha}` (below) |
| `effect`* | `{measure*, definition*, ci_level, scope}`, or `{measure: null, reason*}` where no effect size applies |
| `tables`* | `[{path*, role*, rows*, description*, headline}]` (below) |
| `maps` | `[{path*, kind*, description*, space, measure, contrast, values}]` |
| `figures` | `[{path*, caption*, measure, contrast}]` |
| `decision` | where the analysis defines a decision rule: `{rule*, outcome*, criteria: [{name, description, passed}]}`; `outcome` is `holds`, `does_not_hold` or `not_assessed` |
| `retired` | `{reason*, superseded_by}` — kept on disk, not to be read as current |
| `caveats` | `[sentence, …]` that a reader must see beside the results |
| `references` | `[{label, value}]`, opaque to the spec |

**`inference.correction`.** `p_kind` is `fwe`, `fdr`, `perm` (permutation p of a
single test statistic), `uncorrected` or `none`. `family` says what the correction
is over ("voxels in the skeleton mask, per contrast"; "the 11 measures"). `statement`
is the sentence a methods section would carry ("TFCE, FWE p < 0.05 by 5,000
permutations, per contrast over the skeleton"). A table column may carry its own
`PKind` (§5) where it differs — a cluster table's uncorrected minimum p, say.

**`tables[].role`** — what one row is, so no reader guesses from a filename:

| role | one row is | contract (§6) |
|---|---|---|
| `tests` | one test that was run (a contrast on a measure) | checked |
| `clusters` | one spatial cluster of one test | checked |
| `elements` | one ROI / edge / component of one test | checked |
| `descriptives` | a group summary (means, SDs, n) | not checked |
| `other` | anything else, described in `rows` | not checked |

`rows` says what one row is in words; `headline: true` marks the table a reader
should show first (at most one per analysis).

**`maps[].kind`**: `stat`, `p_corrected`, `p_uncorrected`, `effect`, `mask`,
`background`, `input`, `other`. A `background` map is what the other maps are drawn
on (a template, the mean FA); a reader draws on the `mask` when there is none.
A `p_corrected` / `p_uncorrected` map states what its voxels hold: `values` is `p`
or `one_minus_p` (randomise's convention). A reader does not threshold a p map that
does not say. A map that belongs to one test names its `measure` and
`contrast`, matching that test's row. `space` names the template (e.g. `SIGMA`,
`MNI152NLin2009cAsym`).

## 4. `provenance.json`

| field | content |
|---|---|
| `spec`*, `spec_version`* | as in `analysis.json` |
| `generated_by`* | BIDS `GeneratedBy`: `[{Name*, Version*, CommitID, RequestedRevision, Description}]`, the producing package first, then the tools it ran (FSL, ANTs) |
| `run`* | `{status*, start*, end, id}`; `status` is `completed`, `partial`, `failed` or `running` |
| `inputs` | `[{path*, role, sha256}]` |
| `subjects` | `{n, groups: {name: n}}` |
| `settings` | the parameters the analysis ran with |
| `environment` | `{python, platform}` |

A reader shows a run whose status is not `completed` as such, never silently.

## 5. Tables and column dictionaries

- **CSV or TSV**, chosen by the extension (`.csv` comma, `.tsv` tab); UTF-8; one
  header row; one row per what `rows` says. An empty cell is a missing value.
  Booleans are `true` / `false` (`True` / `False` accepted). Lists inside a cell
  are separated by `"; "`.
- **Column dictionary**: `<table stem>.json` beside the table, BIDS-style — one key
  per column, every column present, no entries for absent columns:

```json
{
  "whole_d": {
    "Description": "Cohen's d of the contrast on each subject's mean over the mask (unselected)",
    "Standard": "effect_size",
    "EffectMeasure": "d",
    "EffectScope": "whole mask"
  },
  "min_p_fwe": {
    "Description": "smallest FWE-corrected voxel p in the map",
    "Standard": "p_value",
    "PKind": "fwe",
    "PScope": "minimum over the mask"
  }
}
```

  `Description` is required; `Units` and `Levels` (`{value: meaning}`) are BIDS's;
  `Standard` maps the column to the vocabulary below, so a reader finds "the
  effect size" without knowing the producer's column names. A standard term is
  claimed by at most one column per table.

**Standard terms** (qualifiers in brackets; * = required with the term):

| term | meaning |
|---|---|
| `measure` | the measured quantity of the row (FA, ReHo, …) |
| `contrast` | test identifier within the analysis; `measure` + `contrast` (+ `facet`) identify a test |
| `contrast_label` | the contrast in words |
| `facet` | any further split of the tests (window, timepoint, family) |
| `element` | per-element rows: the cluster id, ROI name, edge `"A -- B"`, component id |
| `test_kind` | `two_group`, `one_sample`, `regression`, `interaction`, `custom` |
| `tested_direction` | what a positive statistic means, in words: `"cuprizone > control"`, `"mean > 0"` |
| `observed_direction` | the direction the effect actually went, in the same words |
| `group_a`, `group_b` | two-group tests: `group_a` is higher when the statistic is positive |
| `n`, `n_a`, `n_b`, `df` | sample sizes and error degrees of freedom |
| `effect_size` | [`EffectMeasure`*: `d`, `g`, `r`, `beta`, `eta2`, …; `EffectScope`] |
| `effect_ci_low`, `effect_ci_high` | [`CILevel`] bounds on `effect_size` |
| `effect_selected` | true where the effect was computed on elements chosen for significance (inflated) |
| `estimate` | the contrast in raw units [`Units`] |
| `mean_a`, `mean_b`, `mean` | group means / the one-sample mean [`Units`] |
| `stat` | [`StatName`*: `t`, `F`, `z`, …; `StatScope`: `peak`, `element`] |
| `p_value` | [`PKind`*; `PScope`] |
| `significant` | [`Alpha`] |
| `n_significant` | elements (voxels, edges) significant under the correction [`Units`, `PKind`, `Alpha`] |
| `frac_significant` | the same as a fraction of the mask / network |
| `n_voxels`, `volume_mm3` | a cluster's extent |
| `peak_xyz_mm` | [`Space`*] peak coordinate, `"x y z"` |
| `peak_region` | atlas region at the peak |
| `regions` | every region covered, `"name:count; name:count"` |
| `crosses_midline` | the cluster has voxels in both hemispheres |

Columns with no standard term are welcome; their dictionary entry describes them.

## 6. The reporting contract

`neurofaune results check` enforces this on tables whose role is `tests`,
`clusters` or `elements`. A missing item is an error.

**`tests`** — one row per test that was run, significant or not:
- identity: `contrast` (and `measure` where the analysis has several);
- direction: `tested_direction`, or `test_kind` with `group_a` / `group_b` for
  two-group tests;
- magnitude: `effect_size` with its `EffectMeasure` and `effect_ci_low` /
  `effect_ci_high` — or, where no effect applies, `analysis.json`
  `effect = {measure: null, reason}`;
- sample: `n`, or `n_a` and `n_b`;
- significance: `p_value` with its `PKind`; the family it is corrected over is in
  `analysis.json`;
- extent, for voxelwise tests: `n_significant` or `frac_significant`.

**`clusters`** — every cluster under a stated definition:
- which test: `contrast` (and `measure`), matching a `tests` row;
- extent: `n_voxels` or `volume_mm3`;
- location: `peak_xyz_mm` with its `Space`, and `peak_region`;
- magnitude: `stat` with its `StatName` (the peak), or `effect_size` — and
  `effect_selected` wherever an effect is given, since cluster effects are selected;
- significance: `p_value` with its `PKind`.

**`elements`** — one row per ROI / edge / component tested:
- which test: `contrast` (and `measure`); the element: `element`;
- magnitude: `effect_size` with its `EffectMeasure` (signed), or `stat` with `StatName`;
- significance: `p_value` with its `PKind`.

The checker cannot see a test that was run and left out. A producer that knows how
many tests it ran states `"n_rows"` on the table's entry in `analysis.json`, and
the checker counts.

## 7. Conformance

```
neurofaune results check <analysis folder or results root> [--json]
```

checks every analysis folder it finds: the JSON files against the schemas, every
listed file present and inside the folder, every table against its dictionary, the
standard terms and their qualifiers, and the contract (§6). Exit status 0 when every
folder conforms. The schemas in `neurofaune/results/schemas/` are plain JSON Schema
(draft 2020-12) and work with any validator; the checker itself needs only the
standard library, so it can run where neurofaune's imaging dependencies cannot.

## 8. Versioning

`spec_version` is semver. A reader accepts any version with the major version it
was written for and reports, never guesses, a field it does not know. Producers
write the version they were built against. Changes are listed in
`neurofaune/results/CHANGELOG.md`.
