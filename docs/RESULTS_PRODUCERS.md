# Producing results — neurofaune, neurovrai, study scripts

How each producer writes analysis folders that meet `RESULTS_SPEC.md`. The
specification and its checker live in neurofaune; every producer, neurofaune
included, is held to the same checker.

## 1. The recipe (any producer)

1. **Write each table** — CSV or TSV, one row per test / cluster / element, every
   test that was run, nothing truncated.
2. **Describe every column** in a dictionary beside the table, mapping each
   column to a standard term where one applies (with its qualifiers: `PKind` on a
   p value, `EffectMeasure` on an effect size, `Space` on a coordinate).
3. **Write `analysis.json` and `provenance.json`** and check the folder:

```python
from neurofaune.results import provenance, write_analysis, write_columns
from neurofaune.results.write import now

tests.to_csv(out / "tests.csv", index=False)
write_columns(out / "tests.csv", MY_COLUMNS)            # every column described, or KeyError
write_analysis(out, {
    "id": "wmh/poor_vs_good", "title": "WMH burden by glycaemic control",
    "description": "...", "analysis_type": "roi", "modality": "anat",
    "measures": ["total_volume_mm3", "n_lesions"], "role": "exploratory",
    "design": {"n": 80, "groups": {"poor_control": 40, "good_control": 40}},
    "inference": {"method": "Welch t", "correction": {
        "p_kind": "fdr", "family": "the 6 WMH metrics",
        "statement": "Benjamini-Hochberg q < 0.05 over the 6 metrics"}},
    "effect": {"measure": "d", "definition": "pooled-SD Cohen's d", "ci_level": 0.95},
    "tables": [{"path": "tests.csv", "role": "tests", "headline": True, "n_rows": len(tests),
                "rows": "one WMH metric", "description": "every metric, significant or not"}],
}, provenance([{"Name": "neurovrai", "Version": "0.2.0", "CommitID": "..."}],
              status="completed", start=started, end=now()))   # raises if non-conforming
```

4. **Test it**: the producer's own test suite builds a small synthetic analysis
   per analysis type and asserts that `neurofaune.results.check(folder)` passes.
   On real outputs: `neurofaune results check <results root>`.

## 2. Getting the checker and writer

`neurofaune.results` is standard-library only, uses relative imports and imports
nothing else from neurofaune. Three ways in:

- **Depend on neurofaune**, pinned to a commit or tag like any other pin
  (`neurofaune @ git+https://github.com/alexedmon1/neurofaune@<ref>`). neurofaune's
  top-level import is empty, so `import neurofaune.results` loads nothing heavy —
  but installing neurofaune pulls its imaging dependencies (nipype, dipy,
  nilearn < 0.11, …).
- **Run the checker in isolation**, without touching the producer's environment:
  `uvx --from "neurofaune @ git+https://github.com/alexedmon1/neurofaune@<ref>" neurofaune results check <dir>`.
- **Copy `neurofaune/results/`** into the producer as a vendored package, and record
  the neurofaune commit it came from. A copy must be refreshed whenever
  `spec_version` changes; the refresh is the copy, never an edit.

The randomise read-out (`neurofaune.analysis.stats.readout` and `readout_results`)
needs numpy, scipy, pandas and nibabel, and is not standard-library.

## 3. neurofaune

| analysis | writes the spec | how |
|---|---|---|
| TBSS (`analysis.tbss.run_tbss_stats`) | yes | `readout_results.write_readout_results` |
| VBM, voxelwise fMRI (`analysis.randomise_analysis`) | yes | the same |
| ROI extraction / ROI statistics | not yet | an `elements` / `tests` writer |
| covariance networks, NBS, connectomes, fixel, MCCA, classification | not yet | per §6 below, analysis by analysis |

A non-conforming folder at the end of a long run is logged, not raised. Tests
assert conformance for every writer.

## 4. neurovrai

Surveyed on 2026-10-06 at neurovrai `4e830a2` (master), by reading the code.
Nothing was run. The four items marked † were re-checked in the source.

### 4.1 Where it stands

- **No design records.** `design_summary.json` comes in two incompatible schemas
  (neuroaider's, and `stats/design_matrix.py`'s). Neither says what a contrast tests
  or which subject a row is; row ids live in `subject_list.txt`.
- **No test-level read-out.** Nothing like `read_randomise`: no tests table (n, df,
  whole-mask effect with CI, direction, extent) and no clusters table with mm³,
  peak coordinates and named regions.
- **Effect sizes are only maps.** `stats/effect_size.py` assumes a one-sample design
  (d = t / √N) for any `.mat` and is marked untested (`TODO_EFFECT_SIZE_TESTING.md`).
- **The correction is named only in filenames**, never in a table.
- **No provenance**: no version, commit, run id or input record.
- **Discovery is by hardcoded paths**; each module writes its own HTML.

### 4.2 Fix first — the outputs would be described wrongly otherwise

1. † **The TFCE flavour.** `stats/randomise_wrapper.py` always passes `--T2` (2-D
   TFCE, the skeleton setting) when `tfce=True`, and VBM, ASL CBF and the T1w/T2w
   ratio call it on 3-D data. FWE control is still valid under permutation, but the
   enhancement is not the standard 3-D one, and a methods section saying "TFCE"
   would be wrong. Add a `tfce_2d` switch, as neurofaune's wrapper has, and set it
   per analysis.
2. † **ReHo / fALFF variance smoothing.** `func/batch_reho_falff_analysis.py` passes
   `-v 5` commented as "Verbose". In randomise, `-v` is variance smoothing (σ = 5 mm),
   so those statistics are pseudo-t on a smoothed variance. Remove it, or keep it
   deliberately and record it in the design and the inference statement.
3. † **The T1w/T2w cluster threshold.** `anat/t1t2ratio_workflow.py` passes
   `cluster_threshold = 0.95` to `create_enhanced_cluster_report`, which expects a p
   threshold (0.05).
4. † **The TBSS and VBM entry points.** `run_tbss_stats.run_tbss_statistical_analysis`
   and `vbm_workflow.run_vbm_analysis` reference undefined names after randomise
   (`contrasts`, `design_result`, `run_fsl_glm`, `threshold_zstat`, `formula`, …), so
   they would crash. The TBSS path that works is
   `scripts/batch/batch_tbss_cluster_analysis.py`, with a hardcoded study path.
5. **Cluster CSVs are probably empty.** `stats/cluster_report.py` runs FSL `cluster`
   on the t map with `--thresh=0`, ignoring the significance mask it just wrote, and
   splits a header containing "Cluster Index" on whitespace, so no row matches.
6. **Dual regression** accepts `contrast_con` and never passes it on.
7. **Packaging.** statsmodels is imported (`connectome/group_analysis.py`) but not
   declared; the version is 0.2.0 in pyproject, "2.0.0-alpha" in `__version__`.

### 4.3 Getting to the spec, analysis by analysis

**Decision for the author:** whether neurovrai depends on neurofaune (pinned) for
the spec, the design records and the randomise read-out, or copies them (§2).
Recommended: **depend.** neurofaune's randomise wrapper and cluster report began as
copies of neurovrai's and have since diverged — the drift a second copy would
repeat. The modules involved import only numpy, scipy, pandas and nibabel;
neurovrai needs Python ≥ 3.13, which neurofaune supports, and both pin
nilearn < 0.11.

**Randomise analyses** — TBSS, VBM, ReHo / fALFF, ASL CBF, T1w/T2w ratio, dual
regression stage 3:

1. Every design gets a design record (`neurofaune.analysis.stats.design_record`:
   `write_design`, or `write_design_record` from a neuroaider `describe()`). The
   record names every row, says what every column is, and gives each contrast its
   sentence, `test_kind` and, for two-group tests, `group_a` / `group_b`.
2. randomise runs with `--uncorrp`, so uncorrected extents are reported too.
3. `read_randomise(run_dir, data_4d, design.mat, mask, atlas=Atlas(parcellation,
   names, hemispheres), labels={"metric": m})` for every measure. neurovrai builds
   the `Atlas` from its own atlas (JHU, Harvard-Oxford) — `Atlas.from_files` with a
   label table reads SIGMA's format only.
4. `write_readout_results(out, tests, clusters, analysis_type=..., space="MNI152NLin6Asym",
   inference="2-D TFCE" | "3-D TFCE", mask_name="TBSS skeleton" | "brain mask", ...)`.
5. Keep `effect_size.py` maps if wanted, listed as `maps` of kind `effect`. The
   reported effect is the read-out's whole-mask d with its exact CI.

**Edgewise connectivity** (`connectome/group_analysis.compute_group_difference`) — an
`elements` table, one row per edge of the upper triangle (`element` = `"A -- B"`):
`stat` (t, `StatName` "t"), `p_value` with `PKind` "fdr" (BH), the uncorrected p as
an extra column with `PKind` "uncorrected", the group means, `n_a` / `n_b`, and an
`effect_size` (d, or the difference in Fisher z with `EffectMeasure` "z_diff").
`analysis_type` "connectome"; `correction.family` "the N·(N−1)/2 edges".

**NBS** (`compute_network_based_statistic`) — an `elements` table, one row per
component: `element` (component id), `stat` = edges in the component (`StatName`
"component size"), `p_value` with `PKind` "fwe" (NBS controls FWE over components),
and the direction tested. A second table (`role` "other") lists every edge of every
component with its signed t. `analysis_type` "nbs".

**Graph metrics** (`connectome/batch_graph_metrics.py`) — there are no group tests,
so `descriptives` tables only (`group_node_metrics`, `group_global_metrics`), and
`effect = {measure: null, reason: "descriptive; no group test"}`. Group tests, when
added, are `elements` (one row per ROI × metric).

**WMH group comparison** (`scripts/wmh_group_comparison.py`) — a `tests` table: one
row per metric; `contrast` "poor_control > good_control"; `cohens_d` as
`effect_size` with a CI (to add); one p value as the standard `p_value` with its
`PKind` (and say which test), the other as an extra column; `n_a` / `n_b` from the
`*_n` columns.

**Per-subject tables** (`t1t2ratio_summary.csv`, `wmh_summary.csv`) — `descriptives`,
or `other` with a `rows` sentence.

**Provenance** — `provenance([...])` with neurovrai's name, version and commit first,
then FSL. The commit can be read from the installed package's PEP 610
`direct_url.json`, as `neurofaune.provenance.package_provenance` does.

### 4.4 Done when

- neurovrai's test suite builds one synthetic analysis per family above and asserts
  `check(folder)` passes (neurofaune's `tests/unit/test_tbss_readout.py` builds a
  randomise run without FSL, which neurovrai can copy);
- `neurofaune results check` passes on a real neurovrai study's results root;
- the bugs in §4.2 are fixed or deliberately kept and recorded.

## 5. Study scripts

A study's own orchestration (the cuprizone study's `analyses/code/`) writes the same
folders. Scripts that call `read_randomise` pass its tables to
`write_readout_results`. Scripts that build their own tables write their own column
dictionaries. Study records such as a pre-registration id or a finding id go in
`references`; the spec does not know about them.

## 6. Writing a new producer for a new analysis type

Choose the table roles first: what one row is. Then map columns to standard terms
and fill in the contract (§6 of the specification). If a standard term is missing,
propose it in neurofaune with a `spec_version` bump and a `CHANGELOG.md` entry —
never repurpose an existing term.
