# neurofaune.results — specification changelog

## 0.1.0 — 2026-10-06 (draft)

First version: analysis folders (`analysis.json`, `provenance.json`, tables with
BIDS-style column dictionaries), the standard column vocabulary, the reporting
contract for `tests` / `clusters` / `elements` tables, JSON Schemas and a
standard-library checker. Written first by neurofaune's TBSS and VBM /
voxelwise-fMRI read-outs (`neurofaune.analysis.stats.readout_results`).

Amended while still a draft (2026-10-06; every addition optional, nothing removed):
a `background` map kind; p maps' `values` (p / one_minus_p); maps' `facet` and `axes`
(declared orientation) and the analysis's `display.plane`; the per-subgroup terms
`subgroup_effect` / `subgroup_n` with the `Subgroup` qualifier; a tests row giving an
effect must give both interval bounds.


## 0.2.0 — 2026-10-06 (draft)

Identity and vocabulary (RESULTS_SPEC.md §3.1), so a reader can group analyses by what
they are without matching on names:

- an analysis is one modality and one method; `modality` is required and one of
  `anat`, `dwi`, `func`, `msme`, `mrs`, `asl`, `multimodal` (with `modalities`);
- `id` is `<modality>/<analysis_type>/<name>` (`spec.analysis_id` builds one);
- `analysis_type` gains `decoding`, `radiomics`, `spectroscopy`;
- measures take their canonical names from the measure vocabulary
  (`vocab/measures.json`); an unknown measure is warned about, a known one under another
  spelling is an error; a contract table's measures are those `analysis.json` lists;
- `references` labels `registration`, `finding` and `doi` have an agreed meaning.

- **runs** (§3.2): a folder is one run of an analysis, `run: {id, label, supersedes}`
  (required; the writer fills in a time-stamped id when none is given). Runs of one id
  share modality and analysis type; a test held by two runs is an error unless the later
  supersedes the earlier (per test); corrections stay with their runs. `check` checks
  runs across a results root (`check_runs`); the writer checks a new run against its
  siblings. Recommended layout `<root>/<id>/<run id>/` (`spec.run_folder`).

The checker reads 0.1 and 0.2, each folder by the rules of the version it declares.
neurofaune's TBSS and randomise writers now write 0.2 ids (`dwi/tbss/<name>`,
`anat/vbm/<name>`, `func/voxelwise/<name>`).


## 0.3.0 — 2026-10-20 (draft)

The minimum detectable effect (RESULTS_SPEC.md §5.1): beside each test, the smallest
difference its design could have detected, so a finding's size -- and a null -- is read
against what the study could see. Asked for by DUET (OPEN.md #151); the same names and
formula are written by neurofaune, source-analytics and neurovrai.

- standard terms `se` (standard error of the estimate, from the fitted model),
  `mde` (`(t(1 - Alpha/2, df) + t(Power, df)) * se`, normal quantiles without a finite
  df; requires the new `Power` qualifier and `Alpha`), `mde_percent` (percent of the
  reference group's mean) and `mde_note` (why an `mde` is not given, starting
  `not computed: `);
- the producers write power 0.80 and two-sided alpha 0.05, nominal (no
  multiple-comparison correction);
- the checker, for folders that declare 0.3: an `mde` is positive and its `Power` /
  `Alpha` lie in (0, 1); a row without an `mde` carries an `mde_note`, and a row with one
  leaves it empty; `mde_percent` is positive and only beside an `mde`; `se` is >= 0; a
  tests table with neither `mde` nor `mde_note` is warned about. A folder declaring an
  earlier version may not use the 0.3 terms.

The checker reads 0.1, 0.2 and 0.3, each folder by the rules of the version it declares.
neurofaune's randomise read-out (TBSS, VBM, voxelwise fMRI) writes 0.3: `whole_se`,
`whole_mde`, `whole_mde_percent`, `whole_mde_note` on tests rows (the test on each
subject's whole-mask mean), `mde_note` "not computed: cluster selected by the test" on
cluster rows, and a caveat that the voxelwise maps' MDE is not computed (randomise
corrects them).
