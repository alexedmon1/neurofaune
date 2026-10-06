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
