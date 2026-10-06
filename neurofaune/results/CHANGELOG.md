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

