# neurofaune cleanup — pending dead-code sweep (judgment needed)

Follow-up to the legacy-Bruker→BIDS removal + first dead-code sweep (commit
`8593dfd`, ships in v0.5.0-alpha). The items below were flagged by an audit but
NOT removed: each needs a quick verification (grep importers, check
README/CLI/docs refs) and a human call — some may be hand-run tools.

**Before deleting any item:** confirm zero importers, no README/CLI/CI reference,
and it isn't a tool you still run by hand. Run `make check` after deletions.

## Likely-stray scripts

Root-level (duplicates / ad-hoc, not under `scripts/` or `tests/`, unreferenced):
- `build_templates.py` — duplicate of `scripts/build_templates.py`
- `batch_anatomical_preprocessing.py` — overlaps `scripts/batch_preprocess_anat.py`
- `test_adaptive_skull_strip.py`, `test_dwi_workflow.py`, `test_msme_workflow.py`,
  `test_workflows.py` — ad-hoc test scripts at repo root (not in the `tests/` suite)

`scripts/` clusters:
- `scripts/dev_registration/` — numbered R&D (`001_*`…`009_*`) + ~16 PNGs + README;
  the useful part graduated into `neurofaune/registration/`
- Phase-numbered CovNet cluster superseded by `run_covnet_*.py`:
  `scripts/covnet_common.py`, `covnet_prepare.py`, `covnet_nbs.py`,
  `covnet_territory.py`, `covnet_whole_network.py`
- One-offs: `generate_anat_qc_retroactive.py`, `generate_func_qc_retroactive.py`,
  `migrate_qc_structure.py`, `migrate_qc_to_sub_dir.py`, `list_3d_subjects.py`,
  `check_and_scale_sigma_atlas.py`, `process_bruker_session.py`,
  `extract_bruker_mrs.py`, `test_3d_t2w_preprocess.py`
- TBSS thin wrappers superseded by `run_tbss_analysis.py`:
  `run_tbss_prepare.py`, `run_tbss_stats.py`, `run_template_tbss_prepare.py`,
  `run_template_tbss_pipeline.sh`

## Deprecation shim packages (re-export stubs, zero importers)
- `neurofaune/analysis/covnet/__init__.py` → re-exports `network.covnet`
- `neurofaune/analysis/reporting/__init__.py` → re-exports `neurofaune.reporting`
  (keep for back-compat, or drop?)

## Stale docs
- `docs/archive/` — `EXPERIMENTAL_SKULL_STRIPPING_CHANGES.md`, `PHASE1_SUMMARY.md`,
  `SKULL_STRIPPING_COMPARISON.md`, `SKULL_STRIPPING_TROUBLESHOOTING.md`
- `docs/plans/TBSS_IMPLEMENTATION_PLAN.md` — refs non-existent
  `scripts/run_tbss_skeleton.py`, `scripts/run_tbss_slice_qc.py`
- `docs/plans/ANAT_PREPROCESSING_WORKFLOW_PLAN.md` — refs non-existent
  `scripts/build_anat_templates.py`
- Root historical docs (Nov 2025–Apr 2026, superseded by README/CAPABILITIES):
  `BATCH_PROCESSING_GUIDE.md`, `WORKFLOWS_SUMMARY.md`, `PREPROCESSING_EXPLORATION.md`,
  `MODERNIZATION_PLAN.md`, `STATUS.md`, `ROADMAP.md`
- `README.md` — references non-existent `scripts/prepare_auc_lookup.py` (fix the line)

## Minor
- `neurofaune/preprocess/cli.py` module docstring still says it only "exposes the
  config-driven Bruker→BIDS converter" — add the `capabilities` subcommand.

## KEEP — looks dead but is live (do NOT delete)
- `neurofaune/connectome/matrices.py` — imported by
  `network/classification/data_prep.py` (last live remnant of `connectome/`)
- `neurofaune/reporting/index_generator.py`, `section_renderers.py` — used via
  relative imports from `reporting/__init__.py`, `registry.py`, `discover.py`
- `neurofaune/preprocess/qc/{anat/anat_qc, msme/msme_qc, func/confounds_qc,
  dwi/multishell_qc}.py` — re-exported via package `__init__`, used by workflows
- `neurofaune/preprocess/utils/bet4animal.py` — live
- Shared `bruker_convert.py` helpers (`get_bruker_method`, `classify_scan`,
  `extract_bids_metadata`, `extract_bvec_bval`, `convert_bruker_to_nifti`,
  `inventory_session`, `select_best_*_from_inventory`) — used by `utils/bids.py`
