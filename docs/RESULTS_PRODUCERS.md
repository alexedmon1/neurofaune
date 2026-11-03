# Producing results — moved to neurobox-dev

How neurofaune, neurovrai, source-analytics and study scripts write analysis folders that
meet the results specification is now in neurobox-dev's `docs/RESULTS_PRODUCERS.md`
(https://github.com/alexedmon1/neurobox-dev), which moved there with the specification on
2026-11-03. neurofaune's own producers (TBSS, VBM and voxelwise fMRI through
`analysis.stats.readout_results`) are unchanged and keep importing `neurofaune.results`.
