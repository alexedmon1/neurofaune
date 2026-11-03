# Results specification — moved to neurobox-dev

The results specification (`neurofaune.results`, 0.3.0), its JSON Schemas, checker and
writer moved out of neurofaune, unchanged, on 2026-11-03, into **neurobox-dev** (a
working name): https://github.com/alexedmon1/neurobox-dev, `docs/RESULTS_SPEC.md`.

It moved so that tools other than neurofaune (source-analytics, source-localization,
neurovrai) can write and check results without neurofaune's imaging stack. Nothing
changes for neurofaune or its studies: `neurofaune.results` re-exports the package
`neurobox`, `neurofaune results check` and `python -m neurofaune.results check` still
work, and every `analysis.json` keeps `"spec": "neurofaune.results"`.
