# Results specification — moved to Lumière

The results specification (`neurofaune.results`, 0.3.0), its JSON Schemas, checker and
writer moved out of neurofaune, unchanged, on 2026-11-03, into **Lumière** (Lumière until 2026-11-10): https://github.com/alexedmon1/lumiere, `docs/RESULTS_SPEC.md`.
neurofaune now pins its 0.4.0, which adds EEG and what cuprizone's first real run of the
spec found missing; neurofaune's own folders change only in saying what each MDE percent
is a percent of (`PercentOf`), and declare 0.4.0.

It moved so that tools other than neurofaune (source-analytics, source-localization,
neurovrai) can write and check results without neurofaune's imaging stack. Nothing
changes for neurofaune or its studies: `neurofaune.results` re-exports the package
`lumiere`, `neurofaune results check` and `python -m neurofaune.results check` still
work, and every `analysis.json` keeps `"spec": "neurofaune.results"`.
