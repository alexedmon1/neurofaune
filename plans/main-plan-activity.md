# neurofaune implementation-loop — activity log

Append 1–3 lines per loop iteration (newest at top). See `main-plan.md`.

---

## 2026-05-26 — Loop scaffold established
- Added the gate (`make check` = unit + regression), advisory ruff/mypy, pre-commit, and GitHub Actions CI.
- Added the frozen-contract harness in `tests/regression/` (fixtures/ + golden/ + equivalence helper).
- Baseline gate: 119 unit + 2 regression tests passing on `main`.
- No candidate change yet; next loop mints the first real golden (MR-1 / CovNet).
- `$PKG` hash: <fill on commit>
