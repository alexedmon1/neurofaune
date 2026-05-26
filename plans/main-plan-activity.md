# neurofaune implementation-loop — activity log

Append 1–3 lines per loop iteration (newest at top). See `main-plan.md`.

---

## 2026-05-26 — Loop iteration: spearman_matrix refactor (PROMOTED)
- Change: replace `np.apply_along_axis(stats.rankdata, 0, data)` with native `stats.rankdata(data, axis=0)`.
- Candidates compared in worktrees against the `connectivity_spearman_matrix` golden:
  - B `axis=0` (faithful) → gate ✅ → **promoted to main** (`5df2bdb`), CI green.
  - C `axis=1` (wrong axis) → gate ❌ (regression drift) → discarded. Note: both passed all 119 unit tests; only the golden caught C.
- Boundary held both ways. No version bump (internal refactor).

## 2026-05-26 — Loop scaffold established
- Added the gate (`make check` = unit + regression), advisory ruff/mypy, pre-commit, and GitHub Actions CI.
- Added the frozen-contract harness in `tests/regression/` (fixtures/ + golden/ + equivalence helper).
- Baseline gate: 119 unit + 2 regression tests passing on `main`.
- No candidate change yet; next loop mints the first real golden (MR-1 / CovNet).
- `$PKG` hash: <fill on commit>
