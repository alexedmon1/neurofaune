---
author: Alex Edmondson
affiliation: CCHMC
email: alex.edmondson@cchmc.org
package: neurofaune
phase: package-development
repo: https://github.com/alexedmon1/neurofaune
pkg_dir: /home/edm9fd/sandbox/neurofaune
pkg_venv: /home/edm9fd/sandbox/neurofaune/.venv/bin/python
worktrees_dir: /home/edm9fd/sandbox/.worktrees
---

<!-- AI Instructions:
Package-development IRL project for the neurofaune library itself (NOT a study).
This loop "controls the boundary of an implementation": every change to the
package must pass a FROZEN CONTRACT before it is allowed to flow into research.

- $PKG (this repo) holds the library, the gate, and this plan — all git-tracked.
- The CONTRACT for a change = its public interface + invariants + frozen
  regression fixtures/golden in $REF. Candidates are compared against it.
- THE GATE is `make check` (unit + regression). It is the ONLY promotion signal.
  ruff/black/mypy are advisory (`make advisory`) and never block.
- Competing implementations live in separate git worktrees under $WT so the
  main checkout stays clean and candidates are compared side-by-side.
- PROMOTION boundary: a change reaches research ONLY as a git tag. Research IRL
  projects pin `neurofaune @ git+...@<tag>`. Untagged work never reaches research.
- Never weaken a golden to make a candidate pass. Changing a golden is a
  separate, deliberate, reviewed commit (see "Updating a golden" below).
-->

# neurofaune — Implementation-Testing Loop

## 📁 Paths — Single source of truth

- **`$PKG`** — `/home/edm9fd/sandbox/neurofaune` — the library repo (this repo)
- **`$PLAN`** — `$PKG/plans` — this plan, activity log, CSV log
- **`$REF`** — `$PKG/tests/regression` — frozen contract: `fixtures/` + `golden/` + tests
- **`$PY`** — `$PKG/.venv/bin/python`
- **`$WT`** — `/home/edm9fd/sandbox/.worktrees` — one worktree per candidate implementation

Rule: every section below uses these shorthands. New absolute path → add it here first.

---

## 🔧 First-Time Setup — Run once

1. **Sync dev env:** `cd $PKG && make sync`
2. **Confirm the gate is green on a clean tree:** `make check`
3. **(Optional) enable incremental hygiene:** `uv run pre-commit install`
4. **Record the baseline:** tag the current trusted state so research has something to pin:
   ```bash
   git tag -a v0.1.0-alpha -m "baseline before implementation loop" && echo "(push tag only when ready)"
   ```

---

## ✅ Before Each Loop

- **Clean tree:** `cd $PKG && git status`
- **Gate green on main first:** `make check` (never start a candidate from a red baseline)
- **Define the BOUNDARY for this change** in the loop task below: the public
  interface that must not change, the invariants that must hold, and which
  `$REF` fixture/golden encodes "same result".
- If no golden covers the behavior you're about to change, **mint it from the
  trusted implementation first** (see "Updating a golden"), commit it, *then* start.

---

## 🔁 Instruction Loop — Define one change per iteration

<!-- 👤 AUTHOR AREA: edit each loop. -->

### Loop task (current)

- **Change:** <!-- e.g. MR-1: refactor run_classification_analysis.py into ClassificationAnalysis -->
- **Boundary (must NOT change):**
  - Public interface: <!-- signatures / CLI flags callers rely on -->
  - Invariants: <!-- idempotency, config-driven paths, output schema -->
  - Result contract: <!-- which $REF golden encodes "same numbers" -->
- **Candidates:** <!-- e.g. A = current main (baseline), B = class-based refactor -->
- **Expected promotion:** <!-- new tag vX.Y.Z if green -->

### Per-candidate procedure

1. **Spin up a worktree** (keeps main clean, allows side-by-side compare):
   ```bash
   git -C $PKG worktree add $WT/neurofaune-<candidate> -b loop/<candidate>
   cd $WT/neurofaune-<candidate> && make sync
   ```
2. **Implement** the candidate in its worktree.
3. **Run the gate:** `make check`
   - The regression tests compare this candidate's output to the **same**
     committed golden as every other candidate — that is the boundary.
4. **Advisory (optional):** `make advisory` — note new lint/type findings, never blocking.
5. **Record the result** in the CSV log (one row per candidate run).

### Choosing & promoting the winner

- Pick the green candidate that best meets the boundary + design goals.
- Merge it to `main` in `$PKG`, re-run `make check` on main.
- **Tag it:** `git tag -a vX.Y.Z -m "<change>"` — this is the promotion boundary.
- Bump the pin in dependent research projects: `neurofaune @ git+https://github.com/alexedmon1/neurofaune.git@vX.Y.Z`.
- Remove spent worktrees: `git -C $PKG worktree remove $WT/neurofaune-<candidate>`.

### Updating a golden (deliberate, rare)

Only when you have *decided* the new behavior is correct:
```bash
NEUROFAUNE_UPDATE_GOLDEN=1 make regression   # regenerates golden/*.npy
git add tests/regression/golden && git commit -m "golden: <why the result changed>"
```
Never do this just to make a red candidate pass.

### One-Time Instructions
<!-- 👤 AUTHOR AREA: add tasks; move to Completed once done. -->

- [ ] Mint first real golden from a trusted module (candidate for MR-1 / CovNet)
- [ ] Confirm CI is green on GitHub after first push

#### Completed (don't re-run)
<!-- Move checked items here with date -->

---

## 📝 After Each Loop

- **Append to activity log** (`$PLAN/main-plan-activity.md`): change, candidates run,
  which won, gate result, new tag (if any), UTC timestamp, `$PKG` git hash.
- **Append to CSV log** (`$PLAN/main-plan-log.csv`):
  `timestamp,change,candidate,gate_result,advisory_notes,tag,pkg_hash`
- **Commit `$PKG`:** code + plan/log edits. Commit message: `loop: <change> — <outcome>`.
- **Push & tag** only when you intend the change to reach research (outward step).
- **Feedback to AUTHOR:** what passed, what drifted, any boundary that needs revisiting.

---

## 🧪 Test tiers & fixtures

Different domains are reproducible in different ways, so the contract has two tiers.

**Canonical fixtures** — `tests/regression/fixtures.py` is the single registry of
seeded synthetic inputs (subjects×ROIs, connectome, t-map volume, …). Every
regression test and every candidate draws from it, so candidates are comparable.
Inputs regenerate from a generator+seed; only the *golden output* is committed.
Reserve tiny downsampled-real fixtures only where real-data quirks matter; never
put big real data behind the gate.

**Gate tier** (`make check`, blocking, hermetic, every PR) — deterministic boundaries:
- `connectivity/` — matrix/graph math (e.g. `spearman_matrix`, `fisher_z_transform`)
- `voxelwise/` — effect-size / GLM math on small volumes (e.g. `compute_partial_etasq_from_tstat`)
- `qc/` — FD/DVARS, confounds, Dice (add as needed)
- `preprocess/` — **workflow assembly**: freeze the nipype graph / command-lines via
  `_workflow.py` and assert a refactor still emits them — *without running ANTs/FSL*.
  This is how a preprocessing refactor is gated cheaply.

**Integration tier** (`make integration`, slow, local/HPC, **not** PR-blocking, run before a release tag) —
end-to-end preprocessing/stats through ANTs/FSL, which aren't bit-reproducible. Compare
**derived metrics** (Dice, map correlation, intensity summary via `_derived.py`) within
loose tolerance — never raw voxels. Tests skip cleanly when tools are absent.

Mint/refresh any golden deliberately: `NEUROFAUNE_UPDATE_GOLDEN=1 make regression`
(or `make integration`), then commit the changed golden in its own reviewed commit.

## 📌 Conventions

- **The gate is the only promotion signal.** Green `make check` on CI == safe to tag.
- **Research pins tags, never `main`.** A research project upgrades by bumping its pin.
- **Goldens are sacred.** Changing one is a separate reviewed commit with a stated reason.
- **Fixtures stay tiny** (<512 KB, enforced by pre-commit) so the contract is committable.
